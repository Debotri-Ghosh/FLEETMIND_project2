"""The one genuinely LLM-driven step in FleetMind.

Everything else in `agents/` is deterministic rule-based Python. This module
adds a real model call on top of it: the model is given no raw signal and no
freedom to compute, only four callable tools (`agents/tools.py`) that return
numbers the existing rule-based code produced. Its job is to reason over those
numbers and write the two pieces of prose a technician actually reads -- a
diagnosis narrative and a maintenance recommendation -- and to say honestly
when the physical corroboration is weak.

Why this split, rather than letting the model do the analysis: the risk score
is what flips equipment into `maintenance` and files a service request, and
that must stay reproducible, inspectable and testable. So the model never
originates a figure. The system prompt forbids it, the tools are the only
source of numbers, and the structured output is checked against the tool
results before anything is stored (`verify_numbers_are_grounded`).

**Provider: Groq** (`openai/gpt-oss-120b` by default), chosen by the project
owner because that is the API key available. An earlier version of this module
targeted Claude via the Anthropic SDK; the grounding design is unchanged and
provider-independent, but the call shape is not, hence the rewrite.

One Groq-specific constraint shapes the loop below: **JSON mode and tool
calling cannot be combined** -- the API returns `400 json mode cannot be
combined with tool/function calling`. So the conversation runs in two phases:
tools first with no `response_format`, then one final tool-free call with
`response_format={"type": "json_object"}` to collect the structured answer.
Verified against the live API, not assumed.

Generated on demand per service request and cached in `llm_narratives`, so a
demo costs one call per flagged machine rather than one per page view.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from fleetmind.agents.tools import EvidenceContext, openai_tool_definitions, run_tool
from fleetmind.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 4000
# Four tools, and the band-energy one is worth a couple of probes. Anything
# past this is a loop, not reasoning.
MAX_TOOL_ITERATIONS = 8


class LLMNotConfigured(RuntimeError):
    """The LLM layer cannot run: no API key, or the SDK is not installed.

    Raised rather than silently degrading to a canned string, and mapped to a
    503 by the API -- "correctly built but unconfigured" is a different thing
    from "broken", and the rule-based decision exists either way.
    """


SYSTEM_PROMPT = """\
You are a vibration-analysis engineer writing up a predictive-maintenance \
finding for an industrial bearing on a rotating machine (motor, pump, fan or \
compressor) at a customer's manufacturing site.

A 1D CNN classifier has already flagged one vibration window, and a \
rule-based agent chain has already computed a risk score and urgency. Your job \
is NOT to redo that analysis. Your job is to explain it, in the plain prose a \
maintenance planner and a field technician can act on.

Hard rules:
1. Every number you state must come from a tool result in this conversation, \
quoted exactly as the tool returned it. Never estimate, round differently, \
extrapolate or invent a figure. In particular do NOT restate a value in another \
unit or as a percentage -- writing "0.0325%" for a returned 0.000325 is \
originating a number, which is not yours to do; quote the 0.000325 and describe \
it in words instead. If you want a number you do not have, call a tool for it.
2. Do not contradict the risk score or urgency from get_risk_assessment. You \
may explain them; you may not revise them.
3. The characteristic defect frequencies are bearing geometry, not an \
explanation of what the classifier looked at. Never write that the model \
"detected" or "looked at" a frequency.
4. Check the corroboration honestly. Call measure_spectral_band_energy on the \
matched defect band and compare it to where the spectrum's energy actually \
sits. If the energy is not concentrated on the defect frequency, say so \
plainly and let that temper your confidence in the narrative -- a weak match \
reported as strong is worse than no narrative at all.
5. A human decides. Recommend an action; never state that an action has been \
or will be taken.

Call the tools first, then answer. Write for a competent technician: concrete, \
unhedged about what the data says, explicit about what it does not say.\
"""

# Phase 2 of the loop. The shape is spelled out as prose because Groq's JSON
# mode guarantees *valid JSON*, not JSON matching a schema -- so it has to be
# stated here and validated on our side in `_parse_final`.
FINAL_INSTRUCTION = """\
Now write the finding. Reply with ONLY a JSON object, no prose around it, with \
exactly these keys:

{
  "diagnosis_narrative": "2-4 sentences: what the signal shows and what fault it points to, citing the tool-supplied figures.",
  "maintenance_recommendation": "2-4 sentences: what to do, in what timeframe, and what to check or bring. Must match the urgency from get_risk_assessment.",
  "evidence_cited": ["each figure you relied on, as 'name = value unit'"],
  "corroboration": "one of: strong | partial | weak | none"
}

Use "none" for corroboration only when there is no defect frequency to check \
against (a Normal prediction). Every number in your prose must have come from \
a tool result.\
"""

REQUIRED_KEYS = (
    "diagnosis_narrative",
    "maintenance_recommendation",
    "evidence_cited",
    "corroboration",
)
VALID_CORROBORATION = {"strong", "partial", "weak", "none"}


@dataclass
class NarrativeResult:
    diagnosis_narrative: str
    maintenance_recommendation: str
    evidence_cited: list[str]
    corroboration: str
    model: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    ungrounded_numbers: list[str] = field(default_factory=list)


def get_client(api_key: str | None = None):
    """Build a Groq client, or explain why we cannot.

    The key is read from `backend/.env` via Settings rather than relying on the
    SDK's own environment lookup, because every other backend command in this
    project is configured from that file.
    """
    try:
        from groq import Groq
    except ImportError as exc:  # the `llm` extra was not installed
        raise LLMNotConfigured(
            "The groq SDK is not installed. Run "
            '`pip install -e ".[llm]"` from backend/ to enable narrative '
            "generation; the rule-based decision layer works without it."
        ) from exc

    key = api_key or get_settings().groq_api_key
    if not key:
        raise LLMNotConfigured(
            "GROQ_API_KEY is not set. Add it to backend/.env to enable LLM "
            "narrative generation; the rule-based decision layer works "
            "without it."
        )
    return Groq(api_key=key)


def _user_prompt(ctx: EvidenceContext) -> str:
    """Deliberately carries no measurements -- only identifiers and context.

    Anything numeric the model needs must come back from a tool, so that the
    grounding check below is meaningful.
    """
    return (
        f"Service request #{ctx.service_request_id} was filed automatically for "
        f"{ctx.equipment_name or 'an unnamed machine'} at "
        f"{ctx.site_name or 'an unnamed site'}.\n\n"
        f"It was triggered by vibration window #{ctx.window_id}, recorded from "
        f"the {ctx.sensor_position} accelerometer. The classifier "
        f"({ctx.model_version or 'unversioned'}) predicted "
        f"'{ctx.predicted_label}'.\n\n"
        "Use the tools to gather the figures, then write the finding."
    )


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def verify_numbers_are_grounded(
    text: str, tool_results: list[str], tolerance_digits: int = 2
) -> list[str]:
    """Return any number in `text` that does not appear in a tool result.

    A deliberately blunt check, and it is allowed to be: its purpose is to
    catch a hallucinated measurement, not to grade prose. Numbers are matched
    against the raw tool-result JSON both verbatim and rounded, and bare
    integers up to 24 are ignored as counts and HP figures rather than
    measurements. Timeframes need no carve-out: "within 48 hours" comes from
    `rule_based_action`, which is itself a tool result.

    The result is stored alongside the narrative rather than used to reject it,
    so a reviewer can see whether the guard ever fires on real output.
    """
    haystack = " ".join(tool_results)
    haystack_numbers = set(_NUMBER.findall(haystack))
    rounded: set[str] = set()
    for n in haystack_numbers:
        try:
            value = float(n)
        except ValueError:  # pragma: no cover - regex guarantees parseability
            continue
        for digits in range(tolerance_digits + 1):
            rounded.add(f"{round(value, digits):g}")
            rounded.add(str(round(value, digits)))

    ungrounded = []
    for token in _NUMBER.findall(text):
        if token in haystack_numbers or token in rounded:
            continue
        try:
            if float(token) <= 24 and "." not in token:
                continue  # timeframes, HP, small counts
        except ValueError:  # pragma: no cover
            continue
        ungrounded.append(token)
    return ungrounded


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    return (
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def generate_narrative(
    ctx: EvidenceContext,
    client: Any | None = None,
    model: str | None = None,
) -> NarrativeResult:
    """Run the tool loop, then collect the structured write-up.

    A manual loop rather than an agent framework: every tool here is bound to
    one `EvidenceContext`, and we want the full call-and-result trail persisted
    for audit, which the manual loop hands us directly.
    """
    settings = get_settings()
    client = client if client is not None else get_client()
    model = model or settings.groq_model or DEFAULT_MODEL
    tools = openai_tool_definitions()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(ctx)},
    ]
    trail: list[dict[str, Any]] = []
    raw_results: list[str] = []
    input_tokens = output_tokens = 0

    # ── phase 1: let the model gather its figures ──────────────────────
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=MAX_TOKENS,
        )
        used_in, used_out = _usage(response)
        input_tokens += used_in
        output_tokens += used_out

        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])

        if not tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
        )

        for call in tool_calls:
            name = call.function.name
            # Arguments arrive as a JSON string; parse, never string-match.
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            try:
                result = run_tool(ctx, name, arguments)
                is_error = False
            except (KeyError, TypeError, ValueError) as exc:
                result = f"Error running {name}: {exc}"
                is_error = True

            trail.append({"tool": name, "input": arguments, "result": result})
            if not is_error:
                raw_results.append(result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    else:
        raise RuntimeError(
            f"Narrative generation did not converge within {MAX_TOOL_ITERATIONS} tool rounds."
        )

    # ── phase 2: structured answer, no tools ───────────────────────────
    # Groq rejects `response_format` alongside `tools` (400: "json mode cannot
    # be combined with tool/function calling"), so the shape is collected in a
    # separate tool-free call. The tool results are already in `messages`.
    messages.append({"role": "user", "content": FINAL_INSTRUCTION})
    final = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=MAX_TOKENS,
    )
    used_in, used_out = _usage(final)
    input_tokens += used_in
    output_tokens += used_out

    return _parse_final(
        final.choices[0].message.content,
        model,
        trail,
        raw_results,
        input_tokens,
        output_tokens,
    )


def _parse_final(
    text: str | None,
    model: str,
    trail: list[dict[str, Any]],
    raw_results: list[str],
    input_tokens: int,
    output_tokens: int,
) -> NarrativeResult:
    if not text:
        raise RuntimeError("The model returned no content to parse a narrative from.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"The model did not return valid JSON: {exc}") from exc

    # JSON mode guarantees valid JSON, not the right keys -- check ourselves.
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise RuntimeError(f"Narrative JSON is missing required keys: {missing}")

    corroboration = str(data["corroboration"]).strip().lower()
    if corroboration not in VALID_CORROBORATION:
        logger.warning("unexpected corroboration value %r; storing as 'weak'", corroboration)
        corroboration = "weak"

    prose = f"{data['diagnosis_narrative']} {data['maintenance_recommendation']}"
    ungrounded = verify_numbers_are_grounded(prose, raw_results)
    if ungrounded:
        logger.warning(
            "LLM narrative contains %d number(s) absent from tool results: %s",
            len(ungrounded),
            ungrounded,
        )

    evidence = data.get("evidence_cited") or []
    if isinstance(evidence, str):
        evidence = [evidence]

    return NarrativeResult(
        diagnosis_narrative=str(data["diagnosis_narrative"]),
        maintenance_recommendation=str(data["maintenance_recommendation"]),
        evidence_cited=[str(item) for item in evidence],
        corroboration=corroboration,
        model=model,
        tool_calls=trail,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ungrounded_numbers=ungrounded,
    )
