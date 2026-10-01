"""The one genuinely LLM-driven step in FleetMind.

Everything else in `agents/` is deterministic rule-based Python. This module
adds a real Claude call on top of it: the model is given no raw signal and no
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

Model: `claude-sonnet-5`, chosen by the project owner. Generated on demand per
service request and cached in `llm_narratives`, so a demo costs one call per
flagged machine rather than one per page view.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from fleetmind.agents.tools import TOOL_DEFINITIONS, EvidenceContext, run_tool
from fleetmind.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
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
1. Every number you state must come from a tool result in this conversation. \
Never estimate, round differently, extrapolate or invent a figure. If you want \
a number you do not have, call a tool for it.
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

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnosis_narrative": {
            "type": "string",
            "description": (
                "2-4 sentences explaining what the signal shows and what fault "
                "it points to, citing the tool-supplied figures."
            ),
        },
        "maintenance_recommendation": {
            "type": "string",
            "description": (
                "2-4 sentences: what to do, in what timeframe, and what to "
                "check or bring. Must match the urgency from get_risk_assessment."
            ),
        },
        "evidence_cited": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each figure relied on, as 'name = value unit'.",
        },
        "corroboration": {
            "type": "string",
            "enum": ["strong", "partial", "weak", "none"],
            "description": (
                "How well spectral energy supports the predicted defect "
                "frequency. 'none' when there is no defect frequency (Normal)."
            ),
        },
    },
    "required": [
        "diagnosis_narrative",
        "maintenance_recommendation",
        "evidence_cited",
        "corroboration",
    ],
    "additionalProperties": False,
}


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
    """Build an Anthropic client, or explain why we cannot.

    The key is read from `backend/.env` via Settings rather than relying on the
    SDK's own environment lookup, because every other backend command in this
    project is configured from that file.
    """
    try:
        import anthropic
    except ImportError as exc:  # the `llm` extra was not installed
        raise LLMNotConfigured(
            "The anthropic SDK is not installed. Run "
            '`pip install -e ".[llm]"` from backend/ to enable narrative '
            "generation; the rule-based decision layer works without it."
        ) from exc

    key = api_key or get_settings().anthropic_api_key
    if not key:
        raise LLMNotConfigured(
            "ANTHROPIC_API_KEY is not set. Add it to backend/.env to enable "
            "LLM narrative generation; the rule-based decision layer works "
            "without it."
        )
    return anthropic.Anthropic(api_key=key)


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


def generate_narrative(
    ctx: EvidenceContext,
    client: Any | None = None,
    model: str | None = None,
) -> NarrativeResult:
    """Run the tool-use loop and return the model's grounded write-up.

    A manual loop rather than the SDK's beta tool runner: every tool here is
    bound to one `EvidenceContext`, and we want the full call-and-result trail
    persisted for audit, which the manual loop hands us directly.
    """
    settings = get_settings()
    client = client if client is not None else get_client()
    model = model or settings.anthropic_model or DEFAULT_MODEL

    messages: list[dict[str, Any]] = [{"role": "user", "content": _user_prompt(ctx)}]
    trail: list[dict[str, Any]] = []
    raw_results: list[str] = []
    input_tokens = output_tokens = 0

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            thinking={"type": "adaptive"},
            output_config={
                "effort": "medium",
                "format": {
                    "type": "json_schema",
                    "schema": OUTPUT_SCHEMA,
                },
            },
            messages=messages,
        )
        input_tokens += getattr(response.usage, "input_tokens", 0) or 0
        output_tokens += getattr(response.usage, "output_tokens", 0) or 0

        if response.stop_reason == "refusal":
            raise RuntimeError(
                "Claude declined to answer "
                f"({getattr(response.stop_details, 'category', 'unknown')})."
            )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return _parse_final(response, model, trail, raw_results, input_tokens, output_tokens)

        messages.append({"role": "assistant", "content": response.content})

        tool_results: list[dict[str, Any]] = []
        for block in tool_uses:
            # Tool inputs arrive as parsed JSON; never string-match them.
            arguments = dict(block.input or {})
            try:
                result = run_tool(ctx, block.name, arguments)
                is_error = False
            except (KeyError, TypeError, ValueError) as exc:
                result = f"Error running {block.name}: {exc}"
                is_error = True
            trail.append({"tool": block.name, "input": arguments, "result": result})
            if not is_error:
                raw_results.append(result)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    **({"is_error": True} if is_error else {}),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(
        f"Narrative generation did not converge within {MAX_TOOL_ITERATIONS} tool rounds."
    )


def _parse_final(
    response: Any,
    model: str,
    trail: list[dict[str, Any]],
    raw_results: list[str],
    input_tokens: int,
    output_tokens: int,
) -> NarrativeResult:
    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise RuntimeError("Claude returned no text block to parse a narrative from.")
    data = json.loads(text)

    prose = f"{data['diagnosis_narrative']} {data['maintenance_recommendation']}"
    ungrounded = verify_numbers_are_grounded(prose, raw_results)
    if ungrounded:
        logger.warning(
            "LLM narrative contains %d number(s) absent from tool results: %s",
            len(ungrounded),
            ungrounded,
        )

    return NarrativeResult(
        diagnosis_narrative=data["diagnosis_narrative"],
        maintenance_recommendation=data["maintenance_recommendation"],
        evidence_cited=list(data.get("evidence_cited") or []),
        corroboration=data["corroboration"],
        model=model,
        tool_calls=trail,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ungrounded_numbers=ungrounded,
    )
