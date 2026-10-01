"""Tests for the LLM reasoning layer (agents/llm.py + agents/tools.py).

Every test here mocks the Anthropic client with a scripted, deterministic
stand-in, so the suite never makes a network call and never costs money. The
one real API call this feature needs is run by hand and its output recorded in
CLAUDE.md's Progress log -- a live call in CI would be flaky and billed.

What these actually pin down is the property the whole design rests on: the
model writes prose, the deterministic code supplies every number, and a
narrative that quotes a figure no tool returned is detectable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from fleetmind.agents.agents import diagnostic_agent, maintenance_agent, risk_agent
from fleetmind.agents.llm import (
    DEFAULT_MODEL,
    MAX_TOOL_ITERATIONS,
    LLMNotConfigured,
    generate_narrative,
    get_client,
    verify_numbers_are_grounded,
)
from fleetmind.agents.sync_equipment_status import sync_equipment_status
from fleetmind.agents.tools import (
    TOOL_DEFINITIONS,
    EvidenceContext,
    get_bearing_frequency_match,
    get_risk_assessment,
    get_signal_statistics,
    measure_spectral_band_energy,
    run_tool,
)

# ── a scripted stand-in for anthropic.Anthropic ────────────────────────
#
# Deliberately hand-rolled rather than MagicMock: the loop reads
# `block.type` / `.name` / `.input` / `.id` and `response.stop_reason`, and a
# mock that auto-creates those attributes would pass even if the loop read the
# wrong ones.


@dataclass
class FakeText:
    text: str
    type: str = "text"


@dataclass
class FakeToolUse:
    name: str
    input: dict[str, Any]
    id: str = "toolu_fake"
    type: str = "tool_use"


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class FakeResponse:
    content: list[Any]
    stop_reason: str = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_details: Any = None


class FakeMessages:
    def __init__(self, script: list[FakeResponse]):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("fake client ran out of scripted responses")
        return self._script.pop(0)


class FakeAnthropic:
    def __init__(self, script: list[FakeResponse]):
        self.messages = FakeMessages(script)


FINAL_JSON = json.dumps(
    {
        "diagnosis_narrative": (
            "The window shows an RMS of 0.5339 g with 0.7 g peak-to-peak. The "
            "classifier called InnerRace and the risk score is 0.8."
        ),
        "maintenance_recommendation": (
            "Plan maintenance within 48 hours and order a replacement bearing."
        ),
        "evidence_cited": ["rms_g = 0.5339", "risk_score = 0.8"],
        "corroboration": "weak",
    }
)


def _script_full_run() -> list[FakeResponse]:
    """Three no-arg tools, then a band probe, then the answer."""
    return [
        FakeResponse(
            content=[
                FakeToolUse("get_signal_statistics", {}, id="t1"),
                FakeToolUse("get_risk_assessment", {}, id="t2"),
                FakeToolUse("get_bearing_frequency_match", {}, id="t3"),
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(
            content=[
                FakeToolUse(
                    "measure_spectral_band_energy",
                    {"low_hz": 150.0, "high_hz": 170.0},
                    id="t4",
                )
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(content=[FakeText(FINAL_JSON)]),
    ]


@pytest.fixture()
def ctx() -> EvidenceContext:
    return EvidenceContext(
        service_request_id=1,
        window_id=42,
        equipment_name="Bearing Motor A1",
        site_name="Plant A (CWRU 0 HP)",
        depot_id=1,
        load_hp=0,
        nominal_rpm=1797.0,
        sensor_position="DE",
        sampling_rate_hz=12000,
        predicted_label="InnerRace",
        model_confidence=0.9,
        model_version="v1",
        raw_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        stored_risk_score=0.8,
        stored_decision_text="[HIGH] elevated fault repetition rate",
    )


# ── the tools return the rule-based numbers, unchanged ─────────────────


def test_signal_statistics_tool_matches_the_diagnostic_agent(ctx):
    tool = get_signal_statistics(ctx)
    agent = diagnostic_agent(
        {
            "predicted_label": ctx.predicted_label,
            "confidence": ctx.model_confidence,
            "signal_values": ctx.raw_values,
        }
    )
    assert tool["rms_g"] == agent["signal_rms"]
    assert tool["peak_to_peak_g"] == agent["signal_peak"]


def test_risk_tool_matches_the_risk_and_maintenance_agents(ctx):
    tool = get_risk_assessment(ctx)
    state = diagnostic_agent(
        {
            "predicted_label": ctx.predicted_label,
            "confidence": ctx.model_confidence,
            "signal_values": ctx.raw_values,
        }
    )
    state = maintenance_agent(risk_agent({**state, "depot_id": 1, "load_hp": 0}))
    assert tool["risk_score"] == state["risk_score"]
    assert tool["urgency"] == state["maintenance_urgency"]
    assert tool["rule_based_action"] == state["maintenance_action"]


def test_risk_tool_exposes_the_stored_score_for_comparison(ctx):
    """A divergence between recomputed and stored must be visible, not hidden."""
    assert get_risk_assessment(ctx)["stored_risk_score"] == ctx.stored_risk_score


def test_frequency_tool_says_it_is_not_the_model_explanation(ctx):
    result = get_bearing_frequency_match(ctx)
    assert result["matched_frequency"]["name"] == "BPFI"
    assert set(result["all_characteristic_frequencies_hz"]) == {"BPFI", "BPFO", "BSF", "FTF"}
    assert "not model internals" in result["note"]


def test_band_energy_fractions_are_bounded_and_locate_the_peak(ctx):
    result = measure_spectral_band_energy(ctx, 0.0, 6000.0)
    assert 0.0 <= result["fraction_of_total_energy"] <= 1.0
    assert result["peak_frequency_in_band_hz"] is not None


def test_band_energy_rejects_an_inverted_band(ctx):
    with pytest.raises(ValueError, match="greater than"):
        measure_spectral_band_energy(ctx, 500.0, 100.0)


def test_every_declared_tool_is_dispatchable(ctx):
    """A schema with no implementation would 404 the model mid-reasoning."""
    for definition in TOOL_DEFINITIONS:
        args = {"low_hz": 1.0, "high_hz": 2.0} if definition["input_schema"]["required"] else {}
        assert json.loads(run_tool(ctx, definition["name"], args))


def test_unknown_tool_raises_rather_than_returning_nothing(ctx):
    with pytest.raises(KeyError):
        run_tool(ctx, "get_the_answer", {})


# ── the grounding guard ────────────────────────────────────────────────


def test_grounding_accepts_figures_that_came_from_a_tool():
    results = [json.dumps({"rms_g": 0.5339, "risk_score": 0.8})]
    assert verify_numbers_are_grounded("RMS is 0.5339 g, risk 0.8.", results) == []


def test_grounding_flags_an_invented_figure():
    results = [json.dumps({"rms_g": 0.5339})]
    flagged = verify_numbers_are_grounded("RMS is 0.5339 g and BPFO sits at 162.19 Hz.", results)
    assert "162.19" in flagged


def test_grounding_accepts_timeframes_from_the_rule_based_action():
    """ "within 48 hours" is grounded because get_risk_assessment returns it.

    The canned action text is part of a tool result, so a timeframe quoted
    from it is traceable like any other figure -- the guard does not need a
    special case for it.
    """
    results = [json.dumps({"rule_based_action": "Plan maintenance within 48 hours."})]
    assert verify_numbers_are_grounded("Plan maintenance within 48 hours.", results) == []


def test_grounding_ignores_small_bare_integers():
    """Counts and HP figures are prompt/context, not measurements."""
    results = [json.dumps({"rms_g": 0.5339})]
    assert verify_numbers_are_grounded("A 3 HP motor, 4 sites.", results) == []


# ── the tool-use loop ──────────────────────────────────────────────────


def test_generate_narrative_runs_tools_then_returns_structured_prose(ctx):
    client = FakeAnthropic(_script_full_run())

    result = generate_narrative(ctx, client=client)

    assert result.corroboration == "weak"
    assert "InnerRace" in result.diagnosis_narrative
    assert result.maintenance_recommendation
    assert [c["tool"] for c in result.tool_calls] == [
        "get_signal_statistics",
        "get_risk_assessment",
        "get_bearing_frequency_match",
        "measure_spectral_band_energy",
    ]
    # usage accumulates across every round-trip, not just the last one
    assert result.input_tokens == 300
    assert result.output_tokens == 150


def test_generate_narrative_uses_the_configured_model_and_tools(ctx):
    client = FakeAnthropic(_script_full_run())

    generate_narrative(ctx, client=client)

    first = client.messages.calls[0]
    assert first["model"] == DEFAULT_MODEL == "claude-sonnet-5"
    assert [t["name"] for t in first["tools"]] == [t["name"] for t in TOOL_DEFINITIONS]
    assert first["output_config"]["format"]["type"] == "json_schema"


def test_prompt_carries_no_measurements(ctx):
    """The grounding guard is only meaningful if the prompt leaks no numbers.

    The model must have to call a tool to learn the RMS, the risk score or a
    defect frequency -- otherwise it could quote one straight from the prompt
    and the check below would never fire.
    """
    client = FakeAnthropic(_script_full_run())
    generate_narrative(ctx, client=client)

    prompt = client.messages.calls[0]["messages"][0]["content"]
    stats = get_signal_statistics(ctx)
    risk = get_risk_assessment(ctx)
    for value in (stats["rms_g"], stats["peak_to_peak_g"], risk["risk_score"]):
        assert str(value) not in prompt


def test_tool_errors_are_returned_to_the_model_not_raised(ctx):
    """An inverted band must come back as an is_error tool_result, not a 500."""
    client = FakeAnthropic(
        [
            FakeResponse(
                content=[
                    FakeToolUse(
                        "measure_spectral_band_energy",
                        {"low_hz": 500.0, "high_hz": 100.0},
                        id="t1",
                    )
                ],
                stop_reason="tool_use",
            ),
            FakeResponse(content=[FakeText(FINAL_JSON)]),
        ]
    )

    result = generate_narrative(ctx, client=client)

    sent_back = client.messages.calls[1]["messages"][2]["content"][0]
    assert sent_back["is_error"] is True
    assert "greater than" in sent_back["content"]
    # and the failed call is still recorded in the audit trail
    assert result.tool_calls[0]["tool"] == "measure_spectral_band_energy"


def test_generate_narrative_records_ungrounded_numbers_without_failing(ctx):
    """An invented figure is flagged and stored, not silently accepted."""
    invented = json.dumps(
        {
            "diagnosis_narrative": "Energy peaks at 9999.77 Hz.",
            "maintenance_recommendation": "Inspect within 48 hours.",
            "evidence_cited": [],
            "corroboration": "weak",
        }
    )
    client = FakeAnthropic(
        [
            FakeResponse(
                content=[FakeToolUse("get_signal_statistics", {}, id="t1")],
                stop_reason="tool_use",
            ),
            FakeResponse(content=[FakeText(invented)]),
        ]
    )

    result = generate_narrative(ctx, client=client)

    assert "9999.77" in result.ungrounded_numbers


def test_generate_narrative_raises_on_refusal(ctx):
    @dataclass
    class Details:
        category: str = "cyber"

    client = FakeAnthropic(
        [FakeResponse(content=[], stop_reason="refusal", stop_details=Details())]
    )
    with pytest.raises(RuntimeError, match="declined"):
        generate_narrative(ctx, client=client)


def test_generate_narrative_gives_up_rather_than_looping_forever(ctx):
    forever = [
        FakeResponse(
            content=[FakeToolUse("get_signal_statistics", {}, id="t1")], stop_reason="tool_use"
        )
        for _ in range(MAX_TOOL_ITERATIONS + 2)
    ]
    client = FakeAnthropic(forever)
    with pytest.raises(RuntimeError, match="did not converge"):
        generate_narrative(ctx, client=client)


def test_get_client_without_a_key_explains_itself(monkeypatch):
    import fleetmind.agents.llm as llm_module

    class NoKeySettings:
        anthropic_api_key = None
        anthropic_model = DEFAULT_MODEL

    monkeypatch.setattr(llm_module, "get_settings", lambda: NoKeySettings())
    with pytest.raises(LLMNotConfigured, match="ANTHROPIC_API_KEY"):
        get_client()


# ── the API endpoints ──────────────────────────────────────────────────


def _ai_request_id(product_client, product_session) -> int:
    sync_equipment_status(product_session)
    product_session.commit()
    rows = product_client.get("/service-requests", params={"ai_flagged": True}).json()
    assert rows, "fixture should have produced one AI-filed request"
    return rows[0]["id"]


@pytest.fixture()
def patched_llm(monkeypatch):
    """Patch the client factory, not generate_narrative -- the endpoint test
    should exercise the real loop, tools and grounding check."""
    import fleetmind.api.routers.service_requests as router_module

    calls: list[int] = []

    def fake_generate(context, client=None, model=None):
        calls.append(context.service_request_id)
        from fleetmind.agents.llm import generate_narrative as real

        return real(context, client=FakeAnthropic(_script_full_run()))

    monkeypatch.setattr(router_module, "generate_narrative", fake_generate)
    return calls


def test_narrative_is_404_before_generation(product_client, product_session):
    rid = _ai_request_id(product_client, product_session)
    resp = product_client.get(f"/service-requests/{rid}/narrative")
    assert resp.status_code == 404
    assert "no narrative generated yet" in resp.json()["detail"]


def test_post_generates_and_get_returns_the_cached_narrative(
    product_client, product_session, patched_llm
):
    rid = _ai_request_id(product_client, product_session)

    created = product_client.post(f"/service-requests/{rid}/narrative")
    assert created.status_code == 200
    body = created.json()
    assert body["cached"] is False
    assert body["model"] == DEFAULT_MODEL
    assert body["corroboration"] == "weak"
    assert len(body["tool_calls"]) == 4
    assert body["input_tokens"] == 300

    fetched = product_client.get(f"/service-requests/{rid}/narrative").json()
    assert fetched["cached"] is True
    assert fetched["diagnosis_narrative"] == body["diagnosis_narrative"]
    assert len(patched_llm) == 1, "GET must not trigger a second API call"


def test_second_post_serves_the_cache_without_calling_the_api(
    product_client, product_session, patched_llm
):
    rid = _ai_request_id(product_client, product_session)
    product_client.post(f"/service-requests/{rid}/narrative")

    again = product_client.post(f"/service-requests/{rid}/narrative")

    assert again.json()["cached"] is True
    assert len(patched_llm) == 1


def test_force_regenerates_and_replaces_the_cached_row(
    product_client, product_session, patched_llm
):
    rid = _ai_request_id(product_client, product_session)
    product_client.post(f"/service-requests/{rid}/narrative")

    forced = product_client.post(f"/service-requests/{rid}/narrative", params={"force": True})

    assert forced.status_code == 200
    assert forced.json()["cached"] is False
    assert len(patched_llm) == 2
    # the unique FK means a stale row would have raised, not duplicated
    from fleetmind.db.models import LLMNarrative

    assert product_session.query(LLMNarrative).filter_by(service_request_id=rid).count() == 1


def test_narrative_on_a_customer_filed_request_is_a_clear_404(product_client, product_session):
    created = product_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 1, "title": "Odd noise from the pump"},
    )
    rid = created.json()["id"]

    resp = product_client.post(f"/service-requests/{rid}/narrative")

    assert resp.status_code == 404
    assert "no agent decision attached" in resp.json()["detail"]


def test_narrative_without_an_api_key_is_503_not_500(product_client, product_session, monkeypatch):
    rid = _ai_request_id(product_client, product_session)
    import fleetmind.agents.llm as llm_module

    class NoKeySettings:
        anthropic_api_key = None
        anthropic_model = DEFAULT_MODEL

    monkeypatch.setattr(llm_module, "get_settings", lambda: NoKeySettings())

    resp = product_client.post(f"/service-requests/{rid}/narrative")

    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]
