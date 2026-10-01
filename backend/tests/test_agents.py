"""Tests for the FleetMind multi-agent pipeline."""

import math

import pytest

from fleetmind.agents.agents import (
    LOAD_RISK_FACTOR,
    diagnostic_agent,
    maintenance_agent,
    monitor_agent,
    risk_agent,
)
from fleetmind.agents.graph import build_agent_graph
from fleetmind.agents.state import AgentState


def _base_state(**overrides) -> AgentState:
    state = AgentState(
        window_id=1,
        depot_id=1,
        load_hp=0,
        predicted_label="Normal",
        confidence=0.95,
        signal_values=[0.01 * math.sin(i / 10.0) for i in range(2048)],
    )
    state.update(overrides)
    return state


def test_monitor_normal_is_not_anomalous():
    result = monitor_agent(_base_state(predicted_label="Normal"))
    assert result["is_anomalous"] is False
    assert "No anomaly" in result["monitor_note"]


def test_monitor_fault_is_anomalous():
    result = monitor_agent(_base_state(predicted_label="InnerRace"))
    assert result["is_anomalous"] is True
    assert "InnerRace" in result["monitor_note"]


def test_diagnostic_normal_has_high_confidence():
    result = diagnostic_agent(_base_state(predicted_label="Normal"))
    assert result["diagnosis"] == "Normal"
    assert result["diagnosis_confidence"] > 0.5


def test_diagnostic_fault_with_low_rms_is_unconfirmed():
    result = diagnostic_agent(_base_state(predicted_label="Ball", signal_values=[0.001] * 2048))
    assert "unconfirmed" in result["diagnosis"]


def test_diagnostic_computes_signal_stats():
    result = diagnostic_agent(_base_state(predicted_label="OuterRace"))
    assert result["signal_rms"] > 0
    assert result["signal_peak"] > 0


def test_risk_normal_is_low():
    state = _base_state(predicted_label="Normal")
    state = diagnostic_agent(state)
    result = risk_agent(state)
    assert result["risk_score"] < 0.2


def test_risk_outer_race_high_load_is_high():
    state = _base_state(predicted_label="OuterRace", depot_id=4, load_hp=3)
    state = diagnostic_agent(state)
    result = risk_agent(state)
    assert result["risk_score"] > 0.5


def test_risk_factors_populated():
    state = _base_state(predicted_label="InnerRace", depot_id=3, load_hp=2)
    state = diagnostic_agent(state)
    result = risk_agent(state)
    assert len(result["risk_factors"]) >= 1


def test_maintenance_low_risk_is_low_urgency():
    state = _base_state(predicted_label="Normal")
    state = diagnostic_agent(state)
    state = risk_agent(state)
    result = maintenance_agent(state)
    assert result["maintenance_urgency"] == "low"


def test_maintenance_critical_risk():
    state = _base_state(predicted_label="OuterRace", depot_id=4, load_hp=3)
    state = diagnostic_agent(state)
    state = risk_agent(state)
    # Force high risk
    state["risk_score"] = 0.95
    result = maintenance_agent(state)
    assert result["maintenance_urgency"] == "critical"
    assert "SHUTDOWN" in result["maintenance_action"]


def test_full_graph_runs_end_to_end():
    app = build_agent_graph()
    result = app.invoke(_base_state(predicted_label="Ball", depot_id=2, load_hp=1, confidence=0.8))
    assert "maintenance_action" in result
    assert "risk_score" in result
    assert "diagnosis" in result
    assert result["is_anomalous"] is True


def test_full_graph_normal_flow():
    app = build_agent_graph()
    result = app.invoke(_base_state(predicted_label="Normal"))
    assert result["is_anomalous"] is False
    assert result["maintenance_urgency"] == "low"


# --- load-factor keying (Slice G regression) -------------------------------
#
# LOAD_RISK_FACTOR is keyed by motor load in HP, and the CWRU sites map
# id 1 -> 0 HP, 2 -> 1 HP, 3 -> 2 HP, 4 -> 3 HP. Before Slice G the lookup
# used depot_id, so every site got the factor belonging to the next load up
# and the 3 HP site fell through to the 1.0 default. These tests pin the
# correct keying so the off-by-one cannot come back.


def _risk_for(load_hp, depot_id, label="OuterRace"):
    state = _base_state(predicted_label=label, depot_id=depot_id, load_hp=load_hp)
    return risk_agent(diagnostic_agent(state))


def test_risk_load_factor_is_keyed_by_load_hp_not_depot_id():
    """Site 4 (3 HP) must get the 1.3 factor, not the 1.0 default."""
    light = _risk_for(load_hp=0, depot_id=1)
    heavy = _risk_for(load_hp=3, depot_id=4)
    assert heavy["risk_score"] > light["risk_score"]
    # The only difference between the two states is the load factor, so the
    # ratio of the scores is the ratio of the factors (within 4-dp rounding).
    assert heavy["risk_score"] / light["risk_score"] == pytest.approx(
        LOAD_RISK_FACTOR[3] / LOAD_RISK_FACTOR[0], abs=1e-3
    )


def test_risk_scales_monotonically_with_load():
    scores = [_risk_for(load_hp=hp, depot_id=hp + 1)["risk_score"] for hp in (0, 1, 2, 3)]
    assert scores == sorted(scores)
    assert len(set(scores)) == 4, "each load level must produce a distinct risk"


def test_risk_factor_text_names_the_real_motor_load():
    factors = _risk_for(load_hp=2, depot_id=3)["risk_factors"]
    joined = " | ".join(factors)
    assert "2 HP" in joined
    # The old text printed the site id as if it were horsepower -- site 3 is
    # the 2 HP site, so "3 HP" appearing here would be the bug returning.
    assert "3 HP" not in joined


def test_risk_missing_load_hp_uses_neutral_factor_not_a_guess():
    state = _base_state(predicted_label="OuterRace", depot_id=4)
    state.pop("load_hp")
    unknown = risk_agent(diagnostic_agent(state))
    one_hp = _risk_for(load_hp=1, depot_id=2)
    assert unknown["risk_score"] == one_hp["risk_score"]  # factor 1.0
    assert not any("HP" in f for f in unknown["risk_factors"])
