"""Tests for the FleetMind multi-agent pipeline."""

import math

from fleetmind.agents.agents import (
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
    state = _base_state(predicted_label="OuterRace", depot_id=3)
    state = diagnostic_agent(state)
    result = risk_agent(state)
    assert result["risk_score"] > 0.5


def test_risk_factors_populated():
    state = _base_state(predicted_label="InnerRace", depot_id=2)
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
    state = _base_state(predicted_label="OuterRace", depot_id=3)
    state = diagnostic_agent(state)
    state = risk_agent(state)
    # Force high risk
    state["risk_score"] = 0.95
    result = maintenance_agent(state)
    assert result["maintenance_urgency"] == "critical"
    assert "SHUTDOWN" in result["maintenance_action"]


def test_full_graph_runs_end_to_end():
    app = build_agent_graph()
    result = app.invoke(_base_state(predicted_label="Ball", depot_id=2, confidence=0.8))
    assert "maintenance_action" in result
    assert "risk_score" in result
    assert "diagnosis" in result
    assert result["is_anomalous"] is True


def test_full_graph_normal_flow():
    app = build_agent_graph()
    result = app.invoke(_base_state(predicted_label="Normal"))
    assert result["is_anomalous"] is False
    assert result["maintenance_urgency"] == "low"
