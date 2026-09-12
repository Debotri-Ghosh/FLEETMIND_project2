"""The four FleetMind agents.

Each agent is a pure function (state -> state) so it is trivially testable
and can run without an LLM. The optional LLM layer (llm.py) enriches the
diagnosis and maintenance text when AZURE_OPENAI / Ollama credentials are
configured; otherwise the agents use deterministic rule-based logic.
"""

from __future__ import annotations

import math

from fleetmind.agents.state import AgentState

FAULT_SEVERITY: dict[str, float] = {
    "Normal": 0.0,
    "Ball": 0.4,
    "InnerRace": 0.7,
    "OuterRace": 0.85,
}

LOAD_RISK_FACTOR: dict[int, float] = {
    0: 0.8,
    1: 1.0,
    2: 1.15,
    3: 1.3,
}


# ── Monitor ────────────────────────────────────────────────────────────
def monitor_agent(state: AgentState) -> AgentState:
    """Flag non-Normal predictions as anomalous."""
    label = state.get("predicted_label", "Normal")
    confidence = state.get("confidence") or 0.0
    is_anomalous = label != "Normal"
    note = (
        f"Anomaly detected: {label} (confidence {confidence:.2%})"
        if is_anomalous
        else "No anomaly — Normal operation"
    )
    return {**state, "is_anomalous": is_anomalous, "monitor_note": note}


# ── Diagnostic ─────────────────────────────────────────────────────────
def _signal_stats(values: list[float]) -> tuple[float, float]:
    """Return (RMS, peak-to-peak) of the signal window."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean_sq = sum(v * v for v in values) / n
    rms = math.sqrt(mean_sq)
    peak = max(values) - min(values)
    return rms, peak


def diagnostic_agent(state: AgentState) -> AgentState:
    """Confirm/refine the diagnosis using signal statistics."""
    label = state.get("predicted_label", "Normal")
    values = state.get("signal_values") or []
    rms, peak = _signal_stats(values)

    # Rule-based confidence: higher RMS/peak => more confident in non-Normal
    confidence = max(0.6, 1.0 - rms * 5.0) if label == "Normal" else min(0.95, 0.5 + rms * 3.0)

    diagnosis = label
    # Cross-check: very low RMS with a non-Normal flag => downgrade
    if label != "Normal" and rms < 0.05:
        diagnosis = f"{label} (unconfirmed — low signal energy)"
        confidence *= 0.5

    return {
        **state,
        "diagnosis": diagnosis,
        "diagnosis_confidence": round(confidence, 4),
        "signal_rms": round(rms, 6),
        "signal_peak": round(peak, 6),
    }


# ── Risk ───────────────────────────────────────────────────────────────
def risk_agent(state: AgentState) -> AgentState:
    """Compute a 0-1 risk score from fault severity + depot load."""
    label = state.get("predicted_label", "Normal")
    depot_id = state.get("depot_id", 1)
    confidence = state.get("diagnosis_confidence", 0.5)

    severity = FAULT_SEVERITY.get(label, 0.5)
    load_factor = LOAD_RISK_FACTOR.get(depot_id, 1.0)

    # Risk = severity * load_factor, scaled into [0, 1]
    raw = severity * load_factor
    risk = min(1.0, raw * (0.7 + 0.3 * confidence))

    factors: list[str] = []
    if label != "Normal":
        factors.append(f"Fault type: {label} (severity {severity:.2f})")
    if depot_id >= 2:
        factors.append(f"High load: {depot_id}HP depot (factor {load_factor:.2f})")
    if confidence < 0.6:
        factors.append("Low diagnosis confidence")
    if not factors:
        factors.append("No risk factors identified")

    return {
        **state,
        "risk_score": round(risk, 4),
        "risk_factors": factors,
    }


# ── Maintenance ────────────────────────────────────────────────────────
def maintenance_agent(state: AgentState) -> AgentState:
    """Recommend an action based on diagnosis + risk score."""
    risk = state.get("risk_score", 0.0)
    diagnosis = state.get("diagnosis", "Normal")

    if risk < 0.15:
        urgency = "low"
        action = "Continue routine monitoring. No action required."
    elif risk < 0.4:
        urgency = "medium"
        action = f"Schedule inspection within 7 days. Suspected {diagnosis}."
    elif risk < 0.7:
        urgency = "high"
        action = f"Plan maintenance within 48 hours. Likely {diagnosis}. Order replacement bearing."
    else:
        urgency = "critical"
        action = f"IMMEDIATE SHUTDOWN recommended. Critical {diagnosis} detected."

    decision_text = (
        f"[{urgency.upper()}] {action} | Risk: {risk:.2%} | "
        f"Diagnosis: {diagnosis} | Factors: {'; '.join(state.get('risk_factors', []))}"
    )

    return {
        **state,
        "maintenance_action": action,
        "maintenance_urgency": urgency,
        "decision_text": decision_text,
    }
