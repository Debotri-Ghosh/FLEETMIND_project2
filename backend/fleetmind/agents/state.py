"""Shared state for the FleetMind multi-agent system.

The state is a TypedDict that flows through the LangGraph pipeline.
Each agent reads from and writes to its own key.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # --- inputs (set before the graph runs) ---
    window_id: int
    depot_id: int
    # Motor load in HP for this site (depots.load_hp). Distinct from depot_id:
    # LOAD_RISK_FACTOR is keyed by load, and site id 2 is the 1 HP site.
    load_hp: int | None
    predicted_label: str
    confidence: float | None
    signal_values: list[float]

    # --- monitor agent ---
    is_anomalous: bool
    monitor_note: str

    # --- diagnostic agent ---
    diagnosis: str
    diagnosis_confidence: float
    signal_rms: float
    signal_peak: float

    # --- risk agent ---
    risk_score: float  # 0.0 (no risk) .. 1.0 (critical)
    risk_factors: list[str]

    # --- maintenance agent ---
    maintenance_action: str
    maintenance_urgency: str  # low | medium | high | critical

    # --- persistence ---
    agent_decision_id: int | None
    decision_text: str
