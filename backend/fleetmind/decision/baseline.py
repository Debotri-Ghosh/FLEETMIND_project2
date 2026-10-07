"""Run the existing heuristic so it can be scored on identical footing.

This is an adapter, not a reimplementation. It calls the real
`monitor_agent` -> `diagnostic_agent` -> `risk_agent` -> `maintenance_agent`
chain from `agents/agents.py`, with the same inputs the live pipeline supplies,
so the numbers it produces are the numbers the application would produce.

It exists because "the new layer is better" is only a claim if the old layer was
measured the same way. The heuristic is the control.
"""

from __future__ import annotations

from dataclasses import dataclass

from fleetmind.agents.agents import (
    diagnostic_agent,
    maintenance_agent,
    monitor_agent,
    risk_agent,
)
from fleetmind.agents.state import AgentState

#: Urgencies the heuristic emits, weakest first. `maintenance_agent` produces
#: exactly these four strings.
URGENCY_ORDER = ["low", "medium", "high", "critical"]

#: The level at or above which `sync_equipment_status` flips a machine into
#: maintenance and files a service request. Mirrors that module's threshold.
ACTIONABLE_FROM = "high"


@dataclass(frozen=True)
class BaselineDecision:
    """One window's heuristic decision, with the inputs that produced it."""

    window_id: int
    experiment_id: int
    window_index: int
    true_label: str
    predicted_label: str
    model_confidence: float
    risk_score: float
    urgency: str

    @property
    def actionable(self) -> bool:
        """Would this decision alone flag the machine for maintenance?"""
        return URGENCY_ORDER.index(self.urgency) >= URGENCY_ORDER.index(ACTIONABLE_FROM)

    def as_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "experiment_id": self.experiment_id,
            "window_index": self.window_index,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "model_confidence": round(self.model_confidence, 6),
            "risk_score": round(self.risk_score, 6),
            "urgency": self.urgency,
            "actionable": self.actionable,
        }


def decide(
    *,
    window_id: int,
    experiment_id: int,
    window_index: int,
    true_label: str,
    predicted_label: str,
    confidence: float,
    signal_values: list[float],
    load_hp: int | None,
) -> BaselineDecision:
    """Run the real four-agent chain over one window.

    `load_hp` is passed through, never `depot_id` -- see the resolved keying bug
    in CLAUDE.md. Passing the site id here would silently mis-score every site.
    """
    state: AgentState = {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "signal_values": signal_values,
        "load_hp": load_hp,
    }
    for agent in (monitor_agent, diagnostic_agent, risk_agent, maintenance_agent):
        state = agent(state)

    return BaselineDecision(
        window_id=window_id,
        experiment_id=experiment_id,
        window_index=window_index,
        true_label=true_label,
        predicted_label=predicted_label,
        model_confidence=confidence,
        risk_score=float(state["risk_score"]),
        urgency=str(state["maintenance_urgency"]),
    )


def decide_all(
    windows: list[tuple[int, int, int, list[float], str]],
    posteriors: list,
    load_hp: int | None,
) -> list[BaselineDecision]:
    """Heuristic decisions for every held-out window of one site.

    `windows` is what `holdout.holdout_windows()` returned and `posteriors` is
    what `probs.posteriors()` returned from those same windows, so they are
    zipped strictly: a length mismatch means the two were built from different
    partitions, which would silently compare different data.
    """
    decisions: list[BaselineDecision] = []
    for (window_id, experiment_id, window_index, values, true_label), post in zip(
        windows, posteriors, strict=True
    ):
        if post.window_id != window_id:
            raise ValueError(
                f"posterior/window mismatch: {post.window_id} != {window_id}; "
                "the two lists are not the same partition in the same order"
            )
        decisions.append(
            decide(
                window_id=window_id,
                experiment_id=experiment_id,
                window_index=window_index,
                true_label=true_label,
                predicted_label=post.predicted_label,
                confidence=post.confidence,
                signal_values=values,
                load_hp=load_hp,
            )
        )
    return decisions
