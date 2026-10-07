"""Expected-cost decision policy: choose the action that is cheapest on average.

For a machine whose model says it is faulted with probability ``p``, every action
``a`` has an expected cost

    E[cost(a)] = (1 - p) * cost[a][healthy] + p * cost[a][faulted]

and the policy picks the minimum. That replaces the heuristic's invented thresholds
(0.15 / 0.40 / 0.70 on an invented risk score) with boundaries that *follow from*
the stated costs: `boundaries()` derives exactly where each action takes over, so
anyone can check why a machine was or was not flagged.

What this does and does not fix
-------------------------------
It fixes the measured flaw in the heuristic -- the model's probability now actually
reaches the decision (`test_the_policy_responds_to_model_confidence` pins it).

It does **not** fix the model's over-confidence. v1 is over-confident on held-out
data (ECE 0.2146), and an expected-cost decision is only as good as the
probabilities fed to it. See `OPEN_QUESTIONS.md` Q1.
"""

from __future__ import annotations

from dataclasses import dataclass

from fleetmind.decision import costs
from fleetmind.ml.model import LABEL_MAP

NORMAL_INDEX = LABEL_MAP.index("Normal")

#: Actions at which `sync_equipment_status` flags a machine (HIGH and above).
ACTIONABLE = frozenset({"schedule_maintenance", "stop_machine"})


def p_faulted(probabilities: list[float]) -> float:
    """P(any fault) = 1 - P(Normal). Needs no per-fault severity constant."""
    if len(probabilities) != len(LABEL_MAP):
        raise ValueError(f"expected {len(LABEL_MAP)} class probabilities, got {len(probabilities)}")
    return min(1.0, max(0.0, 1.0 - float(probabilities[NORMAL_INDEX])))


def expected_costs(p: float, cost_matrix: dict | None = None) -> dict[str, float]:
    """Expected cost of every action at fault probability ``p``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be a probability, got {p}")
    m = cost_matrix or costs.matrix()
    return {a: (1.0 - p) * m[a][costs.HEALTHY] + p * m[a][costs.FAULTED] for a in costs.ACTIONS}


def choose(p: float, cost_matrix: dict | None = None) -> tuple[str, dict[str, float]]:
    """The cheapest action, and the full table it was chosen from.

    Exact ties go to the less disruptive action (`costs.ACTIONS` order), so the
    choice is deterministic and errs toward not intervening.
    """
    table = expected_costs(p, cost_matrix)
    best = min(costs.ACTIONS, key=lambda a: (table[a], costs.ACTIONS.index(a)))
    return best, table


def boundaries(cost_matrix: dict | None = None, resolution: int = 100_000) -> list[dict]:
    """Where each action is optimal, as contiguous ranges of ``p``.

    Found by scanning rather than solved symbolically, so it stays correct for any
    matrix -- including ones where an action is never optimal, which a symbolic
    pairwise solution would silently miss.
    """
    regions: list[dict] = []
    current: str | None = None
    start = 0.0
    for i in range(resolution + 1):
        p = i / resolution
        action, _ = choose(p, cost_matrix)
        if action != current:
            if current is not None:
                regions.append({"action": current, "from_p": start, "to_p": p})
            current, start = action, p
    regions.append({"action": current, "from_p": start, "to_p": 1.0})
    return regions


def never_optimal(cost_matrix: dict | None = None) -> list[str]:
    chosen = {r["action"] for r in boundaries(cost_matrix)}
    return [a for a in costs.ACTIONS if a not in chosen]


def flag_threshold(cost_matrix: dict | None = None) -> float | None:
    """The smallest ``p`` at which the policy flags a machine; None if it never does."""
    for r in boundaries(cost_matrix):
        if r["action"] in ACTIONABLE:
            return r["from_p"]
    return None


@dataclass(frozen=True)
class PolicyDecision:
    """One window's decision. Exposes what `quality.score()` reads."""

    window_id: int
    experiment_id: int
    window_index: int
    true_label: str
    p_faulted: float
    action: str
    expected_costs: dict[str, float]

    @property
    def actionable(self) -> bool:
        return self.action in ACTIONABLE

    @property
    def urgency(self) -> str:
        return costs.ACTION_TO_URGENCY[self.action]


def decide_all(posteriors: list, cost_matrix: dict | None = None) -> list[PolicyDecision]:
    out: list[PolicyDecision] = []
    for post in posteriors:
        p = p_faulted(post.probabilities)
        action, table = choose(p, cost_matrix)
        out.append(
            PolicyDecision(
                window_id=post.window_id,
                experiment_id=post.experiment_id,
                window_index=post.window_index,
                true_label=post.true_label,
                p_faulted=p,
                action=action,
                expected_costs=table,
            )
        )
    return out


def decision_text(action: str, p: float, table: dict[str, float]) -> str:
    """The persisted decision string, in the format the rest of the system reads.

    Starts with the `[URGENCY]` tag `sync_equipment_status` parses. Every figure in
    it is computed here, so the LLM layer's tools can quote it without originating
    a number.
    """
    urgency = costs.ACTION_TO_URGENCY[action]
    ranked = sorted(table.items(), key=lambda kv: kv[1])
    alternatives = "; ".join(f"{a} {c:.2f}" for a, c in ranked)
    return (
        f"[{urgency}] {costs.ACTION_TEXT[action]} | P(faulted): {p:.2%} | "
        f"Policy: expected cost | Expected costs (relative units): {alternatives}"
    )
