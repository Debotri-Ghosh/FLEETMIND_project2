"""The cost matrix: what each action costs in each true state of the machine.

This is the one place in the decision layer where a human judgement enters, and it
is kept explicit, small and inspectable on purpose. Read `OPEN_QUESTIONS.md` Q2/Q3
for the reasoning; the short version is below.

Two states, not four
--------------------
The heuristic baseline scored each fault class with an invented severity (Ball 0.4,
InnerRace 0.7, OuterRace 0.85). Nothing in CWRU says how those faults differ in
time-to-failure, so a per-class cost would be the same kind of invented number. The
matrix therefore distinguishes only **healthy** from **faulted**, and the policy needs
only ``P(faulted) = 1 - P(Normal)`` -- no per-fault constant anywhere.

Relative units, and why that is enough
--------------------------------------
Costs are in relative units with an unplanned run-to-failure fixed at 100. An
expected-cost decision depends only on the **ratios** between entries: multiplying
the whole matrix by any positive constant changes no decision. So no currency,
downtime figure or ROI number is needed, and none is invented. The ratios themselves
*are* assumptions, which is why `sensitivity()` in `policy.py` reports how decisions
move as the key ratio is varied over two orders of magnitude.

Why "stop machine" is never chosen -- and why that is the honest result
-----------------------------------------------------------------------
Stopping *now* rather than scheduling a repair is only worth its cost if the fault
might fail in the interval. Pricing that needs a degradation rate -- how quickly a
fault progresses to failure -- and CWRU cannot supply one: its defects are seeded at
fixed sizes and no recording runs to failure. Inventing a hazard rate would
manufacture urgency the data cannot support. So ``stop_machine`` is priced honestly
(an immediate stop costs lost production *on top of* the repair) and is therefore
dominated by ``schedule_maintenance`` in every state. The policy reports that it is
never optimal, rather than hiding the action or faking a reason to choose it.
"""

from __future__ import annotations

from dataclasses import dataclass

HEALTHY = "healthy"
FAULTED = "faulted"
STATES = (HEALTHY, FAULTED)

#: Ordered least to most interventionist. This order also breaks exact ties in
#: expected cost: the less disruptive action wins.
ACTIONS = ("monitor", "inspect", "schedule_maintenance", "stop_machine")

#: How each action maps onto the urgency tags `sync_equipment_status` already reads,
#: so the existing flagging, service-request filing and UI keep working unchanged.
#: `sync` flags a machine at HIGH or above, i.e. when maintenance is chosen.
ACTION_TO_URGENCY = {
    "monitor": "LOW",
    "inspect": "MEDIUM",
    "schedule_maintenance": "HIGH",
    "stop_machine": "CRITICAL",
}

ACTION_TEXT = {
    "monitor": "Continue routine monitoring. No action required.",
    "inspect": "Schedule an inspection visit.",
    "schedule_maintenance": "Plan bearing replacement at the next convenient window.",
    "stop_machine": "Stop the machine now.",
}


@dataclass(frozen=True)
class CostPrimitives:
    """The four quantities the whole matrix is built from, in relative units.

    Kept as named primitives rather than a grid of 8 numbers so that every entry in
    the matrix can be traced to a stated assumption.
    """

    #: Unplanned run-to-failure: secondary damage plus unplanned downtime. The unit
    #: anchor -- every other cost is expressed relative to this.
    failure: float = 100.0
    #: A planned bearing replacement at a convenient time. Being well below
    #: `failure` is the entire premise of predictive maintenance; *how far* below is
    #: the key assumption, and the one `sensitivity()` varies.
    repair: float = 20.0
    #: One technician inspection visit.
    visit: float = 2.0
    #: Lost production from stopping immediately instead of at a planned time.
    stop: float = 15.0

    def __post_init__(self) -> None:
        for name in ("failure", "repair", "visit", "stop"):
            if getattr(self, name) < 0:
                raise ValueError(f"cost {name} must be non-negative")
        if not self.repair < self.failure:
            raise ValueError(
                "repair must cost less than failure, or predictive maintenance has no "
                "purpose and every decision degenerates to 'monitor'"
            )


def matrix(p: CostPrimitives | None = None) -> dict[str, dict[str, float]]:
    """cost[action][state], every entry derived from a named primitive.

    monitor              healthy: nothing happens      faulted: runs to failure
    inspect              healthy: a wasted visit       faulted: visit finds it, then a planned repair
    schedule_maintenance healthy: needless repair      faulted: the repair that was needed
    stop_machine         healthy: needless stop+repair faulted: stop, then repair
    """
    p = p or CostPrimitives()
    return {
        "monitor": {HEALTHY: 0.0, FAULTED: p.failure},
        "inspect": {HEALTHY: p.visit, FAULTED: p.visit + p.repair},
        "schedule_maintenance": {HEALTHY: p.repair, FAULTED: p.repair},
        "stop_machine": {HEALTHY: p.stop + p.repair, FAULTED: p.stop + p.repair},
    }
