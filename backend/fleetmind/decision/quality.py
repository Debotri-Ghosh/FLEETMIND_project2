"""Decision quality: false alarms, misses, and how long detection took.

Accuracy is not a decision metric. A maintenance product is judged on how often
it sends a technician to a healthy machine (false alarm) and how often it lets a
faulted one run (miss), and on how quickly it notices.

The unit of a decision here is a **recording**, not a window, because that is
what the product acts on: `sync_equipment_status` flags a *machine*, and one
0.17-second window is not a decision anyone would act on.

* **False alarm** -- a healthy recording where some window produced an
  actionable decision.
* **Miss** -- a faulted recording where no window ever did.
* **Time to detection** -- the index of the first actionable window within a
  faulted recording, in windows. Lower is better; `None` means never detected.

**The limitation that governs this whole module.** No Normal recording is ever
held out (see `holdout.py`), so `n_healthy_recordings` is **0** and the
false-alarm rate is **not measurable** on this split. The report returns `None`
for it rather than `0.0`: a rate of zero would read as "no false alarms", when
the truth is "never tested". That distinction is the entire reason this returns
a structured result instead of a number.
"""

from __future__ import annotations

from dataclasses import dataclass

HEALTHY_LABEL = "Normal"


@dataclass(frozen=True)
class RecordingOutcome:
    """What the decision layer did across one whole recording."""

    experiment_id: int
    true_label: str
    n_windows: int
    n_actionable_windows: int
    first_actionable_index: int | None

    @property
    def is_healthy(self) -> bool:
        return self.true_label == HEALTHY_LABEL

    @property
    def detected(self) -> bool:
        return self.n_actionable_windows > 0

    def as_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "true_label": self.true_label,
            "n_windows": self.n_windows,
            "n_actionable_windows": self.n_actionable_windows,
            "first_actionable_index": self.first_actionable_index,
            "detected": self.detected,
        }


@dataclass(frozen=True)
class QualityReport:
    """Decision quality over the held-out recordings of one site."""

    label: str  # which decision path produced this, e.g. "heuristic baseline"
    n_recordings_examined: int
    n_windows_examined: int
    n_healthy_recordings: int
    n_faulted_recordings: int
    n_false_alarms: int
    n_misses: int
    detection_delays: list[int]
    outcomes: list[RecordingOutcome]

    @property
    def false_alarm_rate(self) -> float | None:
        """None when no healthy recording was held out -- not 0.0.

        Returning 0.0 would assert "we never false-alarm" on the strength of
        having never looked.
        """
        if self.n_healthy_recordings == 0:
            return None
        return self.n_false_alarms / self.n_healthy_recordings

    @property
    def miss_rate(self) -> float | None:
        if self.n_faulted_recordings == 0:
            return None
        return self.n_misses / self.n_faulted_recordings

    @property
    def mean_detection_delay(self) -> float | None:
        """Mean windows until first actionable decision, over DETECTED faults."""
        if not self.detection_delays:
            return None
        return sum(self.detection_delays) / len(self.detection_delays)

    @property
    def measurable(self) -> list[str]:
        """Which rates this split can actually support."""
        ok = []
        if self.n_healthy_recordings:
            ok.append("false_alarm_rate")
        if self.n_faulted_recordings:
            ok.append("miss_rate")
        if self.detection_delays:
            ok.append("mean_detection_delay")
        return ok

    @property
    def not_measurable(self) -> dict[str, str]:
        """Rates this split cannot support, each with the reason."""
        gaps: dict[str, str] = {}
        if self.n_healthy_recordings == 0:
            gaps["false_alarm_rate"] = (
                "no healthy (Normal) recording is held out: each site has exactly one "
                "Normal recording and the stratified experiment split assigns a "
                "single-recording class entirely to train"
            )
        if self.n_faulted_recordings == 0:
            gaps["miss_rate"] = "no faulted recording is held out"
        if not self.detection_delays:
            gaps["mean_detection_delay"] = "no faulted recording was ever detected"
        return gaps

    def as_dict(self) -> dict:
        fa = self.false_alarm_rate
        mr = self.miss_rate
        dd = self.mean_detection_delay
        return {
            "decision_path": self.label,
            "n_recordings_examined": self.n_recordings_examined,
            "n_windows_examined": self.n_windows_examined,
            "n_healthy_recordings": self.n_healthy_recordings,
            "n_faulted_recordings": self.n_faulted_recordings,
            "n_false_alarms": self.n_false_alarms,
            "n_misses": self.n_misses,
            "false_alarm_rate": None if fa is None else round(fa, 6),
            "miss_rate": None if mr is None else round(mr, 6),
            "mean_detection_delay_windows": None if dd is None else round(dd, 4),
            "measurable": self.measurable,
            "not_measurable": self.not_measurable,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


def score(decisions: list, *, label: str) -> QualityReport:
    """Group per-window decisions into per-recording outcomes and score them.

    `decisions` is a list of anything exposing `.experiment_id`, `.true_label`,
    `.window_index` and `.actionable` -- so the heuristic baseline and every
    later policy are scored by the identical code rather than by two
    implementations that might disagree.

    Raises on an empty input rather than returning a report of zeros, because a
    report that examined nothing must not be mistakable for a clean result.
    """
    if not decisions:
        raise ValueError("score() examined no decisions; refusing to report zeros")

    grouped: dict[int, list] = {}
    for d in decisions:
        grouped.setdefault(d.experiment_id, []).append(d)

    outcomes: list[RecordingOutcome] = []
    for experiment_id, group in sorted(grouped.items()):
        group = sorted(group, key=lambda d: d.window_index)
        labels = {d.true_label for d in group}
        if len(labels) != 1:
            raise ValueError(
                f"recording {experiment_id} carries more than one true label: {sorted(labels)}"
            )
        actionable = [d for d in group if d.actionable]
        outcomes.append(
            RecordingOutcome(
                experiment_id=experiment_id,
                true_label=next(iter(labels)),
                n_windows=len(group),
                n_actionable_windows=len(actionable),
                first_actionable_index=actionable[0].window_index if actionable else None,
            )
        )

    healthy = [o for o in outcomes if o.is_healthy]
    faulted = [o for o in outcomes if not o.is_healthy]
    delays = [
        o.first_actionable_index
        for o in faulted
        if o.detected and o.first_actionable_index is not None
    ]

    return QualityReport(
        label=label,
        n_recordings_examined=len(outcomes),
        n_windows_examined=len(decisions),
        n_healthy_recordings=len(healthy),
        n_faulted_recordings=len(faulted),
        n_false_alarms=sum(1 for o in healthy if o.detected),
        n_misses=sum(1 for o in faulted if not o.detected),
        detection_delays=delays,
        outcomes=outcomes,
    )
