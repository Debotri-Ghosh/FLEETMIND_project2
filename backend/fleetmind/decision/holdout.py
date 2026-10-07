"""The canonical held-out split, and an honest description of what it contains.

Every phase of the decision work scores against the partition this module
returns, so that a Phase 3 false-alarm rate is comparable with a Phase 0 one.

It deliberately reproduces `ml/train.py:load_depot_split` exactly -- same
`stratified_experiment_split`, same `frac=0.8`, same `seed=42` -- because the
published per-site accuracies were measured on that split, and a different
partition would make every before/after comparison meaningless.

**The split has a structural limitation that must travel with its numbers.**
Each site holds exactly one Normal recording, and `stratified_experiment_split`
sends any fault group of size one entirely to train. So no Normal window is ever
held out, at any site. Consequences:

* the published accuracies are three-class figures, not four;
* always predicting OuterRace scores exactly 50% on this validation set;
* the false-alarm rate on a healthy machine is **not measurable** here.

`describe_split()` returns those facts as data rather than prose, so a report
cannot quote an accuracy without being able to state what it was measured over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fleetmind.augmentation.exp_split import stratified_experiment_split
from fleetmind.db.models import Experiment, SignalWindow

#: Matches `ml/train.py:load_depot_split`'s defaults. Changing either value
#: invalidates comparison with every published per-site accuracy.
TRAIN_FRACTION = 0.8
SPLIT_SEED = 42


@dataclass(frozen=True)
class SplitDescription:
    """What a split actually contains -- carried alongside any figure from it."""

    depot_id: int
    train_experiment_ids: list[int]
    holdout_experiment_ids: list[int]
    train_class_counts: dict[str, int] = field(default_factory=dict)
    holdout_class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_train_windows(self) -> int:
        return sum(self.train_class_counts.values())

    @property
    def n_holdout_windows(self) -> int:
        return sum(self.holdout_class_counts.values())

    @property
    def classes_absent_from_holdout(self) -> list[str]:
        """Classes present in training but never held out. Normal, in practice."""
        return sorted(set(self.train_class_counts) - set(self.holdout_class_counts))

    @property
    def holdout_majority_class(self) -> str | None:
        if not self.holdout_class_counts:
            return None
        return max(self.holdout_class_counts, key=lambda k: self.holdout_class_counts[k])

    @property
    def holdout_majority_share(self) -> float:
        """The accuracy of always predicting the held-out majority class.

        This -- not 1/n_classes -- is the floor a reported accuracy must clear
        to mean anything.
        """
        total = self.n_holdout_windows
        if not total:
            return 0.0
        return max(self.holdout_class_counts.values()) / total

    def as_dict(self) -> dict:
        return {
            "depot_id": self.depot_id,
            "n_train_recordings": len(self.train_experiment_ids),
            "n_holdout_recordings": len(self.holdout_experiment_ids),
            "n_train_windows": self.n_train_windows,
            "n_holdout_windows": self.n_holdout_windows,
            "train_class_counts": self.train_class_counts,
            "holdout_class_counts": self.holdout_class_counts,
            "classes_absent_from_holdout": self.classes_absent_from_holdout,
            "holdout_majority_class": self.holdout_majority_class,
            "holdout_majority_share": round(self.holdout_majority_share, 6),
            "split": f"experiment-level, frac={TRAIN_FRACTION}, seed={SPLIT_SEED}",
        }


def _experiment_labels(session: Session, depot_id: int) -> list[tuple[int, str]]:
    """(experiment_id, label) for one site, one row per recording.

    The label is read from the recording's windows rather than from
    `experiments.fault_type`, matching how `ml/train.py` groups for the split.
    """
    rows = session.execute(
        select(SignalWindow.experiment_id, SignalWindow.label)
        .join(Experiment, Experiment.id == SignalWindow.experiment_id)
        .where(Experiment.depot_id == depot_id)
    ).all()
    by_experiment = {int(exp_id): str(label) for exp_id, label in rows}
    return sorted(by_experiment.items())


def _class_counts(session: Session, experiment_ids: list[int]) -> dict[str, int]:
    if not experiment_ids:
        return {}
    rows = session.execute(
        select(SignalWindow.label, func.count())
        .where(SignalWindow.experiment_id.in_(experiment_ids))
        .group_by(SignalWindow.label)
    ).all()
    return {str(label): int(n) for label, n in rows}


def split_for_depot(session: Session, depot_id: int) -> SplitDescription:
    """The canonical train / held-out recording split for one site."""
    labelled = _experiment_labels(session, depot_id)
    train_ids, holdout_ids = stratified_experiment_split(
        labelled, frac=TRAIN_FRACTION, seed=SPLIT_SEED
    )
    return SplitDescription(
        depot_id=depot_id,
        train_experiment_ids=sorted(train_ids),
        holdout_experiment_ids=sorted(holdout_ids),
        train_class_counts=_class_counts(session, train_ids),
        holdout_class_counts=_class_counts(session, holdout_ids),
    )


def holdout_windows(
    session: Session, depot_id: int
) -> list[tuple[int, int, int, list[float], str]]:
    """Every held-out window for one site.

    Returns (window_id, experiment_id, window_index, raw_values, true_label),
    ordered by recording then position within it -- the order a sequential
    detector needs, so later phases do not have to re-sort.
    """
    description = split_for_depot(session, depot_id)
    if not description.holdout_experiment_ids:
        return []
    rows = session.execute(
        select(
            SignalWindow.id,
            SignalWindow.experiment_id,
            SignalWindow.window_index,
            SignalWindow.raw_values,
            SignalWindow.label,
        )
        .where(SignalWindow.experiment_id.in_(description.holdout_experiment_ids))
        .order_by(SignalWindow.experiment_id, SignalWindow.window_index)
    ).all()
    return [
        (int(wid), int(exp), int(idx), [float(v) for v in (vals or [])], str(label))
        for wid, exp, idx, vals, label in rows
    ]
