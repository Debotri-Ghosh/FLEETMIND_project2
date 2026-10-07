"""A three-way recording-level split: train / selection / untouched test.

Why the test set is the *existing* holdout rather than a fresh partition
-----------------------------------------------------------------------
The v1 checkpoints on disk were trained on the 12 recordings that
`holdout.split_for_depot()` calls `train_experiment_ids`, and never saw the 4 it
calls `holdout_experiment_ids`. If Phase 1 drew a brand-new three-way partition
over all 16 recordings, the new test set would almost certainly contain
recordings v1 had trained on -- so v1 would be scored on its own training data
and the v1-vs-v2 comparison would be rigged in v1's favour.

So the test set is held fixed at v1's original 4 held-out recordings, and the
train/selection split is carved out of v1's own 12 training recordings. Both
models are then scored on data **neither has ever trained on**, which is the only
arrangement in which the comparison means anything.

The cost of that choice, stated rather than hidden
--------------------------------------------------
Carving a selection set out of the training data leaves v2 with roughly a third
less data to learn from than v1 had. That biases the comparison **against** v2.
`retrain.py` therefore produces two v2 arms from a single training run -- the
last epoch and the best-on-selection epoch -- so that:

* v2-last vs v1        isolates the effect of the smaller training set;
* v2-best vs v2-last   isolates the effect of best-epoch selection;
* v2-best vs v1        answers the practical question.

Without the v2-last control, a v2-best win could not be told apart from noise and
a v2-best loss could not be told apart from simple data starvation.

The Normal-recording limitation propagates
------------------------------------------
Each site has exactly one Normal recording and
`stratified_experiment_split` assigns a single-recording class entirely to its
train side. Applied twice, that keeps the one Normal recording in the innermost
training set: **neither the selection set nor the test set contains a healthy
machine.** So best-epoch selection is made on a three-class criterion, and a
temperature fitted on this selection set would never see a healthy example.
`Split3Description` reports that as data, not as prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from fleetmind.augmentation.exp_split import stratified_experiment_split
from fleetmind.decision import holdout

#: Fraction of v1's training recordings kept for training; the remainder becomes
#: the selection/calibration set. 0.75 of 12 recordings lands at 8 train / 4
#: selection per site under the stratified splitter's per-class rounding.
TRAIN_FRACTION_WITHIN_TRAIN = 0.75

#: Deliberately the same seed as `holdout.SPLIT_SEED` and `ml/train.py`, so the
#: whole chain of partitions is reproducible from one number.
SPLIT_SEED = holdout.SPLIT_SEED


@dataclass(frozen=True)
class Split3Description:
    """Three disjoint recording sets, and what each one contains."""

    depot_id: int
    train_experiment_ids: list[int]
    selection_experiment_ids: list[int]
    test_experiment_ids: list[int]
    train_class_counts: dict[str, int] = field(default_factory=dict)
    selection_class_counts: dict[str, int] = field(default_factory=dict)
    test_class_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        train, selection, test = (
            set(self.train_experiment_ids),
            set(self.selection_experiment_ids),
            set(self.test_experiment_ids),
        )
        if train & selection or train & test or selection & test:
            raise ValueError(
                "the three recording sets must be disjoint; got overlaps "
                f"train&selection={sorted(train & selection)} "
                f"train&test={sorted(train & test)} "
                f"selection&test={sorted(selection & test)}"
            )
        if not train or not selection or not test:
            raise ValueError(
                "every set must be non-empty: "
                f"{len(train)} train, {len(selection)} selection, {len(test)} test"
            )

    @staticmethod
    def _majority_share(counts: dict[str, int]) -> float:
        total = sum(counts.values())
        return max(counts.values()) / total if total else 0.0

    @property
    def test_majority_share(self) -> float:
        """Accuracy of always predicting the test set's majority class."""
        return self._majority_share(self.test_class_counts)

    @property
    def selection_majority_share(self) -> float:
        return self._majority_share(self.selection_class_counts)

    @property
    def classes_absent_from_selection(self) -> list[str]:
        return sorted(set(self.train_class_counts) - set(self.selection_class_counts))

    @property
    def classes_absent_from_test(self) -> list[str]:
        return sorted(set(self.train_class_counts) - set(self.test_class_counts))

    def as_dict(self) -> dict:
        return {
            "depot_id": self.depot_id,
            "recordings": {
                "train": len(self.train_experiment_ids),
                "selection": len(self.selection_experiment_ids),
                "test": len(self.test_experiment_ids),
            },
            "windows": {
                "train": sum(self.train_class_counts.values()),
                "selection": sum(self.selection_class_counts.values()),
                "test": sum(self.test_class_counts.values()),
            },
            "class_counts": {
                "train": self.train_class_counts,
                "selection": self.selection_class_counts,
                "test": self.test_class_counts,
            },
            "classes_absent_from_selection": self.classes_absent_from_selection,
            "classes_absent_from_test": self.classes_absent_from_test,
            "test_majority_share": round(self.test_majority_share, 6),
            "selection_majority_share": round(self.selection_majority_share, 6),
            "train_experiment_ids": sorted(self.train_experiment_ids),
            "selection_experiment_ids": sorted(self.selection_experiment_ids),
            "test_experiment_ids": sorted(self.test_experiment_ids),
            "derivation": (
                "test = v1's original held-out recordings, unchanged, so neither v1 "
                "nor v2 has trained on it; train+selection = v1's original training "
                f"recordings re-split at frac={TRAIN_FRACTION_WITHIN_TRAIN}, "
                f"seed={SPLIT_SEED}"
            ),
        }


def three_way_split(session: Session, depot_id: int) -> Split3Description:
    """Train / selection / test, with test pinned to v1's original holdout."""
    outer = holdout.split_for_depot(session, depot_id)

    labelled = [
        (exp_id, label)
        for exp_id, label in holdout._experiment_labels(session, depot_id)
        if exp_id in set(outer.train_experiment_ids)
    ]
    inner_train, selection = stratified_experiment_split(
        labelled, frac=TRAIN_FRACTION_WITHIN_TRAIN, seed=SPLIT_SEED
    )

    return Split3Description(
        depot_id=depot_id,
        train_experiment_ids=sorted(inner_train),
        selection_experiment_ids=sorted(selection),
        test_experiment_ids=sorted(outer.holdout_experiment_ids),
        train_class_counts=holdout._class_counts(session, inner_train),
        selection_class_counts=holdout._class_counts(session, selection),
        test_class_counts=holdout._class_counts(session, outer.holdout_experiment_ids),
    )


def windows_for(
    session: Session, experiment_ids: list[int]
) -> list[tuple[int, int, int, list[float], str]]:
    """(window_id, experiment_id, window_index, raw_values, label), time-ordered."""
    from sqlalchemy import select

    from fleetmind.db.models import SignalWindow

    if not experiment_ids:
        return []
    rows = session.execute(
        select(
            SignalWindow.id,
            SignalWindow.experiment_id,
            SignalWindow.window_index,
            SignalWindow.raw_values,
            SignalWindow.label,
        )
        .where(SignalWindow.experiment_id.in_(experiment_ids))
        .order_by(SignalWindow.experiment_id, SignalWindow.window_index)
    ).all()
    return [
        (int(wid), int(exp), int(idx), [float(v) for v in (vals or [])], str(label))
        for wid, exp, idx, vals, label in rows
    ]
