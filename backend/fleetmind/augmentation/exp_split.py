"""Experiment-level split utilities (review-honest evaluation).

Phase 2 used a WINDOW-level 80/20 split: windows from the same .mat recording
landed on both sides, inflating accuracy to 100%. Phase 3+ evaluation splits
by EXPERIMENT (whole .mat recordings are held out), so the test set measures
generalization to unseen recordings.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.db.models import Experiment


def experiment_ids_for_depot(session: Session, depot_id: int) -> list[int]:
    rows = session.execute(
        select(Experiment.id).where(Experiment.depot_id == depot_id).order_by(Experiment.id)
    ).all()
    return [r[0] for r in rows]


def experiment_split(
    experiment_ids: list[int], frac: float = 0.7, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Deterministic shuffle split of experiment ids into (train, test)."""
    import random

    rng = random.Random(seed)
    ids = list(experiment_ids)
    rng.shuffle(ids)
    cut = max(1, int(len(ids) * frac))
    return ids[:cut], ids[cut:]


def stratified_experiment_split(
    experiment_faults: list[tuple[int, str]], frac: float = 0.7, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Split experiment ids so every fault_type appears in BOTH train and test.

    Shuffles each fault_type group with the seeded RNG, then rounds per-group
    train counts down (keeping >=1 in train when the group has >=2) and puts
    at least one of every multi-experiment group into test as well.
    """
    import random

    rng = random.Random(seed)
    by_fault: dict[str, list[int]] = {}
    for exp_id, fault in experiment_faults:
        by_fault.setdefault(fault, []).append(exp_id)
    train: list[int] = []
    test: list[int] = []
    for _fault, ids in by_fault.items():
        ids = list(ids)
        rng.shuffle(ids)
        if len(ids) == 1:
            train += ids
            continue
        n_train = max(1, min(len(ids) - 1, int(len(ids) * frac)))
        train += ids[:n_train]
        test += ids[n_train:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
