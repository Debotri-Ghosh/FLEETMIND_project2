"""Healthy recordings a site's model has never seen: the other sites' Normal runs.

Every site has exactly one Normal recording, and the stratified split keeps it in
that site's own training set -- so no healthy machine is ever held out within a
site, and the false-alarm rate has been unmeasurable since Phase 0.

But each per-site model is trained **only on its own site's recordings**
(`ml/train.py:load_depot_split`). Site *k*'s model has therefore never seen site *j*'s
Normal recording for any *j* != *k*. Scoring each model against the other three
sites' healthy recordings gives 4 x 3 = 12 genuinely unseen healthy-machine
evaluations -- the first false-alarm measurement this project can make.

**The caveat travels with every number from here:** the other sites run at a
different motor load, so this is a domain shift. It probably *overstates* the
false-alarm rate a site would see on its own healthy machines, and is always
reported as "cross-site false-alarm rate", never as plain false-alarm rate.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.db.models import Depot, Experiment, SignalWindow

HEALTHY_LABEL = "Normal"


def normal_experiment_ids(session: Session, depot_id: int) -> list[int]:
    rows = session.execute(
        select(SignalWindow.experiment_id)
        .join(Experiment, Experiment.id == SignalWindow.experiment_id)
        .where(Experiment.depot_id == depot_id, SignalWindow.label == HEALTHY_LABEL)
        .distinct()
    ).all()
    return sorted(int(r[0]) for r in rows)


def unseen_healthy_windows(
    session: Session, model_depot_id: int
) -> list[tuple[int, list[tuple[int, int, int, list[float], str]]]]:
    """[(source_depot_id, windows), ...] for every OTHER site's Normal recording.

    Grouped by source site because a caller running the heuristic baseline must use
    the *source* machine's motor load, not the model's site.
    """
    others = [
        int(r[0])
        for r in session.execute(
            select(Depot.id).where(Depot.id != model_depot_id).order_by(Depot.id)
        ).all()
    ]
    out = []
    for source in others:
        ids = normal_experiment_ids(session, source)
        if not ids:
            continue
        rows = session.execute(
            select(
                SignalWindow.id,
                SignalWindow.experiment_id,
                SignalWindow.window_index,
                SignalWindow.raw_values,
                SignalWindow.label,
            )
            .where(SignalWindow.experiment_id.in_(ids))
            .order_by(SignalWindow.experiment_id, SignalWindow.window_index)
        ).all()
        windows = [
            (int(w), int(e), int(i), [float(v) for v in (vals or [])], str(lbl))
            for w, e, i, vals, lbl in rows
        ]
        out.append((source, windows))
    return out


def assert_unseen(session: Session, model_depot_id: int, experiment_ids: list[int]) -> None:
    """Refuse to call a recording 'unseen' if it belongs to the model's own site.

    The whole measurement rests on this. A bug that let a site's own Normal recording
    in would report a false-alarm rate on training data, which looks like a result
    and is not one.
    """
    if not experiment_ids:
        raise ValueError("no healthy recordings to check")
    own = session.execute(
        select(Experiment.id).where(
            Experiment.id.in_(experiment_ids), Experiment.depot_id == model_depot_id
        )
    ).all()
    if own:
        raise AssertionError(
            f"recordings {sorted(r[0] for r in own)} belong to site {model_depot_id}, whose "
            "model trained on them -- they are not unseen"
        )
