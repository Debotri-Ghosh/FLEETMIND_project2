"""Phase 0 report: the before-numbers, from one command.

    python -m fleetmind.decision.report                 # all four sites
    python -m fleetmind.decision.report --depot-id 3
    python -m fleetmind.decision.report --no-write      # print only

Writes `mlruns/decision/phase0/depot_{id}.json` plus a `summary.json`, and
prints a readable summary. Read-only with respect to the database: it writes no
predictions, no decisions and no equipment status, so it is safe to run against
the live demo state at any point in a replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from fleetmind.db.models import Depot
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.decision import baseline, calibration, holdout, probs

OUT_ROOT = Path("mlruns") / "decision" / "phase0"


def report_for_depot(session, depot_id: int, mlruns: Path = probs.MLRUNS) -> dict:
    """Everything Phase 0 measures for one site."""
    row = session.execute(
        select(Depot.name, Depot.load_hp).where(Depot.id == depot_id)
    ).one_or_none()
    if row is None:
        raise ValueError(f"no site with id {depot_id}")
    site_name, load_hp = str(row[0]), (None if row[1] is None else int(row[1]))

    split = holdout.split_for_depot(session, depot_id)
    windows = holdout.holdout_windows(session, depot_id)
    if not windows:
        raise ValueError(f"site {depot_id} has no held-out windows")

    model = probs.load_model(depot_id, mlruns)
    posteriors = probs.posteriors(model, windows)

    calib = calibration.measure(posteriors)
    decisions = baseline.decide_all(windows, posteriors, load_hp)
    qual = quality_score(decisions)

    return {
        "depot_id": depot_id,
        "site_name": site_name,
        "load_hp": load_hp,
        "checkpoint": str(probs.checkpoint_path(depot_id, mlruns)),
        "split": split.as_dict(),
        "calibration": calib.as_dict(),
        "decision_quality": qual.as_dict(),
    }


def quality_score(decisions: list):
    from fleetmind.decision import quality

    return quality.score(decisions, label="heuristic baseline (agents/agents.py)")


def _print_depot(r: dict) -> None:
    s, c, q = r["split"], r["calibration"], r["decision_quality"]
    print(f"\n=== site {r['depot_id']} -- {r['site_name']} ({r['load_hp']} HP) ===")
    print(
        f"  held-out: {s['n_holdout_recordings']} recordings / "
        f"{s['n_holdout_windows']} windows   (train {s['n_train_recordings']}/"
        f"{s['n_train_windows']})"
    )
    print(f"  held-out classes      : {s['holdout_class_counts']}")
    print(f"  ABSENT from held-out  : {s['classes_absent_from_holdout'] or 'none'}")
    print(
        f"  majority-class floor  : always-{s['holdout_majority_class']} scores "
        f"{s['holdout_majority_share']:.4f}"
    )
    print("  --- calibration (measured, not yet corrected) ---")
    print(
        f"  accuracy {c['accuracy']:.4f}   mean confidence {c['mean_confidence']:.4f}   "
        f"over-confidence {c['overconfidence']:+.4f}"
    )
    print(
        f"  ECE {c['ece']:.4f}   Brier {c['brier']:.4f}   "
        f"populated confidence bins {c['n_bins_populated']}/10"
    )
    print(f"  per-class recall      : {c['per_class_recall']}")
    print(f"  predicted class share : {c['predicted_class_shares']}")
    print(f"  --- decision quality: {q['decision_path']} ---")
    print(
        f"  recordings examined {q['n_recordings_examined']}  "
        f"(healthy {q['n_healthy_recordings']}, faulted {q['n_faulted_recordings']}); "
        f"windows examined {q['n_windows_examined']}"
    )
    fa, mr, dd = q["false_alarm_rate"], q["miss_rate"], q["mean_detection_delay_windows"]
    print(f"  false-alarm rate : {'NOT MEASURABLE' if fa is None else f'{fa:.4f}'}")
    print(f"  miss rate        : {'NOT MEASURABLE' if mr is None else f'{mr:.4f}'}")
    print(f"  detection delay  : {'n/a' if dd is None else f'{dd:.2f} windows'}")
    for metric, reason in q["not_measurable"].items():
        print(f"  ! {metric} unmeasurable -- {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depot-id", type=int, default=None, help="one site; default all four")
    parser.add_argument("--mlruns", default="mlruns", help="where the checkpoints live")
    parser.add_argument("--out-dir", default=None, help=f"default {OUT_ROOT}")
    parser.add_argument("--no-write", action="store_true", help="print only")
    args = parser.parse_args()

    mlruns = Path(args.mlruns)
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT

    session = make_session_factory(create_db_engine())()
    try:
        depot_ids = (
            [args.depot_id]
            if args.depot_id is not None
            else [int(r[0]) for r in session.execute(select(Depot.id).order_by(Depot.id)).all()]
        )
        reports = []
        for depot_id in depot_ids:
            r = report_for_depot(session, depot_id, mlruns)
            reports.append(r)
            _print_depot(r)
    finally:
        session.close()

    # A cross-site view, because the single-Normal-recording limitation is only
    # escapable by comparing across sites, and that has to be visible here.
    print("\n=== across sites ===")
    print(
        f"  mean accuracy        : {sum(r['calibration']['accuracy'] for r in reports) / len(reports):.4f}"
    )
    print(
        f"  mean ECE             : {sum(r['calibration']['ece'] for r in reports) / len(reports):.4f}"
    )
    print(
        f"  mean Brier           : {sum(r['calibration']['brier'] for r in reports) / len(reports):.4f}"
    )
    print(
        f"  mean over-confidence : {sum(r['calibration']['overconfidence'] for r in reports) / len(reports):+.4f}"
    )
    healthy = sum(r["decision_quality"]["n_healthy_recordings"] for r in reports)
    print(f"  healthy recordings held out anywhere: {healthy}")
    if healthy == 0:
        print(
            "  => the false-alarm rate is unmeasurable on this split at every site.\n"
            "     Phase 0's finding, not an omission: it needs a leave-one-site-out\n"
            "     design or a split that holds back the single Normal recording."
        )

    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in reports:
            path = out_dir / f"depot_{r['depot_id']}.json"
            path.write_text(json.dumps(r, indent=2), encoding="utf-8")
            print(f"\nwrote {path}")
        summary = out_dir / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "phase": 0,
                    "decision_path_scored": "heuristic baseline (agents/agents.py)",
                    "sites": [
                        {
                            "depot_id": r["depot_id"],
                            "accuracy": r["calibration"]["accuracy"],
                            "ece": r["calibration"]["ece"],
                            "brier": r["calibration"]["brier"],
                            "overconfidence": r["calibration"]["overconfidence"],
                            "majority_class_floor": r["split"]["holdout_majority_share"],
                            "false_alarm_rate": r["decision_quality"]["false_alarm_rate"],
                            "miss_rate": r["decision_quality"]["miss_rate"],
                            "mean_detection_delay_windows": r["decision_quality"][
                                "mean_detection_delay_windows"
                            ],
                            "classes_absent_from_holdout": r["split"][
                                "classes_absent_from_holdout"
                            ],
                        }
                        for r in reports
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {summary}")


if __name__ == "__main__":
    main()
