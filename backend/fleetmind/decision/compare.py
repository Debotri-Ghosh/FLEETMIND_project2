"""Phase 1 report: v1 vs v2 on the untouched test set.

    python -m fleetmind.decision.compare
    python -m fleetmind.decision.compare --depot-id 2 --no-write

Three arms, scored by identical code on identical data that **none** of them
trained on:

* **v1**       `mlruns/depot_N/cnn_state_dict.pt` -- the deployed checkpoint, the
               last epoch of a run over 12 recordings.
* **v2-last**  `mlruns/depot_N_v2/cnn_state_dict_last.pt` -- the last epoch of a
               run over 8 recordings. The control for the smaller training set.
* **v2-best**  `mlruns/depot_N_v2/cnn_state_dict.pt` -- the best-on-selection
               epoch of that same run. The thing being tested.

Reading the three differences:

* v2-last - v1      the cost of carving a selection set out of the training data
* v2-best - v2-last what best-epoch selection is worth, holding data fixed
* v2-best - v1      whether the trade is worth making in practice

This is the only module that opens the test set, and it opens it read-only. It
writes no checkpoints and never touches `mlruns/depot_N`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.db.models import Depot
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.decision import baseline, calibration, probs, quality, retrain, split3
from fleetmind.ml.model import LABEL_MAP, BearingCNN

OUT_ROOT = Path("mlruns") / "decision" / "phase1"


def _load(path: Path) -> BearingCNN:
    model = BearingCNN(num_classes=len(LABEL_MAP))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def arms_for(depot_id: int, mlruns: Path) -> dict[str, Path]:
    v2 = retrain.out_dir_for(depot_id, mlruns)
    return {
        "v1 (deployed, last epoch, 12 recordings)": mlruns
        / f"depot_{depot_id}"
        / "cnn_state_dict.pt",
        "v2-last (last epoch, 8 recordings)": v2 / "cnn_state_dict_last.pt",
        "v2-best (best-on-selection, 8 recordings)": v2 / "cnn_state_dict.pt",
    }


def compare_depot(session: Session, depot_id: int, mlruns: Path = Path("mlruns")) -> dict:
    split = split3.three_way_split(session, depot_id)
    test_windows = split3.windows_for(session, split.test_experiment_ids)
    if not test_windows:
        raise ValueError(f"site {depot_id} has no test windows")

    load_hp = session.execute(select(Depot.load_hp).where(Depot.id == depot_id)).scalar()
    load_hp = None if load_hp is None else int(load_hp)

    results: dict[str, dict] = {}
    for arm, path in arms_for(depot_id, mlruns).items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing -- run `python -m fleetmind.decision.retrain` first"
            )
        posteriors = probs.posteriors(_load(path), test_windows)
        calib = calibration.measure(posteriors)
        decisions = baseline.decide_all(test_windows, posteriors, load_hp)
        qual = quality.score(decisions, label=f"heuristic baseline over {arm}")
        results[arm] = {
            "checkpoint": str(path),
            "calibration": calib.as_dict(),
            "decision_quality": qual.as_dict(),
        }

    def acc(arm_substring: str) -> float:
        for arm, r in results.items():
            if arm.startswith(arm_substring):
                return float(r["calibration"]["accuracy"])
        raise KeyError(arm_substring)

    v1, v2_last, v2_best = acc("v1"), acc("v2-last"), acc("v2-best")
    return {
        "depot_id": depot_id,
        "test_set": {
            "recordings": len(split.test_experiment_ids),
            "windows": sum(split.test_class_counts.values()),
            "class_counts": split.test_class_counts,
            "classes_absent": split.classes_absent_from_test,
            "majority_class_floor": round(split.test_majority_share, 6),
            "trained_on_by_any_arm": False,
        },
        "split": split.as_dict(),
        "arms": results,
        "deltas": {
            "v2_last_minus_v1": round(v2_last - v1, 6),
            "v2_best_minus_v2_last": round(v2_best - v2_last, 6),
            "v2_best_minus_v1": round(v2_best - v1, 6),
        },
    }


def _print(r: dict) -> None:
    t = r["test_set"]
    print(f"\n=== site {r['depot_id']} -- untouched test set ===")
    print(
        f"  {t['recordings']} recordings / {t['windows']} windows  {t['class_counts']}\n"
        f"  classes absent: {t['classes_absent'] or 'none'}   "
        f"majority-class floor: {t['majority_class_floor']:.4f}"
    )
    print(f"  {'arm':<44} {'acc':>7} {'ECE':>7} {'Brier':>7}  per-class recall")
    for arm, a in r["arms"].items():
        c = a["calibration"]
        recall = " ".join(f"{k[:4]}={v:.2f}" for k, v in sorted(c["per_class_recall"].items()))
        print(f"  {arm:<44} {c['accuracy']:>7.4f} {c['ece']:>7.4f} {c['brier']:>7.4f}  {recall}")
    d = r["deltas"]
    print(
        f"  deltas: smaller train set {d['v2_last_minus_v1']:+.4f} | "
        f"best-epoch selection {d['v2_best_minus_v2_last']:+.4f} | "
        f"net v2-best vs v1 {d['v2_best_minus_v1']:+.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depot-id", type=int, default=None)
    parser.add_argument("--mlruns", default="mlruns")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    mlruns = Path(args.mlruns)
    session = make_session_factory(create_db_engine())()
    try:
        depot_ids = (
            [args.depot_id]
            if args.depot_id is not None
            else [int(x[0]) for x in session.execute(select(Depot.id).order_by(Depot.id)).all()]
        )
        reports = [compare_depot(session, d, mlruns) for d in depot_ids]
    finally:
        session.close()

    for r in reports:
        _print(r)

    print("\n=== across sites, on the untouched test set ===")
    arm_names = list(reports[0]["arms"].keys())
    print(f"  {'arm':<44} {'mean acc':>9} {'mean ECE':>9} {'mean Brier':>11}")
    means: dict[str, float] = {}
    for arm in arm_names:
        a = sum(r["arms"][arm]["calibration"]["accuracy"] for r in reports) / len(reports)
        e = sum(r["arms"][arm]["calibration"]["ece"] for r in reports) / len(reports)
        b = sum(r["arms"][arm]["calibration"]["brier"] for r in reports) / len(reports)
        means[arm] = a
        print(f"  {arm:<44} {a:>9.4f} {e:>9.4f} {b:>11.4f}")

    for key, label in (
        ("v2_last_minus_v1", "cost of the smaller training set (v2-last - v1)"),
        ("v2_best_minus_v2_last", "value of best-epoch selection (v2-best - v2-last)"),
        ("v2_best_minus_v1", "net effect (v2-best - v1)"),
    ):
        vals = [r["deltas"][key] for r in reports]
        wins = sum(1 for v in vals if v > 0)
        print(
            f"  {label:<48} mean {sum(vals) / len(vals):+.4f}   "
            f"per site {[f'{v:+.4f}' for v in vals]}   improved at {wins}/{len(vals)} sites"
        )

    floor = reports[0]["test_set"]["majority_class_floor"]
    print(f"\n  majority-class floor on every site's test set: {floor:.4f}")
    print("  n.b. the test set holds no Normal windows, so every figure above is")
    print("       three-class and the false-alarm rate remains unmeasurable.")

    if not args.no_write:
        out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in reports:
            p = out_dir / f"depot_{r['depot_id']}.json"
            p.write_text(json.dumps(r, indent=2), encoding="utf-8")
            print(f"  wrote {p}")
        summary = out_dir / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "phase": 1,
                    "test_set": "v1's original held-out recordings, trained on by no arm",
                    "mean_accuracy_by_arm": {k: round(v, 6) for k, v in means.items()},
                    "per_site": [
                        {
                            "depot_id": r["depot_id"],
                            "deltas": r["deltas"],
                            "accuracy_by_arm": {
                                k: v["calibration"]["accuracy"] for k, v in r["arms"].items()
                            },
                            "ece_by_arm": {
                                k: v["calibration"]["ece"] for k, v in r["arms"].items()
                            },
                        }
                        for r in reports
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {summary}")


if __name__ == "__main__":
    main()
