"""Phase 2 report: temperature scaling, then split-conformal prediction sets.

    python -m fleetmind.decision.phase2
    python -m fleetmind.decision.phase2 --depot-id 3 --alpha 0.1 --no-write

Protocol, and the reason for it
-------------------------------
* **Base model: v2-last.** It is the arm whose selection set was never used to
  pick its weights, so the selection set is genuinely free for calibration. Using
  v2-best would mean fitting the temperature on data that had already chosen the
  checkpoint -- a mild but real reuse of the same sample, which inflates the fit.
* **v2-best is run as a sensitivity check** and reported beside it, so the size of
  that reuse effect is visible rather than argued about.
* **Temperature and q_hat are fitted on the SELECTION set only.** Nothing is
  fitted on the test set; the test set is opened once, read-only, to evaluate.
* **v1 is never loaded here**, and the deployed checkpoints are never written.

Nothing from this module is wired into the API, the pipeline or the UI. It writes
only to `mlruns/decision/phase2/`.
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
from fleetmind.decision import calibration as calib
from fleetmind.decision import conformal, probs, retrain, split3, temperature
from fleetmind.ml.model import LABEL_MAP, BearingCNN

OUT_ROOT = Path("mlruns") / "decision" / "phase2"

ARMS = {
    "v2-last": "cnn_state_dict_last.pt",
    "v2-best": "cnn_state_dict.pt",
}

CAVEATS = [
    "Effective calibration sample is 4 RECORDINGS per site, not ~236 windows. "
    "Windows cut from one continuous recording are strongly autocorrelated, so "
    "they are nearer to 4 exchangeable samples than to 236. Split conformal's "
    "coverage guarantee assumes exchangeability, so the nominal level is "
    "optimistic: empirical coverage near the target is 'consistent with' the "
    "target, not a verification of the guarantee.",
    "Normal is absent from BOTH the calibration and test sets (one Normal "
    "recording per site, which the stratified split keeps entirely in training). "
    "Coverage for the healthy class is therefore UNMEASURABLE and is reported as "
    "None, never as a number.",
    "Temperature scaling cannot change any prediction: dividing all logits by one "
    "positive scalar preserves their order. Accuracy and per-class recall are "
    "identical before and after; only the confidences move.",
    "One seed, one split, 4 sites. Treat small differences between arms as noise.",
]


def _load(path: Path) -> BearingCNN:
    model = BearingCNN(num_classes=len(LABEL_MAP))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def run_depot(
    session: Session,
    depot_id: int,
    *,
    alpha: float = 0.10,
    mlruns: Path = Path("mlruns"),
) -> dict:
    split = split3.three_way_split(session, depot_id)
    selection_windows = split3.windows_for(session, split.selection_experiment_ids)
    test_windows = split3.windows_for(session, split.test_experiment_ids)
    if not selection_windows or not test_windows:
        raise ValueError(f"site {depot_id}: empty selection or test set")

    n_cal_rec = len(split.selection_experiment_ids)
    n_test_rec = len(split.test_experiment_ids)

    arms: dict[str, dict] = {}
    for arm, filename in ARMS.items():
        path = retrain.out_dir_for(depot_id, mlruns) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing -- run `python -m fleetmind.decision.retrain` first"
            )
        model = _load(path)
        selection = probs.posteriors(model, selection_windows)
        test = probs.posteriors(model, test_windows)

        # --- temperature, fitted on the selection set only ---------------
        fit = temperature.fit_temperature(
            [p.logits for p in selection],
            [p.true_label for p in selection],
            n_recordings=n_cal_rec,
        )
        test_scaled = temperature.rescale(test, fit.temperature)
        selection_scaled = temperature.rescale(selection, fit.temperature)

        before = calib.measure(test)
        after = calib.measure(test_scaled)

        # --- conformal, calibrated on the (scaled) selection set ---------
        cal_raw = conformal.calibrate(selection, alpha=alpha, n_recordings=n_cal_rec)
        cal_scaled = conformal.calibrate(selection_scaled, alpha=alpha, n_recordings=n_cal_rec)
        eval_raw = conformal.evaluate(test, cal_raw, n_recordings=n_test_rec)
        eval_scaled = conformal.evaluate(test_scaled, cal_scaled, n_recordings=n_test_rec)

        arms[arm] = {
            "checkpoint": str(path),
            "temperature": fit.as_dict(),
            "calibration_before_temperature": before.as_dict(),
            "calibration_after_temperature": after.as_dict(),
            "accuracy_unchanged_by_temperature": before.accuracy == after.accuracy,
            "conformal_uncalibrated_probabilities": eval_raw.as_dict(),
            "conformal_after_temperature": eval_scaled.as_dict(),
        }

    return {
        "depot_id": depot_id,
        "alpha": alpha,
        "base_arm": "v2-last",
        "sensitivity_arm": "v2-best",
        "calibration_set": {
            "recordings": n_cal_rec,
            "windows": sum(split.selection_class_counts.values()),
            "class_counts": split.selection_class_counts,
            "classes_absent": split.classes_absent_from_selection,
        },
        "test_set": {
            "recordings": n_test_rec,
            "windows": sum(split.test_class_counts.values()),
            "class_counts": split.test_class_counts,
            "classes_absent": split.classes_absent_from_test,
            "majority_class_floor": round(split.test_majority_share, 6),
        },
        "arms": arms,
        "caveats": CAVEATS,
    }


def _print_depot(r: dict) -> None:
    c, t = r["calibration_set"], r["test_set"]
    print(
        f"\n{'=' * 78}\n=== site {r['depot_id']}  (alpha={r['alpha']}, target coverage "
        f"{1 - r['alpha']:.0%})\n{'=' * 78}"
    )
    print(
        f"  calibration: {c['recordings']} recordings / {c['windows']} windows "
        f"{c['class_counts']}  absent: {c['classes_absent']}"
    )
    print(
        f"  test       : {t['recordings']} recordings / {t['windows']} windows "
        f"{t['class_counts']}  absent: {t['classes_absent']}"
    )

    for arm, a in r["arms"].items():
        tag = "BASE" if arm == r["base_arm"] else "sensitivity"
        f, b, af = (
            a["temperature"],
            a["calibration_before_temperature"],
            a["calibration_after_temperature"],
        )
        print(f"\n  --- {arm}  [{tag}] ---")
        print(
            f"  T = {f['temperature']:.4f}  ({f['direction']});  "
            f"NLL on calibration {f['nll_before']:.4f} -> {f['nll_after']:.4f} "
            f"({f['nll_improvement']:+.4f})" + ("  [HIT BOUND]" if f["hit_bound"] else "")
        )
        print(f"  {'on the TEST set':<22} {'acc':>7} {'ECE':>8} {'Brier':>8} {'mean conf':>10}")
        print(
            f"  {'before temperature':<22} {b['accuracy']:>7.4f} {b['ece']:>8.4f} "
            f"{b['brier']:>8.4f} {b['mean_confidence']:>10.4f}"
        )
        print(
            f"  {'after temperature':<22} {af['accuracy']:>7.4f} {af['ece']:>8.4f} "
            f"{af['brier']:>8.4f} {af['mean_confidence']:>10.4f}"
        )
        print(
            f"  {'change':<22} {af['accuracy'] - b['accuracy']:>+7.4f} "
            f"{af['ece'] - b['ece']:>+8.4f} {af['brier'] - b['brier']:>+8.4f} "
            f"{af['mean_confidence'] - b['mean_confidence']:>+10.4f}"
            + ("" if a["accuracy_unchanged_by_temperature"] else "   *** ACCURACY MOVED ***")
        )

        for key, title in (
            ("conformal_uncalibrated_probabilities", "conformal on raw probabilities"),
            ("conformal_after_temperature", "conformal after temperature"),
        ):
            e = a[key]
            cal = e["calibration"]
            print(f"\n  {title}:")
            print(
                f"    q_hat {cal['q_hat']}  (admit a label at p >= "
                f"{cal['probability_floor']:.4f}); rank {cal['quantile_index']} of "
                f"{cal['n_calibration_windows']}; certifiable: "
                f"{cal['certifiable_at_this_alpha']}"
            )
            print(
                f"    empirical coverage {e['empirical_coverage']:.4f} vs target "
                f"{e['target_coverage']:.4f}  ({e['coverage_minus_target']:+.4f})"
            )
            print(
                f"    mean set size {e['mean_set_size']:.4f}   sizes {e['set_size_distribution']}"
            )
            per = "  ".join(
                f"{k[:4]}=" + ("UNMEASURABLE" if v is None else f"{v:.3f}")
                for k, v in e["coverage_by_true_class"].items()
            )
            print(f"    coverage by true class: {per}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depot-id", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.10, help="1 - target coverage")
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
        reports = [run_depot(session, d, alpha=args.alpha, mlruns=mlruns) for d in depot_ids]
    finally:
        session.close()

    for r in reports:
        _print_depot(r)

    print(f"\n{'=' * 78}\n=== across sites\n{'=' * 78}")
    for arm in ARMS:
        print(f"\n  {arm}:")
        temps = [r["arms"][arm]["temperature"]["temperature"] for r in reports]
        print(f"    fitted T per site      : {[f'{t:.3f}' for t in temps]}")
        for stage, key in (
            ("before temperature", "calibration_before_temperature"),
            ("after temperature ", "calibration_after_temperature"),
        ):
            e = sum(r["arms"][arm][key]["ece"] for r in reports) / len(reports)
            b = sum(r["arms"][arm][key]["brier"] for r in reports) / len(reports)
            a = sum(r["arms"][arm][key]["accuracy"] for r in reports) / len(reports)
            print(f"    {stage}     : mean acc {a:.4f}  mean ECE {e:.4f}  mean Brier {b:.4f}")
        for stage, key in (
            ("raw probabilities ", "conformal_uncalibrated_probabilities"),
            ("after temperature ", "conformal_after_temperature"),
        ):
            cov = [r["arms"][arm][key]["empirical_coverage"] for r in reports]
            size = [r["arms"][arm][key]["mean_set_size"] for r in reports]
            target = reports[0]["arms"][arm][key]["target_coverage"]
            hit = sum(1 for c in cov if c >= target)
            print(
                f"    conformal, {stage}: mean coverage {sum(cov) / len(cov):.4f} "
                f"(target {target:.2f}, met at {hit}/{len(cov)} sites)  "
                f"mean set size {sum(size) / len(size):.4f}"
            )

    print(f"\n{'=' * 78}\n=== CAVEATS -- these belong with every number above\n{'=' * 78}")
    for i, caveat in enumerate(CAVEATS, 1):
        print(f"\n  {i}. {caveat}")

    if not args.no_write:
        out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in reports:
            p = out_dir / f"depot_{r['depot_id']}.json"
            p.write_text(json.dumps(r, indent=2), encoding="utf-8")
            print(f"\n  wrote {p}")
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "phase": 2,
                    "alpha": args.alpha,
                    "base_arm": "v2-last",
                    "per_arm": {
                        arm: {
                            "fitted_temperature_per_site": [
                                r["arms"][arm]["temperature"]["temperature"] for r in reports
                            ],
                            "mean_ece_before": sum(
                                r["arms"][arm]["calibration_before_temperature"]["ece"]
                                for r in reports
                            )
                            / len(reports),
                            "mean_ece_after": sum(
                                r["arms"][arm]["calibration_after_temperature"]["ece"]
                                for r in reports
                            )
                            / len(reports),
                            "mean_brier_before": sum(
                                r["arms"][arm]["calibration_before_temperature"]["brier"]
                                for r in reports
                            )
                            / len(reports),
                            "mean_brier_after": sum(
                                r["arms"][arm]["calibration_after_temperature"]["brier"]
                                for r in reports
                            )
                            / len(reports),
                            "mean_conformal_coverage_after_temperature": sum(
                                r["arms"][arm]["conformal_after_temperature"]["empirical_coverage"]
                                for r in reports
                            )
                            / len(reports),
                            "mean_set_size_after_temperature": sum(
                                r["arms"][arm]["conformal_after_temperature"]["mean_set_size"]
                                for r in reports
                            )
                            / len(reports),
                        }
                        for arm in ARMS
                    },
                    "caveats": CAVEATS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
