"""Score the decision the product actually makes: one per machine, not per window.

    python -m fleetmind.decision.recording
    python -m fleetmind.decision.recording --depot-id 2 --no-write

Why this exists
---------------
Every accuracy figure this project quotes is a **window**-level figure, and a
window is 0.17 seconds of vibration. The product does not act on a window: it
flags a *machine*. A recording-level number is therefore the one that matches
what the system does, and it has never been measured.

Method
------
For each held-out recording, average the model's **logits** across all of its
windows, take the argmax of that average, and compare it to the recording's
documented label.

Logit averaging treats the mean as a sum of unnormalised evidence, which is the
natural reading of a log-linear model. It has a real weakness, found by a test
rather than reasoned about: because logits are **unbounded**, one extreme window
can dominate the mean. With one window at logit 40 and nine at logit 1.0 for a
different class, the single outlier wins. Probability averaging is the *more
robust* rule here, since each window contributes at most 1/n, and majority
voting is the most robust of the three and the least sensitive to magnitude.

All three are therefore computed and reported side by side, so the choice of
aggregation is inspectable rather than an invisible assumption. Logit averaging
is reported first because it is what this task specified.

What this is NOT
----------------
It is not a better accuracy figure by assumption. Averaging over ~59 windows
should reduce variance, but it can also wash out a fault that is only visible in
part of a recording, and with **four held-out recordings per site** the estimate
is extremely coarse: one recording is 25% of a site's score. The output prints n
explicitly for exactly that reason.

Read-only: no database writes, no API or UI wiring.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.db.models import Depot
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.decision import holdout, probs
from fleetmind.ml.model import LABEL_MAP

OUT_ROOT = Path("mlruns") / "decision" / "recording_level"

#: The three aggregation rules, all reported so the choice is inspectable.
VOTES = ("logits", "probs", "majority")


@dataclass(frozen=True)
class RecordingDecision:
    """One recording's aggregated decision under one aggregation rule."""

    experiment_id: int
    true_label: str
    predicted_label: str
    n_windows: int
    vote: str

    @property
    def correct(self) -> bool:
        return self.predicted_label == self.true_label

    def as_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "n_windows": self.n_windows,
            "correct": self.correct,
        }


def _argmax_label(values: list[float]) -> str:
    return LABEL_MAP[max(range(len(values)), key=values.__getitem__)]


def aggregate(posteriors: list, vote: str = "logits") -> list[RecordingDecision]:
    """One decision per recording, from that recording's windows.

    Raises on empty input rather than returning an empty list, because a caller
    that scored zero recordings would otherwise compute 0/0 and report something
    that looks like a result.
    """
    if vote not in VOTES:
        raise ValueError(f"vote must be one of {VOTES}, got {vote!r}")
    if not posteriors:
        raise ValueError("aggregate() received no windows; refusing to score nothing")

    grouped: dict[int, list] = {}
    for p in posteriors:
        grouped.setdefault(p.experiment_id, []).append(p)

    out: list[RecordingDecision] = []
    for experiment_id, group in sorted(grouped.items()):
        labels = {p.true_label for p in group}
        if len(labels) != 1:
            raise ValueError(
                f"recording {experiment_id} carries more than one label: {sorted(labels)}"
            )
        true_label = next(iter(labels))
        n = len(group)

        if vote == "logits":
            if any(not p.logits for p in group):
                raise ValueError(
                    "logit averaging needs logits; these posteriors carry none "
                    "(were they built by an older probs.posteriors?)"
                )
            summed = [sum(p.logits[k] for p in group) / n for k in range(len(LABEL_MAP))]
            predicted = _argmax_label(summed)
        elif vote == "probs":
            summed = [sum(p.probabilities[k] for p in group) / n for k in range(len(LABEL_MAP))]
            predicted = _argmax_label(summed)
        else:  # majority vote over per-window argmaxes
            counts: dict[str, int] = {}
            for p in group:
                counts[p.predicted_label] = counts.get(p.predicted_label, 0) + 1
            # LABEL_MAP order breaks ties deterministically.
            predicted = max(LABEL_MAP, key=lambda lbl: (counts.get(lbl, 0), -LABEL_MAP.index(lbl)))

        out.append(
            RecordingDecision(
                experiment_id=experiment_id,
                true_label=true_label,
                predicted_label=predicted,
                n_windows=n,
                vote=vote,
            )
        )
    return out


@dataclass(frozen=True)
class RecordingScore:
    vote: str
    n_recordings_scored: int
    n_correct: int
    n_windows_examined: int
    decisions: list[RecordingDecision]
    classes_absent: list[str]

    @property
    def accuracy(self) -> float:
        if self.n_recordings_scored == 0:
            raise ValueError("no recordings were scored; there is no accuracy to report")
        return self.n_correct / self.n_recordings_scored

    @property
    def per_class(self) -> dict[str, str]:
        """ "correct/total" per true class, as a string so n is never hidden."""
        out: dict[str, str] = {}
        for label in sorted({d.true_label for d in self.decisions}):
            of_class = [d for d in self.decisions if d.true_label == label]
            out[label] = f"{sum(1 for d in of_class if d.correct)}/{len(of_class)}"
        return out

    def as_dict(self) -> dict:
        return {
            "vote": self.vote,
            "n_recordings_scored": self.n_recordings_scored,
            "n_correct": self.n_correct,
            "n_windows_examined": self.n_windows_examined,
            "accuracy": round(self.accuracy, 6),
            "per_class_correct_over_total": self.per_class,
            "classes_absent_from_holdout": self.classes_absent,
            "decisions": [d.as_dict() for d in self.decisions],
        }


def score(posteriors: list, vote: str = "logits", classes_absent: list[str] | None = None):
    decisions = aggregate(posteriors, vote)
    return RecordingScore(
        vote=vote,
        n_recordings_scored=len(decisions),
        n_correct=sum(1 for d in decisions if d.correct),
        n_windows_examined=sum(d.n_windows for d in decisions),
        decisions=decisions,
        classes_absent=classes_absent or [],
    )


def report_for_depot(session: Session, depot_id: int, mlruns: Path = probs.MLRUNS) -> dict:
    """Recording-level and window-level accuracy for the DEPLOYED v1 checkpoint."""
    name = session.execute(select(Depot.name).where(Depot.id == depot_id)).scalar()
    split = holdout.split_for_depot(session, depot_id)
    windows = holdout.holdout_windows(session, depot_id)
    if not windows:
        raise ValueError(f"site {depot_id} has no held-out windows")

    model = probs.load_model(depot_id, mlruns)  # mlruns/depot_N -- the deployed file
    posteriors = probs.posteriors(model, windows)

    window_correct = sum(1 for p in posteriors if p.correct)
    scores = {vote: score(posteriors, vote, split.classes_absent_from_holdout) for vote in VOTES}

    return {
        "depot_id": depot_id,
        "site_name": str(name),
        "checkpoint": str(probs.checkpoint_path(depot_id, mlruns)),
        "holdout": {
            "n_recordings": len(split.holdout_experiment_ids),
            "n_windows": len(windows),
            "class_counts": split.holdout_class_counts,
            "classes_absent": split.classes_absent_from_holdout,
            "majority_class_floor": round(split.holdout_majority_share, 6),
        },
        "window_level": {
            "correct": window_correct,
            "total": len(posteriors),
            "accuracy": round(window_correct / len(posteriors), 6),
        },
        "recording_level": {vote: s.as_dict() for vote, s in scores.items()},
    }


def _print_depot(r: dict) -> None:
    h, w = r["holdout"], r["window_level"]
    print(f"\n=== site {r['depot_id']} -- {r['site_name']} ===")
    print(
        f"  held out: n = {h['n_recordings']} RECORDINGS ({h['n_windows']} windows)  "
        f"{h['class_counts']}"
    )
    print(
        f"  classes absent from held-out: {h['classes_absent'] or 'none'}   "
        f"majority-class floor (window level): {h['majority_class_floor']:.4f}"
    )
    print(f"  window-level    : {w['correct']}/{w['total']} = {w['accuracy']:.4f}")
    for vote in VOTES:
        s = r["recording_level"][vote]
        print(
            f"  recording-level ({vote:<8}): {s['n_correct']}/{s['n_recordings_scored']} = "
            f"{s['accuracy']:.4f}   per class {s['per_class_correct_over_total']}"
        )
    print("  per-recording detail (logit averaging):")
    for d in r["recording_level"]["logits"]["decisions"]:
        mark = "ok " if d["correct"] else "MISS"
        print(
            f"    [{mark}] recording {d['experiment_id']:>3}  true={d['true_label']:<10} "
            f"predicted={d['predicted_label']:<10} over {d['n_windows']} windows"
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
        reports = [report_for_depot(session, d, mlruns) for d in depot_ids]
    finally:
        session.close()

    for r in reports:
        _print_depot(r)

    total_rec = sum(r["recording_level"]["logits"]["n_recordings_scored"] for r in reports)
    total_ok = sum(r["recording_level"]["logits"]["n_correct"] for r in reports)
    total_win = sum(r["window_level"]["total"] for r in reports)
    total_win_ok = sum(r["window_level"]["correct"] for r in reports)

    print(f"\n{'=' * 74}\n=== across sites -- DEPLOYED v1 checkpoints\n{'=' * 74}")
    print(f"  {'site':<6} {'window-level':>22} {'recording-level (logits)':>28}")
    for r in reports:
        w = r["window_level"]
        s = r["recording_level"]["logits"]
        print(
            f"  {r['depot_id']:<6} {w['correct']:>6}/{w['total']:<4} = {w['accuracy']:.4f}  "
            f"{s['n_correct']:>14}/{s['n_recordings_scored']:<2} = {s['accuracy']:.4f}"
        )
    print(
        f"  {'ALL':<6} {total_win_ok:>6}/{total_win:<4} = {total_win_ok / total_win:.4f}  "
        f"{total_ok:>14}/{total_rec:<2} = {total_ok / total_rec:.4f}"
    )
    for vote in VOTES:
        ok = sum(r["recording_level"][vote]["n_correct"] for r in reports)
        n = sum(r["recording_level"][vote]["n_recordings_scored"] for r in reports)
        print(f"  aggregation '{vote}': {ok}/{n} = {ok / n:.4f}")

    print(f"\n{'=' * 74}\n=== READ THESE WITH THE NUMBERS\n{'=' * 74}")
    print(
        f"\n  1. n = {total_rec} RECORDINGS in total, {total_rec // len(reports)} per site. "
        "One recording is\n     "
        f"{100 / (total_rec // len(reports)):.0f}% of a site's score, so a single flip moves a "
        "site by that much.\n     Treat per-site recording-level accuracy as very coarse."
    )
    absent = sorted({c for r in reports for c in r["holdout"]["classes_absent"]})
    print(
        f"\n  2. {', '.join(absent) or 'No class'} is absent from the held-out set at every "
        "site, so this is a\n     THREE-class figure, not four. No healthy machine is ever "
        "scored, and the\n     false-alarm rate on a healthy machine remains unmeasurable."
    )
    print(
        "\n  3. These are the DEPLOYED checkpoints (mlruns/depot_N/cnn_state_dict.pt), the\n"
        "     files that produced every stored prediction and decision."
    )

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
                    "checkpoints": "deployed v1 (mlruns/depot_N/cnn_state_dict.pt)",
                    "n_recordings_total": total_rec,
                    "n_recordings_per_site": total_rec // len(reports),
                    "window_level_overall": {
                        "correct": total_win_ok,
                        "total": total_win,
                        "accuracy": round(total_win_ok / total_win, 6),
                    },
                    "recording_level_overall": {
                        vote: {
                            "correct": sum(
                                r["recording_level"][vote]["n_correct"] for r in reports
                            ),
                            "total": total_rec,
                            "accuracy": round(
                                sum(r["recording_level"][vote]["n_correct"] for r in reports)
                                / total_rec,
                                6,
                            ),
                        }
                        for vote in VOTES
                    },
                    "per_site": [
                        {
                            "depot_id": r["depot_id"],
                            "window_level": r["window_level"],
                            "recording_level_logits": {
                                "correct": r["recording_level"]["logits"]["n_correct"],
                                "total": r["recording_level"]["logits"]["n_recordings_scored"],
                                "accuracy": r["recording_level"]["logits"]["accuracy"],
                            },
                        }
                        for r in reports
                    ],
                    "classes_absent_from_holdout": absent,
                    "three_class_not_four": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
