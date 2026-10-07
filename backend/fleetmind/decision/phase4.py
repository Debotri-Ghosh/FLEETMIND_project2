"""Phase 4 report: the sequential test against the "any window" rules.

    python -m fleetmind.decision.phase4
    python -m fleetmind.decision.phase4 --no-write

Same data as Phase 3 (deployed v1 checkpoints; each site's own held-out faulted
recordings; the other sites' unseen healthy recordings), three decision rules:

* heuristic baseline, flag if ANY window is HIGH/CRITICAL       (Phase 0)
* expected-cost policy, flag if ANY window chooses maintenance  (Phase 3)
* repeated SPRT over each recording's windows in time order     (this phase)

On this data Phase 3 found fault presence perfectly separable, so the headline
error rates cannot improve. What a sequential test is *for* is robustness: one
misclassified window should not flag a healthy machine. So the second half is a
stress test with real evidence: into every healthy recording, inject k consecutive
copies of a REAL faulted window's evidence -- the model's typical (median) faulted
window, and its most confident one -- and find the smallest k that flags it. The
"any window" rule needs k = 1 by construction; the question is what the SPRT needs.

Read-only. Writes `mlruns/decision/phase4/`.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from sqlalchemy import select

from fleetmind.db.models import Depot
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.decision import phase3, policy, probs, sequential

OUT_ROOT = Path("mlruns") / "decision" / "phase4"
MAX_INJECTED = 50


def _by_recording(posteriors: list) -> dict[int, list]:
    grouped: dict[int, list] = {}
    for p in posteriors:
        grouped.setdefault(p.experiment_id, []).append(p)
    for v in grouped.values():
        v.sort(key=lambda p: p.window_index)
    return grouped


def _llrs(posts: list) -> list[float]:
    return [sequential.llr(policy.p_faulted(p.probabilities)) for p in posts]


def _min_k_to_flag(healthy_llrs: list[float], injected: float) -> int | None:
    """Smallest run of injected windows, placed mid-recording, that flags it."""
    mid = len(healthy_llrs) // 2
    for k in range(1, MAX_INJECTED + 1):
        stream = healthy_llrs[:mid] + [injected] * k + healthy_llrs[mid:]
        if sequential.run(stream).flagged:
            return k
    return None


def run_report(session, mlruns: Path = probs.MLRUNS) -> dict:
    depot_ids = [int(r[0]) for r in session.execute(select(Depot.id).order_by(Depot.id)).all()]
    a, b = sequential.boundaries()

    per_site = []
    recordings = []
    stress = []
    for depot_id in depot_ids:
        c = phase3.collect(session, depot_id, mlruns)

        faulted = _by_recording(c["faulted_post"])
        healthy: dict[int, list] = {}
        for h in c["healthy"]:
            healthy.update(_by_recording(h["posteriors"]))

        site: dict[str, Any] = {
            "depot_id": depot_id,
            "faulted": {"n": 0, "flagged": 0, "delays": [], "single_window_flags": 0},
            "healthy": {"n": 0, "flagged": 0, "single_window_flags": 0, "examined": 0},
        }
        for exp_id, posts in faulted.items():
            r = sequential.run(_llrs(posts))
            site["faulted"]["n"] += 1
            site["faulted"]["flagged"] += r.flagged
            site["faulted"]["single_window_flags"] += r.single_window_flag
            if r.flagged:
                site["faulted"]["delays"].append(r.flag_index)
            recordings.append(
                {
                    "model": depot_id,
                    "experiment_id": exp_id,
                    "true_label": posts[0].true_label,
                    **r.as_dict(),
                }
            )
        for exp_id, posts in healthy.items():
            r = sequential.run(_llrs(posts))
            site["healthy"]["n"] += 1
            site["healthy"]["flagged"] += r.flagged
            site["healthy"]["single_window_flags"] += r.single_window_flag
            site["healthy"]["examined"] += r.n_examined
            recordings.append(
                {"model": depot_id, "experiment_id": exp_id, "true_label": "Normal", **r.as_dict()}
            )

        # The "any window" rules on identical data, for the comparison table.
        site["any_window_heuristic"] = phase3._summarise(phase3.heuristic_decisions(c), "h")
        site["any_window_policy"] = phase3._summarise(phase3.policy_decisions(c), "p")

        # Stress test: inject this model's own real faulted evidence.
        f_llrs = sorted(x for posts in faulted.values() for x in _llrs(posts))
        median_llr = statistics.median(f_llrs)
        max_llr = f_llrs[-1]
        # None when the cost matrix never flags at all -- then no k can flag.
        threshold = policy.flag_threshold()
        median_p = statistics.median(policy.p_faulted(p.probabilities) for p in c["faulted_post"])
        for exp_id, posts in healthy.items():
            h_llrs = _llrs(posts)
            stress.append(
                {
                    "model": depot_id,
                    "healthy_experiment_id": exp_id,
                    "typical_faulted_llr": median_llr,
                    "k_typical": _min_k_to_flag(h_llrs, median_llr),
                    "max_faulted_llr": max_llr,
                    "k_max": _min_k_to_flag(h_llrs, max_llr),
                    "any_window_policy_k": 1
                    if threshold is not None and median_p >= threshold
                    else None,
                }
            )
        per_site.append(site)

    def pooled(key: str) -> dict:
        return {
            "n_faulted": sum(s[key]["n_faulted_recordings"] for s in per_site),
            "misses": sum(s[key]["n_misses"] for s in per_site),
            "n_healthy": sum(s[key]["n_healthy_recordings"] for s in per_site),
            "false_alarms": sum(s[key]["n_false_alarms"] for s in per_site),
            "delays": [d for s in per_site for d in s[key]["detection_delays"]],
        }

    sprt = {
        "n_faulted": sum(s["faulted"]["n"] for s in per_site),
        "misses": sum(s["faulted"]["n"] - s["faulted"]["flagged"] for s in per_site),
        "n_healthy": sum(s["healthy"]["n"] for s in per_site),
        "false_alarms": sum(s["healthy"]["flagged"] for s in per_site),
        "delays": [d for s in per_site for d in s["faulted"]["delays"]],
        "single_window_flags_on_faults": sum(s["faulted"]["single_window_flags"] for s in per_site),
        "healthy_windows_examined": sum(s["healthy"]["examined"] for s in per_site),
    }
    return {
        "alpha": sequential.DEFAULT_ALPHA,
        "beta": sequential.DEFAULT_BETA,
        "boundaries": {"A": a, "B": b},
        "faulted_prior": sequential.FAULTED_PRIOR,
        "eps": sequential.EPS,
        "pooled": {
            "any_window_heuristic": pooled("any_window_heuristic"),
            "any_window_policy": pooled("any_window_policy"),
            "sprt": sprt,
        },
        "stress": stress,
        "recordings": recordings,
    }


def _rate(n, d):
    return "n/a" if not d else f"{n}/{d} = {n / d:.1%}"


def _mean(xs):
    return "n/a" if not xs else f"{sum(xs) / len(xs):.2f}"


def _print(r: dict) -> None:
    print("=" * 78)
    print("PHASE 4 -- repeated SPRT vs the 'any window' rules (deployed v1 checkpoints)")
    print("=" * 78)
    print(
        f"\nalpha {r['alpha']}  beta {r['beta']}  ->  A = {r['boundaries']['A']:.4f} (flag), "
        f"B = {r['boundaries']['B']:.4f} (healthy, reset)"
    )
    print(f"training prior P(faulted) = {r['faulted_prior']:.4f} (uniform under class weighting)")

    print(f"\n{'rule':<28}{'misses':>16}{'X-site false alarms':>24}{'mean delay':>14}")
    for key, name in (
        ("any_window_heuristic", "any-window heuristic"),
        ("any_window_policy", "any-window policy"),
        ("sprt", "repeated SPRT"),
    ):
        p = r["pooled"][key]
        print(
            f"{name:<28}{_rate(p['misses'], p['n_faulted']):>16}"
            f"{_rate(p['false_alarms'], p['n_healthy']):>24}{_mean(p['delays']):>14}"
        )
    s = r["pooled"]["sprt"]
    print(
        f"\nSPRT flagged {s['single_window_flags_on_faults']}/{s['n_faulted'] - s['misses']} "
        "detected faults on a SINGLE window (one window crossed A alone)"
    )
    print(
        f"SPRT examined {s['healthy_windows_examined']} healthy windows without a false alarm"
        if s["false_alarms"] == 0
        else f"SPRT raised {s['false_alarms']} false alarms"
    )

    print("\nSTRESS TEST -- inject k consecutive copies of a REAL faulted window's evidence")
    print("into the middle of each unseen healthy recording; smallest k that flags it:")
    print(
        f"  {'model':<6}{'healthy rec':<12}{'typical LLR':>12}{'k':>4}{'max LLR':>10}{'k':>4}"
        f"{'any-window k':>14}"
    )
    for x in r["stress"]:
        print(
            f"  {x['model']:<6}{x['healthy_experiment_id']:<12}{x['typical_faulted_llr']:>12.3f}"
            f"{str(x['k_typical']):>4}{x['max_faulted_llr']:>10.3f}{str(x['k_max']):>4}"
            f"{str(x['any_window_policy_k']):>14}"
        )
    kt = [x["k_typical"] for x in r["stress"] if x["k_typical"]]
    km = [x["k_max"] for x in r["stress"] if x["k_max"]]
    print(
        f"  => typical faulted window: SPRT needs k = {min(kt)}..{max(kt)}; "
        f"most confident window: k = {min(km)}..{max(km)}; any-window rule: k = 1"
    )

    print("\nREAD THESE WITH THE NUMBERS")
    print("  1. Wald's alpha/beta assume independent windows; these are autocorrelated,")
    print("     so achieved rates are measured, not guaranteed.")
    print("  2. One sufficiently confident window can still flag a machine alone (the")
    print("     EPS clip bounds a window at ~13.8 nats, above A). See 'k' for max LLR.")
    print("  3. False alarms are cross-site (other sites' healthy recordings, a domain")
    print("     shift), from only 4 distinct Normal recordings.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlruns", default="mlruns")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    session = make_session_factory(create_db_engine())()
    try:
        report = run_report(session, Path(args.mlruns))
    finally:
        session.close()
    _print(report)
    if not args.no_write:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        path = OUT_ROOT / "report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
