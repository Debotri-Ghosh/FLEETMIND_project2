"""Phase 3 report: the expected-cost policy against the heuristic baseline.

    python -m fleetmind.decision.phase3
    python -m fleetmind.decision.phase3 --no-write

Both decision paths are scored by the identical code (`quality.score`) on identical
data, using the **deployed v1 checkpoints**:

* **misses** on each site's own held-out faulted recordings (16 in total) -- the
  same partition as Phase 0, so the numbers are comparable;
* **cross-site false alarms** on the other sites' Normal recordings, which each
  model has never seen (12 model/recording pairs; see `crosssite.py`).

The second is new: it is the first false-alarm rate this project has measured, and
it applies to the heuristic too, so the comparison is fair in both directions.

Decision unit, deliberately unchanged from Phase 0: a recording counts as flagged
if **any** of its windows produces an actionable decision. That is permissive, and
it is what Phase 4's sequential test exists to improve; keeping it here makes Phase
3 a like-for-like comparison of the *scoring rule* alone.

Read-only. Writes `mlruns/decision/phase3/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.db.models import Depot
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.decision import baseline, costs, crosssite, holdout, policy, probs, quality

OUT_ROOT = Path("mlruns") / "decision" / "phase3"

#: Planned-repair cost relative to an unplanned failure of 100. The sweep spans two
#: orders of magnitude so the conclusion cannot be an artefact of the default 20.
SENSITIVITY_REPAIR = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 95.0)


def _load_hp(session: Session, depot_id: int) -> int | None:
    v = session.execute(select(Depot.load_hp).where(Depot.id == depot_id)).scalar()
    return None if v is None else int(v)


def collect(session: Session, depot_id: int, mlruns: Path = probs.MLRUNS) -> dict:
    """Posteriors for one site's model on its held-out faults and unseen healthy runs."""
    model = probs.load_model(depot_id, mlruns)

    faulted_windows = holdout.holdout_windows(session, depot_id)
    faulted_post = probs.posteriors(model, faulted_windows)

    healthy = []
    source_load_hp: dict[int, int | None] = {}
    for source, windows in crosssite.unseen_healthy_windows(session, depot_id):
        source_load_hp[source] = _load_hp(session, source)
        crosssite.assert_unseen(session, depot_id, sorted({w[1] for w in windows}))
        healthy.append(
            {
                "source_depot_id": source,
                "windows": windows,
                "posteriors": probs.posteriors(model, windows),
            }
        )
    return {
        "depot_id": depot_id,
        "load_hp": _load_hp(session, depot_id),
        "faulted_windows": faulted_windows,
        "faulted_post": faulted_post,
        "healthy": healthy,
        "source_load_hp": source_load_hp,
    }


def heuristic_decisions(c: dict) -> list:
    out = baseline.decide_all(c["faulted_windows"], c["faulted_post"], c["load_hp"])
    for h in c["healthy"]:
        # The heuristic uses the SOURCE machine's motor load: that is the machine
        # being judged, even though another site's model is judging it.
        out += baseline.decide_all(
            h["windows"], h["posteriors"], c["source_load_hp"][h["source_depot_id"]]
        )
    return out


def policy_decisions(c: dict, cost_matrix: dict | None = None) -> list:
    posts = list(c["faulted_post"])
    for h in c["healthy"]:
        posts += h["posteriors"]
    return policy.decide_all(posts, cost_matrix)


def _window_shares(decisions: list) -> dict:
    healthy = [d for d in decisions if d.true_label == quality.HEALTHY_LABEL]
    faulted = [d for d in decisions if d.true_label != quality.HEALTHY_LABEL]
    return {
        "healthy_windows": len(healthy),
        "healthy_windows_actionable": sum(1 for d in healthy if d.actionable),
        "faulted_windows": len(faulted),
        "faulted_windows_actionable": sum(1 for d in faulted if d.actionable),
    }


def _pool(reports: list[dict], key: str) -> dict:
    """Sum counts across sites, then form rates -- never average per-site rates."""
    s = {
        k: sum(r[key][k] for r in reports)
        for k in (
            "n_healthy_recordings",
            "n_false_alarms",
            "n_faulted_recordings",
            "n_misses",
            "healthy_windows",
            "healthy_windows_actionable",
            "faulted_windows",
            "faulted_windows_actionable",
        )
    }
    delays = [d for r in reports for d in r[key]["detection_delays"]]
    s["cross_site_false_alarm_rate"] = (
        s["n_false_alarms"] / s["n_healthy_recordings"] if s["n_healthy_recordings"] else None
    )
    s["miss_rate"] = (
        s["n_misses"] / s["n_faulted_recordings"] if s["n_faulted_recordings"] else None
    )
    s["healthy_window_actionable_share"] = (
        s["healthy_windows_actionable"] / s["healthy_windows"] if s["healthy_windows"] else None
    )
    s["faulted_window_actionable_share"] = (
        s["faulted_windows_actionable"] / s["faulted_windows"] if s["faulted_windows"] else None
    )
    s["mean_detection_delay_windows"] = sum(delays) / len(delays) if delays else None
    return s


def _summarise(decisions: list, label: str) -> dict:
    q = quality.score(decisions, label=label)
    out = {
        "n_healthy_recordings": q.n_healthy_recordings,
        "n_false_alarms": q.n_false_alarms,
        "n_faulted_recordings": q.n_faulted_recordings,
        "n_misses": q.n_misses,
        "detection_delays": q.detection_delays,
        "outcomes": [o.as_dict() for o in q.outcomes],
    }
    out.update(_window_shares(decisions))
    return out


def run(session: Session, mlruns: Path = probs.MLRUNS) -> dict:
    depot_ids = [int(r[0]) for r in session.execute(select(Depot.id).order_by(Depot.id)).all()]
    collected = [collect(session, d, mlruns) for d in depot_ids]

    per_site = []
    for c in collected:
        per_site.append(
            {
                "depot_id": c["depot_id"],
                "healthy_sources": [h["source_depot_id"] for h in c["healthy"]],
                "heuristic": _summarise(heuristic_decisions(c), "heuristic baseline"),
                "policy": _summarise(policy_decisions(c), "expected-cost policy"),
            }
        )

    sensitivity = []
    for repair in SENSITIVITY_REPAIR:
        m = costs.matrix(costs.CostPrimitives(repair=repair))
        rows = [{"policy": _summarise(policy_decisions(c, m), "sweep")} for c in collected]
        pooled = _pool(rows, "policy")
        sensitivity.append(
            {
                "repair_cost": repair,
                "failure_cost": 100.0,
                "flag_threshold_p": policy.flag_threshold(m),
                "never_optimal": policy.never_optimal(m),
                "cross_site_false_alarm_rate": pooled["cross_site_false_alarm_rate"],
                "miss_rate": pooled["miss_rate"],
                "healthy_window_actionable_share": pooled["healthy_window_actionable_share"],
            }
        )

    default = costs.matrix()
    return {
        "checkpoints": "deployed v1 (mlruns/depot_N/cnn_state_dict.pt)",
        "cost_primitives": vars(costs.CostPrimitives()),
        "boundaries": policy.boundaries(default),
        "flag_threshold_p": policy.flag_threshold(default),
        "never_optimal": policy.never_optimal(default),
        "per_site": per_site,
        "pooled": {
            "heuristic": _pool(per_site, "heuristic"),
            "policy": _pool(per_site, "policy"),
        },
        "sensitivity": sensitivity,
        "decision_unit": "recording; flagged if ANY window is actionable (as Phase 0)",
    }


def _fmt(x, pct=True):
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.2f}"


def _print(r: dict) -> None:
    print("=" * 78)
    print("PHASE 3 -- expected-cost policy vs heuristic baseline (deployed v1 checkpoints)")
    print("=" * 78)
    cp = r["cost_primitives"]
    print(
        f"\ncosts (relative units): failure {cp['failure']:.0f}, repair {cp['repair']:.0f}, "
        f"visit {cp['visit']:.0f}, stop {cp['stop']:.0f}"
    )
    print("decision boundaries derived from those costs (P(faulted) ranges):")
    for b in r["boundaries"]:
        print(f"  {b['action']:<22} p in [{b['from_p']:.4f}, {b['to_p']:.4f}]")
    print(f"  never optimal under these costs: {r['never_optimal'] or 'none'}")
    print(f"  => a machine is flagged when P(faulted) >= {r['flag_threshold_p']:.4f}")

    print(
        f"\n{'site':<5}{'path':<11}{'misses':>10}{'X-site false alarms':>22}{'healthy windows flagged':>27}"
    )
    for s in r["per_site"]:
        for key, name in (("heuristic", "heuristic"), ("policy", "policy")):
            q = s[key]
            print(
                f"{s['depot_id']:<5}{name:<11}"
                f"{q['n_misses']:>5}/{q['n_faulted_recordings']:<4}"
                f"{q['n_false_alarms']:>14}/{q['n_healthy_recordings']:<7}"
                f"{q['healthy_windows_actionable']:>17}/{q['healthy_windows']:<9}"
            )

    print("\nPOOLED over all four sites (counts summed, then divided):")
    for key in ("heuristic", "policy"):
        p = r["pooled"][key]
        print(
            f"  {key:<10} miss rate {_fmt(p['miss_rate'])} ({p['n_misses']}/{p['n_faulted_recordings']})"
            f"   cross-site false-alarm rate {_fmt(p['cross_site_false_alarm_rate'])} "
            f"({p['n_false_alarms']}/{p['n_healthy_recordings']})"
        )
        print(
            f"  {'':<10} healthy windows flagged {_fmt(p['healthy_window_actionable_share'])} "
            f"({p['healthy_windows_actionable']}/{p['healthy_windows']})   "
            f"faulted windows flagged {_fmt(p['faulted_window_actionable_share'])}   "
            f"mean detection delay {_fmt(p['mean_detection_delay_windows'], pct=False)} windows"
        )

    print("\nSENSITIVITY -- planned repair cost vs a failure cost of 100:")
    print(
        f"  {'repair':>7} {'flag at P>=':>11} {'miss':>8} {'X-site FA':>10} {'healthy win flagged':>20}  never optimal"
    )
    for s in r["sensitivity"]:
        print(
            f"  {s['repair_cost']:>7.0f} {_fmt(s['flag_threshold_p'], pct=False):>11} "
            f"{_fmt(s['miss_rate']):>8} {_fmt(s['cross_site_false_alarm_rate']):>10} "
            f"{_fmt(s['healthy_window_actionable_share']):>20}  {', '.join(s['never_optimal']) or '-'}"
        )

    print("\nREAD THESE WITH THE NUMBERS")
    print("  1. Cross-site false alarms are measured on OTHER sites' healthy machines,")
    print("     at a different motor load. That domain shift probably overstates the")
    print("     rate a site would see on its own healthy machines.")
    print("  2. A recording counts as flagged if ANY window is actionable -- permissive,")
    print("     deliberately identical to Phase 0. Phase 4 replaces this with a")
    print("     sequential test.")
    print("  3. The policy consumes v1's UNCALIBRATED posterior (OPEN_QUESTIONS Q1). It")
    print("     is only as good as those probabilities, and v1 is over-confident.")
    print("  4. n is small: 16 faulted recordings and 12 healthy model/recording pairs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlruns", default="mlruns")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    session = make_session_factory(create_db_engine())()
    try:
        report = run(session, Path(args.mlruns))
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
