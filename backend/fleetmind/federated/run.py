"""Phase 4 runner: in-process federated simulation (FedAvg / FedProx).

Semantics mirror Flower exactly (sample-weighted FedAvg aggregation of client
deltas after `local_epochs` of local SGD; FedProx adds the proximal term via
mu>0 in `client.local_train`), but run in-process so the M.Tech demo needs no
open ports. `flower_client.DepotClient` is the seam for graduating to a real
`fl.server` deployment.

Pipeline:
  1. For every depot: experiment-split train/test (same honest split as Phase 3).
  2. Baseline: train BearingCNN per depot on its own train partition only.
  3. Federated: T rounds; each round every client trains `local_epochs` from the
     global weights, server aggregates sample-weighted, global model is
     evaluated on each depot's held-out test set.
  4. Write federated.json: per-round global accuracy, per-depot final accuracy
     + per-class recall, and baseline-vs-federated comparison.

Usage:
    python -m fleetmind.federated.run --strategy fedavg --rounds 10 --local-epochs 2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.augmentation.exp_split import stratified_experiment_split
from fleetmind.augmentation.run import load_experiment_rows, train_cnn
from fleetmind.core.config import get_settings
from fleetmind.db.models import Depot
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.federated.client import (
    evaluate_model,
    get_buffers,
    get_parameters,
    local_train,
    recalibrate_bn,
    set_buffers,
    set_parameters,
)
from fleetmind.ml.model import LABEL_MAP, BearingCNN, to_fft_magnitude

MLRUNS = Path("mlruns")


def depot_ids(session: Session) -> list[int]:
    return [r[0] for r in session.execute(select(Depot.id).order_by(Depot.id)).all()]


def average_parameters(client_params: list[list], num_samples: list[int]) -> list:
    """Sample-weighted (FedAvg) aggregation of client parameter lists."""
    import numpy as np

    total = float(sum(num_samples))
    accum = [np.zeros_like(np.asarray(p, dtype="float64")) for p in client_params[0]]
    for params, n in zip(client_params, num_samples, strict=True):
        for i, p in enumerate(params):
            accum[i] = accum[i] + np.asarray(p, dtype="float64") * n
    return [(w / total).tolist() for w in accum]


def average_buffers(client_buffers: list[list], num_samples: list[int]) -> list:
    """Sample-weighted average of BatchNorm running statistics.

    Same weighting as the parameter aggregation. `num_batches_tracked` is an
    integer counter and is averaged here too; `set_buffers` casts it back to
    its original dtype on load.
    """
    import numpy as np

    total = float(sum(num_samples))
    accum = [np.zeros_like(np.asarray(b, dtype="float64")) for b in client_buffers[0]]
    for buffers, n in zip(client_buffers, num_samples, strict=True):
        for i, b in enumerate(buffers):
            accum[i] = accum[i] + np.asarray(b, dtype="float64") * n
    return [(w / total).tolist() for w in accum]


def run_federated(
    strategy: str = "fedavg",
    rounds: int = 10,
    local_epochs: int = 2,
    cnn_epochs: int = 12,
    bsz: int = 64,
    lr: float = 3e-3,
    mu: float = 0.1,
    seed: int = 42,
    out_dir: Path | None = None,
    aggregate_bn: bool = False,
    recalibrate: bool = False,
) -> dict:
    """Run the federated simulation.

    `aggregate_bn` and `recalibrate` both default to False so the published
    result stays byte-for-byte reproducible. They exist to test whether the
    observed majority-class collapse is an artifact of BatchNorm running
    statistics never reaching the global model -- see the Progress log.

    aggregate_bn  also sample-weight-average the BN buffers each round.
    recalibrate   before the final evaluation, re-estimate the global model's
                  BN statistics from each site's own training partition.
    """
    strategy = strategy.lower()
    if strategy not in ("fedavg", "fedprox"):
        raise SystemExit("strategy must be fedavg or fedprox")
    prox_mu = mu if strategy == "fedprox" else 0.0

    # --- Build partitions first (DB session can close before training) ---
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    factory = make_session_factory(engine)
    session = factory()
    ids = depot_ids(session)

    partitions: dict[int, dict] = {}
    for depot_id in ids:
        all_rows = load_experiment_rows(session, depot_id)
        exp_fault = {e: LABEL_MAP[lab] for e, _, lab in all_rows}
        train_exp, test_exp = stratified_experiment_split(
            sorted(exp_fault.items()), frac=0.7, seed=seed
        )
        se_train, se_test = set(train_exp), set(test_exp)
        partitions[depot_id] = {
            "train": [(to_fft_magnitude(v), lab) for e, v, lab in all_rows if e in se_train],
            "test": [(to_fft_magnitude(v), lab) for e, v, lab in all_rows if e in se_test],
        }
        counts = Counter(lab for _, lab in partitions[depot_id]["train"])
        print(
            f"Depot {depot_id}: train={len(partitions[depot_id]['train'])} "
            f"test={len(partitions[depot_id]['test'])} "
            f"counts={[counts.get(c, 0) for c in range(len(LABEL_MAP))]}"
        )
    session.close()

    baselines: dict[int, dict] = {}
    import tempfile

    with tempfile.TemporaryDirectory() as tmpd:
        init_path = Path(tmpd) / "global_state_dict.pt"
        torch.save(BearingCNN(num_classes=len(LABEL_MAP)).state_dict(), init_path)
        for depot_id in ids:
            torch.manual_seed(seed)
            _, res = train_cnn(
                partitions[depot_id]["train"],
                partitions[depot_id]["test"],
                cnn_epochs,
                bsz,
                lr,
                seed,
            )
            baselines[depot_id] = {k: v for k, v in res.items() if k != "predictions"}
            print(f"  baseline depot {depot_id}: acc={res['accuracy']:.3f}")

        global_model = BearingCNN(num_classes=len(LABEL_MAP))
        global_model.load_state_dict(torch.load(init_path, weights_only=True))
        history: list[dict] = []
        for rnd in range(1, rounds + 1):
            base_params = get_parameters(global_model)
            client_params, num_samples = [], []
            client_buffers: list[list] = []
            round_loss = 0.0
            for depot_id in ids:
                client_model = BearingCNN(num_classes=len(LABEL_MAP))
                set_parameters(client_model, base_params)
                stats = local_train(
                    client_model,
                    partitions[depot_id]["train"],
                    epochs=local_epochs,
                    batch_size=bsz,
                    lr=lr,
                    mu=prox_mu,
                    seed=seed + rnd,
                )
                client_params.append(get_parameters(client_model))
                if aggregate_bn:
                    client_buffers.append(get_buffers(client_model))
                num_samples.append(len(partitions[depot_id]["train"]))
                round_loss += stats["loss"] * len(partitions[depot_id]["train"])
            set_parameters(global_model, average_parameters(client_params, num_samples))
            if aggregate_bn:
                set_buffers(global_model, average_buffers(client_buffers, num_samples))
            per_depot = {
                str(d): evaluate_model(global_model, partitions[d]["test"])["accuracy"] for d in ids
            }
            mean_acc = sum(per_depot.values()) / len(per_depot)
            history.append(
                {
                    "round": rnd,
                    "loss": round_loss / sum(num_samples),
                    "mean_accuracy": mean_acc,
                    "per_depot_accuracy": per_depot,
                }
            )
            print(
                f"  round {rnd:2d}/{rounds} loss={history[-1]['loss']:.4f} mean_acc={mean_acc:.3f}"
            )

    out = out_dir or (MLRUNS / "federated" / strategy)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), out / "global_state_dict.pt")

    final: dict[str, dict] = {}
    recalibration_batches: dict[str, int] = {}
    for depot_id in ids:
        if recalibrate:
            # Per site, because each site's input distribution differs -- that
            # non-IID-ness is the whole point of the study. Uses the site's
            # TRAIN partition only; the test partition is never touched.
            recalibration_batches[str(depot_id)] = recalibrate_bn(
                global_model, partitions[depot_id]["train"], batch_size=bsz
            )
        final[str(depot_id)] = evaluate_model(global_model, partitions[depot_id]["test"])
    mean_final = sum(v["accuracy"] for v in final.values()) / len(final)
    mean_base = sum(b["accuracy"] for b in baselines.values()) / len(baselines)

    report = {
        "strategy": strategy,
        "aggregate_bn": aggregate_bn,
        "recalibrate_bn": recalibrate,
        "recalibration_batches": recalibration_batches,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "prox_mu": prox_mu,
        "seed": seed,
        "baselines": {str(k): v for k, v in baselines.items()},
        "mean_baseline_accuracy": mean_base,
        "history": history,
        "final": final,
        "mean_final_accuracy": mean_final,
        "gain_vs_baseline": mean_final - mean_base,
    }
    (out / "federated.json").write_text(json.dumps(report, indent=2))

    print(f"Saved federated report to {out.resolve()}")
    print(
        f"mean baseline={mean_base:.3f} federated={mean_final:.3f} gain={report['gain_vs_baseline']:+.3f}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: federated simulation.")
    parser.add_argument("--strategy", default="fedavg", choices=["fedavg", "fedprox"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--cnn-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aggregate-bn",
        action="store_true",
        help="Also average BatchNorm running statistics each round (off by default).",
    )
    parser.add_argument(
        "--recalibrate-bn",
        action="store_true",
        help="Re-estimate the global model's BN statistics per site before final eval.",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_federated(
        args.strategy,
        args.rounds,
        args.local_epochs,
        args.cnn_epochs,
        args.batch_size,
        args.lr,
        args.mu,
        args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        aggregate_bn=args.aggregate_bn,
        recalibrate=args.recalibrate_bn,
    )


if __name__ == "__main__":
    main()
