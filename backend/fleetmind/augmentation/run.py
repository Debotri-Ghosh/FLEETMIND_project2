"""Phase 3 runner: baseline vs GAN-augmented per-depot CNN (experiment split).

Pipeline for one depot:
  1. Load all windows for the depot, split by EXPERIMENT (whole .mat
     recordings held out) via experiment_split() - review-honest.
  2. Train baseline BearingCNN on train-experiments only.
  3. Train ConditionalWaveGAN on downsampled train windows; generate
     synthetic windows for minority classes up to the majority count.
  4. Re-train BearingCNN on train + synthetic; compare on same test set.
  5. Write comparison.json (before/after accuracy + per-class recall).

Usage:
    python -m fleetmind.augmentation.run --depot-id 1 --epochs 12 --gan-epochs 15
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from sqlalchemy import select
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader, Dataset

from fleetmind.augmentation.exp_split import stratified_experiment_split
from fleetmind.augmentation.gan import ConditionalWaveGAN
from fleetmind.augmentation.sampling import denormalize, downsample, upsample
from fleetmind.core.config import get_settings
from fleetmind.db.models import Experiment, SignalWindow
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.ml.model import LABEL_MAP, BearingCNN

MLRUNS = Path("mlruns")
GAN_SEQ_LEN = 256
WINDOW_LEN = 2048


class _RowsDataset(Dataset):
    def __init__(self, rows: list[tuple[list[float], int]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        values, label = self.rows[idx]
        x = torch.tensor(values, dtype=torch.float32).view(1, -1)
        return x, label


def load_experiment_rows(session: Session, depot_id: int) -> list:
    """Return [(experiment_id, values, label_index)] for a depot."""
    rows = session.execute(
        select(SignalWindow.experiment_id, SignalWindow.raw_values, SignalWindow.label)
        .join(Experiment, Experiment.id == SignalWindow.experiment_id)
        .where(Experiment.depot_id == depot_id)
        .order_by(SignalWindow.id)
    ).all()
    out: list = []
    exp_fault: dict = {}
    for exp_id, raw_values, label in rows:
        if label not in LABEL_MAP:
            continue
        exp_fault.setdefault(exp_id, label)
        values = [float(v) for v in (raw_values or [])]
        if len(values) < WINDOW_LEN:
            values += [0.0] * (WINDOW_LEN - len(values))
        else:
            values = values[:WINDOW_LEN]
        out.append((int(exp_id), values, LABEL_MAP.index(label)))
    return out


def train_cnn(train_rows: list, test_rows: list, epochs: int, bsz: int, lr: float, seed: int):
    torch.manual_seed(seed)
    train_loader = DataLoader(_RowsDataset(train_rows), batch_size=bsz, shuffle=True)
    test_loader = DataLoader(_RowsDataset(test_rows), batch_size=bsz)
    model = BearingCNN(num_classes=len(LABEL_MAP))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    correct, total = 0, 0
    cc: Counter = Counter()
    tc: Counter = Counter()
    preds_all: list = []
    labels_all: list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(1)
            preds_all += preds.tolist()
            labels_all += yb.tolist()
            correct += int((preds == yb).sum())
            total += yb.size(0)
            for p, t in zip(preds.tolist(), yb.tolist(), strict=True):
                tc[t] += 1
                if p == t:
                    cc[t] += 1
    recall = {LABEL_MAP[c]: (cc[c] / tc[c] if tc[c] else 0.0) for c in range(len(LABEL_MAP))}
    metrics = {
        "accuracy": correct / max(total, 1),
        "test_windows": total,
        "per_class_recall": recall,
        "predictions": preds_all,
        "labels": labels_all,
    }
    return model, metrics


def train_gan(train_rows: list, gan_epochs: int, bsz: int, latent: int, seed: int):
    """Train ConditionalWaveGAN on normalized+downsampled train windows."""
    torch.manual_seed(seed)
    normed: list = []
    stats: dict = {}
    pass  # per-class stats built below
    # Global normalization (fix 2026-09-11): per-window normalize() wiped the
    # amplitude differences between classes (Normal std~0.04 vs faults
    # std~0.10+), letting the generator collapse to the class mean. One
    # global (lo, hi) over ALL train windows keeps amplitude discriminative.
    flat = [v for pair in train_rows for v in downsample(pair[0])]
    glo, ghi = float(min(flat)), float(max(flat))
    span = (ghi - glo) if (ghi - glo) > 1e-9 else 1.0

    def gnorm(values: list[float]) -> list[float]:
        return [((v - glo) / span) * 2.0 - 1.0 for v in values]

    normed = [(gnorm(downsample(v)), lab) for v, lab in train_rows]
    stats = {c: (glo, ghi) for c in range(len(LABEL_MAP))}
    loader = DataLoader(_RowsDataset(normed), batch_size=bsz, shuffle=True)
    gan = ConditionalWaveGAN(num_classes=len(LABEL_MAP), latent_dim=latent, seq_len=GAN_SEQ_LEN)
    for _ in range(gan_epochs):
        for xb, yb in loader:
            gan.train_step(xb, yb)
    return gan, stats


def generate_synthetic(gan: ConditionalWaveGAN, stats: dict, needed: dict, seed: int) -> list:
    torch.manual_seed(seed)
    labels: list = []
    for c, n in needed.items():
        labels += [c] * n
    if not labels:
        return []
    synth256 = gan.generate(labels)
    out: list = []
    for i, c in enumerate(labels):
        scaled = synth256[i, 0, :].tolist()
        lo, hi = stats[c]
        out.append((upsample(denormalize(scaled, lo, hi), target_len=WINDOW_LEN), c))
    return out


def run_depot(
    depot_id: int,
    epochs: int,
    gan_epochs: int,
    bsz: int,
    lr: float,
    latent: int,
    seed: int,
    out_dir=None,
) -> dict:
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    factory = make_session_factory(engine)
    session = factory()
    all_rows = load_experiment_rows(session, depot_id)
    session.close()
    if not all_rows:
        raise SystemExit(f"No windows for depot_id={depot_id}")
    exp_ids = sorted({e for e, _, _ in all_rows})
    exp_fault = {e: LABEL_MAP[lab] for e, _, lab in all_rows}
    train_exp, test_exp = stratified_experiment_split(
        sorted(exp_fault.items()), frac=0.7, seed=seed
    )
    se_train, se_test = set(train_exp), set(test_exp)
    train_rows = [(v, lab) for e, v, lab in all_rows if e in se_train]
    test_rows = [(v, lab) for e, v, lab in all_rows if e in se_test]
    counts = Counter(lab for _, lab in train_rows)
    majority = max(counts.values())
    needed = {c: majority - counts.get(c, 0) for c in range(len(LABEL_MAP))}
    print(f"Depot {depot_id}: {len(all_rows)} windows, {len(exp_ids)} experiments")
    print(f"  train_exp={len(train_exp)} test_exp={len(test_exp)}")
    print(f"  train counts={[counts.get(c, 0) for c in range(len(LABEL_MAP))]}")
    print(f"  synthetic needed={[needed[c] for c in range(len(LABEL_MAP))]}")
    _, baseline = train_cnn(train_rows, test_rows, epochs, bsz, lr, seed)
    print(f"  baseline acc={baseline['accuracy']:.3f} recall={baseline['per_class_recall']}")
    gan, stats = train_gan(train_rows, gan_epochs, bsz, latent, seed)
    synth_rows = generate_synthetic(gan, stats, needed, seed)
    print(f"  generated {len(synth_rows)} synthetic windows")
    _, augmented = train_cnn(train_rows + synth_rows, test_rows, epochs, bsz, lr, seed)
    print(f"  augmented acc={augmented['accuracy']:.3f} recall={augmented['per_class_recall']}")
    out = out_dir or (MLRUNS / "augmentation" / f"depot_{depot_id}")
    out.mkdir(parents=True, exist_ok=True)

    def lite(m: dict) -> dict:
        return {k: v for k, v in m.items() if k not in ("predictions", "labels")}

    comparison = {
        "depot_id": depot_id,
        "seed": seed,
        "n_experiments": len(exp_ids),
        "train_experiments": train_exp,
        "test_experiments": test_exp,
        "train_windows": len(train_rows),
        "test_windows": len(test_rows),
        "synthetic_windows": len(synth_rows),
        "baseline": lite(baseline),
        "augmented": lite(augmented),
    }
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2))
    torch.save(gan.gen.state_dict(), out / "generator_state_dict.pt")
    print(f"Saved comparison + generator to {out.resolve()}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: baseline vs GAN-augmented CNN.")
    parser.add_argument("--depot-id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--gan-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_depot(
        args.depot_id,
        args.epochs,
        args.gan_epochs,
        args.batch_size,
        args.lr,
        args.latent_dim,
        args.seed,
    )


if __name__ == "__main__":
    main()
