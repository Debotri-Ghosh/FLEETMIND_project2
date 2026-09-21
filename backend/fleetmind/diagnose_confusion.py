"""Confusion matrix diagnostic, with an option to test FFT-magnitude input."""

from __future__ import annotations

import argparse
from collections import Counter

from fleetmind.augmentation.exp_split import stratified_experiment_split
from fleetmind.augmentation.run import load_experiment_rows, train_cnn
from fleetmind.core.config import get_settings
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.ml.model import LABEL_MAP, to_fft_magnitude


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depot-id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-fft", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    factory = make_session_factory(engine)
    session = factory()

    all_rows = load_experiment_rows(session, args.depot_id)
    session.close()

    exp_fault = {e: LABEL_MAP[lab] for e, _, lab in all_rows}
    train_exp, test_exp = stratified_experiment_split(
        sorted(exp_fault.items()), frac=0.7, seed=args.seed
    )
    se_train, se_test = set(train_exp), set(test_exp)

    if args.use_fft:
        print("Using FFT-magnitude input (512 bins, log-scaled, z-normalized)")
        train_rows = [(to_fft_magnitude(v), lab) for e, v, lab in all_rows if e in se_train]
        test_rows = [(to_fft_magnitude(v), lab) for e, v, lab in all_rows if e in se_test]
    else:
        print("Using raw time-domain input (2048 samples)")
        train_rows = [(v, lab) for e, v, lab in all_rows if e in se_train]
        test_rows = [(v, lab) for e, v, lab in all_rows if e in se_test]

    print(f"Depot {args.depot_id}: train={len(train_rows)} test={len(test_rows)}")

    _, metrics = train_cnn(train_rows, test_rows, args.epochs, 64, 3e-3, args.seed)

    preds = metrics["predictions"]
    labels = metrics["labels"]

    matrix: dict[str, Counter] = {name: Counter() for name in LABEL_MAP}
    for p, t in zip(preds, labels, strict=True):
        matrix[LABEL_MAP[t]][LABEL_MAP[p]] += 1

    print("\nConfusion matrix (rows = true class, cols = predicted class):")
    header = "true\\pred".ljust(12) + "".join(name.ljust(12) for name in LABEL_MAP)
    print(header)
    for true_name in LABEL_MAP:
        row = matrix[true_name]
        total = sum(row.values())
        line = true_name.ljust(12) + "".join(str(row.get(p, 0)).ljust(12) for p in LABEL_MAP)
        line += f"  (n={total})"
        print(line)

    print(f"\nOverall accuracy: {metrics['accuracy']:.3f}")
    print(f"Per-class recall: {metrics['per_class_recall']}")


if __name__ == "__main__":
    main()
