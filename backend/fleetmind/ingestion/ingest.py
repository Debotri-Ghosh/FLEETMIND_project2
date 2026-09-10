"""CWRU ingestion into Postgres (ported from scripts/ingest_cwru_data.py, ORM-based)."""

import argparse
import csv
from pathlib import Path

import scipy.io
from sqlalchemy.orm import Session

from fleetmind.db.models import Depot, Experiment, SignalWindow, seed_depots_payload
from fleetmind.db.session import create_db_engine
from fleetmind.ingestion.windowing import split_windows

DEFAULT_WINDOW_SIZE = 2048  # ~170ms at 12kHz — standard in CWRU literature


def find_de_time_key(mat_dict: dict) -> str | None:
    """CWRU stores the drive-end signal under a key like 'X097_DE_time'."""
    return next((k for k in mat_dict if k.endswith("DE_time")), None)


def seed_depots(session: Session) -> dict[int, int]:
    """Dialect-safe depot seeding (works on Postgres and sqlite for tests)."""
    existing = {d.load_hp: d.id for d in session.query(Depot)}
    for row in seed_depots_payload():
        if row["load_hp"] not in existing:
            depot = Depot(**row)
            session.add(depot)
            session.flush()
            existing[depot.load_hp] = depot.id
    return existing


def ingest(
    manifest_path: Path,
    data_dir: Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    database_url: str | None = None,
) -> tuple[int, int]:
    """Returns (n_experiments, n_windows)."""
    engine = create_db_engine(database_url)
    total_experiments = 0
    total_windows = 0

    with open(manifest_path, newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    with Session(engine) as session:
        depot_id_map = seed_depots(session)
        for row in manifest_rows:
            if "NEEDS_REVIEW" in (row["fault_type"], row["load_hp"], row["fault_diameter_in"]):
                print(f"  [skip] NEEDS_REVIEW row: {row['file_path']}")
                continue

            mat_path = data_dir / row["file_path"]
            if not mat_path.exists():
                print(f"  [skip] missing file: {mat_path}")
                continue

            mat_dict = scipy.io.loadmat(str(mat_path))
            de_key = find_de_time_key(mat_dict)
            if de_key is None:
                print(f"  [skip] no *_DE_time key in {mat_path.name}")
                continue

            signal = mat_dict[de_key].flatten().astype(float).tolist()
            depot_id = depot_id_map.get(int(row["load_hp"]))
            if depot_id is None:
                print(f"  [skip] no depot for load_hp={row['load_hp']}")
                continue

            experiment = Experiment(
                depot_id=depot_id,
                fault_type=row["fault_type"],
                fault_diameter_in=(
                    float(row["fault_diameter_in"]) if row["fault_diameter_in"] else None
                ),
                sensor_position=row["sensor_position"] or "DE",
                sampling_rate_hz=12000,
                source_file=row["file_path"],
                source_variable=de_key,
                n_samples=len(signal),
            )
            session.add(experiment)
            session.flush()

            for w_index, (start, end, values) in enumerate(split_windows(signal, window_size)):
                session.add(
                    SignalWindow(
                        experiment_id=experiment.id,
                        window_index=w_index,
                        start_sample=start,
                        end_sample=end,
                        raw_values=values,
                        label=row["fault_type"],
                    )
                )

            n_windows = len(signal) // window_size
            total_experiments += 1
            total_windows += n_windows
            print(f"  [ok] {mat_path.name}: {n_windows} windows -> experiment {experiment.id}")

        session.commit()

    engine.dispose()
    return total_experiments, total_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CWRU .mat files into Postgres.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/cwru"))
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"{args.manifest} not found — run `python -m fleetmind.ingestion.manifest` first."
        )

    n_exp, n_win = ingest(args.manifest, args.data_dir, args.window_size)
    print(f"\nIngested {n_exp} experiments, {n_win} signal windows.")


if __name__ == "__main__":
    main()
