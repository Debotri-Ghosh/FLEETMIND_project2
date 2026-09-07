"""
Reads data/manifest.csv (see scripts/generate_manifest.py), loads each
referenced .mat file with scipy, splits its signal into fixed-length windows,
and inserts everything into Postgres per db/schema.sql.

Run order:
    1. python scripts/download_cwru_data.py
    2. python scripts/generate_manifest.py     -> review/fix data/manifest.csv
    3. python scripts/apply_schema.py
    4. python scripts/ingest_cwru_data.py       <- this script

Usage:
    python scripts/ingest_cwru_data.py
    python scripts/ingest_cwru_data.py --window-size 4096   # override default
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import scipy.io
import sqlalchemy
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "cwru"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "manifest.csv"

DEFAULT_WINDOW_SIZE = 2048  # ~170ms of signal at 12kHz -- standard choice in CWRU literature


def find_de_time_key(mat_dict: dict) -> str | None:
    """CWRU stores the drive-end signal under a key like 'X097_DE_time'.
    The numeric prefix varies per file, so search for the suffix instead
    of hardcoding a key name."""
    for key in mat_dict:
        if key.endswith("DE_time"):
            return key
    return None


def load_depot_id_map(conn) -> dict:
    rows = conn.execute(sqlalchemy.text("SELECT id, load_hp FROM depots")).fetchall()
    return {load_hp: depot_id for depot_id, load_hp in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"{MANIFEST_PATH} not found — run scripts/generate_manifest.py first.")
        return

    engine = sqlalchemy.create_engine(os.environ["DATABASE_URL"])

    with open(MANIFEST_PATH, newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    bad_rows = [
        r for r in manifest_rows
        if "NEEDS_REVIEW" in (r["fault_type"], r["load_hp"])
        or (r["fault_type"] != "Normal" and r["fault_diameter_in"] == "NEEDS_REVIEW")
    ]
    if bad_rows:
        print(
            f"{len(bad_rows)} row(s) in manifest.csv still say NEEDS_REVIEW — "
            "fix those before ingesting. Skipping them for now, ingesting the rest."
        )

    total_experiments = 0
    total_windows = 0

    with engine.begin() as conn:
        depot_id_map = load_depot_id_map(conn)

        for row in manifest_rows:
            if row in bad_rows:
                continue

            mat_path = DATA_DIR / row["file_path"]
            if not mat_path.exists():
                print(f"  [skip] file listed in manifest but not found on disk: {mat_path}")
                continue

            mat_dict = scipy.io.loadmat(str(mat_path))
            de_key = find_de_time_key(mat_dict)
            if de_key is None:
                print(f"  [skip] no *_DE_time key in {mat_path.name}")
                continue

            signal = mat_dict[de_key].flatten().astype(float)
            load_hp = int(row["load_hp"])
            depot_id = depot_id_map.get(load_hp)
            if depot_id is None:
                print(f"  [skip] no depot seeded for load_hp={load_hp} (file {mat_path.name})")
                continue

            fault_diameter = float(row["fault_diameter_in"]) if row["fault_diameter_in"] else None

            result = conn.execute(
                sqlalchemy.text(
                    """
                    INSERT INTO experiments
                        (depot_id, fault_type, fault_diameter_in, sensor_position,
                         sampling_rate_hz, source_file, source_variable, n_samples)
                    VALUES
                        (:depot_id, :fault_type, :fault_diameter_in, :sensor_position,
                         12000, :source_file, :source_variable, :n_samples)
                    RETURNING id
                    """
                ),
                {
                    "depot_id": depot_id,
                    "fault_type": row["fault_type"],
                    "fault_diameter_in": fault_diameter,
                    "sensor_position": row["sensor_position"] or "DE",
                    "source_file": row["file_path"],
                    "source_variable": de_key,
                    "n_samples": len(signal),
                },
            )
            experiment_id = result.scalar_one()
            total_experiments += 1

            window_size = args.window_size
            n_windows = len(signal) // window_size
            window_rows = []
            for w in range(n_windows):
                start = w * window_size
                end = start + window_size
                window_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "window_index": w,
                        "start_sample": start,
                        "end_sample": end,
                        "raw_values": signal[start:end].tolist(),
                        "label": row["fault_type"],
                    }
                )

            if window_rows:
                conn.execute(
                    sqlalchemy.text(
                        """
                        INSERT INTO signal_windows
                            (experiment_id, window_index, start_sample, end_sample, raw_values, label)
                        VALUES
                            (:experiment_id, :window_index, :start_sample, :end_sample, :raw_values, :label)
                        """
                    ),
                    window_rows,
                )
            total_windows += len(window_rows)
            print(f"  [ok] {mat_path.name}: {len(window_rows)} windows -> experiment_id {experiment_id}")

    print(f"\nIngested {total_experiments} experiments, {total_windows} signal windows.")


if __name__ == "__main__":
    main()
