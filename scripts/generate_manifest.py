"""
Scans data/raw/cwru for .mat files and writes data/manifest.csv — one row per
file, with metadata (fault type, load, fault diameter) parsed from the real
folder structure of the 'CWRU_organized' Kaggle mirror:

    CWRU_organized/
        12k_DE/Load_{0-3}/{Ball,InnerRace,OuterRace}/<code>_<load>.mat
        12k_FE/...    <- excluded: different sensor channel, not used here
        48k_DE/...    <- excluded: different sampling rate, not used here
        Normal/Load_{0-3}/Normal_<load>.mat

We deliberately keep only 12k_DE + Normal, because the database schema
assumes a single consistent sampling rate (12000 Hz) and sensor channel (DE).
12k_FE and 48k_DE are real data too and can be added as a later extension,
but mixing them in now would silently corrupt the sampling_rate_hz metadata.

Usage:
    python scripts/generate_manifest.py
    # then open data/manifest.csv, spot-check a few rows, save
    # then run scripts/ingest_cwru_data.py
"""
import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "cwru"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "manifest.csv"

FAULT_FOLDER_NAMES = {"Normal", "InnerRace", "OuterRace", "Ball"}
DIAMETER_CODE_PATTERN = re.compile(r"(007|014|021|028)")
LOAD_FOLDER_PATTERN = re.compile(r"Load_([0-3])")


def is_target_file(path: Path) -> bool:
    """Keep only 12k_DE and Normal; skip 12k_FE and 48k_DE."""
    parts = path.parts
    if "12k_FE" in parts or "48k_DE" in parts:
        return False
    return "12k_DE" in parts or "Normal" in parts


def guess_fault_type(path: Path) -> str:
    for part in path.parts:
        if part in FAULT_FOLDER_NAMES:
            return part
    return "NEEDS_REVIEW"


def guess_diameter(filename: str, fault_type: str) -> str:
    if fault_type == "Normal":
        return ""
    m = DIAMETER_CODE_PATTERN.search(filename)
    if m:
        return f"0.{m.group(1)}"
    return "NEEDS_REVIEW"


def guess_load(path: Path) -> str:
    for part in path.parts:
        m = LOAD_FOLDER_PATTERN.match(part)
        if m:
            return m.group(1)
    return "NEEDS_REVIEW"


def main() -> None:
    if not DATA_DIR.exists():
        print(f"{DATA_DIR} does not exist yet — extract the dataset there first.")
        return

    all_mat_files = sorted(DATA_DIR.rglob("*.mat"))
    if not all_mat_files:
        print(f"No .mat files found under {DATA_DIR}.")
        return

    mat_files = [f for f in all_mat_files if is_target_file(f)]
    excluded = len(all_mat_files) - len(mat_files)

    rows = []
    for f in mat_files:
        rel = f.relative_to(DATA_DIR)
        fault_type = guess_fault_type(rel)
        diameter = guess_diameter(f.name, fault_type)
        load_hp = guess_load(rel)
        rows.append(
            {
                "file_path": str(rel),
                "fault_type": fault_type,
                "fault_diameter_in": diameter,
                "load_hp": load_hp,
                "sensor_position": "DE",
            }
        )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_path", "fault_type", "fault_diameter_in", "load_hp", "sensor_position"]
        )
        writer.writeheader()
        writer.writerows(rows)

    needs_review = sum(
        1 for r in rows if "NEEDS_REVIEW" in (r["fault_type"], r["fault_diameter_in"], r["load_hp"])
    )
    print(f"Found {len(all_mat_files)} total .mat files; kept {len(mat_files)} (12k_DE + Normal).")
    print(f"Excluded {excluded} files from 12k_FE / 48k_DE (not used in this schema).")
    print(f"Wrote {len(rows)} rows to {MANIFEST_PATH}")
    if needs_review:
        print(f"{needs_review} row(s) marked NEEDS_REVIEW — open the CSV and fix those before ingesting.")
    else:
        print("All rows auto-classified cleanly. Still worth spot-checking a handful before ingesting.")


if __name__ == "__main__":
    main()
