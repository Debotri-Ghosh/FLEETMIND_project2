"""Manifest generation: scan raw .mat files -> data/manifest.csv (ported from scripts/generate_manifest.py)."""

import csv
import re
from pathlib import Path

DATA_DIR = Path("data/raw/cwru")
MANIFEST_PATH = Path("data/manifest.csv")

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
    return f"0.{m.group(1)}" if m else "NEEDS_REVIEW"


def guess_load(path: Path) -> str:
    for part in path.parts:
        m = LOAD_FOLDER_PATTERN.match(part)
        if m:
            return m.group(1)
    return "NEEDS_REVIEW"


def build_rows(data_dir: Path) -> list[dict[str, str]]:
    all_mat_files = sorted(data_dir.rglob("*.mat"))
    rows = []
    for f in (f for f in all_mat_files if is_target_file(f)):
        rel = f.relative_to(data_dir)
        fault_type = guess_fault_type(rel)
        rows.append(
            {
                "file_path": str(rel),
                "fault_type": fault_type,
                "fault_diameter_in": guess_diameter(f.name, fault_type),
                "load_hp": guess_load(rel),
                "sensor_position": "DE",
            }
        )
    return rows


def main(data_dir: Path = DATA_DIR, manifest_path: Path = MANIFEST_PATH) -> None:
    if not data_dir.exists():
        print(f"{data_dir} does not exist yet — download/extract the dataset there first.")
        return

    rows = build_rows(data_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "fault_type",
                "fault_diameter_in",
                "load_hp",
                "sensor_position",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    needs_review = sum(
        1 for r in rows if "NEEDS_REVIEW" in (r["fault_type"], r["fault_diameter_in"], r["load_hp"])
    )
    print(f"Wrote {len(rows)} rows to {manifest_path}")
    if needs_review:
        print(f"{needs_review} row(s) marked NEEDS_REVIEW — fix before ingesting.")


if __name__ == "__main__":
    main()
