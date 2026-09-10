"""CWRU dataset download via Kaggle mirror (ported from scripts/download_cwru_data.py)."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data/raw/cwru")

MANUAL_INSTRUCTIONS = """
Kaggle credentials not found (KAGGLE_USERNAME / KAGGLE_KEY in .env).

Option A — set up Kaggle API (recommended, ~2 minutes):
  1. Go to https://www.kaggle.com/settings -> API -> "Create New Token".
  2. Copy the username/key into .env as KAGGLE_USERNAME / KAGGLE_KEY, re-run.

Option B — manual download from the original source:
  https://engineering.case.edu/bearingdatacenter/download-data-file
  Place .mat files under data/raw/cwru/Normal/ and data/raw/cwru/12k_DE/Load_{0-3}/

Option C — Kaggle mirror via browser:
  https://www.kaggle.com/datasets/esraakhaled299/cwru-data
"""


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    username, key = os.getenv("KAGGLE_USERNAME"), os.getenv("KAGGLE_KEY")
    if not username or not key:
        print(MANUAL_INSTRUCTIONS)
        return

    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed — run: pip install 'fleetmind[ingest]'")
        return

    print("Downloading CWRU mirror from Kaggle (esraakhaled299/cwru-data)...")
    path = kagglehub.dataset_download("esraakhaled299/cwru-data")
    for item in Path(path).iterdir():
        target = DATA_DIR / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    print(f"Copied dataset into: {DATA_DIR}")


if __name__ == "__main__":
    main()
