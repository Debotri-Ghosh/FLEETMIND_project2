"""
Downloads the CWRU Bearing Dataset via the Kaggle mirror (fastest, single
command). Falls back to printing manual-download instructions from the
official Case Western source if Kaggle credentials aren't configured.

Usage:
    python scripts/download_cwru_data.py

Requires either:
  - KAGGLE_USERNAME + KAGGLE_KEY set in .env (get these from
    https://www.kaggle.com/settings -> API -> Create New Token), OR
  - manual download, see the printed instructions below.
"""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "cwru"

MANUAL_INSTRUCTIONS = """
Kaggle credentials not found (KAGGLE_USERNAME / KAGGLE_KEY in .env).

Option A — set up Kaggle API (recommended, ~2 minutes):
  1. Go to https://www.kaggle.com/settings -> API -> "Create New Token".
  2. This downloads kaggle.json. Open it and copy the username/key into
     your .env file under KAGGLE_USERNAME and KAGGLE_KEY.
  3. Re-run this script.

Option B — manual download from the original source (for citation-grade
provenance, cite this as the source regardless of which route you use):
  1. Visit https://engineering.case.edu/bearingdatacenter/download-data-file
  2. Download, at minimum:
       - Normal Baseline Data
       - 12k Drive End Bearing Fault Data  (covers 0/1/2/3 HP load levels
         -> these 4 load levels become your 4 simulated depots)
  3. Place the downloaded .mat files under:
       data/raw/cwru/Normal/
       data/raw/cwru/12k_DE/Load_0/, Load_1/, Load_2/, Load_3/
     (matching folder names keeps the loader script below working as-is)

Option C — Kaggle mirror without the API, via browser:
  https://www.kaggle.com/datasets/esraakhaled299/cwru-data
  Download the zip manually and extract it into data/raw/cwru/
"""


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        print(MANUAL_INSTRUCTIONS)
        return

    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed — run: pip install kagglehub")
        return

    print("Downloading CWRU mirror from Kaggle (esraakhaled299/cwru-data)...")
    path = kagglehub.dataset_download("esraakhaled299/cwru-data")
    print(f"Downloaded to cache: {path}")

    for item in Path(path).iterdir():
        target = DATA_DIR / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    print(f"Copied dataset into: {DATA_DIR}")
    print(
        "Citation reminder: cite the Case Western Reserve University "
        "Bearing Data Center (https://engineering.case.edu/bearingdatacenter) "
        "as the original source in your report, even when using this mirror."
    )


if __name__ == "__main__":
    main()
