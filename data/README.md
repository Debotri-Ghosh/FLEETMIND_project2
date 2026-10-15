# Dataset

**Case Western Reserve University (CWRU) Bearing Data Center** —
https://engineering.case.edu/bearingdatacenter. Please cite it when using this data.

## What is committed here

Only the **64 recordings this project uses** (~186 MB), under
`raw/cwru/CWRU_organized/`:

| Folder | Recordings | What |
|---|---|---|
| `12k_DE/Load_{0-3}/` | 60 | drive-end accelerometer at 12 kHz — 4 Ball, 4 InnerRace, 7 OuterRace per motor load |
| `Normal/Load_{0-3}/` | 4 | one healthy recording per motor load |

`ingestion/manifest.py` keeps exactly `12k_DE` and `Normal` and skips the fan-end
(`12k_FE`) and 48 kHz (`48k_DE`) recordings, so those are not committed. The four motor
loads (0–3 HP) are the four simulated sites, Plant A–D.

`manifest.csv` lists every recording with its fault type, defect size and load.

## Setting the project up on another machine

Create the environment first, with pinned versions (from the repository root):

```powershell
python -m venv venv
cd backend
..\venv\Scripts\python.exe -m pip install -e ".[ml,dev,llm]" -c constraints.txt
```

The recordings are here, but the database is not — it is built from them. Run these
from `backend/`, passing the data paths explicitly (ingestion's defaults are relative
to wherever it is run, and the data lives at the repository root, not in `backend/`):

```powershell
cd backend
..\venv\Scripts\python.exe -m alembic upgrade head
..\venv\Scripts\python.exe -m fleetmind.ingestion.ingest --data-dir ..\data\raw\cwru --manifest ..\data\manifest.csv
..\venv\Scripts\python.exe -m fleetmind.db.seed_product_tables
..\venv\Scripts\python.exe -m fleetmind.db.seed_users
```

The four deployed model checkpoints are committed too, in
`backend/mlruns/depot_{1-4}/`. **Use them rather than retraining:** every accuracy
figure in this project's documents comes from these exact files, and `ml/train.py`
saves the last training epoch, so a retrain would produce different models and
different numbers.
