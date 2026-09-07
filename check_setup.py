"""
Run this after completing setup to confirm every piece is wired up
correctly BEFORE writing any pipeline code.

    python check_setup.py
"""
import importlib
import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CHECKS_PASSED = True


def check(label: str, fn) -> None:
    global CHECKS_PASSED
    try:
        fn()
        print(f"[OK]   {label}")
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        CHECKS_PASSED = False


def check_python_version():
    major, minor = sys.version_info[:2]
    assert (major, minor) == (3, 11), f"Python {major}.{minor} found, need 3.11.x"


def check_packages():
    required = [
        "numpy", "pandas", "scipy", "sklearn", "torch",
        "flwr", "langgraph", "fastapi", "streamlit",
        "psycopg2", "sqlalchemy",
    ]
    missing = [p for p in required if importlib.util.find_spec(p) is None]
    assert not missing, f"missing packages: {missing}"


def check_env_file():
    assert Path(".env").exists(), ".env not found — copy .env.example to .env"
    assert os.getenv("DATABASE_URL"), "DATABASE_URL not set in .env"


def check_postgres():
    import sqlalchemy
    engine = sqlalchemy.create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("SELECT 1"))


def check_cwru_data_present():
    data_dir = Path("data/raw/cwru")
    assert data_dir.exists() and any(data_dir.iterdir()), (
        "data/raw/cwru is empty — run: python scripts/download_cwru_data.py"
    )


def check_mat_file_readable():
    import scipy.io
    data_dir = Path("data/raw/cwru")
    mat_files = list(data_dir.rglob("*.mat"))
    assert mat_files, "no .mat files found under data/raw/cwru"
    sample = scipy.io.loadmat(mat_files[0])
    de_keys = [k for k in sample.keys() if "DE_time" in k]
    assert de_keys, f"{mat_files[0].name} has no *_DE_time key — unexpected file format"


def check_schema_applied():
    import sqlalchemy
    engine = sqlalchemy.create_engine(os.environ["DATABASE_URL"])
    expected = {"depots", "experiments", "signal_windows", "model_predictions", "agent_decisions"}
    with engine.connect() as conn:
        found = {
            row[0]
            for row in conn.execute(
                sqlalchemy.text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                )
            )
        }
    missing = expected - found
    assert not missing, f"missing tables: {missing} — run: python scripts/apply_schema.py"


def check_data_ingested():
    import sqlalchemy
    engine = sqlalchemy.create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        exp_count = conn.execute(sqlalchemy.text("SELECT count(*) FROM experiments")).scalar_one()
        win_count = conn.execute(sqlalchemy.text("SELECT count(*) FROM signal_windows")).scalar_one()
    assert exp_count > 0, "0 rows in experiments — run: python scripts/ingest_cwru_data.py"
    assert win_count > 0, "0 rows in signal_windows — run: python scripts/ingest_cwru_data.py"
    print(f"       ({exp_count} experiments, {win_count} signal windows)", end="")


if __name__ == "__main__":
    check("Python 3.11 active", check_python_version)
    check("Required packages importable", check_packages)
    check(".env file present and populated", check_env_file)
    check("PostgreSQL container reachable", check_postgres)
    check("CWRU raw data present", check_cwru_data_present)
    check("CWRU .mat file readable (scipy)", check_mat_file_readable)
    check("Database schema applied", check_schema_applied)
    check("Data ingested into database", check_data_ingested)

    print()
    if CHECKS_PASSED:
        print("All checks passed — ready to start building the pipeline.")
    else:
        print("Some checks failed — fix the items above before writing pipeline code.")
        sys.exit(1)
