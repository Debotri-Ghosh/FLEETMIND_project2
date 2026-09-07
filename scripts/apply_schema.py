"""
Applies db/schema.sql to the Postgres container defined by DATABASE_URL in .env.
Safe to re-run — every statement in schema.sql uses IF NOT EXISTS / ON CONFLICT.

Usage:
    python scripts/apply_schema.py
"""
from pathlib import Path

import sqlalchemy
from dotenv import load_dotenv
import os

load_dotenv()

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


def main() -> None:
    database_url = os.environ["DATABASE_URL"]
    engine = sqlalchemy.create_engine(database_url)
    sql = SCHEMA_PATH.read_text()

    with engine.begin() as conn:
        # Postgres via psycopg2 can execute a multi-statement script in one call.
        conn.execute(sqlalchemy.text(sql))

    print("Schema applied successfully.")
    with engine.connect() as conn:
        tables = conn.execute(
            sqlalchemy.text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
        ).fetchall()
        print("Tables now in database:")
        for (name,) in tables:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
