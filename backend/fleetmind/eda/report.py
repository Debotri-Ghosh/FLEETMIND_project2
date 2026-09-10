"""EDA report (ported from scripts/eda_report.py, connection-injected for testability)."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sqlalchemy

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR = Path("eda_output")


def class_distribution(conn: sqlalchemy.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT label, COUNT(*) AS window_count FROM signal_windows GROUP BY label ORDER BY label",
        conn,
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(df["label"], df["window_count"])
    plt.xlabel("Fault Class")
    plt.ylabel("Number of Signal Windows")
    plt.title("Signal Window Distribution by Fault Class")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close()
    return df


def waveforms_by_class(conn: sqlalchemy.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT DISTINCT ON (sw.label) sw.label, sw.raw_values, e.sampling_rate_hz
        FROM signal_windows sw
        JOIN experiments e ON e.id = sw.experiment_id
        ORDER BY sw.label, sw.id
        """,
        conn,
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(12, 7))
    for _, row in df.iterrows():
        values = np.asarray(row["raw_values"], dtype=float)[:500]
        plt.plot(np.arange(len(values)), values, label=str(row["label"]))
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Representative Waveforms by Fault Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "waveforms_by_class.png", dpi=200)
    plt.close()
    return df


def summary_statistics(conn: sqlalchemy.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT label, raw_values FROM signal_windows ORDER BY id", conn)
    results = []
    for _, row in df.iterrows():
        values = np.asarray(row["raw_values"], dtype=float)
        if len(values) == 0:
            continue
        results.append(
            {
                "label": row["label"],
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "rms": float(np.sqrt(np.mean(values**2))),
                "peak_to_peak": float(np.ptp(values)),
            }
        )
    stats_df = pd.DataFrame(results)
    summary = (
        stats_df.groupby("label")
        .agg(
            mean=("mean", "mean"),
            std=("std", "mean"),
            min=("min", "mean"),
            max=("max", "mean"),
            rms=("rms", "mean"),
            peak_to_peak=("peak_to_peak", "mean"),
        )
        .reset_index()
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    return summary


def main() -> None:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    engine = sqlalchemy.create_engine(os.environ["DATABASE_URL"])
    OUTPUT_DIR.mkdir(exist_ok=True)
    with engine.connect() as conn:
        print("=== CLASS DISTRIBUTION ===")
        print(class_distribution(conn).to_string(index=False))
        print("=== REPRESENTATIVE WAVEFORMS ===")
        waveforms_by_class(conn)
        print("=== SUMMARY STATISTICS ===")
        print(summary_statistics(conn).to_string(index=False))
    engine.dispose()
    print(f"EDA results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
