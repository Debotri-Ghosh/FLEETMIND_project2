"""
Downloads the APS Failure at Scania Trucks dataset for FleetMind.
Requires: pip install ucimlrepo pandas

Usage: python download_aps.py
"""
import os

DATA_DIR = "data"
APS_DIR = os.path.join(DATA_DIR, "aps_failure")
os.makedirs(APS_DIR, exist_ok=True)


def setup_aps_failure():
    print("\n--- APS Failure at Scania Trucks (FleetMind) ---")
    out_path = os.path.join(APS_DIR, "aps_failure_training_set.csv")

    if os.path.exists(out_path):
        print(f"  already have {out_path}, skipping")
        return

    try:
        from ucimlrepo import fetch_ucirepo
        import pandas as pd

        dataset = fetch_ucirepo(id=421)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([y, X], axis=1)
        df.to_csv(out_path, index=False)
        print("  saved combined dataset to:", out_path)
        print(f"  shape: {df.shape[0]} rows x {df.shape[1]} columns")
    except ImportError:
        print("  'ucimlrepo' not installed.")
        print("  Run: pip install ucimlrepo   (then re-run this script)")
        print("  Or download manually from:")
        print("  https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks")
        print("  Kaggle mirror: https://www.kaggle.com/datasets/uciml/aps-failure-at-scania-trucks-data-set")
        return

    print("\nDone. Real, public dataset from Scania CV AB via UCI ML Repository.")


if __name__ == "__main__":
    setup_aps_failure()
