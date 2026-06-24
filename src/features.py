"""
features.py

Engineers time-based features from the base dataset and saves
the result as model_ready.parquet for use in training.

Usage:
    python src/features.py
"""

import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/processed/base_dataset.parquet")
OUTPUT_PATH = Path("data/processed/model_ready.parquet")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based signals from the timestamp column."""
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def engineer_features() -> None:
    """Run feature engineering and save model-ready dataset."""
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Rows loaded: {df.shape[0]:,}")

    df = add_time_features(df)
    print(f"  Features: {df.columns.tolist()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    engineer_features()