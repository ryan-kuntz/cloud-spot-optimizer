"""
preprocess.py

Parses timestamps on the filtered spot price data produced by
src/ingest.py (which already applies the OS/region/instance filters
during ingestion, so this step only needs to handle timestamps).

Usage:
    python src/preprocess.py
"""

import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/processed/spot_prices_filtered.parquet")


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp column to datetime."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def preprocess() -> None:
    """Parse timestamps on the already-filtered dataset."""
    print(f"Loading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Rows loaded: {df.shape[0]:,}")

    df = parse_timestamps(df)

    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    df.to_parquet(DATA_PATH, index=False)
    print(f"Saved to {DATA_PATH}")


if __name__ == "__main__":
    preprocess()
