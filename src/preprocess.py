"""
preprocess.py

Filters raw spot price data down to relevant OS, regions, and GPU instance
families, then saves the result as a parquet file.

Usage:
    python src/preprocess.py
"""

import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/processed/spot_prices_raw.parquet")
OUTPUT_PATH = Path("data/processed/spot_prices_filtered.parquet")

REGIONS_TO_KEEP = ["use1", "use2", "usw2", "euw1"]  # us-east-1, us-east-2, us-west-2, eu-west-1
# GPU instance families used for ML training — excludes CPU-only instances
GPU_FAMILIES = ["p3", "p4", "p5", "g4dn", "g5", "g6"]


def filter_os(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Linux/UNIX rows — all serious ML workloads run on Linux."""
    return df[df["os"] == "Linux/UNIX"]


def filter_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the four highest-traffic AWS regions."""
    # Availability zone looks like "use1-az1" — split to get region code "use1"
    df["region"] = df["availability_zone"].str.split("-", n=1).str[0]
    return df[df["region"].isin(REGIONS_TO_KEEP)]


def filter_instances(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only GPU instance families; drop bare-metal variants."""
    df["instance_family"] = df["instance_type"].str.split(".").str[0]
    df = df[df["instance_family"].isin(GPU_FAMILIES)]
    df = df[~df["instance_type"].str.contains("metal")]
    return df.drop(columns=["instance_family"])


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp column to datetime."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def preprocess() -> None:
    """Run all preprocessing steps and save filtered parquet."""
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Rows loaded: {df.shape[0]:,}")

    df = filter_os(df)
    print(f"  After OS filter: {df.shape[0]:,}")

    df = filter_regions(df)
    print(f"  After region filter: {df.shape[0]:,}")

    df = filter_instances(df)
    print(f"  After instance filter: {df.shape[0]:,}")

    df = parse_timestamps(df)

    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()