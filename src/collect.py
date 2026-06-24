"""
collect.py

Pulls external macro signals (currently: Henry Hub natural gas prices via
FRED) and merges them onto the filtered spot price data, producing the
base dataset used for EDA and modelling.

Requires a FRED API key set in a .env file:
    FRED_API_KEY=your_key_here

Usage:
    python src/collect.py
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

INPUT_PATH = Path("data/processed/spot_prices_filtered.parquet")
OUTPUT_PATH = Path("data/processed/base_dataset.parquet")

FRED_START_DATE = "2017-01-01"


def load_natgas(api_key: str) -> pd.DataFrame:
    """Pull Henry Hub natural gas spot prices from FRED."""
    fred = Fred(api_key=api_key)
    series = fred.get_series("DHHNGSP", observation_start=FRED_START_DATE)

    df = series.reset_index()
    df.columns = ["date", "natgas_price"]
    df["date"] = pd.to_datetime(df["date"])

    print(f"Natural gas data: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def merge_natgas(spot_df: pd.DataFrame, natgas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge natural gas prices onto spot price data by calendar date.

    Converts timestamps to Eastern time before matching, since FRED
    reports daily prices on US calendar days. Fills weekend/holiday
    gaps with forward fill, then backward fill for series start edge case.
    """
    left_key = (
        spot_df["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    merged = pd.merge(spot_df, natgas_df, how="left", left_on=left_key, right_on="date")
    merged["natgas_price"] = merged["natgas_price"].ffill().bfill()

    return merged


def collect() -> None:
    """Merge external signals onto spot price data and save base dataset."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY not found. Add it to your .env file.")

    print(f"Loading {INPUT_PATH}...")
    spot_df = pd.read_parquet(INPUT_PATH)
    print(f"  Rows: {spot_df.shape[0]:,}")

    print("Pulling natural gas prices from FRED...")
    natgas_df = load_natgas(api_key)

    print("Merging...")
    merged = merge_natgas(spot_df, natgas_df)
    print(f"  Merged rows: {merged.shape[0]:,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    collect()