"""
ingest.py

Loads raw .tsv.zst spot price files from the Zenodo dataset and saves
them as a parquet file for downstream processing.

Usage:
    python src/ingest.py
"""

import io
import zstandard as zstd
import pandas as pd
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/spot_prices_raw.parquet")

COLUMNS = ["availability_zone", "instance_type", "os", "price", "timestamp"]


def load_zst_file(filepath: Path) -> pd.DataFrame:
    """Decompress and load a single .tsv.zst file into a DataFrame."""
    with open(filepath, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            df = pd.read_csv(
                io.TextIOWrapper(reader),
                sep="\t",
                names=COLUMNS,
            )
    return df


def ingest() -> None:
    """Load all .tsv.zst files in the raw data directory and save as parquet."""
    zst_files = sorted(RAW_DATA_DIR.glob("*.tsv.zst"))

    if not zst_files:
        raise FileNotFoundError(f"No .tsv.zst files found in {RAW_DATA_DIR}")

    print(f"Found {len(zst_files)} file(s) to ingest...")

    dfs = []
    for filepath in zst_files:
        print(f"  Loading {filepath.name}...")
        df = load_zst_file(filepath)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {combined.shape[0]:,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    ingest()