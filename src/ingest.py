"""
ingest.py

Loads raw .tsv.zst spot price files from the Zenodo dataset and saves
the filtered result as a parquet file for downstream processing.

Filtering (region/instance/OS) happens per-chunk as each file streams
in, rather than after a full load — a single month's raw file is ~22M
rows and only ~0.15% of rows survive filtering, so materializing the
whole unfiltered month in memory first doesn't scale past a couple of
months.

Usage:
    python src/ingest.py
"""

import io
import zstandard as zstd
import pandas as pd
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/spot_prices_filtered.parquet")

COLUMNS = ["availability_zone", "instance_type", "os", "price", "timestamp"]
CHUNK_SIZE = 1_000_000

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


def filter_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply the region/instance/OS filters to a single chunk."""
    chunk = filter_os(chunk)
    chunk = filter_regions(chunk)
    chunk = filter_instances(chunk)
    return chunk


def load_and_filter_zst_file(filepath: Path) -> pd.DataFrame:
    """Decompress a .tsv.zst file and filter it chunk-by-chunk as it's read."""
    filtered_chunks = []
    with open(filepath, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            chunks = pd.read_csv(
                io.TextIOWrapper(reader),
                sep="\t",
                names=COLUMNS,
                chunksize=CHUNK_SIZE,
            )
            for chunk in chunks:
                filtered_chunks.append(filter_chunk(chunk))
    return pd.concat(filtered_chunks, ignore_index=True)


def ingest() -> None:
    """Load, filter, and combine all .tsv.zst files in the raw data directory."""
    zst_files = sorted(RAW_DATA_DIR.glob("*.tsv.zst"))

    if not zst_files:
        raise FileNotFoundError(f"No .tsv.zst files found in {RAW_DATA_DIR}")

    print(f"Found {len(zst_files)} file(s) to ingest...")

    filtered_dfs = []
    for filepath in zst_files:
        print(f"  Loading and filtering {filepath.name}...")
        df = load_and_filter_zst_file(filepath)
        print(f"    Kept {df.shape[0]:,} rows")
        filtered_dfs.append(df)

    combined = pd.concat(filtered_dfs, ignore_index=True)
    print(f"Total filtered rows: {combined.shape[0]:,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    ingest()
