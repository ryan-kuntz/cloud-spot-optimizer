# src/config.py

from pathlib import Path

# Paths
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
MODEL_PATH = Path("models/spot_price_model.pkl")

# Features
FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "natgas_price",
]

TARGET = "price"
TRAIN_SPLIT = 0.8