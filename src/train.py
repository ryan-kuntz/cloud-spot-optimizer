"""
train.py

Trains a RandomForest model on the model-ready dataset and saves
the trained model to models/spot_price_model.pkl.

Uses a time-based train/test split — never shuffle time series data.

Usage:
    python src/train.py
"""

from config import FEATURES, TARGET, TRAIN_SPLIT, MODEL_PATH

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_PATH = Path("data/processed/model_ready.parquet")


def time_split(df: pd.DataFrame) -> tuple:
    """Split into train/test using time order — no random shuffling."""
    df = df.sort_values("timestamp")
    split_idx = int(len(df) * TRAIN_SPLIT)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


def naive_baseline(y_train: pd.Series, y_test: pd.Series) -> float:
    """Naive baseline: predict last known price for every test observation."""
    naive_pred = y_test.shift(1).fillna(y_train.iloc[-1])
    return mean_absolute_error(y_test, naive_pred)


def train() -> None:
    """Train model, evaluate against naive baseline, and save."""
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)

    train_df, test_df = time_split(df)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("Training RandomForest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    naive_mae = naive_baseline(y_train, y_test)
    improvement = (naive_mae - mae) / naive_mae * 100

    print(f"\nResults:")
    print(f"  MAE:       ${mae:.4f}/hr")
    print(f"  RMSE:      ${rmse:.4f}/hr")
    print(f"  Naive MAE: ${naive_mae:.4f}/hr")
    print(f"  Improvement over naive: {improvement:.1f}%")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()