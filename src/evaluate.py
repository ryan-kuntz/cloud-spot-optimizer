"""
evaluate.py

Loads the trained model and test data, prints evaluation metrics,
and shows feature importances.

Usage:
    python src/evaluate.py
"""

from config import FEATURES, TARGET, TRAIN_SPLIT, MODEL_PATH

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_PATH = Path("data/processed/model_ready.parquet")

def evaluate() -> None:
    """Load model and test set, print metrics and feature importances."""
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH).sort_values("timestamp")

    split_idx = int(len(df) * TRAIN_SPLIT)
    test_df = df.iloc[split_idx:]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nEvaluation Metrics:")
    print(f"  MAE:  ${mae:.4f}/hr")
    print(f"  RMSE: ${rmse:.4f}/hr")

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=False)

    print(f"\nFeature Importances:")
    for feature, importance in importances.items():
        bar = "█" * int(importance * 50)
        print(f"  {feature:<25} {importance:.3f} {bar}")


if __name__ == "__main__":
    evaluate()