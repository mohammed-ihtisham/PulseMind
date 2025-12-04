"""
Train an XGBoost regressor on the same synthetic digital-habits dataset used
by the Random Forest model, matching the configuration described in
`research-paper.md`.

This script is intentionally separate so it does not change the existing
Streamlit app or Random Forest training pipeline. It is meant to reproduce
the external XGBoost baseline results discussed in the paper.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .train import load_dataset
from .utils import get_models_dir


def train_xgboost():
    """
    Train an XGBoost regressor with the hyperparameters specified in
    `research-paper.md` and report MAE and R^2 on the held-out test set.
    """
    print("Loading dataset...")
    df: pd.DataFrame = load_dataset()

    # Same feature set as the Random Forest model
    feature_names = [
        "screen_time_hours",
        "social_media_platforms_used",
        "hours_on_TikTok",
        "sleep_hours",
        "stress_level",
    ]

    X = df[feature_names].values
    y = df["mental_health_score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training XGBoost regressor...")
    model = XGBRegressor(
        max_depth=5,
        n_estimators=300,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    # Train with a simple evaluation set; early stopping was used in the paper’s
    # notebook experiments, but we omit it here to keep this script minimal.
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("Evaluating XGBoost regressor...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("===== XGBoost Model Evaluation =====")
    print(f"MAE: {mae:.3f}")
    print(f"R^2: {r2:.3f}")
    print("====================================")

    # Optionally save the trained XGBoost model alongside the Random Forest.
    models_dir = get_models_dir()
    model_path = models_dir / "mental_health_xgb.pkl"
    joblib.dump(model, model_path)
    print(f"Saved XGBoost model to {model_path}")


if __name__ == "__main__":
    train_xgboost()


