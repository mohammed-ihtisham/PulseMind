import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import shap  # make sure 'shap' is installed

from .utils import (
    get_data_path,
    get_models_dir,
    compute_mental_health_score,
)


def load_dataset() -> pd.DataFrame:
    """
    Load the digital habits vs. mental health dataset and
    add a derived 'mental_health_score' column.
    """
    data_path = get_data_path()
    df = pd.read_csv(data_path)

    # Expecting columns:
    # 'screen_time_hours', 'social_media_platforms_used',
    # 'hours_on_TikTok', 'sleep_hours', 'stress_level', 'mood_score'
    required_cols = {
        "screen_time_hours",
        "social_media_platforms_used",
        "hours_on_TikTok",
        "sleep_hours",
        "stress_level",
        "mood_score",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Create composite mental_health_score
    df["mental_health_score"] = df.apply(
        lambda row: compute_mental_health_score(
            mood_score=row["mood_score"],
            stress_level=row["stress_level"],
        ),
        axis=1,
    )

    return df


def train_model(df: pd.DataFrame):
    """
    Train a RandomForestRegressor to predict mental_health_score from
    a set of interpretable features.
    """
    # Feature set: we exclude mood_score (it's part of the target definition)
    # but include stress_level as a predictor.
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

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Basic evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("===== Model Evaluation =====")
    print(f"MAE: {mae:.3f}")
    print(f"R^2: {r2:.3f}")
    print("============================")

    # Train a SHAP TreeExplainer for per-prediction feature contributions
    # You can reduce background size if you want smaller artifacts.
    print("Fitting SHAP TreeExplainer (this may take a bit)...")
    # Use a subset as background to keep it lightweight
    background_size = min(2000, X_train.shape[0])
    background = shap.sample(pd.DataFrame(X_train, columns=feature_names),
                             background_size, random_state=42)
    explainer = shap.TreeExplainer(model, data=background)

    return model, explainer, feature_names


def save_artifacts(model, explainer, feature_names):
    """
    Save the trained model, SHAP explainer, and feature names to the models/ directory.
    """
    models_dir = get_models_dir()

    model_path = models_dir / "mental_health_model.pkl"
    explainer_path = models_dir / "mental_health_shap_explainer.pkl"
    feature_names_path = models_dir / "feature_names.json"

    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)

    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"Saved model to       {model_path}")
    print(f"Saved explainer to   {explainer_path}")
    print(f"Saved feature names to {feature_names_path}")


def main():
    print("Loading dataset...")
    df = load_dataset()

    print("Training model...")
    model, explainer, feature_names = train_model(df)

    print("Saving artifacts...")
    save_artifacts(model, explainer, feature_names)

    print("Done.")


if __name__ == "__main__":
    main()
