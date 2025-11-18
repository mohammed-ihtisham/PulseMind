# src/predict.py

import json
from typing import Dict, Any, List
import time

import joblib
import numpy as np

from .utils import get_models_dir, categorize_risk

# ---- Module-level caches ----
_MODEL = None
_FEATURE_NAMES: List[str] = []


def timestamp(label: str):
    print(f"[DEBUG] {label} at {time.strftime('%H:%M:%S')}")


def _load_artifacts_once():
    """
    Load model and feature names once and store them in module-level globals.
    """
    global _MODEL, _FEATURE_NAMES

    if _MODEL is not None and _FEATURE_NAMES:
        return

    timestamp("Loading model & feature names")

    models_dir = get_models_dir()
    model_path = models_dir / "mental_health_model.pkl"
    feature_names_path = models_dir / "feature_names.json"

    timestamp("Loading model.pkl...")
    _MODEL = joblib.load(model_path)
    timestamp("Model loaded")

    timestamp("Loading feature_names.json...")
    with open(feature_names_path, "r") as f:
        _FEATURE_NAMES = json.load(f)
    timestamp(f"Loaded feature names: {_FEATURE_NAMES}")


def predict_mental_health(user_features: Dict[str, float]) -> Dict[str, Any]:
    """
    Predict mental health score and provide a simple contribution-style breakdown
    using the model's feature importances instead of SHAP (to avoid hangs).

    user_features example:
    {
        "screen_time_hours": 7.5,
        "social_media_platforms_used": 3,
        "hours_on_TikTok": 2.0,
        "sleep_hours": 6.0,
        "stress_level": 7
    }
    """
    timestamp("Starting predict_mental_health()")

    _load_artifacts_once()

    # Ensure all required features are present
    missing = set(_FEATURE_NAMES) - set(user_features.keys())
    if missing:
        raise ValueError(f"Missing user features: {missing}")

    # Build input row in correct order
    timestamp("Building input feature vector...")
    X = np.array([[user_features[name] for name in _FEATURE_NAMES]])

    # Predict score
    timestamp("Predicting score...")
    predicted_score = float(_MODEL.predict(X)[0])
    risk_category = categorize_risk(predicted_score)
    timestamp(f"Predicted score: {predicted_score:.3f} | Risk: {risk_category}")

    # ---- Simple global-importance-based contributions ----
    timestamp("Computing simple contributions from feature importances...")

    # Feature importances tell us how influential each feature is overall
    importances = _MODEL.feature_importances_  # shape: (n_features,)

    # We'll create a pseudo-contribution based on importance * (feature value)
    raw_contribs = importances * X[0]

    # Normalize contributions so they are comparable
    abs_sum = np.sum(np.abs(raw_contribs)) or 1.0  # avoid div-by-zero
    normalized_contribs = raw_contribs / abs_sum

    # Use the mean prediction as a "baseline" reference if you like,
    # but for simplicity we'll just expose normalized contributions.
    base_value = 0.0  # just a neutral reference point

    contributions = []
    for name, value, raw, norm in zip(
        _FEATURE_NAMES, X[0], raw_contribs, normalized_contribs
    ):
        contributions.append(
            {
                "feature": name,
                "value": float(value),
                "raw_contribution": float(raw),
                "normalized_contribution": float(norm),
                "direction": "increases_score" if raw > 0 else "decreases_score",
                "abs_contribution": float(abs(norm)),
            }
        )

    timestamp("Sorting contributions...")
    contributions_sorted = sorted(
        contributions,
        key=lambda d: d["abs_contribution"],
        reverse=True,
    )

    timestamp("predict_mental_health() complete")
    print("--------")

    return {
        "predicted_score": predicted_score,
        "risk_category": risk_category,
        "base_value": base_value,
        "contributions": contributions_sorted,
    }


def demo():
    timestamp("Demo started")

    example_user = {
        "screen_time_hours": 8.5,
        "social_media_platforms_used": 4,
        "hours_on_TikTok": 3.0,
        "sleep_hours": 5.5,
        "stress_level": 8,
    }

    result = predict_mental_health(example_user)

    print("===== Prediction Result =====")
    print(f"Predicted mental health score: {result['predicted_score']:.3f}")
    print(f"Risk category: {result['risk_category']}")
    print(f"Model base value: {result['base_value']:.3f}")
    print("\nTop contributing factors (by normalized contribution):")
    for contrib in result["contributions"][:5]:
        print(
            f"  - {contrib['feature']}: "
            f"value={contrib['value']}, "
            f"norm_contrib={contrib['normalized_contribution']:.3f} "
            f"({contrib['direction']})"
        )


if __name__ == "__main__":
    demo()
