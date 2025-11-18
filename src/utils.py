from pathlib import Path

# Risk thresholds for the mental health score
# score < -2   -> High risk
# -2 <= score <= 2 -> Medium risk
# score > 2    -> Low risk
HIGH_RISK_THRESHOLD = -2
LOW_RISK_THRESHOLD = 2


def get_project_root() -> Path:
    """
    Returns the project root directory:
    project/
      ├── data/
      ├── models/
      └── src/
    """
    return Path(__file__).resolve().parents[1]


def get_data_path() -> Path:
    """
    Path to the main CSV dataset.
    """
    return get_project_root() / "data" / "digital_habits_vs_mental_health.csv"


def get_models_dir() -> Path:
    """
    Path to the directory where trained models and related artifacts are stored.
    Ensures the directory exists.
    """
    models_dir = get_project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def compute_mental_health_score(mood_score: float, stress_level: float) -> float:
    """
    Compute the composite mental health score.

    Higher is better:
        mental_health_score = mood_score - stress_level
    """
    return mood_score - stress_level


def categorize_risk(score: float) -> str:
    """
    Map a mental health score to a risk category: 'High', 'Medium', or 'Low'.
    """
    if score < HIGH_RISK_THRESHOLD:
        return "High"
    elif score <= LOW_RISK_THRESHOLD:
        return "Medium"
    else:
        return "Low"
