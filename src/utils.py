from pathlib import Path

# Risk thresholds for the mental health score (balanced quintile-based system)
# Critical Risk: score < 0
# High Risk: 0 <= score < 2
# Medium Risk: 2 <= score < 4
# Low Risk: 4 <= score < 6
# Healthy: score >= 6
CRITICAL_RISK_THRESHOLD = 0
HIGH_RISK_THRESHOLD = 2
MEDIUM_RISK_THRESHOLD_LOW = 2
MEDIUM_RISK_THRESHOLD_HIGH = 4
LOW_RISK_THRESHOLD_LOW = 4
LOW_RISK_THRESHOLD_HIGH = 6
HEALTHY_THRESHOLD = 6

# Backward compatibility: old threshold names (for figure generation scripts)
# These represent the old 3-tier system boundaries
OLD_HIGH_RISK_THRESHOLD = -2  # Old high risk threshold
OLD_LOW_RISK_THRESHOLD = 2    # Old low risk threshold


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
    Map a mental health score to a risk category: 'Critical', 'High', 'Medium', 'Low', or 'Healthy'.
    
    Categories based on balanced quintile thresholds [0, 2, 4, 6]:
    - Critical: score < 0
    - High: 0 <= score < 2
    - Medium: 2 <= score < 4
    - Low: 4 <= score < 6
    - Healthy: score >= 6
    """
    if score < CRITICAL_RISK_THRESHOLD:
        return "Critical"
    elif score < HIGH_RISK_THRESHOLD:
        return "High"
    elif score < MEDIUM_RISK_THRESHOLD_HIGH:
        return "Medium"
    elif score < LOW_RISK_THRESHOLD_HIGH:
        return "Low"
    else:
        return "Healthy"
