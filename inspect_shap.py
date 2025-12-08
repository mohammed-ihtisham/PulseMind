"""Inspect the SHAP explainer pickle file."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

explainer_path = Path("models/mental_health_shap_explainer.pkl")
print(f"Loading SHAP explainer from: {explainer_path}")

explainer = joblib.load(explainer_path)

print(f"\n{'='*70}")
print("SHAP Explainer Information")
print(f"{'='*70}")
print(f"Type: {type(explainer)}")
print(f"Class: {explainer.__class__.__name__}")
print(f"Module: {explainer.__class__.__module__}")

print(f"\n{'='*70}")
print("Key Attributes:")
print(f"{'='*70}")

# Check for common SHAP explainer attributes
if hasattr(explainer, 'model'):
    model = explainer.model
    print(f"✓ model: {type(model).__name__}")
    if hasattr(model, 'n_estimators'):
        print(f"    - n_estimators: {model.n_estimators}")
    if hasattr(model, 'feature_importances_'):
        print(f"    - feature_importances_: {model.feature_importances_}")

if hasattr(explainer, 'data'):
    data = explainer.data
    if isinstance(data, (np.ndarray, pd.DataFrame)):
        print(f"✓ data: {type(data).__name__}, shape: {data.shape}")
        if isinstance(data, pd.DataFrame):
            print(f"    - columns: {list(data.columns)}")
    else:
        print(f"✓ data: {type(data).__name__}")

if hasattr(explainer, 'expected_value'):
    print(f"✓ expected_value: {explainer.expected_value}")

if hasattr(explainer, 'feature_names'):
    print(f"✓ feature_names: {explainer.feature_names}")

if hasattr(explainer, 'base_value'):
    print(f"✓ base_value: {explainer.base_value}")

print(f"\n{'='*70}")
print("All Public Attributes:")
print(f"{'='*70}")
attrs = [attr for attr in dir(explainer) if not attr.startswith('_')]
for attr in sorted(attrs):
    try:
        value = getattr(explainer, attr)
        if not callable(value):
            if isinstance(value, (np.ndarray, pd.DataFrame)):
                print(f"  - {attr}: {type(value).__name__}, shape: {value.shape}")
            elif isinstance(value, (list, tuple)) and len(value) < 20:
                print(f"  - {attr}: {type(value).__name__}, length: {len(value)}")
            elif isinstance(value, (int, float, str, bool)) or value is None:
                print(f"  - {attr}: {value}")
            else:
                print(f"  - {attr}: {type(value).__name__}")
    except:
        print(f"  - {attr}: (could not access)")

print(f"\n{'='*70}")
print("Methods:")
print(f"{'='*70}")
methods = [attr for attr in dir(explainer) if not attr.startswith('_') and callable(getattr(explainer, attr, None))]
for method in sorted(methods):
    print(f"  - {method}()")

