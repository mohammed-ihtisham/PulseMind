"""
Generate Figure 3: Global Feature Importance

Two side-by-side plots:
- LEFT: Bar chart of Random Forest feature importances
- RIGHT: SHAP summary plot (beeswarm) with features on y-axis, SHAP values on x-axis
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.train import load_dataset

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,  # Embed fonts in PDF
    'ps.fonttype': 42,
})

print("Loading model...")
models_dir = project_root / "models"
model_path = models_dir / "mental_health_model.pkl"
feature_names_path = models_dir / "feature_names.json"

# Load artifacts
model = joblib.load(model_path)
feature_names = json.load(open(feature_names_path))

print(f"Loaded model: {type(model).__name__}")
print(f"Feature names: {feature_names}")

# Get feature importances
feature_importances = model.feature_importances_
print("\nFeature Importances:")
for name, imp in zip(feature_names, feature_importances):
    print(f"  {name}: {imp:.4f}")

# Create feature display names (more readable)
display_names = {
    "sleep_hours": "Sleep Hours",
    "screen_time_hours": "Screen Time",
    "hours_on_TikTok": "TikTok Hours",
    "social_media_platforms_used": "Platforms Used"
}

# Load dataset for SHAP plot
print("\nLoading dataset for SHAP values...")
df = load_dataset()

# Prepare data for SHAP (use test set for the plot)
from sklearn.model_selection import train_test_split

X = df[feature_names].values
y = df["mental_health_score"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SHAP explainer on the fly
print("Creating SHAP TreeExplainer...")
# For Random Forest, TreeExplainer works well without background data
explainer = shap.TreeExplainer(model)
print("Explainer created successfully")

# Use a sample from test set for SHAP visualization (beeswarm plot)
# Limit to 200 samples for performance (enough for good visualization)
sample_size = min(200, X_test.shape[0])
np.random.seed(42)
sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
X_sample = pd.DataFrame(X_test[sample_indices], columns=feature_names)

print(f"Computing SHAP values for {len(X_sample)} samples...")
# Use values attribute to ensure numpy array
X_sample_values = X_sample.values
shap_values = explainer.shap_values(X_sample_values)
print(f"SHAP values computed successfully")

# Ensure shap_values is 2D array
if isinstance(shap_values, list):
    shap_values = np.array(shap_values)
if shap_values.ndim == 1:
    shap_values = shap_values.reshape(-1, 1)
if shap_values.ndim == 3:
    # Sometimes SHAP returns (n_outputs, n_samples, n_features), take first output
    shap_values = shap_values[0]
print(f"Final SHAP values shape: {shap_values.shape}")

print("Creating Figure 3...")

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# ===== LEFT PLOT: Feature Importances Bar Chart =====
# Sort features by importance (highest first)
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_names = [display_names[feature_names[i]] for i in sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Create horizontal bar chart
colors = sns.color_palette("husl", len(sorted_names))
bars = ax1.barh(range(len(sorted_names)), sorted_importances, color=colors, edgecolor='black', linewidth=0.5)

# Customize left plot
ax1.set_yticks(range(len(sorted_names)))
ax1.set_yticklabels(sorted_names)
ax1.set_xlabel('Feature Importance', fontweight='bold', fontsize=12)
ax1.set_title('Random Forest Feature Importances', fontweight='bold', fontsize=13, pad=15)
ax1.set_xlim(0, max(sorted_importances) * 1.1)
ax1.grid(True, alpha=0.3, linestyle='--', axis='x')

# Add value labels on bars
for i, (bar, imp) in enumerate(zip(bars, sorted_importances)):
    width = bar.get_width()
    ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{imp:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

# Invert y-axis so highest importance is at top
ax1.invert_yaxis()

# ===== RIGHT PLOT: SHAP Summary Plot (Beeswarm) =====
# Create a custom beeswarm plot
# For each feature, plot SHAP values colored by feature value
feature_order = sorted_indices  # Use same order as bar chart

# Prepare data for plotting
n_features = len(feature_names)

# For each feature, plot points colored by feature value
scatter = None
np.random.seed(42)  # For reproducible jitter

for feat_idx in feature_order:
    feature_name = feature_names[feat_idx]
    feature_shap = shap_values[:, feat_idx]
    feature_values = X_sample.iloc[:, feat_idx].values
    
    # Position points with some jitter
    y_pos = list(feature_order).index(feat_idx)
    jitter_amount = 0.15
    jitter = np.random.uniform(-jitter_amount, jitter_amount, len(feature_shap))
    y_scattered = y_pos + jitter
    
    # Color by feature value (normalize to [0, 1] for colormap)
    fmin, fmax = feature_values.min(), feature_values.max()
    if fmax > fmin:
        norm_values = (feature_values - fmin) / (fmax - fmin)
    else:
        norm_values = np.zeros_like(feature_values)
    
    # Create scatter plot
    scatter = ax2.scatter(feature_shap, y_scattered, c=norm_values, cmap='coolwarm', 
                         alpha=0.6, s=15, edgecolors='none', vmin=0, vmax=1,
                         rasterized=True)

# Set y-axis labels
ax2.set_yticks(range(n_features))
ax2.set_yticklabels([display_names[feature_names[i]] for i in feature_order])
ax2.set_xlabel('SHAP Value (Impact on Prediction)', fontweight='bold', fontsize=12)
ax2.set_title('SHAP Summary Plot', fontweight='bold', fontsize=13, pad=15)
ax2.grid(True, alpha=0.3, linestyle='--', axis='x')

# Add vertical line at SHAP = 0
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Add colorbar
if scatter is not None:
    cbar = plt.colorbar(scatter, ax=ax2, label='Feature Value (normalized)', pad=0.02)
    cbar.ax.set_ylabel('Feature Value\n(low → high)', fontsize=9)

# Set y-axis limits
ax2.set_ylim(-0.5, n_features - 0.5)

# Invert y-axis to match left plot (highest importance at top)
ax2.invert_yaxis()

# Adjust layout first, then add title with slightly reduced spacing
plt.tight_layout(rect=[0, 0, 1, 0.92])

# Add overall title
fig.suptitle(
    'Global Feature Importance Analysis',
    fontsize=16,
    fontweight='bold',
    y=0.97
)

# Save figure
output_path = project_root / "figure3_feature_importance.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigure 3 saved to: {output_path}")

# Also save as PDF for paper submission
pdf_path = project_root / "figure3_feature_importance.pdf"
fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure 3 (PDF) saved to: {pdf_path}")

print("\n" + "="*70)
print("Figure 3 Summary")
print("="*70)
print("\nFeature Importances (left plot):")
for name, imp in zip(sorted_names, sorted_importances):
    print(f"  {name:20s}: {imp:.4f}")
print(f"\nSHAP plot (right plot):")
print(f"  Sample size: {len(X_sample)}")
print(f"  Features ordered by importance (top to bottom)")
print("\n✓ Figure 3 generation complete!")

