"""
Generate Figure 2: Distribution of the composite mental health score (mood − stress)
with risk category boundaries.

This figure shows:
- Histogram of mental health score distribution
- Vertical lines at 0, 2, 4, and 6 marking the 5-tier risk thresholds
- Distribution is roughly centered near 3 with tails extending to -8 and 9
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    compute_mental_health_score,
    CRITICAL_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD_HIGH,
    LOW_RISK_THRESHOLD_HIGH,
    categorize_risk
)

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

# Load the dataset
data_path = project_root / "data" / "digital_habits_vs_mental_health.csv"
df = pd.read_csv(data_path)

print(f"Loaded dataset with {len(df):,} rows")

# Compute mental health score
df['mental_health_score'] = df.apply(
    lambda row: compute_mental_health_score(
        mood_score=row['mood_score'],
        stress_level=row['stress_level']
    ),
    axis=1
)

# Get statistics
mean_score = df['mental_health_score'].mean()
median_score = df['mental_health_score'].median()
std_score = df['mental_health_score'].std()
min_score = df['mental_health_score'].min()
max_score = df['mental_health_score'].max()

print(f"\nMental Health Score Statistics:")
print(f"  Mean: {mean_score:.2f}")
print(f"  Median: {median_score:.2f}")
print(f"  Std: {std_score:.2f}")
print(f"  Range: [{min_score:.1f}, {max_score:.1f}]")
print(f"  5-Tier Risk thresholds: {CRITICAL_RISK_THRESHOLD}, {HIGH_RISK_THRESHOLD}, {MEDIUM_RISK_THRESHOLD_HIGH}, {LOW_RISK_THRESHOLD_HIGH}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create histogram
# Use bins that cover the full range from min to max
# Since scores are integers, use integer bins
bin_edges = [x - 0.5 for x in range(int(min_score), int(max_score) + 2)]

n, bins, patches = ax.hist(
    df['mental_health_score'],
    bins=bin_edges,
    edgecolor='black',
    linewidth=0.5,
    alpha=0.7,
    density=False
)

# Color-code bars based on 5-tier risk categories
# Critical: dark red, High: red, Medium: yellow, Low: light green, Healthy: green
for i, patch in enumerate(patches):
    # Calculate bin center
    bin_center = (bins[i] + bins[i + 1]) / 2
    
    # Get category using the categorize_risk function
    category = categorize_risk(bin_center)
    
    if category == "Critical":
        patch.set_facecolor('#991b1b')  # Dark red
    elif category == "High":
        patch.set_facecolor('#ef4444')  # Red
    elif category == "Medium":
        patch.set_facecolor('#facc15')  # Yellow
    elif category == "Low":
        patch.set_facecolor('#86efac')  # Light green
    else:  # Healthy
        patch.set_facecolor('#22c55e')  # Green

# Add vertical lines at all 5-tier risk thresholds
thresholds = [CRITICAL_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD_HIGH, LOW_RISK_THRESHOLD_HIGH]
for thresh in thresholds:
    ax.axvline(thresh, color='red', linestyle='-', linewidth=1.5, alpha=0.7, zorder=3)

# Add vertical line for mean (optional, in a different style)
ax.axvline(mean_score, color='darkgreen', linestyle='--', linewidth=1.5, 
           alpha=0.7, label=f'Mean ({mean_score:.2f})', zorder=2)

# Add color-coded legend entries for risk categories
from matplotlib.patches import Rectangle
critical_patch = Rectangle((0, 0), 1, 1, facecolor='#991b1b', edgecolor='black', linewidth=0.5, alpha=0.7)
high_patch = Rectangle((0, 0), 1, 1, facecolor='#ef4444', edgecolor='black', linewidth=0.5, alpha=0.7)
medium_patch = Rectangle((0, 0), 1, 1, facecolor='#facc15', edgecolor='black', linewidth=0.5, alpha=0.7)
low_patch = Rectangle((0, 0), 1, 1, facecolor='#86efac', edgecolor='black', linewidth=0.5, alpha=0.7)
healthy_patch = Rectangle((0, 0), 1, 1, facecolor='#22c55e', edgecolor='black', linewidth=0.5, alpha=0.7)

# Formatting
ax.set_xlabel('Mental Health Score (mood − stress)', fontweight='bold', fontsize=12)
ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
ax.set_title(
    'Distribution of the Composite Mental Health Score\n'
    'with Risk Category Boundaries',
    fontweight='bold',
    fontsize=13,
    pad=15
)

# Set x-axis limits to show full range with some padding
ax.set_xlim(min_score - 0.5, max_score + 0.5)

# Set x-axis ticks to show integers
ax.set_xticks(range(int(min_score), int(max_score) + 1))

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add legend with 5-tier risk category colors
legend_elements = [
    critical_patch,
    high_patch,
    medium_patch,
    low_patch,
    healthy_patch,
    plt.Line2D([0], [0], color='red', linestyle='-', linewidth=1.5, alpha=0.7, 
               label=f'Thresholds ({CRITICAL_RISK_THRESHOLD}, {HIGH_RISK_THRESHOLD}, {MEDIUM_RISK_THRESHOLD_HIGH}, {LOW_RISK_THRESHOLD_HIGH})'),
    plt.Line2D([0], [0], color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({mean_score:.2f})')
]
legend_labels = [
    f'Critical (score < {CRITICAL_RISK_THRESHOLD})',
    f'High ({CRITICAL_RISK_THRESHOLD} ≤ score < {HIGH_RISK_THRESHOLD})',
    f'Medium ({HIGH_RISK_THRESHOLD} ≤ score < {MEDIUM_RISK_THRESHOLD_HIGH})',
    f'Low ({MEDIUM_RISK_THRESHOLD_HIGH} ≤ score < {LOW_RISK_THRESHOLD_HIGH})',
    f'Healthy (score ≥ {LOW_RISK_THRESHOLD_HIGH})',
    f'Risk Thresholds',
    f'Mean ({mean_score:.2f})'
]
ax.legend(legend_elements, legend_labels, loc='upper right', framealpha=0.9, fontsize=8)

# Add text annotation about thresholds
ax.text(0.02, 0.98,
        f'5-tier thresholds at {CRITICAL_RISK_THRESHOLD}, {HIGH_RISK_THRESHOLD}, {MEDIUM_RISK_THRESHOLD_HIGH}, {LOW_RISK_THRESHOLD_HIGH}\n'
        f'create balanced risk categories based on quintiles.',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
        style='italic')

# Adjust layout
plt.tight_layout()

# Save figure
output_path = project_root / "figure2_target_distribution.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigure 2 saved to: {output_path}")

# Also save as PDF for paper submission
pdf_path = project_root / "figure2_target_distribution.pdf"
fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure 2 (PDF) saved to: {pdf_path}")

# Print summary
print("\n" + "="*70)
print("Distribution Summary")
print("="*70)
print(f"Total observations: {len(df):,}")
print(f"Score range: [{min_score:.1f}, {max_score:.1f}]")
print(f"Mean: {mean_score:.2f}, Median: {median_score:.2f}, Std: {std_score:.2f}")
print(f"\nRisk Categories (5-Tier System):")
critical = (df['mental_health_score'] < CRITICAL_RISK_THRESHOLD).sum()
high = ((df['mental_health_score'] >= CRITICAL_RISK_THRESHOLD) & (df['mental_health_score'] < HIGH_RISK_THRESHOLD)).sum()
medium = ((df['mental_health_score'] >= HIGH_RISK_THRESHOLD) & (df['mental_health_score'] < MEDIUM_RISK_THRESHOLD_HIGH)).sum()
low = ((df['mental_health_score'] >= MEDIUM_RISK_THRESHOLD_HIGH) & (df['mental_health_score'] < LOW_RISK_THRESHOLD_HIGH)).sum()
healthy = (df['mental_health_score'] >= LOW_RISK_THRESHOLD_HIGH).sum()
print(f"  Critical (score < {CRITICAL_RISK_THRESHOLD}): {critical:,} ({100*critical/len(df):.1f}%)")
print(f"  High ({CRITICAL_RISK_THRESHOLD} ≤ score < {HIGH_RISK_THRESHOLD}): {high:,} ({100*high/len(df):.1f}%)")
print(f"  Medium ({HIGH_RISK_THRESHOLD} ≤ score < {MEDIUM_RISK_THRESHOLD_HIGH}): {medium:,} ({100*medium/len(df):.1f}%)")
print(f"  Low ({MEDIUM_RISK_THRESHOLD_HIGH} ≤ score < {LOW_RISK_THRESHOLD_HIGH}): {low:,} ({100*low/len(df):.1f}%)")
print(f"  Healthy (score ≥ {LOW_RISK_THRESHOLD_HIGH}): {healthy:,} ({100*healthy/len(df):.1f}%)")
print("="*70)
print("\n✓ Figure 2 generation complete!")
