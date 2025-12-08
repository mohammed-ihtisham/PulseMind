"""
Generate Figure 1: Distribution of digital habit and mental health variables
in the synthetic dataset (N = 100,000).

This script creates publication-quality distribution plots for all variables
in the dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
data_path = Path(__file__).parent / "data" / "digital_habits_vs_mental_health.csv"
df = pd.read_csv(data_path)

print(f"Loaded dataset with {len(df):,} rows")
print(f"Columns: {list(df.columns)}")

# Define variables and their display names
variables = [
    'screen_time_hours',
    'social_media_platforms_used',
    'hours_on_TikTok',
    'sleep_hours',
    'stress_level',
    'mood_score'
]

display_names = [
    'Screen Time (hours)',
    'Social Media Platforms Used',
    'Hours on TikTok',
    'Sleep Hours',
    'Stress Level',
    'Mood Score'
]

# Create figure with subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Color palette for consistency
colors = sns.color_palette("husl", len(variables))

# Plot histograms for each variable
for i, (var, display_name, color) in enumerate(zip(variables, display_names, colors)):
    ax = axes[i]
    
    # Get statistics first
    mean_val = df[var].mean()
    median_val = df[var].median()
    std_val = df[var].std()
    min_val = df[var].min()
    max_val = df[var].max()
    
    # Special handling for mood_score: include value 1 with 0 frequency
    if var == 'mood_score':
        # Create bins that explicitly include 1 (even though data starts at 2)
        # Use integer bins from 0.5 to max+0.5 with step 1, so each bin centers on an integer
        # This ensures bin for value 1 (0.5 to 1.5) is included and shows 0 frequency
        bin_edges = [x - 0.5 for x in range(1, int(max_val) + 2)]
        
        # Create histogram with these bin edges
        n, bins, patches = ax.hist(
            df[var],
            bins=bin_edges,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7,
            color=color,
            density=False
        )
        
        # Set x-axis to start at 1, ensuring label 1 is visible
        ax.set_xlim(0.5, max_val + 0.5)
        # Set integer ticks on x-axis (including 1)
        ax.set_xticks(range(1, int(max_val) + 1))
    elif var == 'stress_level':
        # Special handling for stress_level: no gaps between bars (like mood_score)
        # Use integer bins from 0.5 to max+0.5 with step 1, so each bin centers on an integer
        bin_edges = [x - 0.5 for x in range(1, int(max_val) + 2)]
        
        # Create histogram with these bin edges
        n, bins, patches = ax.hist(
            df[var],
            bins=bin_edges,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7,
            color=color,
            density=False
        )
        
        # Set x-axis ticks from 1 to 10 with interval of 1
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.5, max_val + 0.5)
    elif var == 'social_media_platforms_used':
        # Special handling for social_media_platforms_used: show x-axis ticks with interval of 1
        # Use integer bins from 0.5 to max+0.5 with step 1, so each bin centers on an integer
        bin_edges = [x - 0.5 for x in range(1, int(max_val) + 2)]
        
        # Create histogram with these bin edges
        n, bins, patches = ax.hist(
            df[var],
            bins=bin_edges,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7,
            color=color,
            density=False
        )
        
        # Set x-axis ticks from 1 to max with interval of 1
        ax.set_xticks(range(1, int(max_val) + 1))
        ax.set_xlim(0.5, max_val + 0.5)
    else:
        # Create histogram for other variables
        n, bins, patches = ax.hist(
            df[var],
            bins=50,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7,
            color=color,
            density=False
        )
    
    # Format title with statistics
    ax.set_title(
        f'{display_name}\n'
        f'μ={mean_val:.2f}, σ={std_val:.2f}, Range=[{min_val:.1f}, {max_val:.1f}]',
        fontsize=11,
        pad=10
    )
    
    ax.set_xlabel(display_name, fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Mean')
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Median')
    
    # Add legend for first subplot only
    if i == 0:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Add overall title
fig.suptitle(
    'Distribution of Digital Habit and Mental Health Variables\n'
    'in the Synthetic Dataset (N = 100,000)',
    fontsize=16,
    fontweight='bold',
    y=0.995
)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save figure
output_path = Path(__file__).parent / "figure1_distributions.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigure 1 saved to: {output_path}")

# Also save as PDF for paper submission
pdf_path = Path(__file__).parent / "figure1_distributions.pdf"
fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure 1 (PDF) saved to: {pdf_path}")

# Print summary statistics
print("\n" + "="*70)
print("Summary Statistics")
print("="*70)
print(df[variables].describe())
print("\n" + "="*70)
print("Variable Ranges (for plausibility check):")
print("="*70)
for var, display_name in zip(variables, display_names):
    min_val = df[var].min()
    max_val = df[var].max()
    print(f"{display_name:30s}: [{min_val:6.2f}, {max_val:6.2f}]")

print("\n✓ Figure 1 generation complete!")

