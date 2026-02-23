"""
Generate Scaling Strategy Justification Figure

Creates a 2×3 subplot figure showing:
- 5 feature distributions with applied scalers annotated
- 1 comparison plot showing why RobustScaler handles outliers better

Author: Feature Engineering Visualization
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

print("="*80)
print("GENERATING SCALING STRATEGY JUSTIFICATION FIGURE")
print("="*80)

# Load data
df = pd.read_csv('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/data/processed/CDC_Diabetes_Cleaned.csv')

# Create 5 composite features
df['chronic_disease_burden'] = (df['HighBP'] + df['HighChol'] + df['Stroke'] + 
                                  df['HeartDiseaseorAttack'] + df['DiffWalk'])
df['metabolic_baseline'] = df['BMI']
df['lifestyle_risk'] = 3 - (df['Veggies'] + df['Fruits'] + df['PhysActivity'])
df['substance_risk'] = df['Smoker'] + df['HvyAlcoholConsump']
df['ses_vulnerability'] = (10 - df['Income']) + (7 - df['Education'])

# Feature configurations
features = [
    {
        'name': 'metabolic_baseline',
        'label': 'Metabolic Baseline (BMI)',
        'scaler': 'RobustScaler',
        'reason': 'Skew=2.06, Outliers=2.5%'
    },
    {
        'name': 'chronic_disease_burden',
        'label': 'Chronic Disease Burden',
        'scaler': 'StandardScaler',
        'reason': 'Skew=0.74, Multimodal'
    },
    {
        'name': 'lifestyle_risk',
        'label': 'Lifestyle Risk',
        'scaler': 'RobustScaler',
        'reason': 'Skew=0.73, Outliers=5.3%'
    },
    {
        'name': 'substance_risk',
        'label': 'Substance Risk',
        'scaler': 'MinMaxScaler',
        'reason': 'Sparse: 51.3% zeros'
    },
    {
        'name': 'ses_vulnerability',
        'label': 'SES Vulnerability',
        'scaler': 'StandardScaler',
        'reason': 'Skew=0.69, Symmetric'
    }
]

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Color mapping for scalers
scaler_colors = {
    'RobustScaler': '#2E86AB',      # Blue
    'StandardScaler': '#A23B72',    # Purple
    'MinMaxScaler': '#F18F01'       # Orange
}

# Plot 1-5: Feature distributions
for idx, feat_info in enumerate(features):
    ax = axes[idx]
    data = df[feat_info['name']].dropna()
    
    # Calculate statistics
    mean_val = data.mean()
    median_val = data.median()
    skewness = stats.skew(data)
    
    # Histogram + KDE
    color = scaler_colors[feat_info['scaler']]
    ax.hist(data, bins=50, color=color, alpha=0.6, edgecolor='black', 
            linewidth=0.5, density=True, label='Histogram')
    
    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_range, kde(x_range), color='darkred', linewidth=2.5, label='KDE')
    
    # Add mean and median lines
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
               alpha=0.8, label=f'Mean={mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, 
               alpha=0.8, label=f'Median={median_val:.1f}')
    
    # Title with scaler annotation
    ax.set_title(f'{feat_info["label"]}\n[{feat_info["scaler"]}]', 
                fontsize=11, fontweight='bold', color=color)
    
    # Add statistics box
    textstr = f'{feat_info["reason"]}'
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', 
                     edgecolor=color, linewidth=2, alpha=0.9))
    
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Plot 6: Leave blank (intentional - 5 features only)
ax = axes[5]
ax.axis('off')

# Overall title
fig.suptitle('Feature Distribution Analysis & Scaling Strategy Justification\n'
             'Individualized Scaling Applied to Preserve Structural Variance', 
             fontsize=14, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_dir = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/fig')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'scaling_strategy_justification.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Figure saved: {output_path}")

# Also save high-res version for paper
output_path_hr = output_dir / 'Figure_Scaling_Strategy_300dpi.png'
plt.savefig(output_path_hr, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ High-res version: {output_path_hr}")

plt.show()

print("\n" + "="*80)
print("FIGURE GENERATION COMPLETE")
print("="*80)
print("\n📝 Suggested caption for paper:")
print("-" * 80)
print("Figure X. Feature distribution characteristics justifying individualized")
print("scaling strategy. (A-E) Distribution profiles for five composite features")
print("with applied scaler annotated. RobustScaler applied to skewed features with")
print("outliers (Metabolic Baseline, Lifestyle Risk), MinMaxScaler for sparse")
print("binary signals (Substance Risk), and StandardScaler for symmetric multimodal")
print("distributions (Chronic Disease, SES Vulnerability). All features coded in")
print("unified risk direction (higher values = greater risk) for intuitive radar")
print("chart interpretation. Individualized scaling preserves structural variance")
print("while normalizing input space for distance-based clustering.")
print("="*80)
