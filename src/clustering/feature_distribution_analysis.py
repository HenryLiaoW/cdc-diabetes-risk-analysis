"""
Feature Distribution Analysis for Scaling Strategy Selection

This script performs comprehensive distribution analysis on 6 composite features:
1. Visualizes distributions (KDE, Histogram, Boxplot)
2. Analyzes distribution characteristics (skewness, outliers, modality)
3. Recommends optimal scaling strategy based on distribution patterns
4. Compares different scalers' effects

Author: Feature Engineering Pipeline
Date: 2026-02-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

print("="*80)
print("FEATURE DISTRIBUTION ANALYSIS - SCALING STRATEGY SELECTION")
print("="*80)

# ============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================================
print("\n[Step 1] Loading data and creating composite features...")
df = pd.read_csv('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/data/processed/CDC_Diabetes_Cleaned.csv')

# Create 6 composite features
df['chronic_disease_burden'] = (df['HighBP'] + df['HighChol'] + df['Stroke'] + 
                                  df['HeartDiseaseorAttack'] + df['DiffWalk'])
df['metabolic_baseline'] = df['BMI']
df['healthy_lifestyle_score'] = df['Veggies'] + df['Fruits'] + df['PhysActivity']
df['substance_risk'] = df['Smoker'] + df['HvyAlcoholConsump']
df['ses_vulnerability'] = (10 - df['Income']) + (7 - df['Education'])
df['medical_engagement'] = df['CholCheck']

feature_names = [
    'metabolic_baseline',
    'chronic_disease_burden',
    'healthy_lifestyle_score',
    'substance_risk',
    'ses_vulnerability',
    'medical_engagement'
]

feature_labels = [
    'Metabolic Baseline (BMI)',
    'Chronic Disease Burden (0-5)',
    'Healthy Lifestyle Score (0-3)',
    'Substance Risk (0-2)',
    'SES Vulnerability (0-17)',
    'Medical Engagement (0-1)'
]

X = df[feature_names]
print(f"✓ Dataset shape: {X.shape}")
print(f"✓ Created {len(feature_names)} composite features")

# ============================================================================
# 2. DISTRIBUTION CHARACTERISTICS ANALYSIS
# ============================================================================
print("\n[Step 2] Analyzing distribution characteristics...")

distribution_stats = []
for feat, label in zip(feature_names, feature_labels):
    data = X[feat].dropna()
    
    # Basic statistics
    mean = data.mean()
    median = data.median()
    std = data.std()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    
    # Skewness and Kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    # Outlier detection (IQR method)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outlier_pct = len(outliers) / len(data) * 100
    
    # Modality check (simple peaks in histogram)
    hist, bins = np.histogram(data, bins=50)
    # Count peaks (local maxima)
    peaks = 0
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > len(data)*0.02:
            peaks += 1
    
    # Determine distribution type
    if abs(skewness) > 1.0:
        dist_type = "Highly Skewed"
    elif abs(skewness) > 0.5:
        dist_type = "Moderately Skewed"
    else:
        dist_type = "Symmetric"
    
    if outlier_pct > 5:
        dist_type += " + Heavy Outliers"
    elif outlier_pct > 1:
        dist_type += " + Mild Outliers"
    
    if peaks >= 3:
        dist_type += " + Multimodal"
    
    # Sparsity check
    zero_pct = (data == 0).sum() / len(data) * 100
    if zero_pct > 50:
        dist_type += " + Sparse"
    
    distribution_stats.append({
        'Feature': label,
        'Mean': mean,
        'Median': median,
        'Std': std,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Outlier_Pct': outlier_pct,
        'Zero_Pct': zero_pct,
        'Dist_Type': dist_type
    })

stats_df = pd.DataFrame(distribution_stats)
print("\n" + "="*80)
print("DISTRIBUTION STATISTICS")
print("="*80)
print(stats_df.to_string(index=False))

# ============================================================================
# 3. SCALING STRATEGY RECOMMENDATION
# ============================================================================
print("\n[Step 3] Recommending scaling strategies based on distribution...")

scaling_recommendations = []
for feat, label in zip(feature_names, feature_labels):
    data = X[feat].dropna()
    skewness = stats.skew(data)
    outlier_pct = len(data[(data < data.quantile(0.25) - 1.5*(data.quantile(0.75)-data.quantile(0.25))) | 
                           (data > data.quantile(0.75) + 1.5*(data.quantile(0.75)-data.quantile(0.25)))]) / len(data) * 100
    zero_pct = (data == 0).sum() / len(data) * 100
    
    # Decision logic
    if abs(skewness) > 1.0 or outlier_pct > 5:
        recommended = "RobustScaler"
        reason = f"Long-tail/Skewed distribution (skew={skewness:.2f}, outliers={outlier_pct:.1f}%)"
    elif zero_pct > 50:
        recommended = "MinMaxScaler"
        reason = f"Sparse/Binary-like distribution (zeros={zero_pct:.1f}%)"
    else:
        recommended = "StandardScaler"
        reason = f"Relatively symmetric distribution (skew={skewness:.2f})"
    
    scaling_recommendations.append({
        'Feature': label,
        'Recommended_Scaler': recommended,
        'Reason': reason
    })

rec_df = pd.DataFrame(scaling_recommendations)
print("\n" + "="*80)
print("SCALING STRATEGY RECOMMENDATIONS")
print("="*80)
print(rec_df.to_string(index=False))

# ============================================================================
# 4. VISUALIZATION - DISTRIBUTION PLOTS
# ============================================================================
print("\n[Step 4] Creating distribution visualizations...")

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/distribution_analysis')
fig_dir = base_dir / timestamp
fig_dir.mkdir(parents=True, exist_ok=True)

print(f"✓ Output directory: {fig_dir}")

# Plot 1: Combined KDE Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (feat, label) in enumerate(zip(feature_names, feature_labels)):
    ax = axes[idx]
    data = X[feat].dropna()
    
    # KDE plot
    data.plot(kind='density', ax=ax, linewidth=2, color='steelblue')
    ax.fill_between(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(), alpha=0.3, color='steelblue')
    
    # Add mean and median lines
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median={median_val:.2f}')
    
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Distribution Analysis: KDE Plots for 6 Composite Features\n(Observe Shape: Symmetric vs Skewed vs Multimodal)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(fig_dir / 'feature_kde_plots.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: feature_kde_plots.png")
plt.close()

# Plot 2: Combined Histogram + Boxplot
fig, axes = plt.subplots(6, 2, figsize=(14, 20))

for idx, (feat, label) in enumerate(zip(feature_names, feature_labels)):
    data = X[feat].dropna()
    
    # Histogram
    ax_hist = axes[idx, 0]
    ax_hist.hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax_hist.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax_hist.axvline(data.median(), color='green', linestyle='--', linewidth=2, label='Median')
    ax_hist.set_title(f'{label} - Histogram', fontsize=11, fontweight='bold')
    ax_hist.set_xlabel('Value', fontsize=9)
    ax_hist.set_ylabel('Frequency', fontsize=9)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)
    
    # Add skewness annotation
    skewness = stats.skew(data)
    ax_hist.text(0.98, 0.98, f'Skewness: {skewness:.2f}', 
                transform=ax_hist.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Boxplot
    ax_box = axes[idx, 1]
    bp = ax_box.boxplot(data, vert=False, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    ax_box.set_title(f'{label} - Boxplot', fontsize=11, fontweight='bold')
    ax_box.set_xlabel('Value', fontsize=9)
    ax_box.grid(True, alpha=0.3)
    
    # Add outlier count
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
    outlier_pct = len(outliers) / len(data) * 100
    ax_box.text(0.02, 0.98, f'Outliers: {outlier_pct:.1f}%', 
                transform=ax_box.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

plt.suptitle('Distribution Analysis: Histograms & Boxplots for 6 Composite Features\n(Identify Outliers and Skewness)', 
             fontsize=14, fontweight='bold', y=0.9995)
plt.tight_layout()
plt.savefig(fig_dir / 'feature_histograms_boxplots.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: feature_histograms_boxplots.png")
plt.close()

# ============================================================================
# 5. SCALER COMPARISON VISUALIZATION
# ============================================================================
print("\n[Step 5] Comparing different scalers' effects...")

fig, axes = plt.subplots(6, 4, figsize=(20, 24))

scalers = {
    'Original': None,
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'MinMaxScaler': MinMaxScaler()
}

for row_idx, (feat, label) in enumerate(zip(feature_names, feature_labels)):
    data = X[[feat]].dropna()
    
    for col_idx, (scaler_name, scaler) in enumerate(scalers.items()):
        ax = axes[row_idx, col_idx]
        
        if scaler is None:
            scaled_data = data[feat]
        else:
            scaled_data = scaler.fit_transform(data)[:, 0]
        
        # KDE plot
        ax.hist(scaled_data, bins=50, color='steelblue', edgecolor='black', alpha=0.7, density=True)
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(scaled_data)
            x_range = np.linspace(scaled_data.min(), scaled_data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
        except:
            pass
        
        if col_idx == 0:
            ax.set_ylabel(label, fontsize=10, fontweight='bold')
        if row_idx == 0:
            ax.set_title(scaler_name, fontsize=11, fontweight='bold')
        
        # Add statistics
        mean_val = scaled_data.mean()
        std_val = scaled_data.std()
        ax.text(0.98, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

plt.suptitle('Scaler Comparison: Distribution Changes for 6 Composite Features\n(Compare Original vs StandardScaler vs RobustScaler vs MinMaxScaler)', 
             fontsize=14, fontweight='bold', y=0.9995)
plt.tight_layout()
plt.savefig(fig_dir / 'scaler_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: scaler_comparison.png")
plt.close()

# ============================================================================
# 6. SAVE ANALYSIS RESULTS
# ============================================================================
print("\n[Step 6] Saving analysis results...")

# Save statistics
stats_df.to_csv(fig_dir / 'distribution_statistics.csv', index=False)
print(f"✓ Saved: distribution_statistics.csv")

# Save recommendations
rec_df.to_csv(fig_dir / 'scaling_recommendations.csv', index=False)
print(f"✓ Saved: scaling_recommendations.csv")

# Save detailed report
report_path = fig_dir / 'distribution_analysis_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("FEATURE DISTRIBUTION ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset Size: {len(df):,} records\n")
    f.write(f"Number of Features: {len(feature_names)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("DISTRIBUTION CHARACTERISTICS\n")
    f.write("="*80 + "\n\n")
    f.write(stats_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("SCALING STRATEGY RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    f.write(rec_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("METHODOLOGY EXPLANATION\n")
    f.write("="*80 + "\n\n")
    f.write("Distribution Classification Criteria:\n\n")
    f.write("1. SKEWNESS ASSESSMENT:\n")
    f.write("   • |Skewness| > 1.0 → Highly Skewed (long tail)\n")
    f.write("   • 0.5 < |Skewness| ≤ 1.0 → Moderately Skewed\n")
    f.write("   • |Skewness| ≤ 0.5 → Symmetric\n\n")
    f.write("2. OUTLIER DETECTION (IQR Method):\n")
    f.write("   • Outlier% > 5% → Heavy Outliers\n")
    f.write("   • 1% < Outlier% ≤ 5% → Mild Outliers\n")
    f.write("   • Outlier% ≤ 1% → Few/No Outliers\n\n")
    f.write("3. SPARSITY CHECK:\n")
    f.write("   • Zero% > 50% → Sparse/Binary-like\n")
    f.write("   • Zero% ≤ 50% → Dense\n\n")
    f.write("="*80 + "\n")
    f.write("SCALING STRATEGY DECISION RULES\n")
    f.write("="*80 + "\n\n")
    f.write("RobustScaler:\n")
    f.write("  • When: Long-tail/Skewed distributions (|skewness| > 1.0) OR Heavy outliers (>5%)\n")
    f.write("  • Why: Uses median and IQR, robust to outliers\n")
    f.write("  • Formula: (X - median) / IQR\n")
    f.write("  • Use Case: BMI with extreme obesity outliers\n\n")
    f.write("StandardScaler:\n")
    f.write("  • When: Relatively symmetric distributions with few outliers\n")
    f.write("  • Why: Centers data at mean=0, std=1, assumes Gaussian-like\n")
    f.write("  • Formula: (X - mean) / std\n")
    f.write("  • Use Case: SES Vulnerability, Healthy Lifestyle Score\n\n")
    f.write("MinMaxScaler:\n")
    f.write("  • When: Sparse/Binary-like distributions (zeros > 50%)\n")
    f.write("  • Why: Preserves zero values, compresses to [0, 1] range\n")
    f.write("  • Formula: (X - min) / (max - min)\n")
    f.write("  • Use Case: Substance Risk (most values are 0)\n\n")
    f.write("="*80 + "\n")
    f.write("FINAL MIXED SCALING CONFIGURATION\n")
    f.write("="*80 + "\n\n")
    
    # Generate final configuration
    for feat, label in zip(feature_names, feature_labels):
        rec = rec_df[rec_df['Feature'] == label]['Recommended_Scaler'].values[0]
        reason = rec_df[rec_df['Feature'] == label]['Reason'].values[0]
        f.write(f"{feat}:\n")
        f.write(f"  Scaler: {rec}\n")
        f.write(f"  Reason: {reason}\n\n")

print(f"✓ Saved: distribution_analysis_report.txt")

print("\n" + "="*80)
print("DISTRIBUTION ANALYSIS COMPLETE")
print("="*80)
print(f"\n✅ All outputs saved to: {fig_dir}")
print(f"\nGenerated files:")
print(f"  1. feature_kde_plots.png - KDE density curves")
print(f"  2. feature_histograms_boxplots.png - Histograms & boxplots")
print(f"  3. scaler_comparison.png - Compare 4 scaling methods")
print(f"  4. distribution_statistics.csv - Numerical statistics")
print(f"  5. scaling_recommendations.csv - Recommended scalers")
print(f"  6. distribution_analysis_report.txt - Detailed report")
print("\n" + "="*80)
print("\n📊 KEY INSIGHT: Check the visualizations to verify:")
print("  • Long-tail features → RobustScaler")
print("  • Symmetric features → StandardScaler")
print("  • Sparse features → MinMaxScaler")
print("="*80)
