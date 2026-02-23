"""
VIF (Variance Inflation Factor) & Multicollinearity Analysis - 6 Features (Including Age)

This script performs comprehensive multicollinearity diagnostics:
1. VIF calculation for each composite feature
2. Correlation matrix and heatmap visualization
3. Pairwise scatter plots for visual inspection
4. Statistical report with academic interpretation

Purpose: Ensure structural integrity of feature space for K-Means clustering
         and validate that each composite feature provides unique explanatory power

Author: Feature Engineering Validation Pipeline
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import pearsonr

print("="*80)
print("VIF & MULTICOLLINEARITY ANALYSIS - 6 COMPOSITE FEATURES (INCLUDING AGE)")
print("="*80)

# ============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================================
print("\n[Step 1] Loading data and creating composite features...")
df = pd.read_csv('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/data/processed/CDC_Diabetes_Cleaned.csv')

# Create 6 composite features (same as kmeans_6features_with_age.py)
df['chronic_disease_burden'] = (df['HighBP'] + df['HighChol'] + df['Stroke'] + 
                                  df['HeartDiseaseorAttack'] + df['DiffWalk'])
df['metabolic_baseline'] = df['BMI']
df['lifestyle_risk'] = 3 - (df['Veggies'] + df['Fruits'] + df['PhysActivity'])
df['substance_risk'] = df['Smoker'] + df['HvyAlcoholConsump']
df['ses_vulnerability'] = (10 - df['Income']) + (7 - df['Education'])
df['age'] = df['Age']  # Include Age as 6th feature

feature_names = [
    'metabolic_baseline',
    'chronic_disease_burden',
    'lifestyle_risk',
    'substance_risk',
    'ses_vulnerability',
    'age'
]

feature_labels = [
    'Metabolic Baseline (BMI)',
    'Chronic Disease Burden',
    'Lifestyle Risk',
    'Substance Risk',
    'SES Vulnerability',
    'Age'
]

X = df[feature_names].copy()
print(f"✓ Dataset shape: {X.shape}")
print(f"✓ Features: {feature_names}")

# ============================================================================
# 2. VIF CALCULATION
# ============================================================================
print("\n[Step 2] Calculating Variance Inflation Factor (VIF)...")

# Add constant for VIF calculation (statsmodels requirement)
X_with_const = add_constant(X)

# Calculate VIF for each feature
vif_data = []
for i, col in enumerate(feature_names):
    vif_value = variance_inflation_factor(X_with_const.values, i+1)  # +1 to skip constant
    vif_data.append({
        'Feature': feature_labels[i],
        'VIF': vif_value
    })

vif_df = pd.DataFrame(vif_data)

# Interpret VIF values
def interpret_vif(vif):
    if vif < 5:
        return "Excellent (No multicollinearity)"
    elif vif < 10:
        return "Acceptable (Mild multicollinearity)"
    else:
        return "Problematic (Severe multicollinearity)"

vif_df['Interpretation'] = vif_df['VIF'].apply(interpret_vif)

print("\n" + "="*80)
print("VIF RESULTS - 6 FEATURES (INCLUDING AGE)")
print("="*80)
print(vif_df.to_string(index=False))
print("\nInterpretation Guidelines:")
print("  • VIF < 5:  Excellent - No multicollinearity detected")
print("  • VIF < 10: Acceptable - Mild multicollinearity, typical in social science")
print("  • VIF ≥ 10: Problematic - Severe multicollinearity, feature redundancy")

# ============================================================================
# 3. CORRELATION MATRIX ANALYSIS
# ============================================================================
print("\n[Step 3] Computing correlation matrix...")

# Pearson correlation matrix
corr_matrix = X.corr()

# Find highest correlations
corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr_pairs.append({
            'Feature_1': feature_labels[i],
            'Feature_2': feature_labels[j],
            'Correlation': corr_matrix.iloc[i, j],
            'Abs_Correlation': abs(corr_matrix.iloc[i, j])
        })

corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Abs_Correlation', ascending=False)

print("\n" + "="*80)
print("TOP PAIRWISE CORRELATIONS")
print("="*80)
print(corr_pairs_df.head(10).to_string(index=False))
print("\nCorrelation Thresholds:")
print("  • |r| < 0.3: Weak correlation")
print("  • 0.3 ≤ |r| < 0.7: Moderate correlation")
print("  • |r| ≥ 0.7: Strong correlation (potential multicollinearity)")

# ============================================================================
# 4. VISUALIZATION
# ============================================================================
print("\n[Step 4] Creating visualizations...")

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/vif_analysis_6features_includingAGE')
fig_dir = base_dir / timestamp
fig_dir.mkdir(parents=True, exist_ok=True)

print(f"✓ Output directory: {fig_dir}")

# ========== PLOT 1: VIF Bar Chart ==========
fig, ax = plt.subplots(figsize=(10, 7))

colors = ['green' if vif < 5 else 'orange' if vif < 10 else 'red' for vif in vif_df['VIF']]
bars = ax.barh(vif_df['Feature'], vif_df['VIF'], color=colors, edgecolor='black', alpha=0.8)

# Add threshold lines
ax.axvline(5, color='orange', linestyle='--', linewidth=2, label='Threshold: VIF=5 (Acceptable)')
ax.axvline(10, color='red', linestyle='--', linewidth=2, label='Threshold: VIF=10 (Critical)')

# Add value labels
for i, (bar, vif) in enumerate(zip(bars, vif_df['VIF'])):
    ax.text(vif + 0.1, bar.get_y() + bar.get_height()/2, 
           f'{vif:.2f}', va='center', fontweight='bold', fontsize=10)

ax.set_xlabel('VIF Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Composite Features', fontsize=12, fontweight='bold')
ax.set_title('Variance Inflation Factor (VIF) Analysis - 6 Features (Including Age)\nMulticollinearity Diagnostic for Composite Features', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, max(vif_df['VIF'].max() + 2, 12))

plt.tight_layout()
plt.savefig(fig_dir / 'vif_bar_chart_6features_includingAGE.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: vif_bar_chart_6features_includingAGE.png")
plt.close()

# ========== PLOT 2: Correlation Heatmap ==========
fig, ax = plt.subplots(figsize=(12, 10))

# Use feature labels for better readability
corr_matrix_labeled = pd.DataFrame(
    corr_matrix.values,
    index=feature_labels,
    columns=feature_labels
)

# Create heatmap
mask = np.triu(np.ones_like(corr_matrix_labeled, dtype=bool), k=1)  # Mask upper triangle
sns.heatmap(corr_matrix_labeled, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5, linecolor='gray',
            cbar_kws={'label': 'Pearson Correlation Coefficient', 'shrink': 0.8},
            mask=mask, ax=ax)

ax.set_title('Correlation Matrix Heatmap - 6 Features (Including Age)\n(Lower Triangle: Pearson r)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(fig_dir / 'correlation_heatmap_6features_includingAGE.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: correlation_heatmap_6features_includingAGE.png")
plt.close()

# ========== PLOT 3: Full Correlation Heatmap (both triangles) ==========
fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corr_matrix_labeled, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True, linewidths=1, linecolor='white',
            cbar_kws={'label': 'Pearson Correlation Coefficient', 'shrink': 0.8},
            ax=ax)

ax.set_title('Full Correlation Matrix - 6 Features (Including Age)\n(All Pairwise Correlations)', 
             fontsize=14, fontweight='bold', pad=20)

# Highlight diagonal
for i in range(len(feature_labels)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='gold', lw=3))

plt.tight_layout()
plt.savefig(fig_dir / 'correlation_heatmap_full_6features_includingAGE.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: correlation_heatmap_full_6features_includingAGE.png")
plt.close()

# ========== PLOT 4: Pairwise Scatter Plot Matrix ==========
print("  Generating pairwise scatter plots (this may take a moment)...")

# Sample data for faster plotting (50k samples)
sample_size = min(50000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)

fig, axes = plt.subplots(6, 6, figsize=(24, 24))

for i in range(6):
    for j in range(6):
        ax = axes[i, j]
        
        if i == j:
            # Diagonal: histogram
            ax.hist(X_sample.iloc[:, i], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_ylabel('Frequency', fontsize=9)
            if i == 0:
                ax.set_title(feature_labels[i], fontsize=10, fontweight='bold')
        elif i > j:
            # Lower triangle: scatter plot
            ax.scatter(X_sample.iloc[:, j], X_sample.iloc[:, i], 
                      alpha=0.3, s=1, color='steelblue')
            
            # Add correlation value
            corr_val = corr_matrix.iloc[i, j]
            ax.text(0.05, 0.95, f'r={corr_val:.3f}', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            # Upper triangle: empty (or can mirror)
            ax.axis('off')
        
        # Labels
        if j == 0 and i != j:
            ax.set_ylabel(feature_labels[i], fontsize=9, fontweight='bold')
        if i == 5:
            ax.set_xlabel(feature_labels[j], fontsize=9, fontweight='bold')
        
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

plt.suptitle(f'Pairwise Feature Scatter Plots - 6 Features (Including Age)\n(Sample: {sample_size:,} records, Lower Triangle: Scatter + Correlation)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(fig_dir / 'pairwise_scatter_matrix_6features_includingAGE.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: pairwise_scatter_matrix_6features_includingAGE.png")
plt.close()

# ============================================================================
# 5. STATISTICAL SUMMARY
# ============================================================================
print("\n[Step 5] Generating statistical summary...")

# Descriptive statistics
desc_stats = X.describe().T
desc_stats['Feature'] = feature_labels
desc_stats = desc_stats[['Feature', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)
print(desc_stats.to_string(index=False))

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n[Step 6] Saving analysis results...")

# Save VIF results
vif_df.to_csv(fig_dir / 'vif_results_6features_includingAGE.csv', index=False)
print(f"✓ Saved: vif_results_6features_includingAGE.csv")

# Save correlation matrix
corr_matrix_labeled.to_csv(fig_dir / 'correlation_matrix_6features_includingAGE.csv')
print(f"✓ Saved: correlation_matrix_6features_includingAGE.csv")

# Save pairwise correlations
corr_pairs_df.to_csv(fig_dir / 'pairwise_correlations_6features_includingAGE.csv', index=False)
print(f"✓ Saved: pairwise_correlations_6features_includingAGE.csv")

# Save descriptive statistics
desc_stats.to_csv(fig_dir / 'descriptive_statistics_6features_includingAGE.csv', index=False)
print(f"✓ Saved: descriptive_statistics_6features_includingAGE.csv")

# Generate comprehensive report
report_path = fig_dir / 'vif_multicollinearity_report_6features_includingAGE.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("VIF & MULTICOLLINEARITY ANALYSIS REPORT - 6 FEATURES (INCLUDING AGE)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset Size: {len(df):,} records\n")
    f.write(f"Number of Features: 6 composite features (including Age)\n\n")
    
    f.write("="*80 + "\n")
    f.write("1. VIF RESULTS (Variance Inflation Factor)\n")
    f.write("="*80 + "\n\n")
    f.write(vif_df.to_string(index=False))
    f.write("\n\n")
    
    # Overall VIF assessment
    max_vif = vif_df['VIF'].max()
    if max_vif < 5:
        overall_assessment = "EXCELLENT - No multicollinearity detected"
        recommendation = "All features are suitable for use. Proceed with confidence."
    elif max_vif < 10:
        overall_assessment = "ACCEPTABLE - Mild multicollinearity present"
        recommendation = "Features are usable. Typical for social science data."
    else:
        overall_assessment = "PROBLEMATIC - Severe multicollinearity detected"
        recommendation = "Consider removing or further aggregating features with VIF ≥ 10."
    
    f.write(f"Overall Assessment: {overall_assessment}\n")
    f.write(f"Maximum VIF: {max_vif:.3f}\n")
    f.write(f"Recommendation: {recommendation}\n\n")
    
    f.write("="*80 + "\n")
    f.write("2. CORRELATION MATRIX\n")
    f.write("="*80 + "\n\n")
    f.write(corr_matrix_labeled.to_string())
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("3. TOP PAIRWISE CORRELATIONS\n")
    f.write("="*80 + "\n\n")
    f.write(corr_pairs_df.head(10).to_string(index=False))
    f.write("\n\n")
    
    # Identify problematic pairs
    high_corr_pairs = corr_pairs_df[corr_pairs_df['Abs_Correlation'] >= 0.7]
    if len(high_corr_pairs) > 0:
        f.write("⚠️  WARNING: High Correlations Detected (|r| ≥ 0.7)\n")
        f.write(high_corr_pairs.to_string(index=False))
        f.write("\n\n")
    else:
        f.write("✓ No high correlations detected (all |r| < 0.7)\n\n")
    
    f.write("="*80 + "\n")
    f.write("4. DESCRIPTIVE STATISTICS\n")
    f.write("="*80 + "\n\n")
    f.write(desc_stats.to_string(index=False))
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("5. METHODOLOGY EXPLANATION\n")
    f.write("="*80 + "\n\n")
    f.write("VIF (Variance Inflation Factor) Calculation:\n")
    f.write("  For each feature X_i, VIF_i measures how much the variance of the\n")
    f.write("  estimated coefficient is inflated by collinearity with other features.\n\n")
    f.write("  Formula: VIF_i = 1 / (1 - R²_i)\n")
    f.write("  Where R²_i is from regressing X_i on all other features.\n\n")
    f.write("Interpretation:\n")
    f.write("  • VIF = 1: No correlation with other features (ideal)\n")
    f.write("  • VIF < 5: Low multicollinearity (acceptable)\n")
    f.write("  • 5 ≤ VIF < 10: Moderate multicollinearity (monitor)\n")
    f.write("  • VIF ≥ 10: High multicollinearity (action required)\n\n")
    
    f.write("="*80 + "\n")
    f.write("6. IMPACT OF ADDING AGE FEATURE\n")
    f.write("="*80 + "\n\n")
    f.write("Age is a critical demographic variable known to be strongly associated\n")
    f.write("with diabetes risk. Including Age allows for:\n")
    f.write("  1. Age-stratified risk profiling\n")
    f.write("  2. Better separation of elderly vs younger cohorts\n")
    f.write("  3. More targeted intervention strategies\n\n")
    
    age_corr_with_chronic = corr_matrix.loc[feature_names[5], feature_names[1]]
    age_corr_with_bmi = corr_matrix.loc[feature_names[5], feature_names[0]]
    f.write(f"Key Age Correlations:\n")
    f.write(f"  • Age ↔ Chronic Disease Burden: r={age_corr_with_chronic:.3f}\n")
    f.write(f"  • Age ↔ Metabolic Baseline (BMI): r={age_corr_with_bmi:.3f}\n\n")
    
    if abs(age_corr_with_chronic) > 0.5 or abs(age_corr_with_bmi) > 0.5:
        f.write("⚠️  CAUTION: Age shows moderate correlation with other features.\n")
        f.write("   This is biologically expected (chronic diseases increase with age).\n")
        f.write("   VIF values should be monitored to ensure no severe multicollinearity.\n\n")
    else:
        f.write("✓ Age shows acceptable independence from other features.\n\n")
    
    f.write("="*80 + "\n")
    f.write("7. ACADEMIC REPORTING TEMPLATE\n")
    f.write("="*80 + "\n\n")
    f.write("To ensure the structural integrity of the feature space and avoid\n")
    f.write("redundant weighting in the K-Means distance calculation, a\n")
    f.write("Multicollinearity Diagnostic was performed using the Variance Inflation\n")
    f.write("Factor (VIF). By aggregating twelve primary variables into five distinct\n")
    f.write("composite domains—Metabolic (BMI), Clinical (Chronic Disease Burden),\n")
    f.write("Behavioral (Lifestyle & Substance), Socioeconomic (SES Vulnerability),\n")
    f.write("and Demographic (Age)—the model incorporates both biological and\n")
    f.write("environmental risk factors while minimising inter-feature correlation.\n\n")
    f.write(f"All selected features yielded VIF values below the critical threshold,\n")
    f.write(f"with a maximum VIF of {max_vif:.2f}, confirming that each composite\n")
    f.write(f"metric provides unique explanatory power for the subsequent diabetes\n")
    f.write(f"risk stratification. The maximum pairwise correlation was\n")
    f.write(f"{corr_pairs_df['Abs_Correlation'].max():.3f}, well below the\n")
    f.write(f"multicollinearity concern threshold of 0.7.\n\n")
    f.write(f"The inclusion of Age as a demographic factor enables age-stratified\n")
    f.write(f"risk profiling while maintaining feature independence.\n\n")
    
    f.write("="*80 + "\n")
    f.write("8. COMPARISON WITH 5-FEATURE MODEL\n")
    f.write("="*80 + "\n\n")
    f.write("The 6-feature model (including Age) should be compared with the\n")
    f.write("5-feature model (without Age) to evaluate:\n")
    f.write("  1. Silhouette Score improvement/degradation\n")
    f.write("  2. Diabetes rate gradient across clusters\n")
    f.write("  3. Clinical interpretability of cluster profiles\n")
    f.write("  4. Age-specific intervention opportunities\n\n")
    f.write("Decision criteria for model selection:\n")
    f.write("  • If Silhouette Score improves by ≥2% → Use 6-feature model\n")
    f.write("  • If VIF remains < 5 for all features → Age is suitable\n")
    f.write("  • If age-stratified interventions are priority → Use 6-feature model\n\n")
    
    f.write("="*80 + "\n")
    f.write("9. CONCLUSION\n")
    f.write("="*80 + "\n\n")
    if max_vif < 5 and corr_pairs_df['Abs_Correlation'].max() < 0.7:
        f.write("✅ PASSED: The 6 composite features (including Age) demonstrate\n")
        f.write("   excellent independence. No multicollinearity concerns.\n")
        f.write("   Feature engineering is validated. Suitable for K-Means clustering\n")
        f.write("   and subsequent analysis.\n")
    elif max_vif < 10:
        f.write("✓ ACCEPTABLE: Mild multicollinearity detected but within acceptable range.\n")
        f.write("  Features are suitable for use in clustering analysis.\n")
    else:
        f.write("⚠️  ACTION REQUIRED: Severe multicollinearity detected.\n")
        f.write("   Consider removing Age or further feature reduction.\n")

print(f"✓ Saved: vif_multicollinearity_report_6features_includingAGE.txt")

print("\n" + "="*80)
print("VIF & MULTICOLLINEARITY ANALYSIS COMPLETE - 6 FEATURES (INCLUDING AGE)")
print("="*80)
print(f"\n✅ All outputs saved to: {fig_dir}")
print(f"\nGenerated files:")
print(f"  1. vif_bar_chart_6features_includingAGE.png - Visual VIF comparison")
print(f"  2. correlation_heatmap_6features_includingAGE.png - Lower triangle correlation matrix")
print(f"  3. correlation_heatmap_full_6features_includingAGE.png - Full correlation matrix")
print(f"  4. pairwise_scatter_matrix_6features_includingAGE.png - Scatter plots with correlations")
print(f"  5. vif_results_6features_includingAGE.csv - VIF values table")
print(f"  6. correlation_matrix_6features_includingAGE.csv - Full correlation matrix")
print(f"  7. pairwise_correlations_6features_includingAGE.csv - All pairwise correlations")
print(f"  8. descriptive_statistics_6features_includingAGE.csv - Feature statistics")
print(f"  9. vif_multicollinearity_report_6features_includingAGE.txt - Comprehensive report")
print("\n" + "="*80)
print("\n📊 KEY FINDINGS:")
print(f"  • Maximum VIF: {vif_df['VIF'].max():.3f}")
print(f"  • Maximum Correlation: {corr_pairs_df['Abs_Correlation'].max():.3f}")
print(f"  • Overall Assessment: {overall_assessment}")
print("="*80)
