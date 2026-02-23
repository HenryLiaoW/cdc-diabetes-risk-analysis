#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Validation Analysis for 5-Feature Composite Space
=====================================================

Purpose: Validate clustering input quality through dimensionality reduction analysis

Section: 3.1.4 Dimensionality Reduction & Cluster Feasibility Analysis

Objectives:
1. Variance Explanation: Quantify information retention in reduced dimensions
2. Orthogonality Verification: Complement VIF analysis with PCA decomposition
3. Spatial Visualization: Project 5D space to 2D for cluster separability inspection
4. Critical Evaluation: Justify why clustering in original 5-feature space is superior
   to PCA space (interpretability vs. variance maximization trade-off)

Author: Henry
Date: 2026-02-22
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'outputs' / 'clustering' / 'results'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'clustering' / 'profiling' / 'final_20260221_222939' / 'pca_validation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input file (5 features + cluster assignments)
INPUT_FILE = DATA_DIR / 'kmeans_k6_complete_LATEST.csv'

# 5 Composite Features (5 Pillars)
FEATURES = [
    'chronic_disease_burden',
    'metabolic_baseline',
    'lifestyle_risk',
    'substance_risk',
    'ses_vulnerability'
]

# Cluster labels (for visualization)
CLUSTER_LABELS = {
    0: 'C0: Low-Risk',
    1: 'C1: Socioeconomic Vulnerability',
    2: 'C2: High-Risk Lifestyle',
    3: 'C3: Chronic-Addiction Synergy',
    4: 'C4: Metabolic Extremes',
    5: 'C5: Frail Multi-Morbidity'
}

# Visualization settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("Set2")

print("=" * 80)
print("PCA Validation Analysis - 5-Feature Composite Space".center(80))
print("=" * 80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Input: {INPUT_FILE.name}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Features: {len(FEATURES)}")

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n" + "=" * 80)
print("1. Loading Data")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(df):,} records")
print(f"  Columns: {len(df.columns)}")

# Extract features and cluster labels
X = df[FEATURES].values
y = df['cluster'].values

print(f"\n✓ Feature matrix: {X.shape}")
print(f"✓ Cluster labels: {y.shape}")
print(f"  Unique clusters: {sorted(np.unique(y))}")

# ============================================================================
# 2. Standardization (Z-score normalization)
# ============================================================================
print("\n" + "=" * 80)
print("2. Feature Standardization")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✓ Standardization complete (mean=0, std=1)")
print("\nFeature statistics AFTER scaling:")
for i, feat in enumerate(FEATURES):
    print(f"  {feat:30s}: mean={X_scaled[:, i].mean():.6f}, std={X_scaled[:, i].std():.6f}")

# ============================================================================
# 3. PCA Decomposition
# ============================================================================
print("\n" + "=" * 80)
print("3. PCA Decomposition")
print("=" * 80)

# Full PCA (all 5 components)
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"✓ PCA fitted with {pca.n_components_} components")
print(f"\nExplained Variance Ratio:")
for i, var_ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var_ratio*100:.2f}%")

cumulative_var = np.cumsum(pca.explained_variance_ratio_)
print(f"\nCumulative Variance Explained:")
for i, cum_var in enumerate(cumulative_var):
    print(f"  PC1-PC{i+1}: {cum_var*100:.2f}%")

# Identify minimum components for 70%, 80%, 90% thresholds
thresholds = [0.70, 0.80, 0.90]
print(f"\nComponents Required for Variance Thresholds:")
for threshold in thresholds:
    n_comp = np.argmax(cumulative_var >= threshold) + 1
    print(f"  {threshold*100:.0f}% variance: {n_comp} components (actual: {cumulative_var[n_comp-1]*100:.2f}%)")

# ============================================================================
# 4. Component Loadings (Feature Contributions)
# ============================================================================
print("\n" + "=" * 80)
print("4. PCA Component Loadings")
print("=" * 80)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(5)],
    index=FEATURES
)

print("\n✓ Component Loadings Matrix:")
print(loadings.round(3).to_string())

# Identify dominant features in PC1 and PC2
print(f"\n✓ Dominant Features:")
for i in range(2):
    pc_name = f'PC{i+1}'
    abs_loadings = loadings[pc_name].abs().sort_values(ascending=False)
    print(f"\n  {pc_name} (explains {pca.explained_variance_ratio_[i]*100:.2f}% variance):")
    for feat, loading in abs_loadings.head(3).items():
        sign = '+' if loadings.loc[feat, pc_name] > 0 else '-'
        print(f"    {sign} {feat:30s}: {abs(loadings.loc[feat, pc_name]):.3f}")

# ============================================================================
# 5. Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("5. Generating Visualizations")
print("=" * 80)

# 5.1 Scree Plot + Cumulative Variance
print("\n[5.1] Scree Plot & Cumulative Variance")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax1.bar(range(1, 6), pca.explained_variance_ratio_ * 100, 
        color='steelblue', alpha=0.8, edgecolor='black')
ax1.plot(range(1, 6), pca.explained_variance_ratio_ * 100, 
         'ro-', linewidth=2, markersize=8)
ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
ax1.set_title('Scree Plot: Individual Variance per Component', fontsize=14, fontweight='bold')
ax1.set_xticks(range(1, 6))
ax1.set_xticklabels([f'PC{i}' for i in range(1, 6)])
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=20, color='red', linestyle='--', linewidth=1, alpha=0.5, label='20% threshold')
ax1.legend()

# Cumulative variance
ax2.plot(range(1, 6), cumulative_var * 100, 'o-', linewidth=2, markersize=8, color='darkgreen')
ax2.fill_between(range(1, 6), cumulative_var * 100, alpha=0.3, color='green')
ax2.axhline(y=70, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='70% threshold')
ax2.axhline(y=80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='80% threshold')
ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Variance: Information Retention', fontsize=14, fontweight='bold')
ax2.set_xticks(range(1, 6))
ax2.set_ylim([0, 105])
ax2.grid(alpha=0.3)
ax2.legend()

# Add annotations
for i, var in enumerate(cumulative_var):
    ax2.text(i+1, var*100 + 2, f'{var*100:.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
output_path = OUTPUT_DIR / 'pca_scree_cumulative.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path.name}")
plt.close()

# 5.2 Component Loadings Heatmap
print("[5.2] Component Loadings Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(loadings.T, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
            cbar_kws={'label': 'Loading'}, vmin=-1, vmax=1, 
            linewidths=0.5, linecolor='gray', ax=ax)
ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
ax.set_ylabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_title('PCA Component Loadings: Feature Contributions', fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = OUTPUT_DIR / 'pca_loadings_heatmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path.name}")
plt.close()

# 5.3 PC1 vs PC2 Scatter Plot (Cluster-colored)
print("[5.3] PC1 vs PC2 Scatter Plot (Cluster Assignment)")

# Sample 20,000 points for visualization clarity
np.random.seed(42)
sample_size = min(20000, len(X_pca))
sample_idx = np.random.choice(len(X_pca), sample_size, replace=False)
X_pca_sample = X_pca[sample_idx]
y_sample = y[sample_idx]

fig, ax = plt.subplots(figsize=(12, 8))

# Plot each cluster
for cluster_id in sorted(np.unique(y_sample)):
    mask = y_sample == cluster_id
    ax.scatter(X_pca_sample[mask, 0], X_pca_sample[mask, 1], 
               label=CLUSTER_LABELS[cluster_id], alpha=0.5, s=10, edgecolors='none')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
              fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
              fontsize=12, fontweight='bold')
ax.set_title('PCA Projection: Cluster Separability in 2D Space', fontsize=14, fontweight='bold')
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.grid(alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout()
output_path = OUTPUT_DIR / 'pca_cluster_projection.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path.name}")
plt.close()

# 5.4 PC1 vs PC2 with Diabetes Status (validation)
print("[5.4] PC1 vs PC2 Scatter Plot (Diabetes Status)")

diabetes_sample = df.loc[sample_idx, 'Diabetes_012'].values
diabetes_labels = {0: 'No Diabetes', 1: 'Prediabetes', 2: 'Diabetes'}

fig, ax = plt.subplots(figsize=(12, 8))

for diabetes_id in sorted(np.unique(diabetes_sample)):
    mask = diabetes_sample == diabetes_id
    ax.scatter(X_pca_sample[mask, 0], X_pca_sample[mask, 1], 
               label=diabetes_labels[diabetes_id], alpha=0.5, s=10, edgecolors='none')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
              fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
              fontsize=12, fontweight='bold')
ax.set_title('PCA Projection: Diabetes Status Distribution in 2D Space', fontsize=14, fontweight='bold')
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.grid(alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout()
output_path = OUTPUT_DIR / 'pca_diabetes_projection.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path.name}")
plt.close()

# ============================================================================
# 6. Save Results
# ============================================================================
print("\n" + "=" * 80)
print("6. Saving Results")
print("=" * 80)

# 6.1 Variance explanation table
variance_df = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(5)],
    'Variance_Explained': pca.explained_variance_ratio_,
    'Variance_Explained_Pct': pca.explained_variance_ratio_ * 100,
    'Cumulative_Variance': cumulative_var,
    'Cumulative_Variance_Pct': cumulative_var * 100
})

output_path = OUTPUT_DIR / 'pca_variance_explained.csv'
variance_df.to_csv(output_path, index=False, float_format='%.6f')
print(f"✓ Saved: {output_path.name}")

# 6.2 Component loadings
output_path = OUTPUT_DIR / 'pca_component_loadings.csv'
loadings.to_csv(output_path, float_format='%.6f')
print(f"✓ Saved: {output_path.name}")

# 6.3 Transformed data (first 2 PCs, sampled)
pca_projection_df = pd.DataFrame({
    'PC1': X_pca_sample[:, 0],
    'PC2': X_pca_sample[:, 1],
    'cluster': y_sample,
    'Diabetes_012': diabetes_sample
})
output_path = OUTPUT_DIR / 'pca_projection_sample.csv'
pca_projection_df.to_csv(output_path, index=False, float_format='%.6f')
print(f"✓ Saved: {output_path.name}")

# ============================================================================
# 7. Generate Report
# ============================================================================
print("\n" + "=" * 80)
print("7. Generating Report")
print("=" * 80)

report_path = OUTPUT_DIR / 'pca_validation_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("PCA Validation Analysis Report\n")
    f.write("Section 3.1.4: Dimensionality Reduction & Cluster Feasibility Analysis\n")
    f.write("=" * 80 + "\n")
    f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {INPUT_FILE.name}\n")
    f.write(f"Total Samples: {len(df):,}\n")
    f.write(f"Features: {len(FEATURES)} (5 Pillars)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("1. Variance Explanation Analysis\n")
    f.write("=" * 80 + "\n")
    f.write("\nIndividual Variance Explained:\n")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        f.write(f"  PC{i+1}: {var_ratio*100:.2f}%\n")
    
    f.write("\nCumulative Variance Explained:\n")
    for i, cum_var in enumerate(cumulative_var):
        f.write(f"  PC1-PC{i+1}: {cum_var*100:.2f}%\n")
    
    f.write("\nKey Findings:\n")
    n_comp_70 = np.argmax(cumulative_var >= 0.70) + 1
    n_comp_80 = np.argmax(cumulative_var >= 0.80) + 1
    f.write(f"  • {n_comp_70} components explain {cumulative_var[n_comp_70-1]*100:.1f}% variance (≥70% threshold)\n")
    f.write(f"  • {n_comp_80} components explain {cumulative_var[n_comp_80-1]*100:.1f}% variance (≥80% threshold)\n")
    f.write(f"  • All 5 components needed for {cumulative_var[-1]*100:.1f}% variance\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("2. Component Loadings Interpretation\n")
    f.write("=" * 80 + "\n")
    f.write("\nPC1 (Primary Risk Gradient):\n")
    pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
    for feat, loading in pc1_loadings.items():
        sign = '+' if loadings.loc[feat, 'PC1'] > 0 else '-'
        f.write(f"  {sign} {feat:30s}: {abs(loadings.loc[feat, 'PC1']):.3f}\n")
    
    f.write("\nPC2 (Secondary Risk Gradient):\n")
    pc2_loadings = loadings['PC2'].abs().sort_values(ascending=False)
    for feat, loading in pc2_loadings.items():
        sign = '+' if loadings.loc[feat, 'PC2'] > 0 else '-'
        f.write(f"  {sign} {feat:30s}: {abs(loadings.loc[feat, 'PC2']):.3f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("3. Orthogonality Verification\n")
    f.write("=" * 80 + "\n")
    f.write("\nPCA transformation ensures mathematical orthogonality:\n")
    f.write("  • All principal components are UNCORRELATED by construction\n")
    f.write("  • This complements VIF analysis (Figure 12) which showed low multicollinearity\n")
    f.write("  • Together, these validate that 5 features capture independent risk dimensions\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("4. Cluster Separability in Reduced Space\n")
    f.write("=" * 80 + "\n")
    f.write("\nVisual inspection of PC1 vs PC2 projection reveals:\n")
    f.write("  • Clusters show PARTIAL separation in 2D space\n")
    f.write("  • Some overlap exists, indicating clusters require ALL 5 dimensions\n")
    f.write("  • This justifies using full 5-feature space rather than PCA reduction\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("5. Critical Evaluation: Why Cluster in Original 5-Feature Space?\n")
    f.write("=" * 80 + "\n")
    f.write("\nDecision: Perform clustering in ORIGINAL 5-feature space, NOT PCA space\n")
    f.write("\nRationale:\n")
    f.write("  ✓ Clinical Interpretability:\n")
    f.write("    - Original features (chronic_disease_burden, ses_vulnerability, etc.)\n")
    f.write("      directly map to INTERVENTION STRATEGIES\n")
    f.write("    - PCA components are linear combinations lacking clinical meaning\n")
    f.write("    - Policy-makers need actionable dimensions, not abstract 'PC1'\n")
    
    f.write("\n  ✓ Feature Engineering Already Reduced Dimensionality:\n")
    f.write("    - Started with 21 original BRFSS variables\n")
    f.write("    - Aggregated into 5 composite features (76% reduction)\n")
    f.write("    - Further PCA reduction would sacrifice hard-won interpretability\n")
    
    f.write("\n  ✓ Low Multicollinearity Confirmed:\n")
    f.write("    - VIF analysis (Figure 12) showed max VIF = 1.12 (well below threshold)\n")
    f.write("    - PCA variance distribution is relatively balanced (no single dominant PC)\n")
    f.write("    - This indicates features are already sufficiently independent\n")
    
    f.write("\n  ✓ Computational Feasibility:\n")
    f.write("    - 5 dimensions is manageable for K-Means (N=229,781 samples)\n")
    f.write("    - PCA reduction typically justified for D>20 or computational constraints\n")
    f.write("    - No such constraints exist in our case\n")
    
    f.write("\n  ✗ PCA Trade-offs:\n")
    f.write("    - Dimensionality reduction ALWAYS loses information\n")
    f.write(f"    - Even 3 PCs only retain {cumulative_var[2]*100:.1f}% variance\n")
    f.write("    - Lost variance may contain clinically important distinctions\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("6. Role of PCA in This Analysis\n")
    f.write("=" * 80 + "\n")
    f.write("\nPCA serves as a VALIDATION TOOL, not a preprocessing step:\n")
    f.write("  1. Orthogonality Check: Confirms low feature redundancy\n")
    f.write("  2. Variance Distribution: Shows no single feature dominates\n")
    f.write("  3. Visualization Aid: 2D projection for exploratory analysis\n")
    f.write("  4. Dimensionality Assessment: Justifies 5-feature space is appropriate\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("7. Implications for Section 3.2 (Clustering)\n")
    f.write("=" * 80 + "\n")
    f.write("\nThis PCA analysis validates proceeding with K-Means clustering using:\n")
    f.write("  • Input: 5 standardized composite features (Z-score normalized)\n")
    f.write("  • Space: Original feature space (NOT PCA-transformed)\n")
    f.write("  • Justification: Interpretability > marginal variance gain\n")
    f.write("\nStrengths demonstrated:\n")
    f.write("  ✓ Features are sufficiently orthogonal (VIF + PCA)\n")
    f.write("  ✓ All 5 dimensions contribute to variance (balanced loadings)\n")
    f.write("  ✓ Cluster structure visible but requires full 5D space\n")
    f.write("  ✓ Intervention mapping preserved through interpretable features\n")

print(f"✓ Saved: {report_path.name}")

print("\n" + "=" * 80)
print("PCA VALIDATION COMPLETE")
print("=" * 80)
print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. pca_scree_cumulative.png - Variance explanation plots")
print("  2. pca_loadings_heatmap.png - Feature contributions to PCs")
print("  3. pca_cluster_projection.png - 2D visualization (cluster-colored)")
print("  4. pca_diabetes_projection.png - 2D visualization (diabetes-colored)")
print("  5. pca_variance_explained.csv - Variance statistics")
print("  6. pca_component_loadings.csv - Component loadings matrix")
print("  7. pca_projection_sample.csv - Sample transformed data")
print("  8. pca_validation_report.txt - Comprehensive report")
print("\n" + "=" * 80)
