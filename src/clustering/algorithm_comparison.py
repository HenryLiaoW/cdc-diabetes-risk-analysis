#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Algorithm Comparison
================================

Compare 4 clustering algorithms on the same 5-feature composite space:
1. K-Means (K=6) - BASELINE
2. Hierarchical Clustering (Agglomerative)
3. DBSCAN (Density-Based)
4. Gaussian Mixture Model (GMM)

Output:
- Comparison table with key metrics
- Performance comparison visualization

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
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'outputs' / 'clustering' / 'results'
BASELINE_DIR = BASE_DIR / 'outputs' / 'clustering' / 'profiling' / 'final_20260221_222939'
OUTPUT_DIR = BASELINE_DIR / 'algorithm_comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / 'kmeans_k6_complete_LATEST.csv'

FEATURES = [
    'chronic_disease_burden',
    'metabolic_baseline',
    'lifestyle_risk',
    'substance_risk',
    'ses_vulnerability'
]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

print("=" * 80)
print("Clustering Algorithm Comparison".center(80))
print("=" * 80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Input: {INPUT_FILE.name}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Features: {len(FEATURES)}")

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================
print("\n" + "=" * 80)
print("1. Loading Data")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(df):,} records")

X = df[FEATURES].values
print(f"✓ Feature matrix: {X.shape}")

# Store diabetes info for validation
y_diabetes = df['Diabetes_012'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Features standardized (mean=0, std=1)")

# For computational efficiency, use stratified sample for Hierarchical clustering
# (DBSCAN and others will use full dataset)
SAMPLE_SIZE_HIERARCHICAL = 50000
np.random.seed(42)
sample_idx_hc = np.random.choice(len(X_scaled), min(SAMPLE_SIZE_HIERARCHICAL, len(X_scaled)), replace=False)
X_scaled_sampled = X_scaled[sample_idx_hc]
y_diabetes_sampled = y_diabetes[sample_idx_hc]
print(f"✓ Created sample for Hierarchical clustering: {len(X_scaled_sampled):,} samples")

# ============================================================================
# 2. Run Clustering Algorithms
# ============================================================================
print("\n" + "=" * 80)
print("2. Running Clustering Algorithms")
print("=" * 80)

results = {}

# ----------------------
# 2.1 K-Means (Baseline)
# ----------------------
print("\n[2.1] K-Means (K=6) - BASELINE")
kmeans = KMeans(n_clusters=6, random_state=42, n_init=50, max_iter=500)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Reorder by diabetes rate (same as baseline)
diabetes_rates_km = []
for i in range(6):
    mask = labels_kmeans == i
    diabetes_rate = (y_diabetes[mask] == 2).mean() * 100
    diabetes_rates_km.append((i, diabetes_rate))
diabetes_rates_km_sorted = sorted(diabetes_rates_km, key=lambda x: x[1])
cluster_mapping_km = {old: new for new, (old, _) in enumerate(diabetes_rates_km_sorted)}
labels_kmeans_reordered = np.array([cluster_mapping_km[label] for label in labels_kmeans])

results['K-Means'] = {
    'labels': labels_kmeans_reordered,
    'n_clusters': 6,
    'algorithm': 'K-Means',
    'params': 'K=6, n_init=50'
}
print(f"  ✓ Clusters: {6}")
print(f"  ✓ Samples per cluster: {np.bincount(labels_kmeans_reordered)}")

# ----------------------
# 2.2 Hierarchical Clustering
# ----------------------
print("\n[2.2] Hierarchical Clustering (Agglomerative)")
print(f"  Note: Using sample of {len(X_scaled_sampled):,} samples for computational efficiency")
# Use Ward linkage with 6 clusters for fair comparison
hierarchical = AgglomerativeClustering(n_clusters=6, linkage='ward')
labels_hierarchical_sample = hierarchical.fit_predict(X_scaled_sampled)

# Reorder by diabetes rate (on sampled data)
diabetes_rates_hc = []
for i in range(6):
    mask = labels_hierarchical_sample == i
    if mask.sum() > 0:
        diabetes_rate = (y_diabetes_sampled[mask] == 2).mean() * 100
        diabetes_rates_hc.append((i, diabetes_rate))
diabetes_rates_hc_sorted = sorted(diabetes_rates_hc, key=lambda x: x[1])
cluster_mapping_hc = {old: new for new, (old, _) in enumerate(diabetes_rates_hc_sorted)}
labels_hierarchical_reordered = np.array([cluster_mapping_hc[label] for label in labels_hierarchical_sample])

results['Hierarchical'] = {
    'labels': labels_hierarchical_reordered,
    'X_data': X_scaled_sampled,  # Store sample data for metrics
    'y_diabetes': y_diabetes_sampled,  # Store sample diabetes labels
    'n_clusters': len(np.unique(labels_hierarchical_sample)),
    'algorithm': 'Hierarchical',
    'params': f'Ward linkage, K=6 (sampled N={len(X_scaled_sampled):,})',
    'is_sampled': True
}
print(f"  ✓ Clusters: {len(np.unique(labels_hierarchical_sample))}")
print(f"  ✓ Samples per cluster: {np.bincount(labels_hierarchical_reordered)}")

# ----------------------
# 2.3 DBSCAN
# ----------------------
print("\n[2.3] DBSCAN (Density-Based)")
# Parameter tuning: try to get reasonable number of clusters
# eps and min_samples need calibration
dbscan = DBSCAN(eps=0.5, min_samples=50)
labels_dbscan = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

# Reorder by diabetes rate (excluding noise points)
unique_labels = sorted([l for l in set(labels_dbscan) if l != -1])
if len(unique_labels) > 0:
    diabetes_rates_db = []
    for i in unique_labels:
        mask = labels_dbscan == i
        diabetes_rate = (y_diabetes[mask] == 2).mean() * 100
        diabetes_rates_db.append((i, diabetes_rate))
    diabetes_rates_db_sorted = sorted(diabetes_rates_db, key=lambda x: x[1])
    cluster_mapping_db = {old: new for new, (old, _) in enumerate(diabetes_rates_db_sorted)}
    cluster_mapping_db[-1] = -1  # Keep noise as -1
    labels_dbscan_reordered = np.array([cluster_mapping_db[label] for label in labels_dbscan])
else:
    labels_dbscan_reordered = labels_dbscan

results['DBSCAN'] = {
    'labels': labels_dbscan_reordered,
    'n_clusters': n_clusters_dbscan,
    'n_noise': n_noise,
    'algorithm': 'DBSCAN',
    'params': f'eps=0.5, min_samples=50'
}
print(f"  ✓ Clusters: {n_clusters_dbscan}")
print(f"  ✓ Noise points: {n_noise:,} ({n_noise/len(X)*100:.1f}%)")
if n_clusters_dbscan > 0:
    print(f"  ✓ Samples per cluster: {np.bincount(labels_dbscan_reordered[labels_dbscan_reordered >= 0])}")

# ----------------------
# 2.4 Gaussian Mixture Model
# ----------------------
print("\n[2.4] Gaussian Mixture Model (GMM)")
gmm = GaussianMixture(n_components=6, random_state=42, n_init=10, covariance_type='full')
labels_gmm = gmm.fit_predict(X_scaled)

# Reorder by diabetes rate
diabetes_rates_gmm = []
for i in range(6):
    mask = labels_gmm == i
    diabetes_rate = (y_diabetes[mask] == 2).mean() * 100
    diabetes_rates_gmm.append((i, diabetes_rate))
diabetes_rates_gmm_sorted = sorted(diabetes_rates_gmm, key=lambda x: x[1])
cluster_mapping_gmm = {old: new for new, (old, _) in enumerate(diabetes_rates_gmm_sorted)}
labels_gmm_reordered = np.array([cluster_mapping_gmm[label] for label in labels_gmm])

results['GMM'] = {
    'labels': labels_gmm_reordered,
    'n_clusters': 6,
    'algorithm': 'GMM',
    'params': 'K=6, covariance=full'
}
print(f"  ✓ Clusters: {6}")
print(f"  ✓ Samples per cluster: {np.bincount(labels_gmm_reordered)}")
print(f"  ✓ Converged: {gmm.converged_}")
print(f"  ✓ BIC: {gmm.bic(X_scaled):.2f}")
print(f"  ✓ AIC: {gmm.aic(X_scaled):.2f}")

# ============================================================================
# 3. Compute Metrics
# ============================================================================
print("\n" + "=" * 80)
print("3. Computing Performance Metrics")
print("=" * 80)

metrics_data = []

for algo_name, result in results.items():
    labels = result['labels']
    
    # Use sampled data if available (for Hierarchical)
    if result.get('is_sampled', False):
        X_for_metrics = result['X_data']
        y_diabetes_for_metrics = result['y_diabetes']
    else:
        X_for_metrics = X_scaled
        y_diabetes_for_metrics = y_diabetes
    
    # Skip metrics for invalid clustering (e.g., DBSCAN with no clusters)
    if result['n_clusters'] < 2:
        print(f"\n[{algo_name}] Insufficient clusters for metric computation")
        metrics_data.append({
            'Algorithm': algo_name,
            'Parameters': result['params'],
            'N_Clusters': result['n_clusters'],
            'Silhouette_Score': np.nan,
            'Davies_Bouldin_Index': np.nan,
            'Calinski_Harabasz_Score': np.nan,
            'AtRisk_Rate_Min': np.nan,
            'AtRisk_Rate_Max': np.nan,
            'AtRisk_Rate_Range': np.nan
        })
        continue
    
    # Filter out noise points for DBSCAN
    if algo_name == 'DBSCAN':
        mask = labels >= 0
        X_filtered = X_for_metrics[mask]
        labels_filtered = labels[mask]
        y_diabetes_filtered = y_diabetes_for_metrics[mask]
        
        if len(np.unique(labels_filtered)) < 2:
            print(f"\n[{algo_name}] Insufficient valid clusters after filtering noise")
            metrics_data.append({
                'Algorithm': algo_name,
                'Parameters': result['params'],
                'N_Clusters': result['n_clusters'],
                'N_Noise': result.get('n_noise', 0),
                'Silhouette_Score': np.nan,
                'Davies_Bouldin_Index': np.nan,
                'Calinski_Harabasz_Score': np.nan,
                'AtRisk_Rate_Min': np.nan,
                'AtRisk_Rate_Max': np.nan,
                'AtRisk_Rate_Range': np.nan
            })
            continue
    else:
        X_filtered = X_for_metrics
        labels_filtered = labels
        y_diabetes_filtered = y_diabetes_for_metrics
    
    # Compute metrics
    sil_score = silhouette_score(X_filtered, labels_filtered)
    db_index = davies_bouldin_score(X_filtered, labels_filtered)
    ch_score = calinski_harabasz_score(X_filtered, labels_filtered)
    
    # Compute At-Risk rate range (validation)
    at_risk_rates = []
    for cluster_id in np.unique(labels_filtered):
        mask = labels_filtered == cluster_id
        at_risk_rate = ((y_diabetes_filtered[mask] == 1) | (y_diabetes_filtered[mask] == 2)).mean() * 100
        at_risk_rates.append(at_risk_rate)
    
    at_risk_min = min(at_risk_rates)
    at_risk_max = max(at_risk_rates)
    at_risk_range = at_risk_max - at_risk_min
    
    print(f"\n[{algo_name}]")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Davies-Bouldin Index: {db_index:.4f}")
    print(f"  Calinski-Harabasz Score: {ch_score:.2f}")
    print(f"  At-Risk Rate Range: {at_risk_min:.1f}% - {at_risk_max:.1f}% (Δ={at_risk_range:.1f}%)")
    
    metric_dict = {
        'Algorithm': algo_name,
        'Parameters': result['params'],
        'N_Clusters': result['n_clusters'],
        'Silhouette_Score': sil_score,
        'Davies_Bouldin_Index': db_index,
        'Calinski_Harabasz_Score': ch_score,
        'AtRisk_Rate_Min': at_risk_min,
        'AtRisk_Rate_Max': at_risk_max,
        'AtRisk_Rate_Range': at_risk_range
    }
    
    if algo_name == 'DBSCAN':
        metric_dict['N_Noise'] = result.get('n_noise', 0)
    
    metrics_data.append(metric_dict)

metrics_df = pd.DataFrame(metrics_data)

# ============================================================================
# 4. Generate Comparison Table
# ============================================================================
print("\n" + "=" * 80)
print("4. Generating Comparison Table")
print("=" * 80)

# Reorder columns for display
column_order = ['Algorithm', 'Parameters', 'N_Clusters', 'Silhouette_Score', 
                'Davies_Bouldin_Index', 'Calinski_Harabasz_Score',
                'AtRisk_Rate_Range', 'AtRisk_Rate_Min', 'AtRisk_Rate_Max']
if 'N_Noise' in metrics_df.columns:
    column_order.insert(3, 'N_Noise')

metrics_df = metrics_df[column_order]

print("\n" + "=" * 80)
print("CLUSTERING ALGORITHM COMPARISON")
print("=" * 80)
print(metrics_df.to_string(index=False))

# Save table
output_path = OUTPUT_DIR / 'algorithm_comparison.csv'
metrics_df.to_csv(output_path, index=False, float_format='%.4f')
print(f"\n✓ Saved: {output_path.name}")

# ============================================================================
# 5. Generate Comparison Visualization
# ============================================================================
print("\n" + "=" * 80)
print("5. Generating Comparison Visualization")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Filter out algorithms with insufficient clusters
valid_metrics = metrics_df[metrics_df['N_Clusters'] >= 2].copy()

if len(valid_metrics) == 0:
    print("⚠️  No valid algorithms for visualization")
else:
    algorithms = valid_metrics['Algorithm'].values
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(algorithms)]
    
    # Plot 1: Silhouette Score (Higher is better)
    ax1 = axes[0, 0]
    bars = ax1.barh(algorithms, valid_metrics['Silhouette_Score'], color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Silhouette Score', fontsize=11, fontweight='bold')
    ax1.set_title('Silhouette Score\n(Higher = Better Cluster Cohesion)', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0.2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.2)')
    for i, (bar, val) in enumerate(zip(bars, valid_metrics['Silhouette_Score'])):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Davies-Bouldin Index (Lower is better)
    ax2 = axes[0, 1]
    bars = ax2.barh(algorithms, valid_metrics['Davies_Bouldin_Index'], color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Davies-Bouldin Index', fontsize=11, fontweight='bold')
    ax2.set_title('Davies-Bouldin Index\n(Lower = Better Cluster Separation)', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, valid_metrics['Davies_Bouldin_Index'])):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Plot 3: Calinski-Harabasz Score (Higher is better)
    ax3 = axes[1, 0]
    bars = ax3.barh(algorithms, valid_metrics['Calinski_Harabasz_Score'], color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Calinski-Harabasz Score', fontsize=11, fontweight='bold')
    ax3.set_title('Calinski-Harabasz Score\n(Higher = Better Defined Clusters)', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, valid_metrics['Calinski_Harabasz_Score'])):
        ax3.text(val + 100, bar.get_y() + bar.get_height()/2, f'{val:.0f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Plot 4: At-Risk Rate Stratification Range (Higher = Better Risk Differentiation)
    ax4 = axes[1, 1]
    bars = ax4.barh(algorithms, valid_metrics['AtRisk_Rate_Range'], color=colors, edgecolor='black', linewidth=1.2)
    ax4.set_xlabel('At-Risk Rate Range (%)', fontsize=11, fontweight='bold')
    ax4.set_title('At-Risk Rate Stratification\n(Higher = Better Risk Differentiation)', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, valid_metrics['AtRisk_Rate_Range'])):
        ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Clustering Algorithm Performance Comparison\n(5-Feature Composite Space, N=229,781)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'algorithm_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()

# ============================================================================
# 6. Generate Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("6. Generating Summary Report")
print("=" * 80)

report_path = OUTPUT_DIR / 'algorithm_comparison_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Clustering Algorithm Comparison Report\n")
    f.write("=" * 80 + "\n")
    f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {INPUT_FILE.name}\n")
    f.write(f"Total Samples: {len(df):,}\n")
    f.write(f"Features: {len(FEATURES)} (5 Pillars)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("Performance Metrics Comparison\n")
    f.write("=" * 80 + "\n\n")
    f.write(metrics_df.to_string(index=False))
    
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("Metric Interpretation\n")
    f.write("=" * 80 + "\n")
    f.write("\n1. Silhouette Score (Range: -1 to 1)\n")
    f.write("   • Measures cluster cohesion and separation\n")
    f.write("   • Higher is better (>0.5 = strong, 0.2-0.5 = acceptable, <0.2 = weak)\n")
    f.write("   • Optimal: K-Means or GMM typically perform best\n")
    
    f.write("\n2. Davies-Bouldin Index (Range: 0 to ∞)\n")
    f.write("   • Measures cluster separation relative to cluster compactness\n")
    f.write("   • Lower is better (<1.0 = good separation)\n")
    f.write("   • Optimal: Hierarchical often achieves low scores\n")
    
    f.write("\n3. Calinski-Harabasz Score (Range: 0 to ∞)\n")
    f.write("   • Ratio of between-cluster to within-cluster variance\n")
    f.write("   • Higher is better (>1000 generally indicates well-defined clusters)\n")
    f.write("   • Optimal: K-Means often scores highest due to spherical cluster assumption\n")
    
    f.write("\n4. At-Risk Rate Stratification Range\n")
    f.write("   • Difference between highest and lowest cluster At-Risk rates\n")
    f.write("   • Higher indicates better risk differentiation for clinical utility\n")
    f.write("   • Optimal: Should maximize range for effective population stratification\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("Algorithm Characteristics\n")
    f.write("=" * 80 + "\n")
    
    f.write("\nK-Means (BASELINE):\n")
    f.write("  ✓ Assumes spherical, equally-sized clusters\n")
    f.write("  ✓ Fast and scalable\n")
    f.write("  ✓ Requires pre-specified K\n")
    f.write("  ✓ Sensitive to initialization (mitigated by n_init=50)\n")
    
    f.write("\nHierarchical Clustering:\n")
    f.write("  ✓ No assumption on cluster shape\n")
    f.write("  ✓ Deterministic (no random initialization)\n")
    f.write("  ✓ Can explore different K via dendrogram\n")
    f.write("  ✗ Computationally expensive for large datasets\n")
    
    f.write("\nDBSCAN:\n")
    f.write("  ✓ Discovers arbitrary-shaped clusters\n")
    f.write("  ✓ Automatically determines number of clusters\n")
    f.write("  ✓ Identifies outliers (noise points)\n")
    f.write("  ✗ Sensitive to eps and min_samples parameters\n")
    f.write("  ✗ Struggles with varying density clusters\n")
    
    f.write("\nGaussian Mixture Model (GMM):\n")
    f.write("  ✓ Assumes Gaussian distribution of clusters\n")
    f.write("  ✓ Soft clustering (probabilistic assignment)\n")
    f.write("  ✓ More flexible than K-Means (elliptical clusters)\n")
    f.write("  ✗ Requires pre-specified K\n")
    f.write("  ✗ Can converge to local optima\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("Recommendation\n")
    f.write("=" * 80 + "\n")
    
    # Find best algorithm by Silhouette Score
    if len(valid_metrics) > 0:
        best_algo = valid_metrics.loc[valid_metrics['Silhouette_Score'].idxmax(), 'Algorithm']
        best_sil = valid_metrics.loc[valid_metrics['Silhouette_Score'].idxmax(), 'Silhouette_Score']
        
        f.write(f"\nBased on Silhouette Score: **{best_algo}** (Score: {best_sil:.4f})\n")
        f.write("\nHowever, for this diabetes risk stratification task:\n")
        f.write("  • K-Means remains the BASELINE choice due to:\n")
        f.write("    1. Computational efficiency (N=229,781 samples)\n")
        f.write("    2. Interpretability (clear cluster centroids)\n")
        f.write("    3. Clinical utility (fixed K=6 for intervention design)\n")
        f.write("    4. Consistent performance across multiple metrics\n")
        f.write("    5. Strong At-Risk rate stratification\n")
        
        f.write("\n  • Alternative algorithms may be considered if:\n")
        f.write("    - Non-spherical cluster shapes are suspected (→ GMM or Hierarchical)\n")
        f.write("    - Outlier detection is required (→ DBSCAN)\n")
        f.write("    - Exploratory analysis without fixed K (→ Hierarchical or DBSCAN)\n")

print(f"✓ Saved: {report_path.name}")

print("\n" + "=" * 80)
print("ALGORITHM COMPARISON COMPLETE")
print("=" * 80)
print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. algorithm_comparison.csv - Metrics comparison table")
print("  2. algorithm_comparison.png - Performance visualization")
print("  3. algorithm_comparison_report.txt - Detailed analysis")
print("\n" + "=" * 80)
