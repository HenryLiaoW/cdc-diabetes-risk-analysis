"""
SHAP Analysis: Diabetes Risk Within Specific Cluster
===================================================
This script analyzes which features contribute most to diabetes risk
WITHIN a specific cluster (not for cluster assignment).

Target: Diabetes_012 (0=No diabetes, 1=Prediabetes, 2=Diabetes)
Subset: Only samples from a specific cluster
Question: What drives diabetes risk in this specific population segment?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
import shap

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# Configuration
# ============================================================================
INPUT_FILE = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/results/kmeans_k6_complete_LATEST.csv')
OUTPUT_DIR = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/profiling/final_20260221_222939/shap_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'chronic_disease_burden',
    'metabolic_baseline',
    'lifestyle_risk',
    'substance_risk',
    'ses_vulnerability'
]

# Display names for visualizations
FEATURE_DISPLAY_NAMES = [
    'chronic_disease_burden',
    'metabolic_baseline',
    'lifestyle_risk',
    'addiction_risk',  # Display as "addiction_risk" instead of "substance_risk"
    'ses_vulnerability'
]

TARGET_CLUSTER = 3  # Analyze Cluster 3 (Moderate Risk Group)
TARGET = 'Diabetes_012'  # Predict diabetes, NOT cluster
RANDOM_STATE = 42
SAMPLE_SIZE = 30000  # For SHAP computation

print("=" * 80)
print(f"SHAP Analysis: Diabetes Risk Within Cluster {TARGET_CLUSTER}".center(80))
print("=" * 80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Input: {INPUT_FILE.name}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Target Cluster: {TARGET_CLUSTER}")
print(f"Target Variable: {TARGET} (Diabetes status)")

# ============================================================================
# 1. Load and Filter Data
# ============================================================================
print("\n" + "=" * 80)
print(f"1. Loading and Filtering Data (Cluster {TARGET_CLUSTER} Only)")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(df):,} total records")

# Filter for target cluster only
df_cluster = df[df['cluster'] == TARGET_CLUSTER].copy()
print(f"✓ Filtered to Cluster {TARGET_CLUSTER}: {len(df_cluster):,} samples ({len(df_cluster)/len(df)*100:.1f}%)")

# Check target distribution
print(f"\n✓ Diabetes distribution in Cluster {TARGET_CLUSTER}:")
diabetes_dist = df_cluster[TARGET].value_counts(normalize=True).sort_index()
for val, pct in diabetes_dist.items():
    count = df_cluster[TARGET].value_counts()[val]
    diabetes_label = {0: 'No Diabetes', 1: 'Prediabetes', 2: 'Diabetes'}.get(val, f'Level {val}')
    print(f"  {diabetes_label}: {pct*100:.2f}% ({count:,} samples)")

# Prepare features and target
X = df_cluster[FEATURES].copy()
y = df_cluster[TARGET].copy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\n✓ Train-test split:")
print(f"  Training: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Testing: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 2. Random Forest Model
# ============================================================================
print("\n" + "=" * 80)
print("2. Training Random Forest for Diabetes Prediction")
print("=" * 80)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'  # Handle imbalanced classes
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')

print(f"✓ Random Forest trained")
print(f"  Accuracy: {rf_acc:.4f}")
print(f"  F1-Score (weighted): {rf_f1:.4f}")

# ============================================================================
# 3. XGBoost Model
# ============================================================================
print("\n" + "=" * 80)
print("3. Training XGBoost for Diabetes Prediction")
print("=" * 80)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')

print(f"✓ XGBoost trained")
print(f"  Accuracy: {xgb_acc:.4f}")
print(f"  F1-Score (weighted): {xgb_f1:.4f}")

# ============================================================================
# 4. SHAP Analysis
# ============================================================================
print("\n" + "=" * 80)
print("4. Computing SHAP Values")
print("=" * 80)

# Sample for SHAP computation
if len(X_train) > SAMPLE_SIZE:
    shap_indices = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
    X_shap = X_train.iloc[shap_indices]
    print(f"✓ Using {SAMPLE_SIZE:,} random samples for SHAP computation")
else:
    X_shap = X_train
    print(f"✓ Using all {len(X_shap):,} training samples for SHAP")

# Random Forest SHAP
print("\nComputing SHAP values for Random Forest...")
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X_shap)
print("✓ Random Forest SHAP complete")

# XGBoost SHAP
print("Computing SHAP values for XGBoost...")
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X_shap)
print("✓ XGBoost SHAP complete")

# ============================================================================
# 5. Extract Mean SHAP Importance (averaged across all diabetes classes)
# ============================================================================
print("\n" + "=" * 80)
print("5. Computing Mean SHAP Importance")
print("=" * 80)

# For TreeExplainer with multiclass, SHAP returns shape (n_samples, n_features, n_classes)
# We want mean |SHAP| across samples and classes, keeping only features
if rf_shap_values.ndim == 3:
    # Shape: (samples, features, classes) -> average across samples and classes
    rf_shap_mean = np.mean(np.abs(rf_shap_values), axis=(0, 2))  # Keep only features
    print(f"  RF: Multiclass SHAP, shape: {rf_shap_values.shape} -> {rf_shap_mean.shape}")
elif isinstance(rf_shap_values, list):
    rf_shap_mean = np.mean([np.abs(rf_shap_values[i]).mean(axis=0) for i in range(len(rf_shap_values))], axis=0)
    print(f"  RF: Multiclass SHAP (list), {len(rf_shap_values)} classes")
else:
    rf_shap_mean = np.abs(rf_shap_values).mean(axis=0)
    print(f"  RF: Binary SHAP, shape: {rf_shap_values.shape}")

if xgb_shap_values.ndim == 3:
    xgb_shap_mean = np.mean(np.abs(xgb_shap_values), axis=(0, 2))  # Keep only features
    print(f"  XGB: Multiclass SHAP, shape: {xgb_shap_values.shape} -> {xgb_shap_mean.shape}")
elif isinstance(xgb_shap_values, list):
    xgb_shap_mean = np.mean([np.abs(xgb_shap_values[i]).mean(axis=0) for i in range(len(xgb_shap_values))], axis=0)
    print(f"  XGB: Multiclass SHAP (list), {len(xgb_shap_values)} classes")
else:
    xgb_shap_mean = np.abs(xgb_shap_values).mean(axis=0)
    print(f"  XGB: Binary SHAP, shape: {xgb_shap_values.shape}")

# Create results dataframe
results_df = pd.DataFrame({
    'Feature': FEATURES,
    'RF_SHAP_Importance': rf_shap_mean,
    'XGB_SHAP_Importance': xgb_shap_mean
})

# Add ranks
results_df['RF_Rank'] = results_df['RF_SHAP_Importance'].rank(ascending=False).astype(int)
results_df['XGB_Rank'] = results_df['XGB_SHAP_Importance'].rank(ascending=False).astype(int)
results_df = results_df.sort_values('RF_SHAP_Importance', ascending=False)

print("\n✓ SHAP Importance Rankings:")
print(results_df.to_string(index=False))

# ============================================================================
# 6. Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("6. Generating Visualizations")
print("=" * 80)

# Determine which diabetes class to visualize (highest prevalence)
if isinstance(rf_shap_values, list):
    diabetes_class_counts = y_train.value_counts().sort_index()
    # Visualize class with most samples (usually class 2 = Diabetes)
    viz_class = 2 if 2 in diabetes_class_counts.index else diabetes_class_counts.idxmax()
    print(f"✓ Visualizing SHAP for Diabetes Class {viz_class} ({['No Diabetes', 'Prediabetes', 'Diabetes'][viz_class]})")
else:
    viz_class = None

# 6.1 SHAP Summary Plot (Beeswarm) - Using Class 2 (Diabetes) for visualization
print(f"\nGenerating SHAP Beeswarm Plot (Cluster {TARGET_CLUSTER})...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Random Forest - shap values for all samples, class 2 (Diabetes)
plt.sca(ax1)
if rf_shap_values.ndim == 3:
    # Shape: (samples, features, classes) - take class 2
    shap.summary_plot(rf_shap_values[:, :, viz_class], X_shap, 
                      feature_names=FEATURE_DISPLAY_NAMES, show=False, max_display=5)
    ax1.set_title(f'Random Forest (Diabetes Class)', fontsize=13, fontweight='bold')
else:
    shap.summary_plot(rf_shap_values, X_shap, feature_names=FEATURE_DISPLAY_NAMES, 
                      show=False, max_display=5)
    ax1.set_title(f'Random Forest', fontsize=13, fontweight='bold')

# XGBoost
plt.sca(ax2)
if xgb_shap_values.ndim == 3:
    shap.summary_plot(xgb_shap_values[:, :, viz_class], X_shap, 
                      feature_names=FEATURE_DISPLAY_NAMES, show=False, max_display=5)
    ax2.set_title(f'XGBoost (Diabetes Class)', fontsize=13, fontweight='bold')
else:
    shap.summary_plot(xgb_shap_values, X_shap, feature_names=FEATURE_DISPLAY_NAMES, 
                      show=False, max_display=5)
    ax2.set_title(f'XGBoost', fontsize=13, fontweight='bold')

plt.suptitle(f'SHAP Value Distribution for Diabetes Prediction (Cluster {TARGET_CLUSTER})', 
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = OUTPUT_DIR / f'shap_diabetes_cluster{TARGET_CLUSTER}_summary.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# 6.2 SHAP Bar Plot - Custom visualization using computed means
print(f"Generating SHAP Bar Plot (Cluster {TARGET_CLUSTER})...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Sort features by importance for display
rf_sorted_idx = np.argsort(rf_shap_mean)
xgb_sorted_idx = np.argsort(xgb_shap_mean)

# Random Forest
y_pos = np.arange(len(FEATURES))
ax1.barh(y_pos, rf_shap_mean[rf_sorted_idx], color='#1f77b4', alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([FEATURE_DISPLAY_NAMES[i] for i in rf_sorted_idx], fontsize=11)
ax1.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax1.set_title(f'Random Forest - Feature Importance (C{TARGET_CLUSTER})', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# XGBoost
ax2.barh(y_pos, xgb_shap_mean[xgb_sorted_idx], color='#ff7f0e', alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([FEATURE_DISPLAY_NAMES[i] for i in xgb_sorted_idx], fontsize=11)
ax2.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax2.set_title(f'XGBoost - Feature Importance (C{TARGET_CLUSTER})', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.suptitle(f'Feature Importance for Diabetes Prediction in Cluster {TARGET_CLUSTER} (Overall)', 
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = OUTPUT_DIR / f'shap_diabetes_cluster{TARGET_CLUSTER}_bar.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# 7. Save Results
# ============================================================================
print("\n" + "=" * 80)
print("7. Saving Results")
print("=" * 80)

# Save feature importance table
output_csv = OUTPUT_DIR / f'diabetes_shap_cluster{TARGET_CLUSTER}.csv'
results_df.to_csv(output_csv, index=False)
print(f"✓ Saved: {output_csv}")

# Save summary report
report_path = OUTPUT_DIR / f'diabetes_shap_cluster{TARGET_CLUSTER}_report.txt'
with open(report_path, 'w') as f:
    f.write(f"=" * 80 + "\n")
    f.write(f"SHAP Analysis: Diabetes Risk Within Cluster {TARGET_CLUSTER}\n")
    f.write(f"=" * 80 + "\n\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Target: {TARGET} (Diabetes status)\n")
    f.write(f"Cluster: {TARGET_CLUSTER}\n")
    f.write(f"Samples: {len(df_cluster):,}\n\n")
    
    f.write(f"Diabetes Distribution:\n")
    for val, pct in diabetes_dist.items():
        count = df_cluster[TARGET].value_counts()[val]
        label = {0: 'No Diabetes', 1: 'Prediabetes', 2: 'Diabetes'}.get(val, f'Level {val}')
        f.write(f"  {label}: {pct*100:.2f}% ({count:,} samples)\n")
    
    f.write(f"\nModel Performance:\n")
    f.write(f"  Random Forest - Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}\n")
    f.write(f"  XGBoost - Accuracy: {xgb_acc:.4f}, F1: {xgb_f1:.4f}\n\n")
    
    f.write(f"SHAP Feature Importance Rankings:\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("=" * 80 + "\n")
    f.write(f"This analysis shows which features drive diabetes risk WITHIN Cluster {TARGET_CLUSTER}.\n")
    f.write("Higher SHAP values = stronger influence on diabetes prediction.\n")
    f.write("\nKey findings:\n")
    
    top_features = results_df.head(3)
    for idx, row in top_features.iterrows():
        f.write(f"  - {row['Feature']}: RF Rank #{row['RF_Rank']}, XGB Rank #{row['XGB_Rank']}\n")

print(f"✓ Saved: {report_path}")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  • shap_diabetes_cluster{TARGET_CLUSTER}_summary.png - SHAP beeswarm plots")
print(f"  • shap_diabetes_cluster{TARGET_CLUSTER}_bar.png - SHAP bar plots")
print(f"  • diabetes_shap_cluster{TARGET_CLUSTER}.csv - Importance rankings")
print(f"  • diabetes_shap_cluster{TARGET_CLUSTER}_report.txt - Summary report")
