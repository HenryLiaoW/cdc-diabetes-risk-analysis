"""
Model Comparison: FN/FP Rate Analysis
======================================
Author: Henry
Date: 2026-02-23
Description: Compare Logistic Regression, Random Forest, and XGBoost on FN/FP metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("MODEL COMPARISON: FALSE NEGATIVE/POSITIVE RATE ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. Load Metrics from All Models
# ============================================================================
print("\n" + "=" * 80)
print("1. Loading Metrics from All Models")
print("=" * 80)

# Load baseline Logistic Regression metrics
lr_metrics = pd.read_csv('outputs/classification/baseline_lr_metrics.csv')
lr_test = lr_metrics[lr_metrics.columns[2]].values  # Test column

# Load Random Forest metrics
rf_metrics = pd.read_csv('outputs/classification/random_forest_metrics.csv')
rf_test = rf_metrics['Test'].values

# Load XGBoost metrics
xgb_metrics = pd.read_csv('outputs/classification/xgboost_metrics.csv')
xgb_test = xgb_metrics['Test'].values

print("✓ All metrics loaded")

# ============================================================================
# 2. Extract FN/FP Statistics
# ============================================================================
print("\n" + "=" * 80)
print("2. Extracting FN/FP Statistics")
print("=" * 80)

# Metric indices (from saved CSV files)
# Order: Accuracy, Recall, Precision, F1, ROC-AUC, FN_Count, FP_Count, FN_Rate, FP_Rate
fn_idx = 5
fp_idx = 6
fnr_idx = 7
fpr_idx = 8

# Create comparison dataframe
comparison_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'FN': [int(lr_test[fn_idx]), int(rf_test[fn_idx]), int(xgb_test[fn_idx])],
    'FN_Rate': [lr_test[fnr_idx], rf_test[fnr_idx], xgb_test[fnr_idx]],
    'FP': [int(lr_test[fp_idx]), int(rf_test[fp_idx]), int(xgb_test[fp_idx])],
    'FP_Rate': [lr_test[fpr_idx], rf_test[fpr_idx], xgb_test[fpr_idx]]
}

comparison_df = pd.DataFrame(comparison_data)

print("\n表 2 - 漏诊率对比")
print("=" * 80)
print(comparison_df.to_string(index=False))

# ============================================================================
# 3. Calculate Improvements
# ============================================================================
print("\n" + "=" * 80)
print("3. Improvement Analysis vs Logistic Regression")
print("=" * 80)

baseline_fn = comparison_df.loc[0, 'FN']
baseline_fnr = comparison_df.loc[0, 'FN_Rate']
baseline_fp = comparison_df.loc[0, 'FP']
baseline_fpr = comparison_df.loc[0, 'FP_Rate']

print("\n【Random Forest vs Logistic Regression】")
rf_fn_reduction = baseline_fn - comparison_df.loc[1, 'FN']
rf_fnr_reduction = baseline_fnr - comparison_df.loc[1, 'FN_Rate']
rf_fnr_pct_improvement = (rf_fnr_reduction / baseline_fnr) * 100

print(f"  FN Reduction:     {rf_fn_reduction:+d} cases ({rf_fnr_reduction:.2%} absolute, {rf_fnr_pct_improvement:.1f}% relative)")
print(f"  Random Forest reduces false negatives by {rf_fnr_pct_improvement:.1f}% compared to Logistic Regression")

print("\n【XGBoost vs Logistic Regression】")
xgb_fn_reduction = baseline_fn - comparison_df.loc[2, 'FN']
xgb_fnr_reduction = baseline_fnr - comparison_df.loc[2, 'FN_Rate']
xgb_fnr_pct_improvement = (xgb_fnr_reduction / baseline_fnr) * 100

print(f"  FN Reduction:     {xgb_fn_reduction:+d} cases ({xgb_fnr_reduction:.2%} absolute, {xgb_fnr_pct_improvement:.1f}% relative)")
print(f"  XGBoost reduces false negatives by {xgb_fnr_pct_improvement:.1f}% compared to Logistic Regression")

print("\n【Best Model】")
best_fn_idx = comparison_df['FN'].idxmin()
best_model = comparison_df.loc[best_fn_idx, 'Model']
print(f"  Lowest FN:  {best_model} with {comparison_df.loc[best_fn_idx, 'FN']} cases ({comparison_df.loc[best_fn_idx, 'FN_Rate']:.2%})")

# ============================================================================
# 4. Create Formatted Table
# ============================================================================
print("\n" + "=" * 80)
print("4. Formatted Comparison Table")
print("=" * 80)

# Format for better readability
formatted_df = comparison_df.copy()
formatted_df['FN_Rate'] = formatted_df['FN_Rate'].apply(lambda x: f"{x:.2%}")
formatted_df['FP_Rate'] = formatted_df['FP_Rate'].apply(lambda x: f"{x:.2%}")
formatted_df['FN'] = formatted_df['FN'].apply(lambda x: f"{x:,}")
formatted_df['FP'] = formatted_df['FP'].apply(lambda x: f"{x:,}")

print("\n表 2 - 漏诊率对比 (可选但加分)")
print("-" * 80)
print(formatted_df.to_string(index=False))
print("-" * 80)

# ============================================================================
# 5. Save Comparison Table
# ============================================================================
print("\n" + "=" * 80)
print("5. Saving Comparison Table")
print("=" * 80)

# Save numeric version
output_path = 'outputs/classification/model_comparison_fn_fp.csv'
comparison_df.to_csv(output_path, index=False)
print(f"✓ Saved: {output_path}")

# Save formatted version
formatted_path = 'outputs/classification/model_comparison_fn_fp_formatted.csv'
formatted_df.to_csv(formatted_path, index=False)
print(f"✓ Saved: {formatted_path}")

# ============================================================================
# 6. Visualization
# ============================================================================
print("\n" + "=" * 80)
print("6. Creating Visualization")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: FN Rate Comparison
ax1 = axes[0]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars1 = ax1.bar(comparison_df['Model'], comparison_df['FN_Rate'] * 100, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('False Negative Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Missed At-Risk Patients (Lower is Better)', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 35])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.2f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotations
ax1.annotate(f'↓{rf_fnr_pct_improvement:.1f}%', 
             xy=(1, comparison_df.loc[1, 'FN_Rate']*100), 
             xytext=(1, 32),
             ha='center', fontsize=10, color='green', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green', lw=2))

ax1.annotate(f'↓{xgb_fnr_pct_improvement:.1f}%', 
             xy=(2, comparison_df.loc[2, 'FN_Rate']*100), 
             xytext=(2, 32),
             ha='center', fontsize=10, color='green', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Subplot 2: FN Count Comparison
ax2 = axes[1]
bars2 = ax2.bar(comparison_df['Model'], comparison_df['FN'], color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('False Negative Count', fontsize=12, fontweight='bold')
ax2.set_title('Number of Missed Diagnoses (Lower is Better)', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()

viz_path = 'outputs/classification/model_comparison_fn_visualization.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_path}")
plt.close()

# ============================================================================
# 7. Key Insight Statement
# ============================================================================
print("\n" + "=" * 80)
print("7. KEY INSIGHT")
print("=" * 80)

print(f"\n💡 POWERFUL STATEMENT:")
print(f"   'Tree models reduce false negatives by {xgb_fnr_pct_improvement:.1f}% compared to Logistic Regression.'")
print(f"   → This means {xgb_fn_reduction} fewer At-Risk patients are missed by XGBoost.")
print(f"   → Medical Impact: Better early detection saves lives and reduces complications.")

print("\n" + "=" * 80)
print("MODEL COMPARISON COMPLETE")
print("=" * 80)
