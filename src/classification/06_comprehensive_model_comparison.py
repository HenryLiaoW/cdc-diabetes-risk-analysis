"""
Comprehensive Model Comparison
===============================
Author: Henry
Date: 2026-02-23
Description: Complete comparison table with all evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("COMPREHENSIVE MODEL COMPARISON - ALL METRICS")
print("=" * 80)

# ============================================================================
# 1. Metrics from All Models (Test Set)
# ============================================================================
print("\n" + "=" * 80)
print("1. Collecting Metrics from All Models")
print("=" * 80)

# Complete metrics extracted from model outputs
model_metrics = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [0.7298, 0.7063, 0.7020],
    'Recall': [0.7079, 0.7520, 0.7585],
    'Precision': [0.3323, 0.3176, 0.3150],
    'F1': [0.4523, 0.4466, 0.4451],
    'ROC-AUC': [0.7917, 0.7949, 0.7961],
    'FN': [2335, 1983, 1931],
    'FP': [11372, 12916, 13188],
    'FN_Rate': [0.2921, 0.2480, 0.2415],
    'FP_Rate': [0.2661, 0.3022, 0.3086]
}

metrics_df = pd.DataFrame(model_metrics)

print("\n✓ Metrics collected")

# ============================================================================
# 2. Display Full Comparison Table
# ============================================================================
print("\n" + "=" * 80)
print("2. Complete Model Evaluation Metrics (Test Set)")
print("=" * 80)

print("\n【Standard Classification Metrics】")
print("-" * 80)
standard_metrics = metrics_df[['Model', 'Accuracy', 'Recall', 'Precision', 'F1', 'ROC-AUC']].copy()
print(standard_metrics.to_string(index=False))

print("\n【Medical Context Metrics (Critical for Healthcare)】")
print("-" * 80)
medical_metrics = metrics_df[['Model', 'FN', 'FN_Rate', 'FP', 'FP_Rate']].copy()
medical_metrics['FN_Rate'] = medical_metrics['FN_Rate'].apply(lambda x: f"{x:.2%}")
medical_metrics['FP_Rate'] = medical_metrics['FP_Rate'].apply(lambda x: f"{x:.2%}")
print(medical_metrics.to_string(index=False))

# ============================================================================
# 3. Model Ranking by Different Criteria
# ============================================================================
print("\n" + "=" * 80)
print("3. Model Ranking by Different Criteria")
print("=" * 80)

print("\n【By ROC-AUC (Overall Discrimination Power)】")
ranked_auc = metrics_df.sort_values('ROC-AUC', ascending=False)[['Model', 'ROC-AUC']]
for i, (idx, row) in enumerate(ranked_auc.iterrows(), 1):
    star = "⭐" if i == 1 else ""
    print(f"  {i}. {row['Model']:25s} ROC-AUC = {row['ROC-AUC']:.4f} {star}")

print("\n【By Recall (Sensitivity - Catch At-Risk Patients)】")
ranked_recall = metrics_df.sort_values('Recall', ascending=False)[['Model', 'Recall']]
for i, (idx, row) in enumerate(ranked_recall.iterrows(), 1):
    star = "⭐" if i == 1 else ""
    print(f"  {i}. {row['Model']:25s} Recall = {row['Recall']:.4f} {star}")

print("\n【By FN Rate (Minimize Missed Diagnoses)】")
ranked_fnr = metrics_df.sort_values('FN_Rate', ascending=True)[['Model', 'FN_Rate']]
for i, (idx, row) in enumerate(ranked_fnr.iterrows(), 1):
    star = "⭐" if i == 1 else ""
    print(f"  {i}. {row['Model']:25s} FNR = {row['FN_Rate']:.2%} {star}")

print("\n【By Precision (Reduce False Alarms)】")
ranked_precision = metrics_df.sort_values('Precision', ascending=False)[['Model', 'Precision']]
for i, (idx, row) in enumerate(ranked_precision.iterrows(), 1):
    star = "⭐" if i == 1 else ""
    print(f"  {i}. {row['Model']:25s} Precision = {row['Precision']:.4f} {star}")

# ============================================================================
# 4. Improvement Analysis
# ============================================================================
print("\n" + "=" * 80)
print("4. Improvement vs Logistic Regression (Baseline)")
print("=" * 80)

baseline_idx = 0  # Logistic Regression

for idx in [1, 2]:  # Random Forest and XGBoost
    model_name = metrics_df.loc[idx, 'Model']
    
    print(f"\n【{model_name} vs Logistic Regression】")
    
    # ROC-AUC improvement
    auc_diff = metrics_df.loc[idx, 'ROC-AUC'] - metrics_df.loc[baseline_idx, 'ROC-AUC']
    auc_pct = (auc_diff / metrics_df.loc[baseline_idx, 'ROC-AUC']) * 100
    print(f"  ROC-AUC:   {metrics_df.loc[idx, 'ROC-AUC']:.4f} ({auc_diff:+.4f}, {auc_pct:+.2f}%)")
    
    # Recall improvement
    recall_diff = metrics_df.loc[idx, 'Recall'] - metrics_df.loc[baseline_idx, 'Recall']
    recall_pct = (recall_diff / metrics_df.loc[baseline_idx, 'Recall']) * 100
    print(f"  Recall:    {metrics_df.loc[idx, 'Recall']:.4f} ({recall_diff:+.4f}, {recall_pct:+.2f}%)")
    
    # FN reduction
    fn_diff = metrics_df.loc[baseline_idx, 'FN'] - metrics_df.loc[idx, 'FN']
    fnr_diff = metrics_df.loc[baseline_idx, 'FN_Rate'] - metrics_df.loc[idx, 'FN_Rate']
    fnr_pct = (fnr_diff / metrics_df.loc[baseline_idx, 'FN_Rate']) * 100
    print(f"  FN:        {metrics_df.loc[idx, 'FN']} ({fn_diff:+d} cases fewer, {fnr_pct:+.1f}% FNR reduction)")
    
    # Precision change
    prec_diff = metrics_df.loc[idx, 'Precision'] - metrics_df.loc[baseline_idx, 'Precision']
    prec_pct = (prec_diff / metrics_df.loc[baseline_idx, 'Precision']) * 100
    print(f"  Precision: {metrics_df.loc[idx, 'Precision']:.4f} ({prec_diff:+.4f}, {prec_pct:+.2f}%)")

# ============================================================================
# 5. Save Complete Comparison Table
# ============================================================================
print("\n" + "=" * 80)
print("5. Saving Complete Comparison Tables")
print("=" * 80)

# Save all metrics to CSV
output_path = 'outputs/classification/model_comparison_complete.csv'
metrics_df.to_csv(output_path, index=False)
print(f"\n✓ Saved: {output_path}")

# Create formatted version for presentation
formatted_df = metrics_df.copy()
formatted_df['Accuracy'] = formatted_df['Accuracy'].apply(lambda x: f"{x:.4f}")
formatted_df['Recall'] = formatted_df['Recall'].apply(lambda x: f"{x:.4f}")
formatted_df['Precision'] = formatted_df['Precision'].apply(lambda x: f"{x:.4f}")
formatted_df['F1'] = formatted_df['F1'].apply(lambda x: f"{x:.4f}")
formatted_df['ROC-AUC'] = formatted_df['ROC-AUC'].apply(lambda x: f"{x:.4f}")
formatted_df['FN'] = formatted_df['FN'].apply(lambda x: f"{x:,}")
formatted_df['FP'] = formatted_df['FP'].apply(lambda x: f"{x:,}")
formatted_df['FN_Rate'] = formatted_df['FN_Rate'].apply(lambda x: f"{x:.2%}")
formatted_df['FP_Rate'] = formatted_df['FP_Rate'].apply(lambda x: f"{x:.2%}")

formatted_path = 'outputs/classification/model_comparison_complete_formatted.csv'
formatted_df.to_csv(formatted_path, index=False)
print(f"✓ Saved: {formatted_path}")

# ============================================================================
# 6. Visualization: Multi-Metric Comparison
# ============================================================================
print("\n" + "=" * 80)
print("6. Creating Comprehensive Visualization")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

models = metrics_df['Model'].values
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# Plot 1: Accuracy
ax1 = axes[0, 0]
bars = ax1.bar(models, metrics_df['Accuracy'], color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0.65, 0.75])
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Recall
ax2 = axes[0, 1]
bars = ax2.bar(models, metrics_df['Recall'], color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax2.set_ylabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
ax2.set_title('Recall Comparison (Higher = Better)', fontsize=14, fontweight='bold')
ax2.set_ylim([0.65, 0.80])
ax2.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Precision
ax3 = axes[0, 2]
bars = ax3.bar(models, metrics_df['Precision'], color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax3.set_title('Precision Comparison', fontsize=14, fontweight='bold')
ax3.set_ylim([0.30, 0.35])
ax3.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: F1 Score
ax4 = axes[1, 0]
bars = ax4.bar(models, metrics_df['F1'], color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax4.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax4.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
ax4.set_ylim([0.42, 0.47])
ax4.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 5: ROC-AUC
ax5 = axes[1, 1]
bars = ax5.bar(models, metrics_df['ROC-AUC'], color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax5.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
ax5.set_title('ROC-AUC Comparison (Higher = Better)', fontsize=14, fontweight='bold')
ax5.set_ylim([0.78, 0.80])
ax5.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 6: False Negative Rate
ax6 = axes[1, 2]
bars = ax6.bar(models, metrics_df['FN_Rate'] * 100, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax6.set_ylabel('False Negative Rate (%)', fontsize=12, fontweight='bold')
ax6.set_title('FN Rate Comparison (Lower = Better)', fontsize=14, fontweight='bold')
ax6.set_ylim([20, 32])
ax6.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add best model indicators
for ax in axes.flat:
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()

viz_path = 'outputs/classification/model_comparison_all_metrics.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {viz_path}")
plt.close()

# ============================================================================
# 7. Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("7. MODEL SELECTION SUMMARY")
print("=" * 80)

best_auc_idx = metrics_df['ROC-AUC'].idxmax()
best_recall_idx = metrics_df['Recall'].idxmax()
best_fnr_idx = metrics_df['FN_Rate'].idxmin()
best_precision_idx = metrics_df['Precision'].idxmax()

print("\n【Best Models by Criterion】")
print(f"  Best ROC-AUC:   {metrics_df.loc[best_auc_idx, 'Model']} ({metrics_df.loc[best_auc_idx, 'ROC-AUC']:.4f})")
print(f"  Best Recall:    {metrics_df.loc[best_recall_idx, 'Model']} ({metrics_df.loc[best_recall_idx, 'Recall']:.4f})")
print(f"  Best FN Rate:   {metrics_df.loc[best_fnr_idx, 'Model']} ({metrics_df.loc[best_fnr_idx, 'FN_Rate']:.2%})")
print(f"  Best Precision: {metrics_df.loc[best_precision_idx, 'Model']} ({metrics_df.loc[best_precision_idx, 'Precision']:.4f})")

print("\n【Recommended Model】")
print("  🏆 XGBoost")
print("     • Highest ROC-AUC (0.7961) - Best overall discrimination")
print("     • Highest Recall (0.7585) - Best at catching At-Risk patients")
print("     • Lowest FN Rate (24.15%) - Fewest missed diagnoses")
print("     • 404 fewer missed diagnoses vs Logistic Regression (17.3% FNR reduction)")

print("\n【Trade-offs】")
print("  Logistic Regression:")
print("    ✓ Best Precision (33.23%) - Fewer false alarms")
print("    ✓ Simplest model - Easy to interpret and deploy")
print("    ✗ Highest FN Rate (29.21%) - Most missed diagnoses")

print("\n  Random Forest:")
print("    ✓ Balanced performance")
print("    ✓ Good FN Rate (24.80%)")
print("    ~ Middle ground between LR and XGBoost")

print("\n  XGBoost:")
print("    ✓ Best overall performance (ROC-AUC, Recall, FNR)")
print("    ✓ Best for medical screening (minimize false negatives)")
print("    ✗ Slightly lower precision (but acceptable trade-off)")

print("\n" + "=" * 80)
print("COMPREHENSIVE MODEL COMPARISON COMPLETE")
print("=" * 80)
