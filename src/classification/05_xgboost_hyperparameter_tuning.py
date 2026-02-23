"""
Step 4.5: XGBoost Hyperparameter Optimization
==============================================
Optimize XGBoost to minimize False Negative Rate using GridSearchCV.

Goal: Improve recall (reduce FNR) while maintaining generalization.

Current Baseline:
- n_estimators: 300
- max_depth: 4
- learning_rate: 0.05
- scale_pos_weight: 5.35
- Test ROC-AUC: ~0.7961
- Test FNR: ~24.15%

Author: AI Assistant
Date: 2026-02-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. Load Prepared Data
# ============================================================================
print("=" * 80)
print("STEP 4.5: XGBoost Hyperparameter Optimization")
print("=" * 80)
print("\n" + "=" * 80)
print("1. Loading Prepared Data")
print("=" * 80)

# Load training and test data
X_train = pd.read_csv('outputs/classification/X_train.csv')
X_test = pd.read_csv('outputs/classification/X_test.csv')
y_train = pd.read_csv('outputs/classification/y_train.csv').values.ravel()
y_test = pd.read_csv('outputs/classification/y_test.csv').values.ravel()

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")
print(f"Features:     {list(X_train.columns)}")
print(f"\nClass distribution (train):")
print(f"  Not At-Risk (0): {(y_train == 0).sum():,} ({(y_train == 0).mean():.1%})")
print(f"  At-Risk (1):     {(y_train == 1).sum():,} ({(y_train == 1).mean():.1%})")

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio: 1:{scale_pos_weight:.2f}")
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# ============================================================================
# 2. Load Baseline XGBoost Results
# ============================================================================
print("\n" + "=" * 80)
print("2. Loading Baseline XGBoost Performance")
print("=" * 80)

baseline_metrics = pd.read_csv('outputs/classification/xgboost_metrics.csv')
print("\nBaseline XGBoost (previous training):")
print(baseline_metrics.to_string(index=False))

# Extract baseline values for comparison
baseline_recall = baseline_metrics.loc[baseline_metrics['Metric'] == 'Recall', 'Test'].values[0]
baseline_fnr = baseline_metrics.loc[baseline_metrics['Metric'] == 'FN_Rate', 'Test'].values[0]
baseline_roc_auc = baseline_metrics.loc[baseline_metrics['Metric'] == 'ROC-AUC', 'Test'].values[0]

print(f"\n📌 Baseline Key Metrics:")
print(f"   Test Recall:  {baseline_recall:.4f}")
print(f"   Test FNR:     {baseline_fnr:.2%}")
print(f"   Test ROC-AUC: {baseline_roc_auc:.4f}")

# ============================================================================
# 3. Define Hyperparameter Search Space
# ============================================================================
print("\n" + "=" * 80)
print("3. Defining Hyperparameter Search Space")
print("=" * 80)

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05, 0.07],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

print("\nSearch Space:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\n📊 Total combinations: {total_combinations}")
print(f"📊 CV folds: 5")
print(f"📊 Total model fits: {total_combinations * 5}")

# Fixed parameters
fixed_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'n_jobs': -1
}

print("\n🔒 Fixed Parameters:")
for param, value in fixed_params.items():
    print(f"  {param}: {value}")

# ============================================================================
# 4. GridSearchCV with Stratified K-Fold
# ============================================================================
print("\n" + "=" * 80)
print("4. Running GridSearchCV (This may take several minutes...)")
print("=" * 80)

# Base estimator
base_xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

# Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV with recall as primary scoring
grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    scoring='recall',  # Primary objective: minimize FNR
    cv=cv,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

print("\n⏳ Training in progress...")
print("   Primary scoring: recall (maximize to minimize FNR)")
print("   Cross-validation: 5-fold stratified\n")

grid_search.fit(X_train, y_train)

print("\n✓ GridSearchCV Complete!")

# ============================================================================
# 5. Extract Best Parameters and CV Results
# ============================================================================
print("\n" + "=" * 80)
print("5. Best Parameters and Cross-Validation Results")
print("=" * 80)

best_params = grid_search.best_params_
best_cv_recall = grid_search.best_score_

print("\n🏆 Best Hyperparameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n📊 Best CV Recall (mean): {best_cv_recall:.4f}")
print(f"   (CV FNR ≈ {(1 - best_cv_recall):.2%})")

# Get best model
best_model = grid_search.best_estimator_

# ============================================================================
# 6. Evaluate Tuned Model on Test Set
# ============================================================================
print("\n" + "=" * 80)
print("6. Evaluating Tuned Model on Test Set")
print("=" * 80)

# Predictions
y_pred_tuned = best_model.predict(X_test)
y_pred_proba_tuned = best_model.predict_proba(X_test)[:, 1]

# Also get train predictions for AUC gap analysis
y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]

# Confusion matrix
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
tn, fp, fn, tp = cm_tuned.ravel()

# Calculate metrics
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)
roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)
roc_auc_train = roc_auc_score(y_train, y_train_pred_proba)

# False Negative Rate and False Positive Rate
fnr_tuned = fn / (fn + tp)
fpr_tuned = fp / (fp + tn)

# AUC gap
auc_gap = roc_auc_train - roc_auc_tuned

print("\n📊 Tuned XGBoost Test Performance:")
print(f"  Accuracy:        {accuracy_tuned:.4f}")
print(f"  Recall:          {recall_tuned:.4f}")
print(f"  Precision:       {precision_tuned:.4f}")
print(f"  F1-Score:        {f1_tuned:.4f}")
print(f"  ROC-AUC (test):  {roc_auc_tuned:.4f}")
print(f"  ROC-AUC (train): {roc_auc_train:.4f}")
print(f"  AUC Gap:         {auc_gap:.4f}")

print(f"\n📊 Medical Priority Metrics:")
print(f"  False Negatives (FN):  {int(fn):,}")
print(f"  False Negative Rate:   {fnr_tuned:.2%}")
print(f"  False Positives (FP):  {int(fp):,}")
print(f"  False Positive Rate:   {fpr_tuned:.2%}")

print(f"\n📊 Confusion Matrix:")
print(f"  TN: {int(tn):,}  |  FP: {int(fp):,}")
print(f"  FN: {int(fn):,}  |  TP: {int(tp):,}")

# ============================================================================
# 7. Compare with Baseline XGBoost
# ============================================================================
print("\n" + "=" * 80)
print("7. Comparison: Baseline vs Tuned XGBoost")
print("=" * 80)

# Calculate improvements
fnr_absolute_improvement = baseline_fnr - fnr_tuned
fnr_percent_improvement = (fnr_absolute_improvement / baseline_fnr) * 100
roc_auc_gain = roc_auc_tuned - baseline_roc_auc
recall_gain = recall_tuned - baseline_recall

print(f"\n📈 Improvements:")
print(f"  FNR Reduction:")
print(f"    Baseline FNR: {baseline_fnr:.2%}")
print(f"    Tuned FNR:    {fnr_tuned:.2%}")
print(f"    Absolute:     {fnr_absolute_improvement:.2%}")
print(f"    Relative:     {fnr_percent_improvement:.2f}%")

print(f"\n  Recall Improvement:")
print(f"    Baseline:     {baseline_recall:.4f}")
print(f"    Tuned:        {recall_tuned:.4f}")
print(f"    Gain:         {recall_gain:+.4f}")

print(f"\n  ROC-AUC:")
print(f"    Baseline:     {baseline_roc_auc:.4f}")
print(f"    Tuned:        {roc_auc_tuned:.4f}")
print(f"    Gain:         {roc_auc_gain:+.4f}")

# Create comparison table
comparison_data = {
    'Model': ['XGBoost Baseline', 'XGBoost Tuned'],
    'Recall': [baseline_recall, recall_tuned],
    'FNR': [baseline_fnr, fnr_tuned],
    'ROC_AUC': [baseline_roc_auc, roc_auc_tuned],
    'n_estimators': [300, best_params['n_estimators']],
    'max_depth': [4, best_params['max_depth']],
    'learning_rate': [0.05, best_params['learning_rate']],
    'subsample': [0.8, best_params['subsample']],
    'colsample_bytree': [0.8, best_params['colsample_bytree']]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + "=" * 80)
print("Comparison Table: XGBoost Baseline vs Tuned")
print("=" * 80)
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv('outputs/classification/xgboost_tuning_comparison.csv', index=False)
print("\n✓ Saved: outputs/classification/xgboost_tuning_comparison.csv")

# ============================================================================
# 8. Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("8. Generating Visualizations")
print("=" * 80)

# ========== Visualization 1: FNR Comparison Bar Chart ==========
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Baseline\nXGBoost', 'Tuned\nXGBoost']
fnr_values = [baseline_fnr * 100, fnr_tuned * 100]
colors = ['#e74c3c', '#27ae60']

bars = ax.bar(models, fnr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, fnr_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{val:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add improvement annotation
if fnr_absolute_improvement > 0:
    ax.annotate('', xy=(1, fnr_values[1]), xytext=(1, fnr_values[0]),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(1.15, (fnr_values[0] + fnr_values[1])/2,
            f'↓ {fnr_absolute_improvement*100:.2f}%\n({fnr_percent_improvement:.1f}% reduction)',
            ha='left', va='center', fontsize=11, color='blue', fontweight='bold')

ax.set_ylabel('False Negative Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('XGBoost Hyperparameter Tuning Impact on FNR\n(Lower is Better)',
             fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(0, max(fnr_values) * 1.3)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/classification/xgboost_fnr_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/classification/xgboost_fnr_comparison.png")
plt.close()

# ========== Visualization 2: ROC Curve (Tuned Model) ==========
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate ROC curve
fpr_roc, tpr_roc, thresholds = roc_curve(y_test, y_pred_proba_tuned)
roc_auc_value = auc(fpr_roc, tpr_roc)

# Plot ROC curve
ax.plot(fpr_roc, tpr_roc, color='#2ecc71', lw=3, 
        label=f'Tuned XGBoost (AUC = {roc_auc_value:.4f})')

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')

# Mark operating point
operating_fpr = fpr_tuned
operating_tpr = recall_tuned
ax.plot(operating_fpr, operating_tpr, 'ro', markersize=12, 
        label=f'Operating Point (TPR={operating_tpr:.3f}, FPR={operating_fpr:.3f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate (Recall)', fontsize=13, fontweight='bold')
ax.set_title('ROC Curve: Tuned XGBoost Model', fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3, linestyle='--')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('outputs/classification/xgboost_tuned_roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/classification/xgboost_tuned_roc_curve.png")
plt.close()

# ========== Visualization 3: Metrics Comparison Heatmap ==========
fig, ax = plt.subplots(figsize=(10, 5))

metrics_comparison = pd.DataFrame({
    'Baseline': [baseline_recall, baseline_roc_auc, 1 - baseline_fnr],
    'Tuned': [recall_tuned, roc_auc_tuned, 1 - fnr_tuned]
}, index=['Recall', 'ROC-AUC', 'Sensitivity (1-FNR)'])

sns.heatmap(metrics_comparison, annot=True, fmt='.4f', cmap='RdYlGn', 
            cbar_kws={'label': 'Score'}, linewidths=2, linecolor='black',
            vmin=0.7, vmax=0.85, ax=ax, annot_kws={'fontsize': 12, 'fontweight': 'bold'})

ax.set_title('XGBoost Performance Metrics: Baseline vs Tuned\n(Higher is Better)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Model Version', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/classification/xgboost_metrics_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/classification/xgboost_metrics_heatmap.png")
plt.close()

# ============================================================================
# 9. Save Tuned Model Results
# ============================================================================
print("\n" + "=" * 80)
print("9. Saving Tuned Model Results")
print("=" * 80)

# Save tuned metrics
tuned_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1_Score', 'ROC_AUC', 
               'FN', 'FN_Rate', 'FP', 'FP_Rate', 'AUC_Gap'],
    'Train': ['-', '-', '-', '-', roc_auc_train, '-', '-', '-', '-', '-'],
    'Test': [accuracy_tuned, recall_tuned, precision_tuned, f1_tuned, 
             roc_auc_tuned, int(fn), fnr_tuned, int(fp), fpr_tuned, auc_gap]
})

tuned_metrics.to_csv('outputs/classification/xgboost_tuned_metrics.csv', index=False)
print("✓ Saved: outputs/classification/xgboost_tuned_metrics.csv")

# Save predictions
tuned_predictions = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred_tuned,
    'Predicted_Probability': y_pred_proba_tuned
})
tuned_predictions.to_csv('outputs/classification/xgboost_tuned_predictions.csv', index=False)
print("✓ Saved: outputs/classification/xgboost_tuned_predictions.csv")

# Save best parameters
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv('outputs/classification/xgboost_best_params.csv', index=False)
print("✓ Saved: outputs/classification/xgboost_best_params.csv")

# ============================================================================
# 10. Final Summary and Recommendations
# ============================================================================
print("\n" + "=" * 80)
print("10. FINAL SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print("\n🎯 Optimization Objective: Minimize False Negative Rate")
print("   (Medical Priority: Catch more at-risk patients)")

print(f"\n📊 Results:")
print(f"   ✓ Best CV Recall: {best_cv_recall:.4f}")
print(f"   ✓ Test Recall:    {recall_tuned:.4f}")
print(f"   ✓ Test FNR:       {fnr_tuned:.2%}")
print(f"   ✓ Test ROC-AUC:   {roc_auc_tuned:.4f}")

print(f"\n📈 Improvement vs Baseline:")
print(f"   • FNR reduced by:    {fnr_absolute_improvement:.2%} (absolute)")
print(f"   • FNR reduction:     {fnr_percent_improvement:.2f}% (relative)")
print(f"   • Recall improved:   {recall_gain:+.4f}")
print(f"   • ROC-AUC improved:  {roc_auc_gain:+.4f}")

print(f"\n🔍 Medical Interpretation:")
if fnr_absolute_improvement > 0.01:  # More than 1% FNR reduction
    fn_baseline = int(baseline_fnr * len(y_test))
    fn_reduction = fn_baseline - int(fn)
    print(f"   ✅ MEANINGFUL IMPROVEMENT")
    print(f"   • Approximately {fn_reduction} fewer missed diagnoses")
    print(f"   • This represents {fn_reduction / fn_baseline * 100:.1f}% fewer false negatives")
    print(f"   • Medical benefit: Significant - more at-risk patients identified")
elif fnr_absolute_improvement > 0:
    print(f"   ⚠️  MARGINAL IMPROVEMENT")
    print(f"   • Small reduction in missed diagnoses")
    print(f"   • Medical benefit: Limited - minor improvement in sensitivity")
else:
    print(f"   ❌ NO IMPROVEMENT")
    print(f"   • FNR did not improve")
    print(f"   • Baseline model may already be optimal")

print(f"\n📊 Generalization Assessment:")
if auc_gap < 0.05:
    print(f"   ✅ Excellent generalization (AUC gap = {auc_gap:.4f})")
    print(f"   • Model is not overfitting")
elif auc_gap < 0.10:
    print(f"   ✓ Good generalization (AUC gap = {auc_gap:.4f})")
    print(f"   • Acceptable train-test performance gap")
else:
    print(f"   ⚠️  Possible overfitting (AUC gap = {auc_gap:.4f})")
    print(f"   • Consider regularization or simpler model")

print(f"\n🏆 Best Hyperparameters:")
for param, value in best_params.items():
    baseline_value = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }.get(param, 'N/A')
    
    if value != baseline_value:
        print(f"   • {param}: {value} (changed from {baseline_value})")
    else:
        print(f"   • {param}: {value} (unchanged)")

print(f"\n💡 Recommendation:")
if fnr_absolute_improvement > 0.01 and auc_gap < 0.10:
    print("   ✅ ADOPT TUNED MODEL")
    print("   • Meaningful FNR reduction achieved")
    print("   • Good generalization maintained")
    print("   • Medical benefit justifies the complexity")
elif fnr_absolute_improvement > 0:
    print("   ⚠️  CONSIDER TUNED MODEL")
    print("   • Marginal improvement in recall")
    print("   • Evaluate if complexity increase is justified")
    print("   • May be useful in high-stakes medical scenarios")
else:
    print("   ❌ KEEP BASELINE MODEL")
    print("   • No improvement in FNR")
    print("   • Additional complexity not justified")
    print("   • Baseline hyperparameters are already optimal")

print("\n" + "=" * 80)
print("XGBOOST HYPERPARAMETER OPTIMIZATION COMPLETE")
print("=" * 80)
print("\n✓ All results saved to outputs/classification/")
print("✓ Visualizations generated")
print("✓ Comparison table created")
