"""
Random Forest Classification Model for Diabetes Risk Prediction
=================================================================
Author: Henry
Date: 2026-02-23
Description: Train Random Forest with balanced class weights and SHAP analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("STEP 3: RANDOM FOREST MODEL - ENSEMBLE LEARNING")
print("=" * 80)

# ============================================================================
# 1. Load Prepared Data
# ============================================================================
print("\n" + "=" * 80)
print("1. Loading Prepared Data")
print("=" * 80)

X_train = pd.read_csv('outputs/classification/X_train.csv')
X_test = pd.read_csv('outputs/classification/X_test.csv')
y_train = pd.read_csv('outputs/classification/y_train.csv').values.ravel()
y_test = pd.read_csv('outputs/classification/y_test.csv').values.ravel()

print(f"Training Set: X={X_train.shape}, y={y_train.shape}")
print(f"Test Set: X={X_test.shape}, y={y_test.shape}")
print(f"Features: {list(X_train.columns)}")
print(f"Class Distribution (Train): {np.bincount(y_train)} ({(y_train==1).mean()*100:.1f}% At-Risk)")

# ============================================================================
# 2. Training Random Forest
# ============================================================================
print("\n" + "=" * 80)
print("2. Training Random Forest")
print("=" * 80)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=200,
    min_samples_leaf=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Model Configuration:")
print(f"  n_estimators: {rf_model.n_estimators}")
print(f"  max_depth: {rf_model.max_depth}")
print(f"  min_samples_split: {rf_model.min_samples_split}")
print(f"  min_samples_leaf: {rf_model.min_samples_leaf}")
print(f"  class_weight: {rf_model.class_weight}")
print(f"  random_state: {rf_model.random_state}")

print("\nTraining...")
rf_model.fit(X_train, y_train)
print("✓ Training Complete")

# ============================================================================
# 3. Making Predictions
# ============================================================================
print("\n" + "=" * 80)
print("3. Making Predictions")
print("=" * 80)

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_train_proba = rf_model.predict_proba(X_train)[:, 1]
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

print("✓ Predictions Generated")

# ============================================================================
# 4. Performance Metrics
# ============================================================================
print("\n" + "=" * 80)
print("4. Performance Metrics")
print("=" * 80)

# Calculate metrics for both train and test sets
train_metrics = {
    'Accuracy': accuracy_score(y_train, y_train_pred),
    'Recall': recall_score(y_train, y_train_pred),
    'Precision': precision_score(y_train, y_train_pred),
    'F1': f1_score(y_train, y_train_pred),
    'ROC-AUC': roc_auc_score(y_train, y_train_proba)
}

test_metrics = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'F1': f1_score(y_test, y_test_pred),
    'ROC-AUC': roc_auc_score(y_test, y_test_proba)
}

print("\nTraining Set Performance:")
print("-" * 80)
for metric, value in train_metrics.items():
    print(f"  {metric:12s}: {value:.4f}")

print("\nTest Set Performance:")
print("-" * 80)
for metric, value in test_metrics.items():
    print(f"  {metric:12s}: {value:.4f}")

# ============================================================================
# 5. Confusion Matrix and Medical Metrics
# ============================================================================
print("\n" + "=" * 80)
print("5. Confusion Matrix and Medical Metrics")
print("=" * 80)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print("\nTraining Set Confusion Matrix:")
print(cm_train)
tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
print("\nBreakdown:")
print(f"  True Negatives:  {tn_train:,}")
print(f"  False Positives: {fp_train:,}")
print(f"  False Negatives: {fn_train:,}")
print(f"  True Positives:  {tp_train:,}")

print("\n" + "-" * 80)
print("Test Set Confusion Matrix:")
print(cm_test)
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
# Convert to Python integers
tn_test, fp_test, fn_test, tp_test = int(tn_test), int(fp_test), int(fn_test), int(tp_test)
print("\nBreakdown:")
print(f"  True Negatives:  {tn_test:,}")
print(f"  False Positives: {fp_test:,}")
print(f"  False Negatives: {fn_test:,}")
print(f"  True Positives:  {tp_test:,}")

# Calculate critical medical metrics
fnr_test_value = fn_test / (fn_test + tp_test) if (fn_test + tp_test) > 0 else 0.0
fpr_test_value = fp_test / (fp_test + tn_test) if (fp_test + tn_test) > 0 else 0.0

# Ensure scalar float values
fnr_test = float(fnr_test_value)
fpr_test = float(fpr_test_value)

print(f"\n{'*' * 80}")
print(f"CRITICAL MEDICAL METRICS (Test Set):")
print(f"{'*' * 80}")
print(f"  False Negative Rate (Missed At-Risk): {fnr_test:.2%} ({fn_test}/{fn_test + tp_test})")
print(f"  False Positive Rate (False Alarms):   {fpr_test:.2%} ({fp_test}/{fp_test + tn_test})")
print(f"  → Medical Context: FNR={fnr_test:.2%} means {fnr_test:.0%} of At-Risk patients are missed")

# ============================================================================
# 6. Detailed Classification Report
# ============================================================================
print("\n" + "=" * 80)
print("6. Detailed Classification Report")
print("=" * 80)

print("\nTest Set Classification Report:")
target_names = ['Not At-Risk', 'At-Risk']
print(classification_report(y_test, y_test_pred, target_names=target_names, digits=4))

# ============================================================================
# 7. SHAP Analysis (Feature Importance)
# ============================================================================
print("\n" + "=" * 80)
print("7. SHAP Analysis - Feature Importance")
print("=" * 80)

print("\nCalculating SHAP values (TreeExplainer)...")
print("Note: Using sample of 5000 instances for efficiency")

# Use a sample for SHAP calculation
sample_size = min(5000, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=42)

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_sample)

# For binary classification, shap_values is a list [class_0_shap, class_1_shap]
# We want class 1 (At-Risk) SHAP values
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)

# Ensure it's a proper 1D numpy array
mean_abs_shap = np.array(mean_abs_shap).flatten()

# Verify dimensions match
if len(mean_abs_shap) != len(X_train.columns):
    print(f"Warning: SHAP values length ({len(mean_abs_shap)}) != features length ({len(X_train.columns)})")
    # Truncate or pad if necessary
    if len(mean_abs_shap) > len(X_train.columns):
        mean_abs_shap = mean_abs_shap[:len(X_train.columns)]

shap_importance = pd.DataFrame({
    'Feature': list(X_train.columns),
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("\n✓ SHAP values calculated")
print("\nFeature Importance (SHAP - sorted by mean absolute SHAP value):")
print(shap_importance.to_string(index=False))

# ============================================================================
# 8. ROC Curve Calculation
# ============================================================================
print("\n" + "=" * 80)
print("8. ROC Curve Calculation")
print("=" * 80)

fpr_roc_train, tpr_roc_train, _ = roc_curve(y_train, y_train_proba)
fpr_roc_test, tpr_roc_test, _ = roc_curve(y_test, y_test_proba)
roc_auc_train = auc(fpr_roc_train, tpr_roc_train)
roc_auc_test = auc(fpr_roc_test, tpr_roc_test)

print(f"Train ROC-AUC: {roc_auc_train:.4f}")
print(f"Test ROC-AUC:  {roc_auc_test:.4f}")

# ============================================================================
# 9. Generating Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("9. Generating Visualizations")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# --- Subplot 1: Confusion Matrix (Test Set) ---
ax1 = axes[0, 0]
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'})
ax1.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)

# Add FN/FP annotations
ax1.text(0.5, 1.5, f'FN={fn_test}\nFNR={fnr_test:.2%}', 
         ha='center', va='center', fontsize=11, color='red', fontweight='bold')
ax1.text(1.5, 0.5, f'FP={fp_test}\nFPR={fpr_test:.2%}', 
         ha='center', va='center', fontsize=11, color='orange', fontweight='bold')

# --- Subplot 2: ROC Curve ---
ax2 = axes[0, 1]
ax2.plot(fpr_roc_train, tpr_roc_train, color='blue', lw=2, 
         label=f'Train ROC (AUC = {roc_auc_train:.4f})', alpha=0.6)
ax2.plot(fpr_roc_test, tpr_roc_test, color='red', lw=2, 
         label=f'Test ROC (AUC = {roc_auc_test:.4f})')
ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve - Train vs Test', fontsize=14, fontweight='bold')
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(True, alpha=0.3)

# --- Subplot 3: SHAP Feature Importance Bar Chart ---
ax3 = axes[1, 0]
shap_importance_sorted = shap_importance.sort_values('Mean_Abs_SHAP', ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(shap_importance_sorted)))
ax3.barh(shap_importance_sorted['Feature'], shap_importance_sorted['Mean_Abs_SHAP'], 
         color=colors, edgecolor='black', linewidth=0.5)
ax3.set_xlabel('Mean Absolute SHAP Value', fontsize=12)
ax3.set_title('Feature Importance (SHAP Values)', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# --- Subplot 4: SHAP Summary Plot ---
ax4 = axes[1, 1]
shap.summary_plot(shap_values_class1, X_test_sample, plot_type="dot", 
                  show=False, plot_size=None)
plt.sca(ax4)
ax4.set_title('SHAP Summary Plot (Class: At-Risk)', fontsize=14, fontweight='bold')

plt.tight_layout()

output_path = 'outputs/classification/random_forest_evaluation.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")
plt.close()

# ============================================================================
# 10. Saving Results
# ============================================================================
print("\n" + "=" * 80)
print("10. Saving Results")
print("=" * 80)

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': list(train_metrics.keys()) + ['FN_Count', 'FP_Count', 'FN_Rate', 'FP_Rate'],
    'Train': list(train_metrics.values()) + [fn_train, fp_train, 
                                               fn_train/(fn_train+tp_train), 
                                               fp_train/(fp_train+tn_train)],
    'Test': list(test_metrics.values()) + [fn_test, fp_test, fnr_test, fpr_test]
})
metrics_path = 'outputs/classification/random_forest_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"✓ Saved: {metrics_path}")

# Save SHAP importance
shap_path = 'outputs/classification/random_forest_shap_importance.csv'
shap_importance.to_csv(shap_path, index=False)
print(f"✓ Saved: {shap_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Probability_At_Risk': y_test_proba
})
predictions_path = 'outputs/classification/random_forest_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"✓ Saved: {predictions_path}")

# ============================================================================
# 11. RANDOM FOREST SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("11. RANDOM FOREST SUMMARY")
print("=" * 80)

print("\nModel: Random Forest (class_weight='balanced', n_estimators=300, max_depth=None)")

print(f"\n【Train vs Test Performance - Overfitting Check】")
auc_diff = train_metrics['ROC-AUC'] - test_metrics['ROC-AUC']
overfitting_status = 'Minimal overfitting' if abs(auc_diff) < 0.02 else \
                     'Moderate overfitting' if abs(auc_diff) < 0.10 else \
                     'Significant overfitting - Consider regularization'
print(f"  Train ROC-AUC: {train_metrics['ROC-AUC']:.4f}")
print(f"  Test ROC-AUC:  {test_metrics['ROC-AUC']:.4f}")
print(f"  Difference:    {auc_diff:+.4f} ({overfitting_status})")

if abs(auc_diff) >= 0.10:
    print(f"\n  ⚠️  OVERFITTING DETECTED:")
    print(f"      Train AUC ({train_metrics['ROC-AUC']:.4f}) >> Test AUC ({test_metrics['ROC-AUC']:.4f})")
    print(f"      Model memorizes training patterns but fails to generalize.")
    print(f"      Recommendations: Increase min_samples_leaf, reduce max_depth, or add more data.")

print(f"\n【Test Set Performance】")
print(f"  Accuracy:  {test_metrics['Accuracy']:.4f}")
print(f"  Recall:    {test_metrics['Recall']:.4f}  (Sensitivity - % of At-Risk correctly identified)")
print(f"  Precision: {test_metrics['Precision']:.4f}  (% of predicted At-Risk actually At-Risk)")
print(f"  F1 Score:  {test_metrics['F1']:.4f}  (Harmonic mean of Precision and Recall)")
print(f"  ROC-AUC:   {test_metrics['ROC-AUC']:.4f}  (Area Under ROC Curve)")

print(f"\n【Medical Context - Confusion Matrix Metrics】")
print(f"  False Negatives (FN): {fn_test:,}  → {fnr_test:.2%} of At-Risk patients MISSED")
print(f"  False Positives (FP): {fp_test:,}  → {fpr_test:.2%} of healthy flagged as At-Risk")
print(f"  True Positives (TP):  {tp_test:,}  → Correctly identified At-Risk")
print(f"  True Negatives (TN):  {tn_test:,}  → Correctly identified healthy")

print(f"\n【Feature Importance - Top 3 SHAP Values】")
for idx, row in shap_importance.head(3).iterrows():
    print(f"  {row['Feature']:25s}: {row['Mean_Abs_SHAP']:.4f}")

print("\n" + "=" * 80)
print("RANDOM FOREST MODEL EVALUATION COMPLETE")
print("=" * 80)
