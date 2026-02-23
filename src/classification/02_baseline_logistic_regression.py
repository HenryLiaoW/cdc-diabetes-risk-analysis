"""
Classification Task - Step 2: Baseline Model (Logistic Regression)
Train and evaluate Logistic Regression as baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 2: BASELINE MODEL - LOGISTIC REGRESSION")
print("=" * 80)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'classification'
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. Load Prepared Data
# ============================================================================
print("\n" + "=" * 80)
print("1. Loading Prepared Data")
print("=" * 80)

X_train = pd.read_csv(OUTPUT_DIR / 'X_train.csv')
X_test = pd.read_csv(OUTPUT_DIR / 'X_test.csv')
y_train = pd.read_csv(OUTPUT_DIR / 'y_train.csv').values.ravel()
y_test = pd.read_csv(OUTPUT_DIR / 'y_test.csv').values.ravel()

print(f"Training Set: X={X_train.shape}, y={y_train.shape}")
print(f"Test Set: X={X_test.shape}, y={y_test.shape}")
print(f"Features: {list(X_train.columns)}")
print(f"Class Distribution (Train): {np.bincount(y_train)} ({y_train.mean():.1%} At-Risk)")

# ============================================================================
# 2. Train Logistic Regression with Balanced Weights
# ============================================================================
print("\n" + "=" * 80)
print("2. Training Logistic Regression")
print("=" * 80)

# Use class_weight='balanced' to handle class imbalance
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

print("Model Configuration:")
print(f"  Solver: lbfgs")
print(f"  Class Weight: balanced")
print(f"  Max Iterations: 1000")
print(f"  Random State: 42")

print("\nTraining...")
lr_model.fit(X_train, y_train)
print("✓ Training Complete")

# ============================================================================
# 3. Make Predictions
# ============================================================================
print("\n" + "=" * 80)
print("3. Making Predictions")
print("=" * 80)

# Predictions on training set
y_train_pred = lr_model.predict(X_train)
y_train_proba = lr_model.predict_proba(X_train)[:, 1]

# Predictions on test set
y_test_pred = lr_model.predict(X_test)
y_test_proba = lr_model.predict_proba(X_test)[:, 1]

print("✓ Predictions Generated")

# ============================================================================
# 4. Evaluate Performance Metrics
# ============================================================================
print("\n" + "=" * 80)
print("4. Performance Metrics")
print("=" * 80)

def calculate_metrics(y_true, y_pred, y_proba, dataset_name=""):
    """Calculate and return all metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }
    
    print(f"\n{dataset_name} Set Performance:")
    print("-" * 60)
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    
    return metrics

# Training metrics
train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, "Training")

# Test metrics
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, "Test")

# ============================================================================
# 5. Confusion Matrix
# ============================================================================
print("\n" + "=" * 80)
print("5. Confusion Matrix")
print("=" * 80)

# Calculate confusion matrices
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print("\nTraining Set Confusion Matrix:")
print(cm_train)
print(f"\nBreakdown:")
print(f"  True Negatives:  {cm_train[0,0]:,}")
print(f"  False Positives: {cm_train[0,1]:,}")
print(f"  False Negatives: {cm_train[1,0]:,}")
print(f"  True Positives:  {cm_train[1,1]:,}")

print("\n" + "-" * 60)
print("Test Set Confusion Matrix:")
print(cm_test)
print(f"\nBreakdown:")
print(f"  True Negatives:  {cm_test[0,0]:,}")
print(f"  False Positives: {cm_test[0,1]:,}")
print(f"  False Negatives: {cm_test[1,0]:,}")
print(f"  True Positives:  {cm_test[1,1]:,}")

# Calculate critical medical metrics
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
fnr_test = fn_test / (fn_test + tp_test)  # False Negative Rate (Miss Rate)
fpr_test = fp_test / (fp_test + tn_test)  # False Positive Rate (False Alarm Rate)

print(f"\nCritical Medical Metrics (Test Set):")
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
print(classification_report(y_test, y_test_pred, 
                          target_names=['Not At-Risk', 'At-Risk'],
                          digits=4))

# ============================================================================
# 7. Feature Coefficients
# ============================================================================
print("\n" + "=" * 80)
print("7. Feature Importance (Logistic Regression Coefficients)")
print("=" * 80)

coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Coefficients (sorted by absolute value):")
print(coefficients.to_string(index=False))

# ============================================================================
# 8. Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("8. Generating Visualizations")
print("=" * 80)

# Create figure with 3 subplots
fig = plt.figure(figsize=(18, 5))

# Subplot 1: Confusion Matrix (Test Set)
ax1 = plt.subplot(1, 3, 1)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not At-Risk', 'At-Risk'],
            yticklabels=['Not At-Risk', 'At-Risk'],
            cbar_kws={'label': 'Count'})
ax1.set_xlabel('Predicted', fontweight='bold', fontsize=12)
ax1.set_ylabel('Actual', fontweight='bold', fontsize=12)
ax1.set_title('Confusion Matrix (Test Set)', fontweight='bold', fontsize=14, pad=15)

# Subplot 2: ROC Curve
ax2 = plt.subplot(1, 3, 2)
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
ax2.plot(fpr, tpr, color='darkorange', lw=2.5, 
         label=f'ROC Curve (AUC = {test_metrics["ROC-AUC"]:.4f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random Classifier')
ax2.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
ax2.set_title('ROC Curve (Test Set)', fontweight='bold', fontsize=14, pad=15)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(alpha=0.3, linestyle=':')

# Subplot 3: Feature Coefficients
ax3 = plt.subplot(1, 3, 3)
colors = ['green' if c > 0 else 'red' for c in coefficients['Coefficient']]
bars = ax3.barh(range(len(coefficients)), coefficients['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(coefficients)))
ax3.set_yticklabels(coefficients['Feature'], fontsize=10)
ax3.set_xlabel('Coefficient Value', fontweight='bold', fontsize=12)
ax3.set_title('Logistic Regression Coefficients', fontweight='bold', fontsize=14, pad=15)
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax3.grid(axis='x', alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'baseline_logistic_regression_evaluation.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {OUTPUT_DIR / 'baseline_logistic_regression_evaluation.png'}")
plt.close()

# ============================================================================
# 9. Save Results
# ============================================================================
print("\n" + "=" * 80)
print("9. Saving Results")
print("=" * 80)

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1', 'ROC-AUC'],
    'Train': [train_metrics[m] for m in ['Accuracy', 'Recall', 'Precision', 'F1', 'ROC-AUC']],
    'Test': [test_metrics[m] for m in ['Accuracy', 'Recall', 'Precision', 'F1', 'ROC-AUC']]
})
metrics_df.to_csv(OUTPUT_DIR / 'baseline_lr_metrics.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'baseline_lr_metrics.csv'}")

# Save feature coefficients
coefficients.to_csv(OUTPUT_DIR / 'baseline_lr_coefficients.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'baseline_lr_coefficients.csv'}")

# Save predictions
predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_test_pred,
    'y_proba': y_test_proba
})
predictions_df.to_csv(OUTPUT_DIR / 'baseline_lr_predictions.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'baseline_lr_predictions.csv'}")

# ============================================================================
# 10. Summary
# ============================================================================
print("\n" + "=" * 80)
print("10. BASELINE MODEL SUMMARY")
print("=" * 80)

print("\nModel: Logistic Regression (class_weight='balanced')")

print(f"\n【Train vs Test Performance - Overfitting Check】")
print(f"  Train ROC-AUC: {train_metrics['ROC-AUC']:.4f}")
print(f"  Test ROC-AUC:  {test_metrics['ROC-AUC']:.4f}")
print(f"  Difference:    {train_metrics['ROC-AUC'] - test_metrics['ROC-AUC']:+.4f} ({'Minimal overfitting' if abs(train_metrics['ROC-AUC'] - test_metrics['ROC-AUC']) < 0.02 else 'Check overfitting'})")

print(f"\n【Test Set Performance】")
print(f"  Accuracy:  {test_metrics['Accuracy']:.4f}")
print(f"  Recall:    {test_metrics['Recall']:.4f}  (Sensitivity - % of At-Risk correctly identified)")
print(f"  Precision: {test_metrics['Precision']:.4f}  (% of predicted At-Risk actually At-Risk)")
print(f"  F1 Score:  {test_metrics['F1']:.4f}  (Harmonic mean of Precision and Recall)")
print(f"  ROC-AUC:   {test_metrics['ROC-AUC']:.4f}  (Area Under ROC Curve)")

# Recalculate FN/FP rates for summary
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
fnr_test = fn_test / (fn_test + tp_test)
fpr_test = fp_test / (fp_test + tn_test)

print(f"\n【Medical Context - Confusion Matrix Metrics】")
print(f"  False Negatives (FN): {fn_test:,}  → {fnr_test:.2%} of At-Risk patients MISSED")
print(f"  False Positives (FP): {fp_test:,}  → {fpr_test:.2%} of healthy flagged as At-Risk")
print(f"  True Positives (TP):  {tp_test:,}  → Correctly identified At-Risk")
print(f"  True Negatives (TN):  {tn_test:,}  → Correctly identified healthy")

print(f"\n【Feature Importance - Top 3 Coefficients】")
for i, row in coefficients.head(3).iterrows():
    print(f"  {row['Feature']:25s}: {row['Coefficient']:+.4f}")

print("\n" + "=" * 80)
print("BASELINE MODEL EVALUATION COMPLETE")
print("=" * 80)
