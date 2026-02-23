"""
Export SHAP Summary Plot (Standalone)
======================================
Author: Henry
Date: 2026-02-23
Description: Generate standalone SHAP Summary Plot for best model (XGBoost)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EXPORTING SHAP SUMMARY PLOT (Class: At-Risk)")
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

# Calculate scale_pos_weight
n_negative = np.sum(y_train == 0)
n_positive = np.sum(y_train == 1)
scale_pos_weight = n_negative / n_positive

# ============================================================================
# 2. Train XGBoost Model (Best Performer)
# ============================================================================
print("\n" + "=" * 80)
print("2. Training XGBoost Model")
print("=" * 80)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("✓ Training Complete")

# ============================================================================
# 3. Calculate SHAP Values
# ============================================================================
print("\n" + "=" * 80)
print("3. Calculating SHAP Values")
print("=" * 80)

# Use full test set for comprehensive SHAP analysis
print(f"Using full test set: {len(X_test):,} samples")

print("\nCalculating SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

print("✓ SHAP values calculated")

# ============================================================================
# 4. Generate SHAP Summary Plot
# ============================================================================
print("\n" + "=" * 80)
print("4. Generating SHAP Summary Plot")
print("=" * 80)

# Create figure with appropriate size (wider to accommodate color bar)
fig = plt.figure(figsize=(10, 7))

# Generate SHAP summary plot
shap.summary_plot(
    shap_values, 
    X_test, 
    plot_type="dot",
    show=False
)

# Enhance styling
ax = plt.gca()
ax.set_title('SHAP Summary Plot (Class: At-Risk) - XGBoost Model', 
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.0)

# Save high-resolution image
output_path = 'outputs/classification/shap_summary_plot_standalone.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")
plt.close()

# ============================================================================
# 5. Summary Statistics
# ============================================================================
print("\n" + "=" * 80)
print("5. SHAP Feature Importance Summary")
print("=" * 80)

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'Feature': list(X_train.columns),
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("\nFeature Importance (sorted by mean absolute SHAP):")
print(shap_importance.to_string(index=False))

# Save to CSV
csv_path = 'outputs/classification/shap_summary_importance.csv'
shap_importance.to_csv(csv_path, index=False)
print(f"\n✓ Saved: {csv_path}")

print("\n" + "=" * 80)
print("SHAP SUMMARY PLOT EXPORT COMPLETE")
print("=" * 80)

print("\n📊 Key Insights:")
print(f"  • Most important feature: {shap_importance.iloc[0]['Feature']}")
print(f"    Mean |SHAP| = {shap_importance.iloc[0]['Mean_Abs_SHAP']:.4f}")
print(f"\n  • Least important feature: {shap_importance.iloc[-1]['Feature']}")
print(f"    Mean |SHAP| = {shap_importance.iloc[-1]['Mean_Abs_SHAP']:.4f}")
print(f"\n  • SHAP values based on {len(X_test):,} test samples")
print(f"  • Model: XGBoost (Best performing model)")
