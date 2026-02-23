"""
Classification Task - Step 1: Data Preparation
Prepare engineered features and target for At-Risk prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CLASSIFICATION TASK - DATA PREPARATION")
print("=" * 80)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'classification'
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. Load Raw Data
# ============================================================================
print("\n" + "=" * 80)
print("1. Loading Raw Data")
print("=" * 80)

df = pd.read_csv(DATA_DIR / 'raw' / 'CDC Diabetes Dataset.csv')
print(f"Dataset: {len(df):,} samples, {df.shape[1]} raw features")

# ============================================================================
# 2. Engineer 5 Composite Features
# ============================================================================
print("\n" + "=" * 80)
print("2. Engineering Composite Features")
print("=" * 80)

# Feature 1: Chronic Disease Burden (0-5 scale)
df['Chronic_Disease_Burden'] = (
    df['HighBP'].astype(int) +
    df['HighChol'].astype(int) +
    df['Stroke'].astype(int) +
    df['HeartDiseaseorAttack'].astype(int) +
    df['DiffWalk'].astype(int)
)
print(f"✓ Chronic Disease Burden: Mean={df['Chronic_Disease_Burden'].mean():.2f}, Range=[{df['Chronic_Disease_Burden'].min()}, {df['Chronic_Disease_Burden'].max()}]")

# Feature 2: SES Vulnerability (0-17 scale, higher = more vulnerable)
# Keep original granularity: (10-Income) + (7-Education)
df['SES_Vulnerability'] = (10 - df['Income']) + (7 - df['Education'])
print(f"✓ SES Vulnerability: Mean={df['SES_Vulnerability'].mean():.2f}, Range=[{df['SES_Vulnerability'].min()}, {df['SES_Vulnerability'].max()}]")

# Feature 3: Lifestyle Risk (0-3 scale, healthy behaviors only)
# Exclude substance use (tracked separately)
df['Lifestyle_Risk'] = 3 - (df['Veggies'] + df['Fruits'] + df['PhysActivity'])
print(f"✓ Lifestyle Risk: Mean={df['Lifestyle_Risk'].mean():.2f}, Range=[{df['Lifestyle_Risk'].min()}, {df['Lifestyle_Risk'].max()}]")

# Feature 4: BMI - keep original continuous values
df['BMI'] = df['BMI']  # Already exists, ensure it's in the right format
print(f"✓ BMI: Mean={df['BMI'].mean():.2f}, Std={df['BMI'].std():.2f}")

# Feature 5: Substance Risk (0-2 scale)
df['Substance_Risk'] = df['Smoker'] + df['HvyAlcoholConsump']
print(f"✓ Substance Risk: Mean={df['Substance_Risk'].mean():.2f}, Range=[{df['Substance_Risk'].min()}, {df['Substance_Risk'].max()}]")

# ============================================================================
# 3. Create Target Variable: At-Risk (0/1)
# ============================================================================
print("\n" + "=" * 80)
print("3. Creating Target Variable")
print("=" * 80)

# At-Risk = Prediabetes (1) OR Diabetes (2)
df['At_Risk'] = (df['Diabetes_012'] >= 1).astype(int)

print(f"Target Distribution:")
print(f"  Not At-Risk (0): {(df['At_Risk'] == 0).sum():,} ({(df['At_Risk'] == 0).mean():.1%})")
print(f"  At-Risk (1):     {(df['At_Risk'] == 1).sum():,} ({(df['At_Risk'] == 1).mean():.1%})")
print(f"  Class Imbalance Ratio: 1:{(df['At_Risk'] == 0).sum() / (df['At_Risk'] == 1).sum():.2f}")

# ============================================================================
# 4. Prepare Feature Matrix and Target
# ============================================================================
print("\n" + "=" * 80)
print("4. Preparing Feature Matrix")
print("=" * 80)

feature_cols = [
    'Chronic_Disease_Burden',
    'BMI',
    'Lifestyle_Risk',
    'Substance_Risk',
    'SES_Vulnerability'
]

X = df[feature_cols].copy()
y = df['At_Risk'].copy()

print(f"Feature Matrix X: {X.shape}")
print(f"Target Vector y: {y.shape}")
print(f"\nFeature Statistics:")
print(X.describe().round(2))

# Check for missing values
missing = X.isnull().sum()
if missing.sum() > 0:
    print(f"\n⚠ Missing values detected:")
    print(missing[missing > 0])
else:
    print("\n✓ No missing values in features")

# ============================================================================
# 5. Stratified Train-Test Split (80/20)
# ============================================================================
print("\n" + "=" * 80)
print("5. Stratified Train-Test Split")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Training Set:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  At-Risk Rate: {y_train.mean():.1%}")

print(f"\nTest Set:")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")
print(f"  At-Risk Rate: {y_test.mean():.1%}")

# Verify stratification
print(f"\n✓ Stratification Preserved:")
print(f"  Original:  {y.mean():.4f}")
print(f"  Training:  {y_train.mean():.4f}")
print(f"  Test:      {y_test.mean():.4f}")

# ============================================================================
# 6. Save Prepared Data
# ============================================================================
print("\n" + "=" * 80)
print("6. Saving Prepared Data")
print("=" * 80)

# Save full feature engineered dataset
df_features = df[feature_cols + ['At_Risk']].copy()
df_features.to_csv(OUTPUT_DIR / 'features_engineered.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'features_engineered.csv'}")

# Save train-test splits
X_train.to_csv(OUTPUT_DIR / 'X_train.csv', index=False)
X_test.to_csv(OUTPUT_DIR / 'X_test.csv', index=False)
y_train.to_csv(OUTPUT_DIR / 'y_train.csv', index=False)
y_test.to_csv(OUTPUT_DIR / 'y_test.csv', index=False)

print(f"✓ Saved: {OUTPUT_DIR / 'X_train.csv'}")
print(f"✓ Saved: {OUTPUT_DIR / 'X_test.csv'}")
print(f"✓ Saved: {OUTPUT_DIR / 'y_train.csv'}")
print(f"✓ Saved: {OUTPUT_DIR / 'y_test.csv'}")

# ============================================================================
# 7. Feature Engineering Summary
# ============================================================================
print("\n" + "=" * 80)
print("7. Summary: Ready for Classification")
print("=" * 80)

print("\nFeature Definitions (synchronized with clustering task):")
print("  1. Chronic_Disease_Burden: HighBP + HighChol + Stroke + HeartDisease + DiffWalk (0-5)")
print("  2. BMI: Body Mass Index (continuous)")
print("  3. Lifestyle_Risk: 3 - (Veggies + Fruits + PhysActivity) (0-3)")
print("  4. Substance_Risk: Smoker + HvyAlcoholConsump (0-2)")
print("  5. SES_Vulnerability: (10-Income) + (7-Education) (0-17)")

print(f"\nTarget:")
print(f"  At_Risk: 1 if Prediabetes or Diabetes, else 0")

print(f"\nData Split:")
print(f"  Training: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test:     {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nClass Weights Recommendation:")
print(f"  For tree-based models: Use class_weight='balanced'")
print(f"  Calculated weights: {{0: {len(y)/(2*sum(y==0)):.3f}, 1: {len(y)/(2*sum(y==1)):.3f}}}")

print(f"\nNote: SMOTE not applied - using class_weight='balanced' instead")

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE")
print("=" * 80)
