"""
CDC Diabetes Dataset - Data Cleaning and Preprocessing Script

This script performs comprehensive data cleaning and validation for the CDC BRFSS 2015 dataset.
Steps include:
1. Load raw data
2. Check for missing values
3. Check for duplicates
4. Validate data ranges and types
5. Handle outliers
6. Save cleaned data

Author: Data Cleaning Pipeline
Date: 2026-02-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print("CDC DIABETES DATASET - DATA CLEANING PIPELINE")
print("="*80)

# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================
print("\n[STEP 1] Loading raw data...")
data_path = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/data/raw/CDC Diabetes Dataset.csv')
df = pd.read_csv(data_path)

print(f"✓ Data loaded successfully")
print(f"  • Shape: {df.shape}")
print(f"  • Columns: {df.columns.tolist()}")

# ============================================================================
# 2. CHECK MISSING VALUES
# ============================================================================
print("\n[STEP 2] Checking for missing values...")
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100

if missing_counts.sum() == 0:
    print("✓ No missing values found in the dataset")
else:
    print(f"⚠️  Found missing values:")
    for col in missing_counts[missing_counts > 0].index:
        print(f"  • {col}: {missing_counts[col]} ({missing_percent[col]:.2f}%)")

# ============================================================================
# 3. CHECK DUPLICATES
# ============================================================================
print("\n[STEP 3] Checking for duplicate rows...")
duplicates = df.duplicated().sum()

if duplicates == 0:
    print("✓ No duplicate rows found")
else:
    print(f"⚠️  Found {duplicates} duplicate rows ({duplicates/len(df)*100:.2f}%)")
    print("  Action: Removing duplicates...")
    df = df.drop_duplicates()
    print(f"✓ Duplicates removed. New shape: {df.shape}")

# ============================================================================
# 4. VALIDATE DATA RANGES
# ============================================================================
print("\n[STEP 4] Validating data ranges and types...")

# Define expected ranges based on CDC BRFSS 2015 codebook
expected_ranges = {
    'Diabetes_012': (0, 2),
    'HighBP': (0, 1),
    'HighChol': (0, 1),
    'CholCheck': (0, 1),
    'BMI': (10, 100),  # Reasonable BMI range
    'Smoker': (0, 1),
    'Stroke': (0, 1),
    'HeartDiseaseorAttack': (0, 1),
    'PhysActivity': (0, 1),
    'Fruits': (0, 1),
    'Veggies': (0, 1),
    'HvyAlcoholConsump': (0, 1),
    'AnyHealthcare': (0, 1),
    'NoDocbcCost': (0, 1),
    'GenHlth': (1, 5),
    'MentHlth': (0, 30),
    'PhysHlth': (0, 30),
    'DiffWalk': (0, 1),
    'Sex': (0, 1),
    'Age': (1, 13),
    'Education': (1, 6),
    'Income': (1, 8)
}

validation_issues = []
for col, (min_val, max_val) in expected_ranges.items():
    if col in df.columns:
        out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
        if len(out_of_range) > 0:
            validation_issues.append(f"{col}: {len(out_of_range)} values out of range [{min_val}, {max_val}]")

if len(validation_issues) == 0:
    print("✓ All features are within expected ranges")
else:
    print("⚠️  Validation issues found:")
    for issue in validation_issues:
        print(f"  • {issue}")

# ============================================================================
# 5. DATA TYPE VALIDATION
# ============================================================================
print("\n[STEP 5] Checking data types...")

# Binary features should be int (0 or 1)
binary_features = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex'
]

# Ordinal features should be int
ordinal_features = ['Diabetes_012', 'GenHlth', 'Age', 'Education', 'Income']

# Continuous features can be float
continuous_features = ['BMI', 'MentHlth', 'PhysHlth']

print("✓ Data type summary:")
print(f"  • Binary features (0/1): {len(binary_features)}")
print(f"  • Ordinal features (categorical): {len(ordinal_features)}")
print(f"  • Continuous features: {len(continuous_features)}")

# Check for unexpected data types
type_issues = []
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        type_issues.append(f"{col}: {df[col].dtype}")

if len(type_issues) > 0:
    print("⚠️  Non-numeric columns found:")
    for issue in type_issues:
        print(f"  • {issue}")
else:
    print("✓ All columns are numeric")

# ============================================================================
# 6. OUTLIER DETECTION
# ============================================================================
print("\n[STEP 6] Detecting outliers in continuous features...")

outlier_summary = []
for col in continuous_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers
    upper_bound = Q3 + 3 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percent = (outlier_count / len(df)) * 100
    
    outlier_summary.append({
        'Feature': col,
        'Outliers': outlier_count,
        'Percent': outlier_percent,
        'Min': df[col].min(),
        'Max': df[col].max(),
        'Q1': Q1,
        'Q3': Q3
    })

outlier_df = pd.DataFrame(outlier_summary)
print("\nOutlier Detection Summary (3×IQR method):")
print(outlier_df.to_string(index=False))

# Note: For BMI, outliers are expected (extreme obesity cases) and should be kept
print("\n✓ Note: Outliers in BMI are expected (extreme obesity) and will be retained")
print("  Recommendation: Use RobustScaler for modeling instead of removing outliers")

# ============================================================================
# 7. CLASS DISTRIBUTION CHECK
# ============================================================================
print("\n[STEP 7] Checking target variable distribution...")
diabetes_dist = df['Diabetes_012'].value_counts().sort_index()
diabetes_percent = (diabetes_dist / len(df)) * 100

print("\nDiabetes_012 Distribution:")
print(f"  • Class 0 (No diabetes): {diabetes_dist[0]:,} ({diabetes_percent[0]:.2f}%)")
print(f"  • Class 1 (Prediabetes): {diabetes_dist[1]:,} ({diabetes_percent[1]:.2f}%)")
print(f"  • Class 2 (Diabetes): {diabetes_dist[2]:,} ({diabetes_percent[2]:.2f}%)")

imbalance_ratio = diabetes_dist.max() / diabetes_dist.min()
print(f"\n⚠️  Class imbalance ratio: {imbalance_ratio:.2f}:1")
print("  Recommendation: Consider using SMOTE or class weights in modeling")

# ============================================================================
# 8. VISUALIZATION OF DATA QUALITY
# ============================================================================
print("\n[STEP 8] Generating data quality visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Missing values heatmap (if any)
ax1 = axes[0, 0]
if df.isnull().sum().sum() > 0:
    sns.heatmap(df.isnull(), cbar=False, ax=ax1, cmap='viridis')
    ax1.set_title('Missing Values Heatmap', fontweight='bold')
else:
    ax1.text(0.5, 0.5, '✓ No Missing Values\nDataset is complete!', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            transform=ax1.transAxes)
    ax1.axis('off')

# Plot 2: Class distribution
ax2 = axes[0, 1]
diabetes_dist.plot(kind='bar', ax=ax2, color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
ax2.set_title('Target Variable Distribution', fontweight='bold')
ax2.set_xlabel('Diabetes Status (0=No, 1=Pre, 2=Yes)')
ax2.set_ylabel('Count')
ax2.set_xticklabels(['No Diabetes', 'Prediabetes', 'Diabetes'], rotation=0)
for i, v in enumerate(diabetes_dist):
    ax2.text(i, v + 5000, f'{v:,}\n({diabetes_percent[i]:.1f}%)', 
            ha='center', fontweight='bold')

# Plot 3: Outliers in continuous features
ax3 = axes[1, 0]
outlier_data = outlier_df[['Feature', 'Percent']]
outlier_data.plot(x='Feature', y='Percent', kind='bar', ax=ax3, 
                 color='#e74c3c', edgecolor='black', legend=False)
ax3.set_title('Outlier Percentage by Feature (3×IQR)', fontweight='bold')
ax3.set_xlabel('Feature')
ax3.set_ylabel('Outlier %')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
ax3.axhline(y=5, color='orange', linestyle='--', label='5% threshold', alpha=0.7)
ax3.legend()

# Plot 4: Data quality summary
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
DATA QUALITY SUMMARY
{'='*40}

Total Records: {len(df):,}
Total Features: {df.shape[1]}

Missing Values: {df.isnull().sum().sum():,}
Duplicate Rows: {duplicates:,}
Data Type Issues: {len(type_issues)}

Range Violations: {len(validation_issues)}
Outliers (BMI): {outlier_df[outlier_df['Feature']=='BMI']['Outliers'].values[0]:,}

Class Imbalance: {imbalance_ratio:.2f}:1

Status: {'✓ CLEAN' if len(validation_issues)==0 and len(type_issues)==0 else '⚠️ NEEDS REVIEW'}
"""
ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
output_path = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/data_quality_report.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Data quality report saved: {output_path}")
plt.close()

# ============================================================================
# 9. SAVE CLEANED DATA
# ============================================================================
print("\n[STEP 9] Saving cleaned data...")

# Save cleaned data (even if no changes were made, this validates the pipeline)
cleaned_data_path = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/data/processed/CDC_Diabetes_Cleaned.csv')
df.to_csv(cleaned_data_path, index=False)
print(f"✓ Cleaned data saved: {cleaned_data_path}")
print(f"  • Shape: {df.shape}")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DATA CLEANING COMPLETE")
print("="*80)
print(f"\n✓ Original records: {len(df):,}")
print(f"✓ Final records: {len(df):,}")
print(f"✓ Features: {df.shape[1]}")
print(f"\n📊 Data Quality Score: {'EXCELLENT' if len(validation_issues)==0 and len(type_issues)==0 and duplicates==0 else 'GOOD'}")

if len(validation_issues) == 0 and len(type_issues) == 0 and duplicates == 0:
    print("\n✅ Dataset is clean and ready for analysis!")
else:
    print("\n⚠️  Some issues detected. Review the report above.")

print("\nNext Steps:")
print("  1. Use 'CDC_Diabetes_Cleaned.csv' for modeling")
print("  2. Apply RobustScaler for handling BMI outliers")
print("  3. Consider SMOTE or class weights for imbalance")
print("  4. Review 'data_quality_report.png' for visual summary")
print("\n" + "="*80)
