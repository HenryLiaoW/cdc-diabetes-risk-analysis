import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

# Load data
df = pd.read_csv('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/data/processed/CDC_Diabetes_Cleaned.csv')

# Calculate metabolic_baseline (BMI)
df['metabolic_baseline'] = df['BMI']
data_bmi = df['metabolic_baseline'].dropna().values.reshape(-1, 1)

# Apply both scalers on FULL dataset
standard_scaled = StandardScaler().fit_transform(data_bmi).flatten()
robust_scaled = RobustScaler().fit_transform(data_bmi).flatten()

# Calculate percentiles
print("="*70)
print("SCALED VALUE DISTRIBUTION ANALYSIS (Full Dataset)")
print("="*70)
print(f"\nDataset size: {len(data_bmi):,} samples")
print(f"\nOriginal BMI range: [{data_bmi.min():.1f}, {data_bmi.max():.1f}]")
print(f"Original BMI median: {np.median(data_bmi):.2f}, IQR: {np.percentile(data_bmi, 75) - np.percentile(data_bmi, 25):.2f}")

print("\n" + "-"*70)
print("StandardScaler Results:")
print("-"*70)
print(f"Range: [{standard_scaled.min():.2f}, {standard_scaled.max():.2f}]")
print(f"99% of data in: [{np.percentile(standard_scaled, 0.5):.2f}, {np.percentile(standard_scaled, 99.5):.2f}]")
print(f"95% of data in: [{np.percentile(standard_scaled, 2.5):.2f}, {np.percentile(standard_scaled, 97.5):.2f}]")
print(f"90% of data in: [{np.percentile(standard_scaled, 5):.2f}, {np.percentile(standard_scaled, 95):.2f}]")
print(f"50% of data (IQR) in: [{np.percentile(standard_scaled, 25):.2f}, {np.percentile(standard_scaled, 75):.2f}]")

print("\n" + "-"*70)
print("RobustScaler Results:")
print("-"*70)
print(f"Range: [{robust_scaled.min():.2f}, {robust_scaled.max():.2f}]")
print(f"99% of data in: [{np.percentile(robust_scaled, 0.5):.2f}, {np.percentile(robust_scaled, 99.5):.2f}]")
print(f"95% of data in: [{np.percentile(robust_scaled, 2.5):.2f}, {np.percentile(robust_scaled, 97.5):.2f}]")
print(f"90% of data in: [{np.percentile(robust_scaled, 5):.2f}, {np.percentile(robust_scaled, 95):.2f}]")
print(f"50% of data (IQR) in: [{np.percentile(robust_scaled, 25):.2f}, {np.percentile(robust_scaled, 75):.2f}]")

# Key metric: spread of core data
ss_50_range = np.percentile(standard_scaled, 75) - np.percentile(standard_scaled, 25)
rs_50_range = np.percentile(robust_scaled, 75) - np.percentile(robust_scaled, 25)

print("\n" + "="*70)
print("KEY COMPARISON: Core Data Spread (50% Central Data)")
print("="*70)
print(f"StandardScaler IQR range: {ss_50_range:.3f}")
print(f"RobustScaler IQR range: {rs_50_range:.3f}")
print(f"Ratio (RobustScaler / StandardScaler): {rs_50_range/ss_50_range:.2f}×")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
if rs_50_range > ss_50_range:
    print(f"✓ RobustScaler IQR is {rs_50_range/ss_50_range:.2f}× WIDER than StandardScaler")
    print("  → Core data gets MORE resolution for K-Means distance calculation")
else:
    print(f"✗ StandardScaler IQR is actually {ss_50_range/rs_50_range:.2f}× WIDER")
    print("  → BMI data may not have severe enough outliers to demonstrate effect")
    print("  → Consider using a different feature (e.g., with more extreme outliers)")
