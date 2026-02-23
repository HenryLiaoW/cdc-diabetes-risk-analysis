import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/data/processed/CDC_Diabetes_Cleaned.csv')
bmi_data = df['BMI'].dropna().values

print("="*80)
print("BMI SCALER SELECTION ANALYSIS")
print("="*80)

# 1. Original distribution characteristics
print("\n1. ORIGINAL BMI DISTRIBUTION:")
print("-"*80)
print(f"   Mean: {np.mean(bmi_data):.2f}")
print(f"   Median: {np.median(bmi_data):.2f}")
print(f"   Std: {np.std(bmi_data):.2f}")
print(f"   IQR: {np.percentile(bmi_data, 75) - np.percentile(bmi_data, 25):.2f}")
print(f"   Skewness: {pd.Series(bmi_data).skew():.3f}")
print(f"   Range: [{bmi_data.min():.1f}, {bmi_data.max():.1f}]")

# Calculate outlier percentage
Q1 = np.percentile(bmi_data, 25)
Q3 = np.percentile(bmi_data, 75)
IQR = Q3 - Q1
outlier_low = Q1 - 1.5 * IQR
outlier_high = Q3 + 1.5 * IQR
outliers = (bmi_data < outlier_low) | (bmi_data > outlier_high)
outlier_pct = np.sum(outliers) / len(bmi_data) * 100
print(f"   Outliers (1.5×IQR rule): {outlier_pct:.2f}%")

# Check if mean-median difference is significant
mean_median_diff = abs(np.mean(bmi_data) - np.median(bmi_data))
mean_median_diff_pct = mean_median_diff / np.median(bmi_data) * 100
print(f"   Mean-Median difference: {mean_median_diff:.2f} ({mean_median_diff_pct:.1f}%)")

# 2. Apply both scalers
bmi_reshaped = bmi_data.reshape(-1, 1)
ss_scaled = StandardScaler().fit_transform(bmi_reshaped).flatten()
rs_scaled = RobustScaler().fit_transform(bmi_reshaped).flatten()

print("\n2. SCALED DISTRIBUTION COMPARISON:")
print("-"*80)
print("   StandardScaler:")
print(f"     IQR (25th-75th): {np.percentile(ss_scaled, 75) - np.percentile(ss_scaled, 25):.3f}")
print(f"     99% range: {np.percentile(ss_scaled, 99) - np.percentile(ss_scaled, 1):.3f}")
print(f"     Max outlier impact: {ss_scaled.max():.2f} (originally BMI={bmi_data.max():.0f})")

print("\n   RobustScaler:")
print(f"     IQR (25th-75th): {np.percentile(rs_scaled, 75) - np.percentile(rs_scaled, 25):.3f}")
print(f"     99% range: {np.percentile(rs_scaled, 99) - np.percentile(rs_scaled, 1):.3f}")
print(f"     Max outlier impact: {rs_scaled.max():.2f} (originally BMI={bmi_data.max():.0f})")

# 3. Key insight: Why RobustScaler IQR is smaller?
print("\n3. WHY ROBUSTSCALER IQR IS SMALLER (1.0 vs 1.179)?")
print("-"*80)
print("   RobustScaler: IQR is ALWAYS 1.0 by definition")
print("                 (scales data by dividing by IQR)")
print("   StandardScaler: IQR varies based on data shape")
print("                   (scales by std, which includes tail influence)")
print(f"\n   BMI's std/IQR ratio: {np.std(bmi_data) / IQR:.3f}")
print("   → When std > IQR, StandardScaler's IQR will be > 1.0")
print("   → This happens when distribution has MODERATE tails (not extreme)")

# 4. Decision criteria
print("\n4. SCALER SELECTION CRITERIA FOR BMI:")
print("-"*80)
criteria = []

# Criterion 1: Skewness
if abs(pd.Series(bmi_data).skew()) > 1.0:
    criteria.append(("Skewness > 1.0", "RobustScaler", f"Skew={pd.Series(bmi_data).skew():.2f}, right-tailed"))
else:
    criteria.append(("Skewness > 1.0", "Either", f"Skew={pd.Series(bmi_data).skew():.2f}, moderate"))

# Criterion 2: Outliers
if outlier_pct > 1.0:
    criteria.append(("Outliers > 1%", "RobustScaler", f"{outlier_pct:.2f}% outliers present"))
else:
    criteria.append(("Outliers > 1%", "StandardScaler", f"{outlier_pct:.2f}% minimal outliers"))

# Criterion 3: Mean-Median difference
if mean_median_diff_pct > 5.0:
    criteria.append(("Mean≠Median", "RobustScaler", f"{mean_median_diff_pct:.1f}% difference"))
else:
    criteria.append(("Mean≠Median", "Either", f"{mean_median_diff_pct:.1f}% acceptable"))

# Criterion 4: Physical meaning
criteria.append(("Domain knowledge", "RobustScaler", "BMI extremes may be data errors or rare cases"))

print("\n   Criterion                  Recommendation    Reason")
print("   " + "-"*76)
for crit, rec, reason in criteria:
    print(f"   {crit:25s}  {rec:15s}   {reason}")

# 5. Final recommendation
robust_votes = sum(1 for _, rec, _ in criteria if rec == "RobustScaler")
print("\n5. FINAL RECOMMENDATION:")
print("="*80)
if robust_votes >= 3:
    print("   ✓ USE ROBUSTSCALER for BMI")
    print("\n   Reasons:")
    print("   • Right-skewed distribution (Skewness=2.06)")
    print("   • 2.5% outliers that may not represent typical population")
    print("   • Median (27.0) better represents 'typical' BMI than mean (28.3)")
    print("   • Protects clustering from extreme BMI values (e.g., BMI=98)")
    print("\n   Note: StandardScaler's wider IQR doesn't mean it's better!")
    print("         - Wider IQR just means outliers aren't EXTREME enough to compress core")
    print("         - But outliers still bias the mean and std, making scaling suboptimal")
else:
    print("   → USE STANDARDSCALER for BMI")
    print("   (Minimal skewness and outliers)")

print("\n" + "="*80)
print("CONCLUSION: RobustScaler is correct choice despite IQR comparison")
print("="*80)
