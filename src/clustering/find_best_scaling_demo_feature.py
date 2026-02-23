import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

# Load data
df = pd.read_csv('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/data/processed/CDC_Diabetes_Cleaned.csv')

# Define all raw features to test
test_features = {
    'BMI': 'BMI',
    'PhysHlth': 'PhysHlth',  # Days of poor physical health
    'MentHlth': 'MentHlth',  # Days of poor mental health
    'Age': 'Age',
}

print("="*80)
print("SEARCHING FOR BEST FEATURE TO DEMONSTRATE ROBUSTSCALER ADVANTAGE")
print("="*80)
print("\nCriteria: StandardScaler IQR should be SMALLER than RobustScaler IQR")
print("          (indicating core data compression due to outliers)\n")

results = []

for name, col in test_features.items():
    if col not in df.columns:
        continue
        
    data = df[col].dropna().values.reshape(-1, 1)
    
    # Apply both scalers
    ss_scaled = StandardScaler().fit_transform(data).flatten()
    rs_scaled = RobustScaler().fit_transform(data).flatten()
    
    # Calculate IQR for both
    ss_iqr = np.percentile(ss_scaled, 75) - np.percentile(ss_scaled, 25)
    rs_iqr = np.percentile(rs_scaled, 75) - np.percentile(rs_scaled, 25)
    
    # Calculate compression ratio (lower is better for demonstration)
    ratio = ss_iqr / rs_iqr
    
    # Original distribution stats
    orig_skew = pd.Series(data.flatten()).skew()
    orig_p99 = np.percentile(data, 99)
    orig_p1 = np.percentile(data, 1)
    orig_range_ratio = (orig_p99 - orig_p1) / np.percentile(data, 50)
    
    results.append({
        'Feature': name,
        'Orig_Skewness': orig_skew,
        'Orig_Range_Ratio': orig_range_ratio,
        'SS_IQR': ss_iqr,
        'RS_IQR': rs_iqr,
        'Compression_Ratio': ratio,
        'Demo_Quality': 'EXCELLENT' if ratio < 0.8 else ('GOOD' if ratio < 0.95 else 'POOR')
    })

# Sort by compression ratio (ascending = more compressed by StandardScaler)
results_df = pd.DataFrame(results).sort_values('Compression_Ratio')

print("-"*80)
print("COMPARISON RESULTS (sorted by compression ratio)")
print("-"*80)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
best = results_df.iloc[0]
if best['Compression_Ratio'] < 0.95:
    print(f"✓ Use {best['Feature']} for demonstration!")
    print(f"  StandardScaler IQR = {best['SS_IQR']:.3f}")
    print(f"  RobustScaler IQR = {best['RS_IQR']:.3f}")
    print(f"  Compression Ratio = {best['Compression_Ratio']:.2f}× ({(1-best['Compression_Ratio'])*100:.0f}% compression)")
else:
    print("✗ None of the tested features show significant compression!")
    print("  Alternatives:")
    print("  1. Create a SYNTHETIC comparison with artificial outliers")
    print("  2. Focus on showing IQR's INVARIANCE property (RS_IQR always = 1.0)")
    print("  3. Use scatter plot to show outlier handling instead of distribution")
