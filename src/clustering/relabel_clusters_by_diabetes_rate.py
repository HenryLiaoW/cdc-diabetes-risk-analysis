"""
Relabel Clusters by Diabetes Prevalence Rate
=============================================
Reads the original clustering results and relabels clusters 
in ascending order of diabetes/prediabetes prevalence.

Output: New CSV with relabeled cluster column (0-5)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
INPUT_FILE = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/results/kmeans_k6_complete_LATEST.csv')
OUTPUT_FILE = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/results/kmeans_k6_complete_relabeled.csv')

print("=" * 80)
print("Relabeling Clusters by Diabetes Prevalence Rate".center(80))
print("=" * 80)

# Load data
print(f"\nLoading data from: {INPUT_FILE.name}")
df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(df):,} records")

# Calculate diabetes prevalence for each cluster
print("\nCalculating diabetes prevalence for each original cluster:")
cluster_diabetes_rates = []

for c in range(6):
    cluster_data = df[df['cluster'] == c]
    size = len(cluster_data)
    at_risk_count = (cluster_data['Diabetes_012'] > 0).sum()
    at_risk_rate = at_risk_count / size * 100
    
    cluster_diabetes_rates.append({
        'original_cluster': c,
        'size': size,
        'at_risk_count': at_risk_count,
        'at_risk_rate': at_risk_rate
    })
    
    print(f"  Original C{c}: Size={size:,}, At-Risk={at_risk_count:,} ({at_risk_rate:.1f}%)")

# Sort by diabetes rate (ascending)
cluster_diabetes_rates_df = pd.DataFrame(cluster_diabetes_rates)
cluster_diabetes_rates_df = cluster_diabetes_rates_df.sort_values('at_risk_rate')

print("\n" + "=" * 80)
print("Creating Mapping (Original → New Label):")
print("=" * 80)

# Create mapping dictionary
mapping = {}
for new_label, row in enumerate(cluster_diabetes_rates_df.itertuples()):
    original = row.original_cluster
    mapping[original] = new_label
    print(f"  Original C{original} → New C{new_label} (Size={row.size:,}, At-Risk={row.at_risk_rate:.1f}%)")

# Apply mapping to create new cluster column
print("\n" + "=" * 80)
print("Relabeling Data")
print("=" * 80)

df['cluster_relabeled'] = df['cluster'].map(mapping)
df['cluster_original'] = df['cluster']  # Keep original for reference
df['cluster'] = df['cluster_relabeled']  # Replace cluster column with relabeled version
df.drop('cluster_relabeled', axis=1, inplace=True)

# Verify new distribution
print("\nVerifying new cluster distribution:")
for c in range(6):
    cluster_data = df[df['cluster'] == c]
    size = len(cluster_data)
    at_risk_rate = (cluster_data['Diabetes_012'] > 0).sum() / size * 100
    print(f"  New C{c}: Size={size:,}, At-Risk Rate={at_risk_rate:.1f}%")

# Save
print("\n" + "=" * 80)
print("Saving Relabeled Data")
print("=" * 80)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.drop('cluster_original', axis=1, inplace=True)  # Remove original cluster column
df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Saved to: {OUTPUT_FILE}")
print(f"  Total records: {len(df):,}")
print(f"  Columns: {len(df.columns)}")

print("\n" + "=" * 80)
print("Relabeling Complete!")
print("=" * 80)
print(f"\nNew file can be used directly for SHAP analysis.")
print(f"Cluster labels now range from 0 (lowest risk) to 5 (highest risk).")
