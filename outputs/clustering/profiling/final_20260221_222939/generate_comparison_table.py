#!/usr/bin/env python
"""
Generate comparison table for 5-feature vs 6-feature (Age) vs 6-feature (Sex) models
"""

import pandas as pd
from pathlib import Path

# File paths
results_5feat = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/log/5features_composite_kmeans_results_20260221_222110.csv')
results_6feat_age = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/log/6features_includingAGE_composite_kmeans_results_20260222_135530.csv')
results_6feat_sex = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/log/6features_includingSEX_composite_kmeans_results_20260222_135730.csv')

# Read results
df_5feat = pd.read_csv(results_5feat)
df_6feat_age = pd.read_csv(results_6feat_age)
df_6feat_sex = pd.read_csv(results_6feat_sex)

# Extract metrics
# Silhouette Score for K=6
sil_k6_5feat = df_5feat[df_5feat['K'] == 6]['Silhouette_Score'].values[0]
sil_k6_6feat_age = df_6feat_age[df_6feat_age['K'] == 6]['Silhouette_Score'].values[0]
sil_k6_6feat_sex = df_6feat_sex[df_6feat_sex['K'] == 6]['Silhouette_Score'].values[0]

# Average Silhouette Score (all K values)
sil_avg_5feat = df_5feat['Silhouette_Score'].mean()
sil_avg_6feat_age = df_6feat_age['Silhouette_Score'].mean()
sil_avg_6feat_sex = df_6feat_sex['Silhouette_Score'].mean()

# VIF and Correlation (from manual observations)
max_vif_5feat = 1.120
max_corr_5feat = 0.251

max_vif_6feat_age = 1.354
max_corr_6feat_age = 0.414

max_vif_6feat_sex = 1.125
max_corr_6feat_sex = 0.251

# Create comparison table
comparison_data = {
    'Dimension': [
        'Silhouette Score (K=6)',
        'Silhouette Score (Average)',
        'Max VIF',
        'Max Correlation'
    ],
    '5-Feature': [
        f'{sil_k6_5feat:.4f}',
        f'{sil_avg_5feat:.4f}',
        f'{max_vif_5feat:.3f}',
        f'{max_corr_5feat:.3f}'
    ],
    '6-Feature (including Age)': [
        f'{sil_k6_6feat_age:.4f}',
        f'{sil_avg_6feat_age:.4f}',
        f'{max_vif_6feat_age:.3f}',
        f'{max_corr_6feat_age:.3f}'
    ],
    '6-Feature (including Sex)': [
        f'{sil_k6_6feat_sex:.4f}',
        f'{sil_avg_6feat_sex:.4f}',
        f'{max_vif_6feat_sex:.3f}',
        f'{max_corr_6feat_sex:.3f}'
    ]
}

df_comparison = pd.DataFrame(comparison_data)

# Determine winner for each metric
winners = []
for i, row in df_comparison.iterrows():
    metric = row['Dimension']
    vals = [
        float(row['5-Feature']),
        float(row['6-Feature (including Age)']),
        float(row['6-Feature (including Sex)'])
    ]
    
    if 'Silhouette' in metric:
        # Higher is better
        winner_idx = vals.index(max(vals))
    else:
        # Lower is better (VIF and Correlation)
        winner_idx = vals.index(min(vals))
    
    winner_names = ['5-Feature', '6-Feature (Age)', '6-Feature (Sex)']
    winners.append(winner_names[winner_idx])

df_comparison['Winner'] = winners

# Print table
print("\n" + "="*100)
print("FEATURE COMPARISON: 5-FEATURE vs 6-FEATURE (AGE) vs 6-FEATURE (SEX)")
print("="*100)
print(df_comparison.to_string(index=False))
print("="*100)

# Save to CSV
output_file = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/profiling/final_20260221_222939/feature_comparison_table.csv')
df_comparison.to_csv(output_file, index=False)
print(f"\n✓ Comparison table saved to: {output_file}")

# Print interpretation
print("\n" + "="*100)
print("INTERPRETATION")
print("="*100)
print("\n1. Silhouette Score (K=6):")
print("   - Measures cluster separation quality for K=6 configuration")
print(f"   - Winner: {df_comparison.loc[0, 'Winner']} with score {df_comparison.loc[0, df_comparison.loc[0, 'Winner'].replace('6-Feature (Age)', '6-Feature (including Age)').replace('6-Feature (Sex)', '6-Feature (including Sex)')]}")
print("   - Higher is better (>0.5 = good, >0.7 = excellent)")

print("\n2. Silhouette Score (Average):")
print("   - Average performance across all K values (K=2 to 7)")
print(f"   - Winner: {df_comparison.loc[1, 'Winner']}")
print("   - Indicates overall clustering stability")

print("\n3. Max VIF:")
print("   - Variance Inflation Factor (multicollinearity indicator)")
print(f"   - Winner: {df_comparison.loc[2, 'Winner']} with VIF {df_comparison.loc[2, df_comparison.loc[2, 'Winner'].replace('6-Feature (Age)', '6-Feature (including Age)').replace('6-Feature (Sex)', '6-Feature (including Sex)')]}")
print("   - Lower is better (<5 = excellent, <10 = acceptable)")
print("   - All models show excellent multicollinearity control")

print("\n4. Max Correlation:")
print("   - Maximum pairwise feature correlation")
print(f"   - Winner: {df_comparison.loc[3, 'Winner']} with r = {df_comparison.loc[3, df_comparison.loc[3, 'Winner'].replace('6-Feature (Age)', '6-Feature (including Age)').replace('6-Feature (Sex)', '6-Feature (including Sex)')]}")
print("   - Lower is better (<0.3 = weak, <0.7 = moderate)")
print("   - All models show weak to moderate correlations")

print("\n" + "="*100)
print("OVERALL ASSESSMENT")
print("="*100)

# Count wins
win_counts = df_comparison['Winner'].value_counts().to_dict()
print(f"\nWin Count:")
for model, count in win_counts.items():
    print(f"  • {model}: {count}/4 metrics")

# Overall recommendation
if win_counts.get('5-Feature', 0) >= 2:
    print("\n✅ RECOMMENDED: 5-Feature Model")
    print("   Rationale: Best balance of clustering performance and feature independence")
elif win_counts.get('6-Feature (Sex)', 0) >= 2:
    print("\n✅ RECOMMENDED: 6-Feature (Sex) Model")
    print("   Rationale: Improved clustering with minimal multicollinearity cost")
else:
    print("\n⚠️  MIXED RESULTS: Consider research goals")
    print("   • If interpretability is key → 5-Feature")
    print("   • If demographic insights needed → 6-Feature (Sex)")
    print("   • Age shows weakest performance (VIF=1.354, introduces moderate correlation)")

print("\n" + "="*100)
