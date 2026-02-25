#!/usr/bin/env python
"""
Generate cluster summary table with core features identification
"""

import pandas as pd
from pathlib import Path

# Load cluster profiles
profiles_file = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/profiling/final_20260221_222939/cluster_profiles.csv')
df = pd.read_csv(profiles_file)

# Define feature ranges and thresholds for identifying core features
feature_info = {
    'metabolic_baseline': {'name': 'BMI', 'name_cn': 'BMI', 'range': (12, 98), 'high_threshold': 35, 'very_high': 40},
    'chronic_disease_burden': {'name': 'Chronic Disease', 'name_cn': 'chronic disease burden', 'range': (0, 5), 'high_threshold': 2.0, 'very_high': 2.5},
    'lifestyle_risk': {'name': 'Lifestyle Risk', 'name_cn': 'lifestyle risk', 'range': (0, 3), 'high_threshold': 1.5, 'very_high': 2.0},
    'substance_risk': {'name': 'Substance Risk', 'name_cn': 'substance use risk', 'range': (0, 2), 'high_threshold': 1.0, 'very_high': 1.5},
    'ses_vulnerability': {'name': 'SES Vulnerability', 'name_cn': 'socioeconomic vulnerability', 'range': (3, 15), 'high_threshold': 8.0, 'very_high': 10.0}
}

# Label translations
label_translations = {
    'Low-Risk': '低风险组',
    'Socioeconomic Vulnerability': '社会经济脆弱组',
    'High-Risk Lifestyle': '高风险生活方式组',
    'Chronic-Addiction Synergy': '慢病-成瘾协同组',
    'Metabolic Extremes': '代谢极端组',
    'Frail Multi-Morbidity': '虚弱多重慢病组',
    # Legacy labels (for backward compatibility)
    'Low-Moderate Risk Group': '低-中等风险组',
    'Moderate-High Risk Group': '中高风险组',
    'High Diabetes Risk Group': '高糖尿病风险组',
    'Poor Lifestyle Group': '不良生活方式组',
    'Metabolic Risk Group': '代谢风险组',
    'Multi-Morbidity Group': '多重慢病组',
    'Substance Risk Group': 'substance use risk组',
    'Vulnerable Group': '脆弱群体组'
}

# Identify core features for each cluster
def identify_core_features(row):
    """Identify the most prominent features for a cluster"""
    core_features = []
    
    # Check metabolic_baseline (BMI)
    if row['metabolic_baseline'] >= feature_info['metabolic_baseline']['very_high']:
        core_features.append(f"极高BMI ({row['metabolic_baseline']:.1f})")
    elif row['metabolic_baseline'] >= feature_info['metabolic_baseline']['high_threshold']:
        core_features.append(f"高BMI ({row['metabolic_baseline']:.1f})")
    
    # Check chronic_disease_burden
    if row['chronic_disease_burden'] >= feature_info['chronic_disease_burden']['very_high']:
        core_features.append(f"极高chronic disease burden ({row['chronic_disease_burden']:.2f})")
    elif row['chronic_disease_burden'] >= feature_info['chronic_disease_burden']['high_threshold']:
        core_features.append(f"高chronic disease burden ({row['chronic_disease_burden']:.2f})")
    
    # Check lifestyle_risk
    if row['lifestyle_risk'] >= feature_info['lifestyle_risk']['very_high']:
        core_features.append(f"极高lifestyle risk ({row['lifestyle_risk']:.2f})")
    elif row['lifestyle_risk'] >= feature_info['lifestyle_risk']['high_threshold']:
        core_features.append(f"高lifestyle risk ({row['lifestyle_risk']:.2f})")
    
    # Check substance_risk
    if row['substance_risk'] >= feature_info['substance_risk']['high_threshold']:
        core_features.append(f"高substance use risk ({row['substance_risk']:.2f})")
    
    # Check ses_vulnerability
    if row['ses_vulnerability'] >= feature_info['ses_vulnerability']['very_high']:
        core_features.append(f"极高socioeconomic vulnerability ({row['ses_vulnerability']:.1f})")
    elif row['ses_vulnerability'] >= feature_info['ses_vulnerability']['high_threshold']:
        core_features.append(f"高socioeconomic vulnerability ({row['ses_vulnerability']:.1f})")
    
    # If no high features, identify the relatively higher ones
    if not core_features:
        # Find top 2 features relative to their ranges
        feature_scores = {}
        for feat in ['metabolic_baseline', 'chronic_disease_burden', 'lifestyle_risk', 'substance_risk', 'ses_vulnerability']:
            min_val, max_val = feature_info[feat]['range']
            normalized = (row[feat] - min_val) / (max_val - min_val)
            feature_scores[feat] = normalized
        
        # Get top 2
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        for feat, score in sorted_features:
            if score > 0.3:  # Only include if reasonably elevated
                core_features.append(f"{feature_info[feat]['name_cn']} ({row[feat]:.2f})")
    
    return "；".join(core_features) if core_features else "整体低风险"

# Create summary table
summary_data = []
for idx, row in df.iterrows():
    summary_data.append({
        '簇': int(row['Cluster']),
        '人数': f"{int(row['Size']):,} ({row['Percent']})",
        '糖尿病率': f"{row['Diabetes_Rate']:.1f}%",
        '标签': label_translations.get(row['Label'], row['Label']),
        '核心特征': identify_core_features(row)
    })

df_summary = pd.DataFrame(summary_data)

# Print table
print("\n" + "="*120)
print("聚类总结表 - 5特征K-Means聚类 (K=6)")
print("="*120)
print(df_summary.to_string(index=False))
print("="*120)

# Save to CSV
output_file = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/profiling/final_20260221_222939/cluster_summary_table_cn.csv')
df_summary.to_csv(output_file, index=False)
print(f"\n✓ 聚类总结表已保存至: {output_file}")

# Also create a detailed version with all feature values
detailed_data = []
for idx, row in df.iterrows():
    detailed_data.append({
        '簇': int(row['Cluster']),
        '人数': int(row['Size']),
        '百分比': row['Percent'],
        '糖尿病率': f"{row['Diabetes_Rate']:.2f}%",
        '标签': label_translations.get(row['Label'], row['Label']),
        'BMI': f"{row['metabolic_baseline']:.2f}",
        'chronic disease burden': f"{row['chronic_disease_burden']:.2f}",
        'lifestyle risk': f"{row['lifestyle_risk']:.2f}",
        'substance use risk': f"{row['substance_risk']:.2f}",
        'socioeconomic vulnerability': f"{row['ses_vulnerability']:.2f}",
        '核心特征': identify_core_features(row)
    })

df_detailed = pd.DataFrame(detailed_data)
detailed_output = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/profiling/final_20260221_222939/cluster_summary_detailed_cn.csv')
df_detailed.to_csv(detailed_output, index=False)
print(f"✓ 详细总结已保存至: {detailed_output}")

# Print notes
print("\n" + "="*120)
print("说明")
print("="*120)
print("\n• 核心特征: 基于特征值相对range识别出的显著特征")
print("• 阈值设定:")
print("  - BMI: 高 ≥35, 极高 ≥40")
print("  - chronic disease burden: 高 ≥2.0, 极高 ≥2.5")
print("  - lifestyle risk: 高 ≥1.5, 极高 ≥2.0")
print("  - socioeconomic vulnerability: 高 ≥8.0, 极高 ≥10.0")
print("  - substance use risk: 高 ≥1.0")
print("\n• 簇按糖尿病患病率排序 (簇0=最低, 簇5=highest)")
print("="*120)
