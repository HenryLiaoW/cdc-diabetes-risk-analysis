"""
Cluster Profiling Analysis for K-Means Results

This script performs comprehensive cluster characterization:
1. Load cluster assignments from kmeans.py output (or recompute if missing)
2. Feature mean comparison across clusters
3. Diabetes prevalence analysis by cluster
4. Cluster labeling based on dominant characteristics
5. Intervention strategy recommendations
6. Visualization (radar charts, heatmaps, comparison tables)

Author: Cluster Analysis Pipeline
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import shutil
import glob

print("="*80)
print("CLUSTER PROFILING ANALYSIS")
print("="*80)

# ============================================================================
# 1. DATA LOADING & CLUSTER ASSIGNMENT
# ============================================================================
print("\n[Step 1] Loading K-Means results from kmeans.py...")

# Load K=6 cluster assignments from results directory
results_file = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/results/kmeans_k6_complete_LATEST.csv')

if not results_file.exists():
    raise FileNotFoundError(
        f"❌ K-Means results file not found: {results_file}\n"
        f"   Please run kmeans.py first to generate clustering results."
    )

df = pd.read_csv(results_file)
print(f"✓ Loaded K-Means results: {results_file}")
print(f"  • Samples: {len(df):,}")
print(f"  • Clusters (K=6): {df['cluster'].value_counts().sort_index().to_dict()}")

# ============================================================================
# 2. FEATURE ENGINEERING & CLUSTER REORDERING
# ============================================================================
print("\n[Step 2] Verifying composite features...")

# Verify composite features exist (they should already be in the loaded data)
feature_names = ['BMI', 'chronic_disease_burden', 
                 'lifestyle_risk', 'substance_risk', 
                 'ses_vulnerability']

# Double-check they exist, if not recreate (safety check)
if not all(col in df.columns for col in feature_names):
    print("⚠️ Composite features missing, recreating...")
    df['chronic_disease_burden'] = (df['HighBP'] + df['HighChol'] + df['Stroke'] + 
                                      df['HeartDiseaseorAttack'] + df['DiffWalk'])
    # BMI already exists as a column, no need to create
    df['lifestyle_risk'] = 3 - (df['Veggies'] + df['Fruits'] + df['PhysActivity'])
    df['substance_risk'] = df['Smoker'] + df['HvyAlcoholConsump']
    df['ses_vulnerability'] = (10 - df['Income']) + (7 - df['Education'])

print(f"✓ Composite features verified: {feature_names}")

# Reorder clusters by diabetes rate (C0→C5 ascending)
print(f"\n[Step 3] Reordering clusters by diabetes prevalence...")
df['cluster_original'] = df['cluster'].copy()
diabetes_rates = []
for cluster_id in range(6):
    cluster_mask = df['cluster_original'] == cluster_id
    diabetes_rate = (df.loc[cluster_mask, 'Diabetes_012'] == 2).mean() * 100
    diabetes_rates.append((cluster_id, diabetes_rate))

# Sort by diabetes rate
diabetes_rates_sorted = sorted(diabetes_rates, key=lambda x: x[1])

# Create mapping: new_cluster_id <- old_cluster_id
cluster_mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(diabetes_rates_sorted)}
print(f"  Cluster mapping (old→new): {cluster_mapping}")

# Apply remapping
df['cluster'] = df['cluster_original'].map(cluster_mapping)
print(f"  Remapped cluster sizes: {df['cluster'].value_counts().sort_index().to_dict()}")

# ============================================================================
# 3. CLUSTER PROFILING - FEATURE MEANS
# ============================================================================
print("\n[Step 4] Computing cluster feature profiles...")

# Calculate mean for each feature in each cluster
profile_data = []
for cluster_id in range(6):
    cluster_mask = df['cluster'] == cluster_id
    profile = {
        'Cluster': cluster_id,
        'Size': cluster_mask.sum(),
        'Percent': f"{cluster_mask.sum()/len(df)*100:.1f}%"
    }
    
    # Feature means
    for feat in feature_names:
        profile[feat] = df.loc[cluster_mask, feat].mean()
    
    # Diabetes prevalence
    diabetes_dist = df.loc[cluster_mask, 'Diabetes_012'].value_counts(normalize=True)
    profile['Diabetes_Rate'] = diabetes_dist.get(2, 0) * 100  # Class 2 = diabetes
    profile['Prediabetes_Rate'] = diabetes_dist.get(1, 0) * 100  # Class 1 = prediabetes
    profile['Healthy_Rate'] = diabetes_dist.get(0, 0) * 100  # Class 0 = no diabetes
    profile['AtRisk_Rate'] = profile['Prediabetes_Rate'] + profile['Diabetes_Rate']  # At-Risk = Prediabetes + Diabetes
    
    profile_data.append(profile)

profile_df = pd.DataFrame(profile_data)

# Display profile table
print("\n" + "="*80)
print("CLUSTER FEATURE PROFILES")
print("="*80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(profile_df.to_string(index=False))

# ============================================================================
# 4. CLUSTER LABELING BASED ON CHARACTERISTICS
# ============================================================================
print("\n[Step 4] Generating cluster labels based on dominant characteristics...")

cluster_labels = {}
cluster_descriptions = {}
intervention_strategies = {}

# Define cluster labels based on clinical interpretation
CLUSTER_DEFINITIONS = {
    0: {
        'label': 'Low-Risk',
        'desc': 'Healthy baseline population with minimal risk factors',
        'intervention': 'Preventive care, health promotion, annual screening'
    },
    1: {
        'label': 'Socioeconomic Vulnerability',
        'desc': 'High socioeconomic barriers limiting healthcare access',
        'intervention': 'Free screening, community health workers, simplified education materials'
    },
    2: {
        'label': 'High-Risk Lifestyle',
        'desc': 'Poor behavioral profiles with high intervention potential',
        'intervention': 'Nutrition education, community exercise programs, cooking classes'
    },
    3: {
        'label': 'Chronic-Addiction Synergy',
        'desc': 'Chronic disease burden compounded by addiction behaviors',
        'intervention': 'Integrated chronic disease management, addiction treatment support'
    },
    4: {
        'label': 'Metabolic Extremes',
        'desc': 'Severe obesity driving metabolic syndrome and diabetes',
        'intervention': 'Weight management programs, dietary counseling, bariatric evaluation'
    },
    5: {
        'label': 'Frail Multi-Morbidity',
        'desc': 'Multiple chronic conditions with complex care needs',
        'intervention': 'Intensive case management, medication adherence, specialist referrals'
    }
}

for idx, row in profile_df.iterrows():
    cluster_id = int(row['Cluster'])
    
    # Assign pre-defined labels
    label = CLUSTER_DEFINITIONS[cluster_id]['label']
    desc = CLUSTER_DEFINITIONS[cluster_id]['desc']
    intervention = CLUSTER_DEFINITIONS[cluster_id]['intervention']
    
    cluster_labels[cluster_id] = label
    cluster_descriptions[cluster_id] = desc
    intervention_strategies[cluster_id] = intervention

# Add labels to profile
profile_df['Label'] = profile_df['Cluster'].map(cluster_labels)
profile_df['Description'] = profile_df['Cluster'].map(cluster_descriptions)
profile_df['Intervention_Strategy'] = profile_df['Cluster'].map(intervention_strategies)

print("\n" + "="*80)
print("CLUSTER LABELS & INTERVENTION STRATEGIES")
print("="*80)
for cluster_id in range(6):
    print(f"\n【Cluster {cluster_id}: {cluster_labels[cluster_id]}】")
    print(f"  Size: {profile_df.loc[cluster_id, 'Size']:,} ({profile_df.loc[cluster_id, 'Percent']})")
    print(f"  Description: {cluster_descriptions[cluster_id]}")
    print(f"  At-Risk Rate: {profile_df.loc[cluster_id, 'AtRisk_Rate']:.1f}% (Prediabetes + Diabetes)")
    print(f"  Intervention: {intervention_strategies[cluster_id]}")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
print("\n[Step 5] Creating visualizations...")

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/profiling')
fig_dir = base_dir / timestamp
fig_dir.mkdir(parents=True, exist_ok=True)

print(f"✓ Output directory: {fig_dir}")

# ========== PLOT 1: Feature Comparison Heatmap ==========
fig, ax = plt.subplots(figsize=(14, 8))

# Z-score standardization for heatmap colors (measures deviation from mean)
# This makes color represent "relative high/low" rather than absolute values
heatmap_data = profile_df[feature_names].copy()
for col in feature_names:
    col_mean = heatmap_data[col].mean()
    col_std = heatmap_data[col].std()
    if col_std > 0:
        heatmap_data[col] = (heatmap_data[col] - col_mean) / col_std  # Z-score
    else:
        heatmap_data[col] = 0

# Create heatmap: colors show Z-score, annotations show actual values
sns.heatmap(heatmap_data.T, annot=profile_df[feature_names].T, fmt='.2f',
            cmap='RdYlGn_r', center=0, vmin=-2, vmax=2,
            cbar_kws={'label': 'Z-Score (Standard Deviations from Mean)'},
            xticklabels=[f"C{i}: {cluster_labels[i][:20]}" for i in range(6)],
            yticklabels=['BMI', 'Chronic Disease', 'Lifestyle Risk', 
                        'Addiction Risk', 'SES Vulnerability'],
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_title('Cluster Feature Profile Comparison\n(Colors: Z-score deviation | Numbers: Actual mean values)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Cluster Groups', fontsize=12, fontweight='bold')
ax.set_ylabel('Composite Features', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'cluster_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: cluster_heatmap.png")

# ========== PLOT 2: At-Risk Prevalence by Cluster (Prediabetes + Diabetes) ==========
fig, ax = plt.subplots(figsize=(14, 6))

# Use shortened cluster labels for x-axis
cluster_labels_display = {
    0: 'C0\nLow-Risk',
    1: 'C1\nSocioeconomic\nVulnerability',
    2: 'C2\nHigh-Risk\nLifestyle',
    3: 'C3\nChronic-Addiction\nSynergy',
    4: 'C4\nMetabolic\nExtremes',
    5: 'C5\nFrail\nMulti-Morbidity'
}

x = np.arange(6)
width = 0.35

# Two-category visualization: No Diabetes vs At-Risk (Prediabetes + Diabetes merged)
bars1 = ax.bar(x - width/2, profile_df['Healthy_Rate'], width, 
               label='No Diabetes', color='#2ecc71', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, profile_df['AtRisk_Rate'], width, 
               label='At-Risk (Prediabetes + Diabetes)', color='#e74c3c', edgecolor='black', linewidth=1.2)

ax.set_xlabel('Cluster Groups', fontsize=12, fontweight='bold')
ax.set_ylabel('Prevalence (%)', fontsize=12, fontweight='bold')
ax.set_title('At-Risk Prevalence by Cluster\n(Validation of Risk Stratification)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([cluster_labels_display[i] for i in range(6)], fontsize=9)
ax.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 100)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 1:  # Only label if > 1%
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'diabetes_prevalence.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: diabetes_prevalence.png")

# ========== PLOT 3: Radar Chart for Each Cluster ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

# Normalize features to 0-1 for radar chart
radar_data = profile_df[feature_names].copy()
for col in feature_names:
    col_min = radar_data[col].min()
    col_max = radar_data[col].max()
    radar_data[col] = (radar_data[col] - col_min) / (col_max - col_min) if col_max > col_min else 0

angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

feature_labels = ['BMI', 'Chronic\nDisease', 'Lifestyle\nRisk', 
                  'Addiction\nRisk', 'SES\nVulnerability']

for idx, ax in enumerate(axes):
    values = radar_data.iloc[idx].tolist()
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2, label=cluster_labels[idx], color=f'C{idx}')
    ax.fill(angles, values, alpha=0.25, color=f'C{idx}')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7)
    ax.set_title(f"Cluster {idx}: {cluster_labels[idx]}\n(n={profile_df.loc[idx, 'Size']:,}, At-Risk={profile_df.loc[idx, 'AtRisk_Rate']:.1f}%)", 
                 fontsize=10, fontweight='bold', pad=15)
    ax.grid(True)

plt.suptitle('Cluster Feature Profiles - Radar Charts\n(Normalized 0-1 scale)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(fig_dir / 'cluster_radar.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: cluster_radar.png")
plt.close('all')

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n[Step 6] Saving profiling results...")

# Save detailed profile CSV
profile_output = fig_dir / 'cluster_profiles.csv'
profile_df.to_csv(profile_output, index=False)
print(f"✓ Saved: cluster_profiles.csv")

# Save intervention recommendations
intervention_output = fig_dir / 'intervention_strategies.txt'
with open(intervention_output, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("CLUSTER-SPECIFIC INTERVENTION STRATEGIES\n")
    f.write("="*80 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Population: {len(df):,}\n")
    f.write(f"Clustering Method: K-Means (K=6)\\n")
    f.write(f"Features: 5 composite indicators\\n\\n")
    
    for cluster_id in range(6):
        f.write("="*80 + "\n")
        f.write(f"CLUSTER {cluster_id}: {cluster_labels[cluster_id].upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Population Size: {profile_df.loc[cluster_id, 'Size']:,} ({profile_df.loc[cluster_id, 'Percent']})\n\n")
        f.write("RISK PROFILE:\n")
        f.write(f"  • Chronic Disease Burden: {profile_df.loc[cluster_id, 'chronic_disease_burden']:.2f}/5.0\n")
        f.write(f"  • BMI: {profile_df.loc[cluster_id, 'BMI']:.1f}\n")
        f.write(f"  • Lifestyle Risk: {profile_df.loc[cluster_id, 'lifestyle_risk']:.2f}/3.0\n")
        f.write(f"  • Addiction Risk: {profile_df.loc[cluster_id, 'substance_risk']:.2f}/2.0\n")
        f.write(f"  • SES Vulnerability: {profile_df.loc[cluster_id, 'ses_vulnerability']:.1f}/17.0\n\n")
        f.write("DIABETES PREVALENCE:\n")
        f.write(f"  • No Diabetes: {profile_df.loc[cluster_id, 'Healthy_Rate']:.1f}%\n")
        f.write(f"  • At-Risk (Prediabetes + Diabetes): {profile_df.loc[cluster_id, 'AtRisk_Rate']:.1f}%\n")
        f.write(f"    - Prediabetes: {profile_df.loc[cluster_id, 'Prediabetes_Rate']:.1f}%\n")
        f.write(f"    - Diabetes: {profile_df.loc[cluster_id, 'Diabetes_Rate']:.1f}%\n\n")
        f.write("CLINICAL DESCRIPTION:\n")
        f.write(f"  {cluster_descriptions[cluster_id]}\n\n")
        f.write("RECOMMENDED INTERVENTIONS:\n")
        f.write(f"  {intervention_strategies[cluster_id]}\n\n")
        
        # Financial feasibility note
        if profile_df.loc[cluster_id, 'ses_vulnerability'] > 10:
            f.write("IMPLEMENTATION CONSIDERATIONS:\n")
            f.write("  ⚠️  High SES vulnerability - Prioritize low-cost interventions:\n")
            f.write("     • Community-based programs (free)\n")
            f.write("     • Peer support groups\n")
            f.write("     • Public health messaging via social media\n")
            f.write("     • Mobile health clinics\n\n")
        
        # Communication strategy note
        if profile_df.loc[cluster_id, 'ses_vulnerability'] > 10:
            f.write("COMMUNICATION STRATEGY:\n")
            f.write("  • Use simple language (6th-grade reading level)\n")
            f.write("  • Visual aids and infographics\n")
            f.write("  • Community health workers for outreach\n")
            f.write("  • Multilingual materials if applicable\n\n")
        
        f.write("\n")

print(f"✓ Saved: intervention_strategies.txt")

# Copy kmeans.py log files to profiling directory
log_dir = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/log')
kmeans_log_pattern = str(log_dir / '5features_composite_kmeans_run_*.txt')
kmeans_logs = sorted(glob.glob(kmeans_log_pattern), reverse=True)

if kmeans_logs:
    # Copy the most recent kmeans log file
    latest_log = Path(kmeans_logs[0])
    log_copy_dest = fig_dir / f'kmeans_log_{latest_log.stem.split("_")[-1]}.txt'
    shutil.copy2(latest_log, log_copy_dest)
    print(f"✓ Saved: kmeans_log_{latest_log.stem.split('_')[-1]}.txt (copied from {latest_log.name})")
    
    # Also copy the corresponding results CSV
    kmeans_results_pattern = str(log_dir / '5features_composite_kmeans_results_*.csv')
    kmeans_results = sorted(glob.glob(kmeans_results_pattern), reverse=True)
    if kmeans_results:
        latest_results = Path(kmeans_results[0])
        results_copy_dest = fig_dir / f'kmeans_results_{latest_results.stem.split("_")[-1]}.csv'
        shutil.copy2(latest_results, results_copy_dest)
        print(f"✓ Saved: kmeans_results_{latest_results.stem.split('_')[-1]}.csv (copied from {latest_results.name})")
else:
    print("⚠️  Warning: No kmeans log files found to copy")

print("\n" + "="*80)
print("CLUSTER PROFILING COMPLETE")
print("="*80)
print(f"\n✅ All outputs saved to: {fig_dir}")
print(f"   (Timestamped folder: {timestamp})")
print(f"\nGenerated files:")
print(f"  1. cluster_profiles.csv")
print(f"  2. cluster_heatmap.png")
print(f"  3. diabetes_prevalence.png")
print(f"  4. cluster_radar.png")
print(f"  5. intervention_strategies.txt")
if kmeans_logs:
    print(f"  6. kmeans_log_{Path(kmeans_logs[0]).stem.split('_')[-1]}.txt (from kmeans.py)")
    if glob.glob(str(log_dir / '5features_composite_kmeans_results_*.csv')):
        print(f"  7. kmeans_results_{Path(kmeans_logs[0]).stem.split('_')[-1]}.csv (from kmeans.py)")
print("\n" + "="*80)
