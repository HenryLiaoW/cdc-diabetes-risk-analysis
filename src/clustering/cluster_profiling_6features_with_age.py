"""
Cluster Profiling Analysis for K=6 K-Means Results - 6 Features (Including Age)

This script performs comprehensive cluster characterization:
1. Feature mean comparison across clusters
2. Diabetes prevalence analysis by cluster
3. Cluster labeling based on dominant characteristics
4. Intervention strategy recommendations
5. Visualization (radar charts, heatmaps, comparison tables)

Author: Cluster Analysis Pipeline
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

print("="*80)
print("CLUSTER PROFILING ANALYSIS - K=6 (6 FEATURES INCLUDING AGE)")
print("="*80)

# ============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================================
print("\n[Step 1] Loading data and recreating composite features...")
df = pd.read_csv('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/data/processed/CDC_Diabetes_Cleaned.csv')

# Recreate composite features (same as kmeans_6features_with_age.py)
df['chronic_disease_burden'] = (df['HighBP'] + df['HighChol'] + df['Stroke'] + 
                                  df['HeartDiseaseorAttack'] + df['DiffWalk'])
# BMI already exists as a column
df['healthy_lifestyle_score'] = df['Veggies'] + df['Fruits'] + df['PhysActivity']
df['substance_risk'] = df['Smoker'] + df['HvyAlcoholConsump']
df['ses_vulnerability'] = (10 - df['Income']) + (7 - df['Education'])
df['age'] = df['Age']  # Include Age as 6th feature

feature_names = ['BMI', 'chronic_disease_burden', 
                 'healthy_lifestyle_score', 'substance_risk', 
                 'ses_vulnerability', 'age']

X_cluster = df[feature_names]
print(f"✓ Dataset shape: {X_cluster.shape}")

# ============================================================================
# 2. SCALING & CLUSTERING
# ============================================================================
print("\n[Step 2] Applying data-driven mixed scaling and K-Means clustering (K=6)...")

# Data-driven preprocessing (including Age)
robust_features = ['BMI', 'healthy_lifestyle_score']
standard_features = ['chronic_disease_burden', 'ses_vulnerability', 'age']
minmax_features = ['substance_risk']

preprocessor = ColumnTransformer(
    transformers=[
        ('robust', RobustScaler(), robust_features),
        ('standard', StandardScaler(), standard_features),
        ('minmax', MinMaxScaler(), minmax_features)
    ],
    remainder='drop'
)

X_scaled = preprocessor.fit_transform(X_cluster)

print("✓ Data-driven mixed scaling applied (6 features including Age):")
print("  • RobustScaler: BMI, Lifestyle (skewed/outliers)")
print("  • StandardScaler: ChronicDisease, SES, Age (symmetric/ordinal)")
print("  • MinMaxScaler: SubstanceRisk (51.3% sparse)")

# Train K-Means with K=6
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df['cluster_original'] = kmeans.fit_predict(X_scaled)

print(f"✓ Clustering complete")
print(f"  Original cluster sizes: {df['cluster_original'].value_counts().sort_index().to_dict()}")

# Reorder clusters by diabetes rate (ascending)
print("\n[Step 2b] Reordering clusters by diabetes rate (C0→C5 ascending)...")
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
print("\n[Step 3] Computing cluster feature profiles...")

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
    
    profile_data.append(profile)

profile_df = pd.DataFrame(profile_data)

# Display profile table
print("\n" + "="*80)
print("CLUSTER FEATURE PROFILES (6 FEATURES INCLUDING AGE)")
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

for idx, row in profile_df.iterrows():
    cluster_id = int(row['Cluster'])
    
    # Labeling logic based on feature values (including age)
    chronic = row['chronic_disease_burden']
    lifestyle = row['healthy_lifestyle_score']
    substance = row['substance_risk']
    ses = row['ses_vulnerability']
    bmi = row['BMI']
    age = row['age']  # Age factor
    diabetes_rate = row['Diabetes_Rate']
    
    # Determine label
    if chronic >= 2.5:
        label = "Multi-Morbidity Group"
        desc = f"High chronic disease burden (avg={chronic:.2f}), age={age:.1f}, diabetes rate {diabetes_rate:.1f}%"
        intervention = "Intensive case management, medication adherence, specialist referrals"
    elif age >= 10:  # Elderly group (age bracket 10+ typically represents 65+ years)
        label = "Elderly High-Risk Group"
        desc = f"Advanced age (avg={age:.1f}), chronic conditions (avg={chronic:.2f}), diabetes {diabetes_rate:.1f}%"
        intervention = "Geriatric care coordination, fall prevention, chronic disease monitoring"
    elif substance >= 1.0:
        label = "Substance Risk Group"
        desc = f"High substance use risk (avg={substance:.2f}), age={age:.1f}, requires cessation support"
        intervention = "Smoking cessation programs, alcohol counseling, behavioral therapy"
    elif ses >= 10:
        label = "Vulnerable Group"
        desc = f"High socioeconomic vulnerability (avg={ses:.1f}), age={age:.1f}, financial/educational barriers"
        intervention = "Free screening, community health workers, simplified education materials"
    elif lifestyle <= 1.0:
        label = "Poor Lifestyle Group"
        desc = f"Low healthy behaviors (avg={lifestyle:.2f}), age={age:.1f}, high intervention potential"
        intervention = "Nutrition education, community exercise programs, cooking classes"
    elif bmi >= 32:
        label = "Metabolic Risk Group"
        desc = f"Elevated BMI (avg={bmi:.1f}), age={age:.1f}, metabolic syndrome concerns"
        intervention = "Weight management programs, dietary counseling, physical activity"
    else:
        label = "Moderate Risk Group"
        desc = f"Mixed risk profile, age={age:.1f}, tailored interventions needed"
        intervention = "Risk-stratified monitoring, lifestyle modification support"
    
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
    print(f"  Diabetes Rate: {profile_df.loc[cluster_id, 'Diabetes_Rate']:.1f}%")
    print(f"  Intervention: {intervention_strategies[cluster_id]}")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
print("\n[Step 5] Creating visualizations...")

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/profiling_6features_includingAGE')
fig_dir = base_dir / timestamp
fig_dir.mkdir(parents=True, exist_ok=True)

print(f"✓ Output directory: {fig_dir}")

# ========== PLOT 1: Feature Comparison Heatmap ==========
fig, ax = plt.subplots(figsize=(14, 9))

# Z-score standardization for heatmap colors
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
            yticklabels=['BMI', 'Chronic Disease', 'Healthy Lifestyle', 
                        'Substance Risk', 'SES Vulnerability', 'Age'],
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_title('Cluster Feature Profile Comparison - 6 Features (Including Age)\n(Colors: Z-score deviation | Numbers: Actual mean values)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Cluster Groups', fontsize=12, fontweight='bold')
ax.set_ylabel('Composite Features', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'cluster_heatmap_6features_includingAGE.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: cluster_heatmap_6features_includingAGE.png")

# ========== PLOT 2: Diabetes Prevalence by Cluster ==========
fig, ax = plt.subplots(figsize=(12, 6))

cluster_labels_short = [f"C{i}\n{cluster_labels[i][:15]}" for i in range(6)]
x = np.arange(6)
width = 0.25

bars1 = ax.bar(x - width, profile_df['Healthy_Rate'], width, 
               label='No Diabetes', color='#2ecc71', edgecolor='black')
bars2 = ax.bar(x, profile_df['Prediabetes_Rate'], width, 
               label='Prediabetes', color='#f39c12', edgecolor='black')
bars3 = ax.bar(x + width, profile_df['Diabetes_Rate'], width, 
               label='Diabetes', color='#e74c3c', edgecolor='black')

ax.set_xlabel('Cluster Groups', fontsize=12, fontweight='bold')
ax.set_ylabel('Prevalence (%)', fontsize=12, fontweight='bold')
ax.set_title('Diabetes Prevalence by Cluster - 6 Features (Including Age)\n(Validation of Risk Stratification)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(cluster_labels_short, fontsize=9)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 1:  # Only label if > 1%
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(fig_dir / 'diabetes_prevalence_6features_includingAGE.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: diabetes_prevalence_6features_includingAGE.png")

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

feature_labels = ['BMI', 'Chronic\nDisease', 'Healthy\nLifestyle', 
                  'Substance\nRisk', 'SES\nVulnerability', 'Age']

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
    ax.set_title(f"Cluster {idx}: {cluster_labels[idx]}\n(n={profile_df.loc[idx, 'Size']:,}, DM={profile_df.loc[idx, 'Diabetes_Rate']:.1f}%)", 
                 fontsize=10, fontweight='bold', pad=15)
    ax.grid(True)

plt.suptitle('Cluster Feature Profiles - Radar Charts (6 Features Including Age)\n(Normalized 0-1 scale)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(fig_dir / 'cluster_radar_6features_includingAGE.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: cluster_radar_6features_includingAGE.png")
plt.close('all')

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n[Step 6] Saving profiling results...")

# Save detailed profile CSV
profile_output = fig_dir / 'cluster_profiles_6features_includingAGE.csv'
profile_df.to_csv(profile_output, index=False)
print(f"✓ Saved: cluster_profiles_6features_includingAGE.csv")

# Save intervention recommendations
intervention_output = fig_dir / 'intervention_strategies_6features_includingAGE.txt'
with open(intervention_output, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("CLUSTER-SPECIFIC INTERVENTION STRATEGIES - 6 FEATURES (INCLUDING AGE)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Population: {len(df):,}\n")
    f.write(f"Clustering Method: K-Means (K=6)\n")
    f.write(f"Features: 6 composite indicators (including Age)\n\n")
    
    for cluster_id in range(6):
        f.write("="*80 + "\n")
        f.write(f"CLUSTER {cluster_id}: {cluster_labels[cluster_id].upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Population Size: {profile_df.loc[cluster_id, 'Size']:,} ({profile_df.loc[cluster_id, 'Percent']})\n\n")
        f.write("RISK PROFILE:\n")
        f.write(f"  • Age: {profile_df.loc[cluster_id, 'age']:.1f} (ordinal, 1-13 scale)\n")
        f.write(f"  • Chronic Disease Burden: {profile_df.loc[cluster_id, 'chronic_disease_burden']:.2f}/5.0\n")
        f.write(f"  • BMI: {profile_df.loc[cluster_id, 'BMI']:.1f}\n")
        f.write(f"  • Healthy Lifestyle Score: {profile_df.loc[cluster_id, 'healthy_lifestyle_score']:.2f}/3.0\n")
        f.write(f"  • Substance Risk: {profile_df.loc[cluster_id, 'substance_risk']:.2f}/2.0\n")
        f.write(f"  • SES Vulnerability: {profile_df.loc[cluster_id, 'ses_vulnerability']:.1f}/17.0\n\n")
        f.write("DIABETES PREVALENCE:\n")
        f.write(f"  • No Diabetes: {profile_df.loc[cluster_id, 'Healthy_Rate']:.1f}%\n")
        f.write(f"  • Prediabetes: {profile_df.loc[cluster_id, 'Prediabetes_Rate']:.1f}%\n")
        f.write(f"  • Diabetes: {profile_df.loc[cluster_id, 'Diabetes_Rate']:.1f}%\n\n")
        f.write("CLINICAL DESCRIPTION:\n")
        f.write(f"  {cluster_descriptions[cluster_id]}\n\n")
        f.write("RECOMMENDED INTERVENTIONS:\n")
        f.write(f"  {intervention_strategies[cluster_id]}\n\n")
        
        # Age-specific considerations
        if profile_df.loc[cluster_id, 'age'] >= 10:
            f.write("AGE-SPECIFIC CONSIDERATIONS:\n")
            f.write("  ⚠️  Elderly population - Special attention required:\n")
            f.write("     • Polypharmacy management\n")
            f.write("     • Cognitive assessment for medication adherence\n")
            f.write("     • Home health visits\n")
            f.write("     • Caregiver education and support\n\n")
        elif profile_df.loc[cluster_id, 'age'] <= 4:
            f.write("AGE-SPECIFIC CONSIDERATIONS:\n")
            f.write("  • Younger adults - Prevention focus:\n")
            f.write("     • Early lifestyle modification\n")
            f.write("     • Digital health interventions\n")
            f.write("     • Workplace wellness programs\n")
            f.write("     • Family-based interventions\n\n")
        
        # Financial feasibility note
        if profile_df.loc[cluster_id, 'ses_vulnerability'] > 10:
            f.write("IMPLEMENTATION CONSIDERATIONS:\n")
            f.write("  ⚠️  High SES vulnerability - Prioritize low-cost interventions:\n")
            f.write("     • Community-based programs (free)\n")
            f.write("     • Peer support groups\n")
            f.write("     • Public health messaging via social media\n")
            f.write("     • Mobile health clinics\n\n")
        
        f.write("\n")

print(f"✓ Saved: intervention_strategies_6features_includingAGE.txt")

print("\n" + "="*80)
print("CLUSTER PROFILING COMPLETE - 6 FEATURES (INCLUDING AGE)")
print("="*80)
print(f"\n✅ All outputs saved to: {fig_dir}")
print(f"   (Timestamped folder: {timestamp})")
print(f"\nGenerated files:")
print(f"  1. cluster_profiles_6features_includingAGE.csv")
print(f"  2. cluster_heatmap_6features_includingAGE.png")
print(f"  3. diabetes_prevalence_6features_includingAGE.png")
print(f"  4. cluster_radar_6features_includingAGE.png")
print(f"  5. intervention_strategies_6features_includingAGE.txt")
print("\n" + "="*80)
