"""
Demographic Risk Analysis: Identifying Vulnerable Populations

This analysis addresses the scenario requirement:
"Design targeted health interventions aimed at reducing diabetes prevalence, 
particularly in VULNERABLE GROUPS."

We need to identify WHO is at highest risk based on non-modifiable factors,
so that public health resources can be allocated appropriately.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

print("=" * 80)
print("DEMOGRAPHIC RISK ANALYSIS: IDENTIFYING VULNERABLE POPULATIONS")
print("=" * 80)

# Load data
df = pd.read_csv(DATA_DIR / 'raw' / 'CDC Diabetes Dataset.csv')
print(f"\nLoaded data: {len(df):,} records")

# Define demographic features (non-modifiable factors)
# These determine WHO needs intervention, not WHAT intervention they need
demographic_features = {
    'Age': 'Age Group',
    'Sex': 'Sex',
    'Income': 'Income Level',
    'Education': 'Education Level'
}

baseline_rate = (df['Diabetes_012'] > 0).sum() / len(df)
print(f"\nBaseline diabetes/prediabetes rate: {baseline_rate*100:.2f}%")

# ============================================================================
# Analysis 1: Diabetes Rate by Individual Demographic Factors
# ============================================================================
print("\n" + "=" * 80)
print("DIABETES RATES BY DEMOGRAPHIC FACTORS")
print("=" * 80)

demographic_summary = []

# Age analysis (detailed breakdown)
print("\n【AGE GRADIENT】")
age_labels = {1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
              6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
              11: '70-74', 12: '75-79', 13: '80+'}
for age in sorted(df['Age'].unique()):
    subset = df[df['Age'] == age]
    rate = (subset['Diabetes_012'] > 0).sum() / len(subset) * 100
    risk_ratio = rate / (baseline_rate * 100)
    demographic_summary.append({
        'Factor': 'Age',
        'Category': age_labels.get(age, f'Age_{age}'),
        'Diabetes Rate (%)': f"{rate:.2f}",
        'Sample Size': len(subset),
        'Risk Ratio': f"{risk_ratio:.2f}x"
    })
    print(f"  {age_labels.get(age, f'Age_{age}'):8s}: {rate:5.2f}% (n={len(subset):,}) - {risk_ratio:.2f}x baseline")

# Sex analysis
print("\n【SEX DISTRIBUTION】")
sex_labels = {0: 'Female', 1: 'Male'}
for sex in sorted(df['Sex'].unique()):
    subset = df[df['Sex'] == sex]
    rate = (subset['Diabetes_012'] > 0).sum() / len(subset) * 100
    risk_ratio = rate / (baseline_rate * 100)
    demographic_summary.append({
        'Factor': 'Sex',
        'Category': sex_labels[sex],
        'Diabetes Rate (%)': f"{rate:.2f}",
        'Sample Size': len(subset),
        'Risk Ratio': f"{risk_ratio:.2f}x"
    })
    print(f"  {sex_labels[sex]:8s}: {rate:5.2f}% (n={len(subset):,}) - {risk_ratio:.2f}x baseline")

# Income analysis
print("\n【INCOME DISPARITY】")
income_labels = {1: '<$10k', 2: '$10-15k', 3: '$15-20k', 4: '$20-25k',
                5: '$25-35k', 6: '$35-50k', 7: '$50-75k', 8: '>$75k'}
for income in sorted(df['Income'].unique()):
    subset = df[df['Income'] == income]
    rate = (subset['Diabetes_012'] > 0).sum() / len(subset) * 100
    risk_ratio = rate / (baseline_rate * 100)
    demographic_summary.append({
        'Factor': 'Income',
        'Category': income_labels.get(income, f'Income_{income}'),
        'Diabetes Rate (%)': f"{rate:.2f}",
        'Sample Size': len(subset),
        'Risk Ratio': f"{risk_ratio:.2f}x"
    })
    print(f"  {income_labels.get(income, f'Income_{income}'):10s}: {rate:5.2f}% (n={len(subset):,}) - {risk_ratio:.2f}x baseline")

# Education analysis
print("\n【EDUCATION GRADIENT】")
edu_labels = {1: 'Never/Kindergarten', 2: 'Elementary', 3: 'Some HS', 
              4: 'HS Grad', 5: 'Some College', 6: 'College Grad'}
for edu in sorted(df['Education'].unique()):
    subset = df[df['Education'] == edu]
    rate = (subset['Diabetes_012'] > 0).sum() / len(subset) * 100
    risk_ratio = rate / (baseline_rate * 100)
    demographic_summary.append({
        'Factor': 'Education',
        'Category': edu_labels.get(edu, f'Edu_{edu}'),
        'Diabetes Rate (%)': f"{rate:.2f}",
        'Sample Size': len(subset),
        'Risk Ratio': f"{risk_ratio:.2f}x"
    })
    print(f"  {edu_labels.get(edu, f'Edu_{edu}'):20s}: {rate:5.2f}% (n={len(subset):,}) - {risk_ratio:.2f}x baseline")

# Save demographic summary
demo_df = pd.DataFrame(demographic_summary)
demo_df.to_csv(OUTPUT_DIR / 'demographic_risk_by_factor.csv', index=False)

# ============================================================================
# Analysis 2: Multi-Dimensional Vulnerable Groups
# ============================================================================
print("\n" + "=" * 80)
print("IDENTIFYING HIGH-RISK DEMOGRAPHIC COMBINATIONS")
print("=" * 80)

# Create demographic profile
# Simplify to avoid too many combinations
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 6, 10, 13], labels=['Young (18-49)', 'Middle (50-64)', 'Senior (65+)'])
df['Income_Group'] = pd.cut(df['Income'], bins=[0, 4, 6, 8], labels=['Low (<$25k)', 'Medium ($25-50k)', 'High (>$50k)'])
df['Sex_Label'] = df['Sex'].map({0: 'Female', 1: 'Male'})

# Calculate risk for all combinations
demographic_combinations = df.groupby(['Age_Group', 'Income_Group', 'Sex_Label']).agg({
    'Diabetes_012': ['count', lambda x: (x > 0).sum(), lambda x: (x > 0).mean()]
}).reset_index()

demographic_combinations.columns = ['Age_Group', 'Income_Group', 'Sex', 'Total', 'Diabetic_Count', 'Diabetes_Rate']
demographic_combinations['Diabetes_Rate_Pct'] = demographic_combinations['Diabetes_Rate'] * 100
demographic_combinations['Risk_Ratio'] = demographic_combinations['Diabetes_Rate'] / baseline_rate

# Filter combinations with at least 1000 samples for reliability
demographic_combinations = demographic_combinations[demographic_combinations['Total'] >= 1000].copy()
demographic_combinations = demographic_combinations.sort_values('Diabetes_Rate', ascending=False)

# Top 10 highest risk groups
print("\n【TOP 10 HIGHEST RISK DEMOGRAPHIC GROUPS】")
print("(These are the vulnerable populations requiring targeted intervention)")
print("-" * 80)
for idx, row in demographic_combinations.head(10).iterrows():
    print(f"\n{list(demographic_combinations.head(10).index).index(idx) + 1}. {row['Age_Group']} + {row['Income_Group']} + {row['Sex']}")
    print(f"   Diabetes Rate: {row['Diabetes_Rate_Pct']:.2f}%")
    print(f"   Sample Size: {row['Total']:,} ({row['Diabetic_Count']:,} with diabetes)")
    print(f"   Risk Ratio: {row['Risk_Ratio']:.2f}x baseline")

# Top 10 lowest risk groups
print("\n" + "=" * 80)
print("【TOP 10 LOWEST RISK DEMOGRAPHIC GROUPS】")
print("-" * 80)
for idx, row in demographic_combinations.tail(10).iterrows():
    print(f"\n{list(demographic_combinations.tail(10).index).index(idx) + 1}. {row['Age_Group']} + {row['Income_Group']} + {row['Sex']}")
    print(f"   Diabetes Rate: {row['Diabetes_Rate_Pct']:.2f}%")
    print(f"   Sample Size: {row['Total']:,} ({row['Diabetic_Count']:,} with diabetes)")
    print(f"   Risk Ratio: {row['Risk_Ratio']:.2f}x baseline")

# Save results
demographic_combinations.to_csv(OUTPUT_DIR / 'demographic_risk_combinations.csv', index=False)

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "=" * 80)
print("Generating visualizations...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Demographic Risk Analysis: Identifying Vulnerable Populations', 
             fontsize=18, fontweight='bold')

# Plot 1: Age gradient
ax1 = axes[0, 0]
age_data = df.groupby('Age')['Diabetes_012'].apply(lambda x: (x > 0).mean() * 100).reset_index()
age_data['Age_Label'] = age_data['Age'].map(age_labels)
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(age_data)))
bars = ax1.bar(range(len(age_data)), age_data['Diabetes_012'], color=colors, 
               edgecolor='black', linewidth=1.5)
ax1.axhline(baseline_rate * 100, color='black', linestyle='--', linewidth=2, 
            label=f'Baseline: {baseline_rate*100:.1f}%', alpha=0.7)
ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax1.set_ylabel('Diabetes Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Age Gradient: Diabetes Risk Increases with Age', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(age_data)))
ax1.set_xticklabels(age_data['Age_Label'], rotation=45, ha='right', fontsize=9)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# Plot 2: Income disparity
ax2 = axes[0, 1]
income_data = df.groupby('Income')['Diabetes_012'].apply(lambda x: (x > 0).mean() * 100).reset_index()
income_data['Income_Label'] = income_data['Income'].map(income_labels)
colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(income_data)))
bars = ax2.bar(range(len(income_data)), income_data['Diabetes_012'], color=colors,
               edgecolor='black', linewidth=1.5)
ax2.axhline(baseline_rate * 100, color='black', linestyle='--', linewidth=2, 
            label=f'Baseline: {baseline_rate*100:.1f}%', alpha=0.7)
ax2.set_xlabel('Income Level', fontsize=12, fontweight='bold')
ax2.set_ylabel('Diabetes Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Income Disparity: Low Income = Higher Diabetes Risk', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(income_data)))
ax2.set_xticklabels(income_data['Income_Label'], rotation=45, ha='right', fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# Plot 3: Top 10 vulnerable groups
ax3 = axes[1, 0]
top10 = demographic_combinations.head(10).copy()
top10['Label'] = top10['Age_Group'].astype(str) + '\n' + top10['Income_Group'].astype(str) + '\n' + top10['Sex'].astype(str)
colors_vulnerable = plt.cm.Reds(np.linspace(0.5, 0.9, len(top10)))
bars = ax3.barh(range(len(top10)), top10['Diabetes_Rate_Pct'], color=colors_vulnerable,
                edgecolor='black', linewidth=1.5)
ax3.set_yticks(range(len(top10)))
ax3.set_yticklabels(top10['Label'], fontsize=8)
ax3.set_xlabel('Diabetes Rate (%)', fontsize=12, fontweight='bold')
ax3.set_title('TOP 10 Vulnerable Populations\n(Highest Risk Groups)', fontsize=14, fontweight='bold')
ax3.axvline(baseline_rate * 100, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

for i, (bar, val) in enumerate(zip(bars, top10['Diabetes_Rate_Pct'])):
    ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

# Plot 4: Education gradient
ax4 = axes[1, 1]
edu_data = df.groupby('Education')['Diabetes_012'].apply(lambda x: (x > 0).mean() * 100).reset_index()
edu_data['Edu_Label'] = edu_data['Education'].map(edu_labels)
colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(edu_data)))
bars = ax4.bar(range(len(edu_data)), edu_data['Diabetes_012'], color=colors,
               edgecolor='black', linewidth=1.5)
ax4.axhline(baseline_rate * 100, color='black', linestyle='--', linewidth=2, 
            label=f'Baseline: {baseline_rate*100:.1f}%', alpha=0.7)
ax4.set_xlabel('Education Level', fontsize=12, fontweight='bold')
ax4.set_ylabel('Diabetes Rate (%)', fontsize=12, fontweight='bold')
ax4.set_title('Education Gradient: Higher Education = Lower Risk', fontsize=14, fontweight='bold')
ax4.set_xticks(range(len(edu_data)))
ax4.set_xticklabels(edu_data['Edu_Label'], rotation=45, ha='right', fontsize=8)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_demographic_risk_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: 10_demographic_risk_analysis.png")

# ============================================================================
# Key Findings Summary
# ============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS: WHO NEEDS INTERVENTION MOST?")
print("=" * 80)

highest_risk = demographic_combinations.iloc[0]
lowest_risk = demographic_combinations.iloc[-1]

summary = f"""
CRITICAL PUBLIC HEALTH INSIGHT:
The diabetes burden is NOT equally distributed - certain demographic groups 
face dramatically higher risk through NO FAULT OF THEIR OWN.

1. AGE GRADIENT (Non-modifiable):
   - Young adults (18-24): 1.4% diabetes rate (0.09x baseline)
   - Seniors (70-74): 21.8% diabetes rate (1.38x baseline)
   - Implication: Screening programs must prioritize older adults

2. INCOME DISPARITY (Socioeconomic):
   - Low income (<$10k): ~26% diabetes rate (1.65x baseline)
   - High income (>$75k): ~8% diabetes rate (0.51x baseline)
   - Implication: 3.25x difference! Healthcare access and food security matter
   
3. EDUCATION GRADIENT (Social determinant):
   - Low education: ~22% diabetes rate
   - College graduate: ~9% diabetes rate
   - Implication: 2.44x difference - health literacy affects outcomes

4. INTERSECTIONALITY MATTERS:
   Highest Risk Group: {highest_risk['Age_Group']} + {highest_risk['Income_Group']} + {highest_risk['Sex']}
   → Diabetes Rate: {highest_risk['Diabetes_Rate_Pct']:.2f}% ({highest_risk['Risk_Ratio']:.2f}x baseline)
   → Sample Size: {highest_risk['Total']:,} people
   
   Lowest Risk Group: {lowest_risk['Age_Group']} + {lowest_risk['Income_Group']} + {lowest_risk['Sex']}
   → Diabetes Rate: {lowest_risk['Diabetes_Rate_Pct']:.2f}% ({lowest_risk['Risk_Ratio']:.2f}x baseline)
   → Sample Size: {lowest_risk['Total']:,} people
   
   Risk Ratio: {highest_risk['Diabetes_Rate_Pct'] / lowest_risk['Diabetes_Rate_Pct']:.2f}x difference

WHY THIS MATTERS FOR PUBLIC HEALTH POLICY:
✓ Can't tell seniors to "become younger" - but CAN provide free screening
✓ Can't tell poor people to "earn more" - but CAN subsidize healthy food/healthcare
✓ Can't change education history - but CAN provide health education programs

TARGETED INTERVENTION STRATEGY:
Priority 1: Senior + Low Income populations (Risk Ratio > 2.0x)
Priority 2: Middle-aged + Low/Medium Income (Risk Ratio 1.5-2.0x)
Priority 3: Universal lifestyle interventions (exercise, nutrition education)

The lifestyle analysis shows WHAT to do (exercise, eat vegetables).
This demographic analysis shows WHO to help first (vulnerable populations).
Together, they inform effective, equitable public health interventions.
"""

print(summary)

with open(OUTPUT_DIR / 'demographic_risk_summary.txt', 'w') as f:
    f.write(summary)

print("\n✓ Generated files:")
print("  1. demographic_risk_by_factor.csv - Diabetes rates for each demographic factor")
print("  2. demographic_risk_combinations.csv - Multi-dimensional risk profiles")
print("  3. 10_demographic_risk_analysis.png - 4 visualization charts")
print("  4. demographic_risk_summary.txt - Key findings and policy implications")
print("\nDemographic risk analysis complete!")
print("\nCRITICAL REFLECTION:")
print("Non-modifiable factors determine WHO is vulnerable.")
print("Modifiable factors determine WHAT interventions work.")
print("Both are essential for designing equitable, effective public health policy.")
