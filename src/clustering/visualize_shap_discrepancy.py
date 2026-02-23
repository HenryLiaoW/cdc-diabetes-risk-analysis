import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 使用绝对路径指向SHAP分析结果目录
data_path = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/profiling/final_20260221_222939/shap_analysis/feature_importance_results.csv')
# 如果文件不存在，请确保路径正确
if not data_path.exists():
    print("Warning: CSV not found. Please ensure feature_importance_results.csv is in the directory.")
    # 这里为了演示，创建一个模拟数据结构（实际运行时请注释掉）
    df = pd.DataFrame({
        'Feature': ['chronic_disease_burden', 'ses_vulnerability', 'lifestyle_risk', 'BMI', 'substance_risk'],
        'RF_Gini_Importance': [0.245, 0.296, 0.228, 0.228, 0.001],
        'RF_SHAP_Importance': [0.121, 0.119, 0.096, 0.066, 0.001],
        'XGB_Gain_Importance': [0.321, 0.296, 0.282, 0.097, 0.002],
        'XGB_SHAP_Importance': [1.749, 1.845, 1.400, 1.244, 0.020],
        'RF_Gini_Importance_Rank': [2, 1, 3, 4, 5],
        'RF_SHAP_Importance_Rank': [1, 2, 3, 4, 5],
        'XGB_Gain_Importance_Rank': [1, 2, 3, 4, 5],
        'XGB_SHAP_Importance_Rank': [2, 1, 3, 4, 5]
    })
else:
    df = pd.read_csv(data_path)

# 配置
feature_labels = {
    'chronic_disease_burden': 'Chronic Disease Burden',
    'ses_vulnerability': 'SES Vulnerability',
    'lifestyle_risk': 'Lifestyle Risk',
    'BMI': 'BMI',
    'substance_risk': 'Substance Risk'
}
df['Label'] = df['Feature'].map(feature_labels)

# 计算平均排名来确定统一的特征顺序（便于对比）
df['Avg_Rank'] = (df['RF_SHAP_Importance_Rank'] + df['XGB_SHAP_Importance_Rank']) / 2
df_sorted = df.sort_values('Avg_Rank', ascending=False)  # 降序，重要的在上面
feature_order = df_sorted['Label'].tolist()

# 创建画布 (只保留1行2列的条形图对比)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# RF SHAP - 使用统一顺序
df_plot_rf = df.set_index('Label').loc[feature_order].reset_index()
bars1 = ax1.barh(df_plot_rf['Label'], df_plot_rf['RF_SHAP_Importance'], color='#4682B4', alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_title('Random Forest: SHAP Global Importance', fontweight='bold', fontsize=13)
ax1.set_xlabel('Mean |SHAP Value|', fontweight='bold', fontsize=11)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# 标注排名
for i, (bar, label) in enumerate(zip(bars1, df_plot_rf['Label'])):
    rank = df[df['Label'] == label]['RF_SHAP_Importance_Rank'].values[0]
    ax1.text(bar.get_width() * 0.05, bar.get_y() + bar.get_height()/2, 
             f'#{int(rank)}', va='center', fontsize=11, fontweight='bold', color='white')

# XGB SHAP - 使用统一顺序
df_plot_xgb = df.set_index('Label').loc[feature_order].reset_index()
bars2 = ax2.barh(df_plot_xgb['Label'], df_plot_xgb['XGB_SHAP_Importance'], color='#8B4513', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_title('XGBoost: SHAP Global Importance', fontweight='bold', fontsize=13)
ax2.set_xlabel('Mean |SHAP Value|', fontweight='bold', fontsize=11)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# 标注排名
for i, (bar, label) in enumerate(zip(bars2, df_plot_xgb['Label'])):
    rank = df[df['Label'] == label]['XGB_SHAP_Importance_Rank'].values[0]
    ax2.text(bar.get_width() * 0.05, bar.get_y() + bar.get_height()/2, 
             f'#{int(rank)}', va='center', fontsize=11, fontweight='bold', color='white')

plt.suptitle('SHAP Feature Importance Comparison: RF vs XGBoost\n⚠️ Notice: Top 2 features swap positions', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

output_path = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/profiling/final_20260221_222939/shap_analysis/feature_importance_audit.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")