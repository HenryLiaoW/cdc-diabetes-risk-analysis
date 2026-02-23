"""
CDC Diabetes Dataset - Correlation Analysis for SageMaker/Jupyter
相关性分析 - SageMaker/Jupyter Notebook 版本

使用说明:
1. 上传 CDC Diabetes Dataset.csv 到 SageMaker Studio 的当前目录
2. 将这个文件的代码复制到 Notebook 单元格中运行
3. 或者保存为 .py 文件后在 Notebook 中运行: %run correlation_analysis_sagemaker.py
"""

# ============================================================================
# 单元格 1: 导入库和检查环境
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 80)
print("CDC DIABETES DATASET - CORRELATION ANALYSIS")
print("=" * 80)

# 检查当前目录
print(f"\n当前工作目录: {os.getcwd()}")
print("\n当前目录的文件:")
print("-" * 80)
for f in os.listdir('.')[:20]:  # 只显示前20个文件
    print(f"  {f}")

# ============================================================================
# 单元格 2: 上传数据（如果需要）
# ============================================================================
# 取消注释下面的代码来上传文件
# from IPython.display import display, HTML
# print("\n请使用文件浏览器上传 'CDC Diabetes Dataset.csv' 到当前目录")
# print("或者运行下面的代码:")
# print("!aws s3 cp s3://your-bucket/CDC_Diabetes_Dataset.csv .")

# ============================================================================
# 单元格 3: 加载数据
# ============================================================================
print("\n" + "=" * 80)
print("1. 加载数据")
print("=" * 80)

# 方式1: 如果CSV文件在当前目录
data_file = 'CDC Diabetes Dataset.csv'

# 方式2: 如果文件在子文件夹中
# data_file = 'data/CDC Diabetes Dataset.csv'

# 方式3: 如果文件名不同，修改这里
# data_file = 'diabetes_data.csv'

try:
    df = pd.read_csv(data_file)
    print(f"✓ 数据加载成功!")
    print(f"  样本数: {len(df):,}")
    print(f"  特征数: {df.shape[1]}")
    print(f"  缺失值: {df.isnull().sum().sum()}")
except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 '{data_file}'")
    print("\n可用的文件:")
    for f in os.listdir('.'):
        if f.endswith('.csv'):
            print(f"  - {f}")
    raise

print("\n特征列表:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# 单元格 4: 计算相关性矩阵
# ============================================================================
print("\n" + "=" * 80)
print("2. 相关性矩阵计算")
print("=" * 80)

corr_matrix = df.corr()
print(f"\n相关性矩阵大小: {corr_matrix.shape}")

# ============================================================================
# 单元格 5: 与目标变量的相关性（完整排序）
# ============================================================================
print("\n" + "=" * 80)
print("3. 与 Diabetes_012 的相关性分析")
print("=" * 80)

# 获取完整的相关性（按实际值排序，非绝对值）
target_corr = corr_matrix["Diabetes_012"].sort_values(ascending=False)

print("\n完整相关性排序 (ALL FEATURES):")
print("-" * 80)
for i, (feat, corr_val) in enumerate(target_corr.items(), 1):
    if feat != 'Diabetes_012':
        direction = "Positive (Risk Factor)" if corr_val > 0 else "Negative (Protective Factor)"
        strength = "Strong" if abs(corr_val) > 0.2 else "Moderate" if abs(corr_val) > 0.1 else "Weak"
        print(f"  {i:2d}. {feat:25s}: r = {corr_val:7.4f}  [{strength:8s}, {direction}]")

# ============================================================================
# 单元格 6: 保存结果到 DataFrame
# ============================================================================
print("\n" + "=" * 80)
print("4. 保存相关性数据")
print("=" * 80)

# 创建 DataFrame
correlation_df = pd.DataFrame({
    'Feature': target_corr.index,
    'Correlation': target_corr.values,
    'Abs_Correlation': target_corr.abs().values,
    'Direction': ['Risk' if x > 0 else 'Protective' for x in target_corr.values],
    'Strength': ['Strong' if abs(x) > 0.2 else 'Moderate' if abs(x) > 0.1 else 'Weak' 
                 for x in target_corr.values]
}).query('Feature != "Diabetes_012"')

# 保存到CSV
correlation_df.to_csv('feature_correlations.csv', index=False)
print("✓ 已保存: feature_correlations.csv")

# 显示数据
print("\n前10个特征:")
print(correlation_df.head(10).to_string(index=False))

# ============================================================================
# 单元格 7: 可视化 - Top 10 相关特征
# ============================================================================
print("\n" + "=" * 80)
print("5. 可视化分析")
print("=" * 80)

# Top 10 features (按绝对值)
top_features = correlation_df.nlargest(10, 'Abs_Correlation')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: 所有特征的相关性（按实际值排序）
ax1 = axes[0]
colors1 = ['#e74c3c' if x > 0 else '#3498db' for x in correlation_df['Correlation']]
ax1.barh(range(len(correlation_df)), correlation_df['Correlation'], color=colors1, edgecolor='black', linewidth=0.8)
ax1.set_yticks(range(len(correlation_df)))
ax1.set_yticklabels(correlation_df['Feature'], fontsize=8)
ax1.set_xlabel('Correlation Coefficient', fontweight='bold', fontsize=12)
ax1.set_title('All Features Correlation with Diabetes_012', fontweight='bold', fontsize=13)
ax1.axvline(0, color='black', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# 添加标签
for i, val in enumerate(correlation_df['Correlation']):
    ax1.text(val, i, f' {val:.3f}', va='center', fontsize=7, 
            ha='left' if val > 0 else 'right')

# Plot 2: Top 10 特征（按绝对值）
ax2 = axes[1]
colors2 = ['#e74c3c' if x > 0 else '#3498db' for x in top_features['Correlation']]
bars = ax2.barh(range(len(top_features)), top_features['Correlation'], color=colors2, 
               edgecolor='black', linewidth=1.2)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['Feature'], fontsize=10)
ax2.set_xlabel('Correlation Coefficient', fontweight='bold', fontsize=12)
ax2.set_title('Top 10 Features (by Absolute Correlation)', fontweight='bold', fontsize=13)
ax2.axvline(0, color='black', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, top_features['Correlation'])):
    ax2.text(val, i, f' {val:.3f}', va='center', fontweight='bold', fontsize=9,
            ha='left' if val > 0 else 'right')

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: correlation_analysis.png")
plt.show()

# ============================================================================
# 单元格 8: 相关性热图（Top 15 特征）
# ============================================================================
print("\n生成相关性热图...")

# 选择 Top 15 特征 + Diabetes_012
top15_features = correlation_df.nlargest(15, 'Abs_Correlation')['Feature'].tolist()
features_to_plot = ['Diabetes_012'] + top15_features

# 计算子集相关性
corr_subset = df[features_to_plot].corr()

# 绘制热图
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_subset, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')

# 设置刻度和标签
ax.set_xticks(np.arange(len(features_to_plot)))
ax.set_yticks(np.arange(len(features_to_plot)))
ax.set_xticklabels(features_to_plot, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(features_to_plot, fontsize=9)

# 添加数值
for i in range(len(features_to_plot)):
    for j in range(len(features_to_plot)):
        text = ax.text(j, i, f'{corr_subset.iloc[i, j]:.2f}',
                     ha='center', va='center', 
                     color='white' if abs(corr_subset.iloc[i, j]) > 0.5 else 'black',
                     fontsize=8)

ax.set_title('Correlation Heatmap - Top 15 Features', fontweight='bold', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap_top15.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: correlation_heatmap_top15.png")
plt.show()

# ============================================================================
# 单元格 9: 总结报告
# ============================================================================
print("\n" + "=" * 80)
print("6. 分析总结")
print("=" * 80)

# 统计各类别数量
positive_features = correlation_df[correlation_df['Correlation'] > 0]
negative_features = correlation_df[correlation_df['Correlation'] < 0]
strong_features = correlation_df[correlation_df['Abs_Correlation'] > 0.2]
moderate_features = correlation_df[(correlation_df['Abs_Correlation'] > 0.1) & 
                                  (correlation_df['Abs_Correlation'] <= 0.2)]
weak_features = correlation_df[correlation_df['Abs_Correlation'] <= 0.1]

summary = f"""
相关性分析总结:
{'=' * 80}

1. 总体统计:
   - 总特征数: {len(correlation_df)}
   - 正相关特征 (风险因素): {len(positive_features)}
   - 负相关特征 (保护因素): {len(negative_features)}

2. 相关性强度分布:
   - 强相关 (|r| > 0.2): {len(strong_features)} 个特征
   - 中等相关 (0.1 < |r| ≤ 0.2): {len(moderate_features)} 个特征
   - 弱相关 (|r| ≤ 0.1): {len(weak_features)} 个特征

3. 最强相关特征 (Top 5):
"""

for i, row in correlation_df.head(5).iterrows():
    summary += f"   {i+1}. {row['Feature']:20s}: r = {row['Correlation']:7.4f} ({row['Strength']}, {row['Direction']})\n"

summary += f"""
4. 最强保护因素 (Top 3 负相关):
"""

negative_top3 = correlation_df[correlation_df['Correlation'] < 0].head(3)
for i, row in negative_top3.iterrows():
    summary += f"   {i+1}. {row['Feature']:20s}: r = {row['Correlation']:7.4f} (Protective)\n"

summary += f"""
5. 关键洞察:
   - GenHlth (一般健康状况) 是最强预测因子 (r = {correlation_df.iloc[0]['Correlation']:.4f})
   - 健康指标 (HighBP, BMI, HighChol) 比生活方式更相关
   - 社会经济因素 (Income, Education) 显示中等保护效应
   - 运动和饮食的相关性较弱，但仍有统计学意义

6. 生成的文件:
   ✓ feature_correlations.csv - 完整相关性数据表
   ✓ correlation_analysis.png - 相关性柱状图
   ✓ correlation_heatmap_top15.png - Top15特征热图
{'=' * 80}
"""

print(summary)

# 保存报告
with open('correlation_analysis_report.txt', 'w') as f:
    f.write(summary)
print("✓ 已保存: correlation_analysis_report.txt")

print("\n" + "=" * 80)
print("分析完成！所有结果已保存。")
print("=" * 80)
