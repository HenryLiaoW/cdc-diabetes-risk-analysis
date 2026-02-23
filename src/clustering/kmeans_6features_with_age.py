from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("="*80)
print("K-MEANS CLUSTERING ANALYSIS - 6 FEATURES INCLUDING AGE")
print("="*80)

# 1. 准备聚类特征
print("\n[Step 1] Loading data and preparing features...")
df = pd.read_csv('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/data/processed/CDC_Diabetes_Cleaned.csv')

# 特征工程：基于学术最佳实践的维度聚合 + Age
print("\n[Step 1.1] Feature Engineering - Creating 6 composite indicators (including Age)...")

# 1. 慢性病负担 (0-5)：多重慢病严重程度
df['chronic_disease_burden'] = (df['HighBP'] + df['HighChol'] + df['Stroke'] + 
                                  df['HeartDiseaseorAttack'] + df['DiffWalk'])

# 2. 代谢基准：BMI（保持原始连续值，最核心的糖尿病预测因子）
df['metabolic_baseline'] = df['BMI']

# 3. 生活方式风险 (0-3)：不健康行为的聚合（统一风险方向）
df['lifestyle_risk'] = 3 - (df['Veggies'] + df['Fruits'] + df['PhysActivity'])

# 4. 物质使用风险 (0-2)：成瘾行为指标
df['substance_risk'] = df['Smoker'] + df['HvyAlcoholConsump']

# 5. 社会经济脆弱性 (0-17)：精准定位弱势群体
df['ses_vulnerability'] = (10 - df['Income']) + (7 - df['Education'])

# 6. 年龄 (Age)：重要的糖尿病风险因子
df['age'] = df['Age']

print(f"✓ Created 6 composite features (including Age):")
print(f"  1. chronic_disease_burden (0-5): Sum of 5 chronic conditions")
print(f"  2. metabolic_baseline: BMI (continuous)")
print(f"  3. lifestyle_risk (0-3): 3 - (Veggies + Fruits + PhysActivity)")
print(f"  4. substance_risk (0-2): Smoker + HvyAlcoholConsump")
print(f"  5. ses_vulnerability (0-17): (10-Income) + (7-Education)")
print(f"  6. age: Age (1-13, ordinal 5-year brackets)")

# 定义特征组（基于分布分析的数据驱动缩放策略）
# Age使用StandardScaler（假设为ordinal编码的年龄组）

robust_features = ['metabolic_baseline', 'lifestyle_risk']
standard_features = ['chronic_disease_burden', 'ses_vulnerability', 'age']
minmax_features = ['substance_risk']

all_features = robust_features + standard_features + minmax_features
n_features = len(all_features)
X_cluster = df[all_features]

# 创建日志目录和文件名（使用特征数作为前缀并标注包含Age）
log_dir = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/log')
log_dir.mkdir(parents=True, exist_ok=True)

# 创建时间戳用于日志文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"{n_features}features_includingAGE_composite_kmeans_run_{timestamp}.txt"
results_csv = log_dir / f"{n_features}features_includingAGE_composite_kmeans_results_{timestamp}.csv"

# 创建日志文件并写入头部信息
with open(log_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("K-MEANS CLUSTERING ANALYSIS LOG - 6 FEATURES INCLUDING AGE\n")
    f.write("="*80 + "\n")
    f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Number of Features: {n_features} (including Age)\n")
    f.write(f"Log File: {log_file}\n")
    f.write("="*80 + "\n\n")

print(f"✓ Dataset shape: {X_cluster.shape}")
print(f"\n✓ Data-Driven Mixed Scaling Strategy (6 features, including Age):")
print(f"  • RobustScaler (2 features): {robust_features}")
print(f"    → Skewed/Outliers: BMI(skew=2.06), Lifestyle(skew=-0.73, outliers=5.3%)")
print(f"  • StandardScaler (3 features): {standard_features}")
print(f"    → Symmetric/Ordinal: ChronicDisease, SES, Age")
print(f"  • MinMaxScaler (1 feature): {minmax_features}")
print(f"    → Sparse: SubstanceRisk(51.3% zeros)")

# 记录到日志
with open(log_file, 'a', encoding='utf-8') as f:
    f.write("[Step 1] Feature Engineering - Composite Indicators\n")
    f.write("="*60 + "\n")
    f.write("Strategy: Dimension aggregation following CRISP-DM framework\n")
    f.write("          Address feature redundancy and scale conflicts\n\n")
    f.write(f"Dataset shape: {X_cluster.shape}\n")
    f.write(f"Total features: {n_features}\n\n")
    f.write("Composite Feature Design:\n")
    f.write("="*60 + "\n")
    f.write("1. chronic_disease_burden (0-5)\n")
    f.write("   Formula: HighBP + HighChol + Stroke + HeartDiseaseorAttack + DiffWalk\n")
    f.write("   Purpose: Multi-morbidity severity grading\n")
    f.write("   Academic Value: Identify high-risk populations with multiple conditions\n\n")
    f.write("2. metabolic_baseline (continuous)\n")
    f.write("   Formula: BMI (original value)\n")
    f.write("   Purpose: Core physiological diabetes predictor\n")
    f.write("   Academic Value: Most significant continuous risk factor\n\n")
    f.write("3. lifestyle_risk (0-3)\n")
    f.write("   Formula: Veggies + Fruits + PhysActivity\n")
    f.write("   Purpose: Modifiable behavior intervention potential\n")
    f.write("   Academic Value: Actionable composite for health promotion\n\n")
    f.write("4. substance_risk (0-2)\n")
    f.write("   Formula: Smoker + HvyAlcoholConsump\n")
    f.write("   Purpose: Addiction behavior intervention targeting\n")
    f.write("   Academic Value: Identify cessation support needs\n\n")
    f.write("5. ses_vulnerability (0-17)\n")
    f.write("   Formula: (10-Income) + (7-Education)\n")
    f.write("   Purpose: Vulnerable group identification (FAIR requirement 39)\n")
    f.write("   Academic Value: Tailor financial/communication strategies\n\n")
    f.write("6. age (1-13)\n")
    f.write("   Formula: Age (original ordinal encoding)\n")
    f.write("   Purpose: Demographic age factor (strong diabetes predictor)\n")
    f.write("   Academic Value: Enable age-stratified risk profiling\n\n")
    f.write("Design Rationale:\n")
    f.write("  ✓ Avoid feature redundancy (no double-counting)\n")
    f.write("  ✓ Data-driven scaling based on distribution analysis\n")
    f.write("  ✓ Equal-weight assumption for public health fairness\n")
    f.write("  ✓ XGBoost will learn true weights in classification phase\n\n")
    f.write("Scaling Strategy (Distribution-Based):\n")
    f.write(f"  • RobustScaler: {robust_features}\n")
    f.write("    Reason: Long-tail/Skewed distributions with outliers\n")
    f.write(f"  • StandardScaler: {standard_features}\n")
    f.write("    Reason: Relatively symmetric distributions or ordinal variables\n")
    f.write(f"  • MinMaxScaler: {minmax_features}\n")
    f.write("    Reason: Sparse/Binary-like distribution (51.3% zeros)\n\n")

# 2. 数据驱动的混合缩放：基于分布分析结果
print("\n[Step 2] Applying data-driven mixed scaling strategy...")
preprocessor = ColumnTransformer(
    transformers=[
        ('robust', RobustScaler(), robust_features),       # 2 features: BMI, Lifestyle
        ('standard', StandardScaler(), standard_features), # 3 features: ChronicDisease, SES, Age
        ('minmax', MinMaxScaler(), minmax_features)        # 1 feature: SubstanceRisk (sparse)
    ],
    remainder='drop'  # No passthrough needed
)

X_scaled = preprocessor.fit_transform(X_cluster)
print("✓ Data-driven mixed scaling applied (6 features including Age):")
print("  • RobustScaler: metabolic_baseline, lifestyle_risk")
print("  • StandardScaler: chronic_disease_burden, ses_vulnerability, age")
print("  • MinMaxScaler: substance_risk (51.3% sparse)")

# 记录到日志
with open(log_file, 'a', encoding='utf-8') as f:
    f.write("[Step 2] Data-Driven Feature Scaling (6 features including Age)\n")
    f.write("Mixed scaling strategy (based on distribution analysis):\n")
    f.write("  • RobustScaler: metabolic_baseline, lifestyle_risk\n")
    f.write("    → Reason: Long-tail/Skewed distributions (|skewness| > 0.7 or outliers > 4%)\n")
    f.write("  • StandardScaler: chronic_disease_burden, ses_vulnerability, age\n")
    f.write("    → Reason: Relatively symmetric distributions or ordinal variables\n")
    f.write("  • MinMaxScaler: substance_risk\n")
    f.write("    → Reason: Sparse distribution (51.3% zeros)\n")
    f.write(f"Scaled data shape: {X_scaled.shape}\n")
    f.write("\nNote: Scaling strategy determined by distribution analysis report\n")
    f.write("      (see outputs/clustering/distribution_analysis/)\n")
    f.write("      Age added as ordinal demographic factor with StandardScaler\n\n")

# 3. 寻找最优 K (手肘法 + 轮廓系数)
print("\n[Step 3] Evaluating different K values (K=2 to 7)...")
print("⚠️  Note: Using 50,000 sample for Silhouette Score (faster computation)")

# 为 Silhouette Score 创建样本（避免 O(n²) 对大数据集太慢）
sample_size = min(50000, len(X_scaled))
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]

# 记录到日志
with open(log_file, 'a', encoding='utf-8') as f:
    f.write("[Step 3] K-Means Evaluation\n")
    f.write(f"K range: 2 to 7\n")
    f.write(f"Sample size for Silhouette Score: {sample_size:,}\n")
    f.write(f"Sample ratio: {sample_size/len(X_scaled)*100:.2f}%\n")
    f.write("="*60 + "\n\n")

inertia = []
silhouette_scores = []
K_range = range(2, 8)  # K=2到7，对大多数应用足够

# 用于保存详细结果的列表
detailed_results = []

for k in K_range:
    print(f"  Testing K={k}...", end=' ')
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    
    # 计算轮廓系数 (使用样本数据，范围 -1 到 1，越接近 1 越好)
    labels_sample = kmeans.predict(X_sample)
    sil_score = silhouette_score(X_sample, labels_sample)
    silhouette_scores.append(sil_score)
    
    # 计算每个簇的大小
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    print(f"Inertia={kmeans.inertia_:.0f}, Silhouette={sil_score:.4f}")
    
    # 记录到日志文件
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"K = {k}\n")
        f.write(f"  Inertia: {kmeans.inertia_:.2f}\n")
        f.write(f"  Silhouette Score: {sil_score:.4f}\n")
        f.write(f"  Cluster Sizes: {cluster_sizes}\n")
        f.write(f"  Cluster Centers shape: {kmeans.cluster_centers_.shape}\n")
        f.write("\n")
    
    # 保存详细结果用于CSV
    detailed_results.append({
        'K': k,
        'Inertia': kmeans.inertia_,
        'Silhouette_Score': sil_score,
        'Cluster_Sizes': str(cluster_sizes),
        'N_Samples': len(X_scaled),
        'N_Features': X_scaled.shape[1],
        'Sample_Size': sample_size
    })

# 找到最优 K (silhouette score 最高)
best_k = K_range[np.argmax(silhouette_scores)]
best_silhouette = max(silhouette_scores)
print(f"\n✓ Best K based on Silhouette Score: {best_k} (score={best_silhouette:.4f})")

# 保存结果到CSV
results_df = pd.DataFrame(detailed_results)
results_df.to_csv(results_csv, index=False)

# 写入总结到日志
with open(log_file, 'a', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("RESULTS SUMMARY\n")
    f.write("="*60 + "\n")
    f.write(f"Best K: {best_k}\n")
    f.write(f"Best Silhouette Score: {best_silhouette:.4f}\n")
    f.write(f"\nAll Results:\n")
    for i, k in enumerate(K_range):
        f.write(f"  K={k}: Inertia={inertia[i]:.2f}, Silhouette={silhouette_scores[i]:.4f}\n")
    f.write(f"\nResults saved to: {results_csv}\n")
    f.write("="*60 + "\n")

print(f"✓ Results saved to: {results_csv}")
print(f"✓ Log saved to: {log_file}")

# ============================================================================
# 4. TRAIN FINAL MODEL WITH K=6 AND SAVE COMPLETE RESULTS
# ============================================================================
print(f"\n[Step 4] Training final K=6 model and saving complete dataset for visualization...")

# Train K-Means with K=6 (fixed for radar chart visualization)
FINAL_K = 6
kmeans_final = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df_with_clusters = df.copy()
df_with_clusters['cluster'] = cluster_labels

# Create output directory
results_dir = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/results')
results_dir.mkdir(parents=True, exist_ok=True)

# Save complete dataset with cluster assignments
results_file = results_dir / f'kmeans_k{FINAL_K}_with_age_complete_{timestamp}.csv'
df_with_clusters.to_csv(results_file, index=False)

# Also save a "LATEST" version for cluster_profiling.py
latest_file = results_dir / f'kmeans_k{FINAL_K}_with_age_complete_LATEST.csv'
df_with_clusters.to_csv(latest_file, index=False)

# Calculate and save cluster statistics
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
silhouette_k6 = silhouette_score(X_scaled, cluster_labels, sample_size=sample_size)

with open(log_file, 'a', encoding='utf-8') as f:
    f.write("\n" + "="*60 + "\n")
    f.write(f"FINAL MODEL (K={FINAL_K}) - FOR VISUALIZATION\n")
    f.write("="*60 + "\n")
    f.write(f"Silhouette Score: {silhouette_k6:.4f}\n")
    f.write(f"Inertia: {kmeans_final.inertia_:.2f}\n\n")
    f.write("Cluster Distribution:\n")
    for cluster_id, count in cluster_counts.items():
        pct = (count / len(cluster_labels)) * 100
        f.write(f"  Cluster {cluster_id}: {count:,} samples ({pct:.2f}%)\n")
    f.write(f"\nComplete dataset saved to:\n")
    f.write(f"  • {results_file}\n")
    f.write(f"  • {latest_file} (for cluster_profiling.py)\n")
    f.write("="*60 + "\n")

print(f"✓ Complete dataset with K={FINAL_K} clusters saved:")
print(f"  • {results_file}")
print(f"  • {latest_file}")
print(f"\n  Cluster distribution:")
for cluster_id, count in cluster_counts.items():
    pct = (count / len(cluster_labels)) * 100
    print(f"    Cluster {cluster_id}: {count:,} samples ({pct:.2f}%)")
print(f"  Silhouette Score: {silhouette_k6:.4f}")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
print(f"\n[Step 5] Generating evaluation plots...")

# 可视化：手肘图 + 轮廓系数图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 左图：手肘图 (Elbow Method)
ax1.plot(K_range, inertia, 'bo-', markersize=8, linewidth=2, label='Inertia')
ax1.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=11, fontweight='bold')
ax1.set_title('Elbow Method for Optimal K - 6 Features (Including Age)', fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 右图：轮廓系数图 (Silhouette Score)
ax2.plot(K_range, silhouette_scores, 'ro-', markersize=8, linewidth=2, label='Silhouette Score')
ax2.axhline(y=best_silhouette, color='g', linestyle='--', alpha=0.7, label=f'Best Score ({best_silhouette:.4f})')
ax2.axvline(x=best_k, color='g', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax2.scatter([best_k], [best_silhouette], color='green', s=200, zorder=5, marker='*', 
           edgecolors='black', linewidth=2)
ax2.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
ax2.set_title('Silhouette Score Analysis - 6 Features (Including Age)', fontsize=13, fontweight='bold', pad=15)
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')

# 添加解释文本
ax2.text(0.05, 0.95, 'Interpretation:\n• Score > 0.5: Good separation\n• Score > 0.7: Strong separation\n• Score < 0.3: Weak separation',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
fig_dir = Path('/Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project/outputs/clustering/fig')
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / f'{n_features}features_includingAGE_kmeans_evaluation_{timestamp}.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Evaluation plot saved: {fig_path}")
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE - 6 FEATURES (INCLUDING AGE)")
print("="*80)

# 写入完成日志
with open(log_file, 'a', encoding='utf-8') as f:
    f.write(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n")
