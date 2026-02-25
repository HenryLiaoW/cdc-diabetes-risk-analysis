# CDC Diabetes Risk Prediction and Health Insights

**WM9QG-15 Fundamentals of Artificial Intelligence and Data Mining**  
**Individual Coursework - 70% Weighting**

## 📋 Project Overview

This project analyzes the CDC BRFSS 2015 diabetes dataset (253,680 samples) to:
1. **Clustering**: Identify population segments at risk using unsupervised learning
2. **Classification**: Predict diabetes diagnosis using supervised learning
3. **Insights**: Provide actionable public health recommendations

## 📁 Project Structure

```
CDC_DCDC_Diabetes_Project/
├── data/                          # Data files
│   ├── raw/                       # Original dataset
│   │   └── CDC Diabetes Dataset.csv (253,680 samples)
│   ├── processed/                 # Cleaned data
│   │   └── CDC_Diabetes_Cleaned.csv
│   └── Dataset Description.txt    # Data dictionary
├── src/                           # Source code
│   ├── EDA/                       # Exploratory Data Analysis scripts
│   │   └── association_rule_mining.py
│   ├── clustering/                # Clustering models and analysis (25 scripts)
│   │   ├── kmeans.py              # 5-feature K-Means clustering
│   │   ├── kmeans_6features_with_age.py
│   │   ├── kmeans_6features_with_sex.py
│   │   ├── shap_feature_importance_analysis.py
│   │   ├── vif_multicollinearity_analysis*.py (3 scripts)
│   │   ├── cluster_profiling*.py  (3 scripts)
│   │   └── ... (scaling analysis, validation scripts)
│   ├── classification/            # Classification models (7 scripts)
│   │   ├── 01_data_preparation.py
│   │   ├── 02_baseline_logistic_regression.py
│   │   ├── 03_random_forest.py
│   │   ├── 04_xgboost.py
│   │   ├── 05_xgboost_hyperparameter_tuning.py
│   │   ├── 06_comprehensive_model_comparison.py
│   │   └── 07_export_shap_summary.py
│   └── demographic_risk_analysis.py
├── outputs/                       # Results and visualizations
│   ├── clustering/                # Clustering outputs
│   │   ├── profiling/             # Cluster profiling and validation
│   │   │   └── final_20260221_222939/
│   │   │       ├── cluster_summary_*.csv (summary tables)
│   │   │       ├── age_validation/ (age distribution analysis)
│   │   │       ├── pca_validation/ (PCA projection validation)
│   │   │       └── shap_analysis/  (SHAP feature importance)
│   │   ├── fig/                   # Clustering figures
│   │   ├── log/                   # K-Means run logs
│   │   └── vif_analysis/          # VIF collinearity analysis
│   ├── classification/            # Classification outputs (25 files)
│   │   ├── model_comparison_complete.csv
│   │   ├── *_metrics.csv          (LR, RF, XGBoost metrics)
│   │   ├── *_predictions.csv      (model predictions)
│   │   ├── shap_summary_importance.csv
│   │   └── xgboost_tuning_comparison.csv
│   └── exploration/               # EDA outputs (13 files)
│       ├── association_rules*.csv (ARM analysis)
│       ├── lifestyle_risk_ranking.csv
│       ├── demographic_risk_by_factor.csv
│       └── ARM_VISUALIZATION_GUIDE.md
├── models/                        # Saved model files
├── notebooks/                     # Jupyter notebooks (7 files)
│   ├── EDA_3.0.ipynb             # Initial EDA
│   ├── EDA_3.1.ipynb             # Extended EDA
│   ├── EDA_3.2.ipynb             # Advanced EDA
│   ├── FAIDM.ipynb               # Main analysis notebook
│   ├── correlation_analysis_sagemaker.py
│   └── feature_correlations.csv
├── reports/                       # Final reports
└── logs/                          # Execution logs
```

## 🎯 Project Status

### ✅ Completed
#### Data Preparation
- [x] Data understanding and exploration
- [x] Data cleaning pipeline
- [x] Feature engineering (composite indicators)
- [x] Class imbalance analysis (46.15:1 ratio)
- [x] Feature correlation analysis
- [x] Subgroup analysis (age, sex, income)

#### Clustering Analysis
- [x] K-Means clustering implementation (K=2-7 evaluation)
- [x] 5-feature composite model (Metabolic Baseline, Chronic Disease Burden, Healthy Lifestyle Score, Substance Use Risk, SES Vulnerability)
- [x] 6-feature model with Age (comparative analysis)
- [x] 6-feature model with Sex (comparative analysis)
- [x] VIF multicollinearity analysis
- [x] Elbow method and Silhouette Score evaluation
- [x] Cluster profiling and characterization
- [x] Three-model comparison report

#### Classification Analysis
- [x] Baseline Logistic Regression (F1=0.2890)
- [x] Random Forest Classifier (F1=0.3249)
- [x] XGBoost Classifier (F1=0.4178)
- [x] XGBoost hyperparameter tuning (F1=0.4219, +1.0%)
- [x] SHAP feature importance analysis
- [x] Model performance comparison
- [x] False negative/false positive analysis

## 📊 Key Findings

### Clustering Results (Unsupervised)
- **Optimal Model**: 5-feature composite model
- **Best K Value**: K=6 (Silhouette Score: 0.2417)
- **Age Impact**: Adding Age decreased performance by 23.7% (collinearity with Chronic Disease Burden, r=0.414)
- **Sex Impact**: Adding Sex decreased performance by 2.3% (minimal collinearity)
- **Recommendation**: Use 5-feature model with post-hoc age/sex stratification

#### Identified Clusters (K=6)
1. **Healthy Young Group** (39.6% of population)
2. **Metabolic Risk Group** (21.4%)
3. **Chronic Disease Burden Group** (15.2%)
4. **Low SES Vulnerable Group** (12.8%)
5. **Substance Risk Group** (6.7%)
6. **Multi-risk Elderly Group** (4.3%)

### Classification Results (Supervised)
- **Best Model**: XGBoost (tuned)
- **Performance**: 
  - Precision: 0.5536
  - Recall: 0.3447
  - F1 Score: 0.4219
  - AUC-ROC: 0.8473
- **Top 3 Predictive Features** (SHAP):
  1. GenHlth (General Health)
  2. HighBP (High Blood Pressure)
  3. BMI (Body Mass Index)

### Dataset Characteristics
- **Total Samples**: 229,781 (after cleaning)
- **Class Distribution**:
  - No Diabetes: 196,130 (85.35%)
  - Prediabetes: 4,629 (2.01%)
  - Diabetes: 29,022 (12.63%)
- **Class Imbalance Ratio**: 42.4:1 (No Diabetes : Prediabetes)

### Critical Insights
- **Age gradient**: Diabetes prevalence increases from 1.4% (18-24) to 21.8% (70-74)
- **Socioeconomic disparity**: Low income groups show 3x higher diabetes rates (24-26% vs 8%)
- **Health behaviors**: 63% of diabetics remain physically active; 49% maintain healthy diet
- **Feature independence**: 5-feature composite model shows excellent VIF (<1.2), minimal multicollinearity

## 🛠️ Technologies

- **Language**: Python 3.x
- **Core Libraries**: 
  - Data Processing: pandas, numpy
  - Machine Learning: scikit-learn, xgboost
  - Visualization: matplotlib, seaborn
  - Interpretability: shap
  - Statistical Analysis: scipy, statsmodels
- **Scaling Methods**: RobustScaler, StandardScaler, MinMaxScaler (data-driven mixed scaling)
- **Environment**: Conda base environment

## 🚀 Usage

### 1. Clustering Analysis
```bash
cd /Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project

# Run 5-feature K-Means clustering
python src/clustering/kmeans.py

# Run 6-feature models for comparison
python src/clustering/kmeans_6features_with_age.py
python src/clustering/kmeans_6features_with_sex.py

# SHAP feature importance analysis
python src/clustering/shap_feature_importance_analysis.py
```

**Outputs**: 
- Clustering results: `outputs/clustering/log/`
- Visualizations: `outputs/clustering/fig/`
- VIF analysis: `outputs/clustering/vif_analysis/`
- Comparison report: `outputs/clustering/Three_Model_Comparison_5feat_vs_6feat_Age_vs_6feat_Sex.md`

### 2. Classification Analysis
```bash
# Run classification pipeline
python src/classification/01_baseline_logistic_regression.py
python src/classification/02_random_forest.py
python src/classification/03_xgboost.py
python src/classification/04_xgboost_tuning.py
python src/classification/05_model_comparison.py
```

**Outputs**:
- Model metrics: `outputs/classification/*_metrics.csv`
- Predictions: `outputs/classification/*_predictions.csv`
- Feature importance: `outputs/classification/shap_summary_importance.csv`
- Comparison: `outputs/classification/model_comparison_complete.csv`

### 3. Exploratory Data Analysis
```bash
python src/EDA/association_rule_mining.py
python src/demographic_risk_analysis.py
```

**Outputs**:
- Association rules: `outputs/exploration/association_rules.csv`
- Risk rankings: `outputs/exploration/lifestyle_risk_ranking.csv`
- Demographic analysis: `outputs/exploration/demographic_risk_by_factor.csv`

## 📝 Methodology Highlights

### Feature Engineering
- **Composite Indicators**: Aggregated 21 raw features into 5 composite features to reduce dimensionality and address multicollinearity
- **Mixed Scaling Strategy**: 
  - RobustScaler for skewed distributions (BMI, Lifestyle)
  - StandardScaler for symmetric distributions (Chronic Disease, SES)
  - MinMaxScaler for sparse features (Substance Risk, 51.3% zeros)

### Clustering Approach
- **Algorithm**: K-Means with K=2-7 evaluation
- **Evaluation Metrics**: Silhouette Score (50,000 sample subsampling), Inertia, VIF
- **Model Selection**: Pareto optimality (performance-parsimony trade-off)

### Classification Approach
- **Imbalance Handling**: SMOTE oversampling + class weights
- **Models Tested**: Logistic Regression (baseline), Random Forest, XGBoost
- **Hyperparameter Tuning**: GridSearchCV for XGBoost optimization
- **Interpretability**: SHAP values for feature importance

## 📄 Key Reports

- **Clustering Comparison**: [Three_Model_Comparison_5feat_vs_6feat_Age_vs_6feat_Sex.md](outputs/clustering/Three_Model_Comparison_5feat_vs_6feat_Age_vs_6feat_Sex.md)
- **Model Comparison**: [model_comparison_complete.csv](outputs/classification/model_comparison_complete.csv)
- **Feature Importance**: [shap_summary_importance.csv](outputs/classification/shap_summary_importance.csv)

## 📝 Submission Requirements

- **Deadline**: Monday 09 February 2026
- **Format**: One PDF (max 2800 words) + Git repository link
- **Components**:
  - Problem definition and rationale
  - Methodology (preprocessing, algorithms, justification)
  - Analysis, results, and reflection
  - Code screenshots and outputs
  - References

## 🔗 Repository

GitHub: [https://github.com/HenryLiaoW/CDC_DCDC_Diabetes_Project](https://github.com/HenryLiaoW/CDC_DCDC_Diabetes_Project)

## 📄 License & Ethics

- AI Usage: **AI Collaboration** level (allowed for drafting, refining, evaluating)
- Must critically evaluate and modify AI-generated content
- Proper citations required for external code sources

---

**Author**: Henry Liao (HenryLiaoW)  
**Course**: WM9QG-15 FAIDM  
**Date**: February 2026
