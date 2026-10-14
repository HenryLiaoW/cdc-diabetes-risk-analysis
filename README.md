# CDC Diabetes Risk Analysis (BRFSS 2015)

Individual coursework for *Fundamentals of AI and Data Mining* (MSc Applied AI, University of Warwick / WMG, 2025/26).

The CDC BRFSS 2015 diabetes survey (253,680 respondents, 21 health and lifestyle indicators) is analysed from three angles:

1. **Exploration / association rules** — which lifestyle and demographic combinations go with diabetes risk.
2. **Clustering (unsupervised)** — segment the population into risk groups with K-Means on composite health scores.
3. **Classification (supervised)** — predict "at risk" (pre-diabetes or diabetes, 15.8% of respondents) and explain the models with SHAP.

## Key results

### Clustering — K-Means, K = 6 on five composite scores

Five composite features were built from the raw indicators (Metabolic Baseline/BMI, Chronic Disease Burden, Lifestyle Risk, Substance Risk, SES Vulnerability). K = 2 has the highest silhouette (0.28) but only separates "healthy vs. unhealthy"; K = 6 is the best of the finer solutions (silhouette 0.24, vs. 0.23–0.24 for K = 3–7) and gives interpretable, actionable segments, so it was chosen on elbow + silhouette + interpretability. Adding Age or Sex as a sixth feature *lowered* cluster quality (Age is collinear with Chronic Disease Burden; checked with VIF), so they are used only for post-hoc profiling.

| Cluster | Share of population | Diabetes rate | Profile |
|---|---|---|---|
| 0 | 34.1% | 5.6% | Low risk overall |
| 1 | 16.6% | 10.0% | High SES vulnerability |
| 2 | 13.6% | 11.5% | Poor lifestyle |
| 3 | 17.2% | 23.1% | High chronic-disease burden |
| 4 | 6.2% | 29.8% | Very high BMI (mean 45) |
| 5 | 12.3% | 35.3% | Multi-morbidity + lifestyle + SES |

Diabetes rate was **not** used for clustering; it is computed afterwards and rises monotonically across the clusters (5.6% → 35.3%), which is the main validation that the segments are meaningful. PCA projection and a within-cluster SHAP analysis are in `outputs/clustering/profiling/`.

### Classification — at-risk vs. not (stratified 80/20 split, class-weighted)

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression (`class_weight=balanced`) | 0.792 | 0.708 | 0.332 | 0.452 | 0.730 |
| Random Forest (`class_weight=balanced`) | 0.795 | 0.752 | 0.318 | 0.447 | 0.706 |
| XGBoost (`scale_pos_weight`) | 0.796 | 0.759 | 0.315 | 0.445 | 0.702 |

The three models are within 0.005 ROC-AUC of each other: on this survey data the ceiling is set by the features, not the model. Class weighting was chosen over SMOTE so that recall on the minority class is prioritised (a missed at-risk person costs more than a false alarm in a screening setting). Hyper-parameter tuning of XGBoost gave only ~+1% F1.

SHAP (mean |SHAP|, XGBoost): Chronic Disease Burden 0.85 ≫ BMI 0.45 > SES Vulnerability 0.23 > Lifestyle 0.08 > Substance use 0.06.

### Association rules (Apriori, mlxtend)

Strongest rules for "At-Risk" have lift ≈ 3.4, e.g. `Poor general health & Obese BMI & High blood pressure → At-Risk` (confidence 0.54 vs. 0.16 base rate). Lifestyle-combination analysis: the highest-risk lifestyle profile has 8× the diabetes rate of the lowest. One counter-intuitive finding — diabetics report *more* exercise and fruit/veg intake — is discussed in `outputs/exploration/critical_reflection.txt` as likely reverse causation (behaviour change after diagnosis).

## Repository layout

```
src/
  EDA/              association_rule_mining.py, demographic_risk_analysis.py
  clustering/       data_cleaning.py, kmeans*.py, vif_*.py, pca_validation.py,
                    cluster_profiling*.py, shap_*.py, algorithm_comparison.py
  classification/   01_data_preparation.py ... 07_export_shap_summary.py (run in order)
notebooks/          EDA_3.0 / 3.1 / 3.2 — exploratory analysis
outputs/
  exploration/      association rules, lifestyle / demographic risk tables, reflections
  clustering/       final run: cluster profiles, PCA / age / VIF validation, SHAP
  classification/   metrics, model comparison, SHAP importance, figures
data/               Dataset Description.txt (data dictionary); CSVs are not committed
```

## Data

`data/raw/CDC Diabetes Dataset.csv` is the Kaggle release of the CDC BRFSS 2015 survey
(["Diabetes Health Indicators Dataset"](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), file `diabetes_012_health_indicators_BRFSS2015.csv`).
Download it into `data/raw/` and run `src/clustering/data_cleaning.py` to produce `data/processed/CDC_Diabetes_Cleaned.csv`.

## Reproduce

```bash
pip install -r requirements.txt
python src/clustering/data_cleaning.py
python src/clustering/kmeans.py && python src/clustering/cluster_profiling.py
for f in src/classification/0*.py; do python "$f"; done
python src/EDA/association_rule_mining.py
```

## Limitations

- Survey data is self-reported and cross-sectional, so the models describe association, not causation.
- The composite scores are hand-designed; a different grouping of the raw indicators would change the clusters.
- Precision is low (~0.32) at the recall-oriented operating point; the models are suited to screening, not diagnosis.
