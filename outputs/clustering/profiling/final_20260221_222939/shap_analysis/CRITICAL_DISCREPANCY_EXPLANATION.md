# SHAP Feature Importance Discrepancy - Explanation & Resolution

**Date**: 2026-02-22  
**Issue**: RF vs XGBoost show different SHAP rankings, and SHAP rankings differ from built-in importance

---

## 🔍 The Three Different Measurements

### 1. **Built-in Feature Importance** (Model-Specific)
- **RF uses Gini Impurity**: Measures how much each feature reduces node impurity during tree splits
- **XGB uses Gain**: Measures average gain across all splits using that feature

**RF Built-in Rankings**:
1. ses_vulnerability (0.2966)
2. chronic_disease_burden (0.2456)
3. lifestyle_risk (0.2289)
4. metabolic_baseline (0.2281)
5. substance_risk (0.0009)

**XGB Built-in Rankings**:
1. chronic_disease_burden (0.3211)
2. ses_vulnerability (0.2969)
3. lifestyle_risk (0.2823)
4. metabolic_baseline (0.0980)
5. substance_risk (0.0017)

**Why they differ**: Gini measures node purity, Gain measures splitting benefit. Different algorithms prioritize different splitting strategies.

---

### 2. **Overall SHAP Importance** (Averaged Across All 6 Clusters)
Measures which features best **distinguish all clusters from each other globally**.

**RF Overall SHAP Rankings**:
1. chronic_disease_burden (0.1215)
2. ses_vulnerability (0.1193)
3. lifestyle_risk (0.0966)
4. metabolic_baseline (0.0664)
5. substance_risk (0.0013)

**XGB Overall SHAP Rankings**:
1. **ses_vulnerability (1.8453)** ← Top feature
2. **chronic_disease_burden (1.7494)** ← Second
3. lifestyle_risk (1.4004)
4. metabolic_baseline (1.2443)
5. substance_risk (0.0205)

**⚠️ KEY FINDING**: RF and XGB **DISAGREE** on the top 2 features!
- RF says: chronic_disease_burden > ses_vulnerability
- XGB says: ses_vulnerability > chronic_disease_burden

**This is shown in**: `feature_importance_comparison.png` (bottom 2 panels)

---

### 3. **Cluster 5-Specific SHAP Importance**
Measures which features most strongly **predict Cluster 5 assignment specifically** (not other clusters).

**Question answered**: "If I want to know whether a patient belongs to C5 (Frail Multi-Morbidity), which features should I look at?"

**This is shown in**: 
- `shap_bar_rf.png` (RF SHAP for Cluster 5 only)
- `shap_bar_xgb.png` (XGB SHAP for Cluster 5 only)

**Why it's different from Overall SHAP**:
- Overall SHAP: "What separates C0 from C1 from C2 from C3 from C4 from C5?"
- Cluster 5 SHAP: "What makes C5 different from all others?"

These can and SHOULD differ! For example:
- `ses_vulnerability` might be critical for separating C1 from others (overall importance high)
- `chronic_disease_burden` might be critical for identifying C5 specifically (C5-specific importance high)

---

## ❓ Why Do RF and XGBoost Disagree on SHAP Rankings?

### Root Cause: Different Model Architectures Lead to Different Feature Interactions

**Random Forest (Bagging)**:
- Trains 100 independent trees on random subsets
- Trees vote equally (democratic ensemble)
- Each tree sees ~63% of data (bootstrap sampling)
- **Effect**: More sensitive to **frequent splitting patterns**
- Interpretation: "Which feature appears in splits most often across all trees?"

**XGBoost (Boosting)**:
- Trains 100 sequential trees, each correcting previous errors
- Later trees weighted more heavily (autocratic ensemble)
- Each tree sees 100% of data but focuses on hard-to-classify samples
- **Effect**: More sensitive to **residual error patterns**
- Interpretation: "Which feature helps correct misclassifications?"

### Why ses_vulnerability vs chronic_disease_burden Flip Rankings:

**Hypothesis 1: Class Imbalance Sensitivity**
- C0 (Healthy): 28.5% of data
- C1 (SES Vulnerability): 16.6% of data
- C5 (Frail Multi-Morbidity): 12.3% of data

XGBoost's boosting mechanism focuses more on minority classes (C4, C5), where `chronic_disease_burden` is critical.
RF treats all samples equally, so majority classes (C0, C1) where `ses_vulnerability` discriminates strongly have more influence.

**Hypothesis 2: Feature Interaction Patterns**
- `ses_vulnerability` may have **additive main effects** → captured well by RF's averaging
- `chronic_disease_burden` may have **multiplicative interactions** with other features → captured better by XGBoost's residual correction

**Hypothesis 3: Multicollinearity Handling**
- VIF analysis showed moderate correlation between features (VIF = 1.69-2.18)
- RF splits on one feature per node → orthogonalizes features naturally
- XGBoost can use correlated features in gradient updates → captures shared variance differently

---

## 🔬 Validation: Are These Differences Meaningful or Noise?

### Evidence SUPPORTING Real Differences:
1. **Magnitude**: The flip is in the top 2 features (most important), not low-ranked noise
2. **Consistency**: substance_risk ranks #5 in ALL methods (no flipping in unimportant features)
3. **Clinical plausibility**: Both interpretations make sense:
   - SES barriers could be the primary clustering driver (XGB view)
   - Chronic disease burden could be the primary clustering driver (RF view)

### Evidence SUGGESTING Minimal Practical Impact:
1. **Rank difference is only 1 position** (swap between #1 and #2, not #1 vs #5)
2. **Values are close**:
   - XGB SHAP: ses_vulnerability (1.8453) vs chronic_disease_burden (1.7494) = 5.2% difference
   - RF SHAP: chronic_disease_burden (0.1215) vs ses_vulnerability (0.1193) = 1.8% difference
3. **Both features are critical**: The disagreement is "which comes first" not "which matters at all"

---

## 📊 Recommended Actions

### Option 1: **Report Both Rankings with Caveats** (Conservative)
Present both RF and XGBoost rankings, explain the methodological differences, and state:
> "Feature importance rankings show minor inconsistencies between algorithms due to different ensemble strategies. However, both models agree that chronic_disease_burden and ses_vulnerability are the two most important features for cluster separation, with differences <6% in SHAP values. Substance_risk consistently ranks lowest across all methods."

### Option 2: **Use Ensemble Average** (Pragmatic)
Compute mean SHAP importance across RF and XGBoost:
```
Average SHAP = (RF_SHAP + XGB_SHAP) / 2
```
This would give a single consensus ranking.

### Option 3: **Prioritize XGBoost** (Methodological Justification)
Argue that XGBoost's boosting mechanism better handles:
- Class imbalance (unequal cluster sizes)
- Complex feature interactions (multiplicative effects)
- Residual error patterns (misclassified boundary cases)

Therefore, accept XGBoost SHAP rankings as ground truth:
1. ses_vulnerability
2. chronic_disease_burden
3. lifestyle_risk
4. metabolic_baseline
5. substance_risk

---

## 🛠️ Code Fix Required

The current report (`shap_analysis_report.txt`) has **HARDCODED** rankings on lines 468-472:
```python
f.write("     Rank 1: chronic_disease_burden\n")
f.write("     Rank 2: metabolic_baseline (BMI)\n")
f.write("     Rank 3: ses_vulnerability\n")
```

These don't match the actual SHAP data! Need to:
1. Compute rankings dynamically from `results_df`
2. Present RF and XGBoost rankings separately
3. Document the discrepancy explicitly

---

## 📌 Summary

| Method | Top Feature | 2nd Feature | Difference |
|--------|------------|-------------|------------|
| RF Built-in (Gini) | ses_vulnerability | chronic_disease_burden | 17.2% |
| XGB Built-in (Gain) | chronic_disease_burden | ses_vulnerability | 7.5% |
| **RF SHAP (Overall)** | **chronic_disease_burden** | **ses_vulnerability** | **1.8%** |
| **XGB SHAP (Overall)** | **ses_vulnerability** | **chronic_disease_burden** | **5.2%** |

**Conclusion**: The ranking inconsistency is **real but small in magnitude**. Both algorithms agree these are the top 2 features, differing only in order. The clinical interpretation remains stable: **chronic disease burden and socioeconomic vulnerability are the primary drivers of cluster separation**.

**Recommendation**: Use **Option 1** (report both) or **Option 3** (prioritize XGBoost) depending on journal requirements for methodological transparency vs. decisiveness.
