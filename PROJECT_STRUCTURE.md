# Project Structure Documentation

## Directory Organization

### `/data/`
**Purpose**: Data files (raw and processed)  
**Contents**:
- `raw/` - Original unmodified datasets
  - `CDC Diabetes Dataset.csv` - Main dataset (253,680 samples, 22 columns)
  - `Dataset Description.txt` - Feature descriptions and coding information
- `CDC_Diabetes_Cleaned.csv` - Cleaned dataset (duplicates removed)

**Note**: Raw data should never be modified. Keep backups if needed.

---

### `/src/`
**Purpose**: Source code and scripts  
**Current Files**:
- `data_exploration.py` - Data understanding and exploratory data analysis (EDA)

**TODO**:
- `preprocessing.py` - Data cleaning, feature engineering, scaling
- `clustering.py` - K-Means, DBSCAN implementations
- `classification.py` - Decision Trees, Logistic Regression, etc.
- `evaluation.py` - Model evaluation metrics and visualizations
- `utils.py` - Helper functions

---

### `/outputs/`
**Purpose**: Generated figures and visualizations  
**Current Files** (from data_exploration.py):
1. `01_class_distribution.png` - Class imbalance visualization
2. `02_correlation_heatmap.png` - Feature correlation matrix
3. `03_top_correlations.png` - Top features correlated with diabetes
4. `04_age_analysis.png` - Diabetes distribution by age groups
5. `05_sex_analysis.png` - Diabetes distribution by sex
6. `06_income_analysis.png` - Diabetes distribution by income level

**Future Files**:
- Clustering visualizations (elbow plots, silhouette scores)
- Classification results (confusion matrices, ROC curves)
- Feature importance plots

---

### `/models/`
**Purpose**: Saved trained models  
**Format**: `.pkl`, `.joblib`, or `.h5` files  
**Naming Convention**: `{model_type}_{timestamp}.pkl`

Example:
- `kmeans_clusters_20260209.pkl`
- `random_forest_classifier_20260209.pkl`

---

### `/notebooks/`
**Purpose**: Jupyter notebooks for interactive analysis (optional)  
**Use Cases**:
- Quick experiments
- Interactive visualizations
- Presentation materials

---

### `/reports/`
**Purpose**: Generated reports and documentation  
**TODO**:
- `reflective_report.pdf` - Final submission report (max 2800 words)
- `methodology.md` - Detailed methodology notes
- `findings.md` - Key findings and insights

---

### `/doc/`
**Purpose**: Assignment briefs and reference materials  
**Contents**:
- Assignment brief PDF
- Marking rubric
- Scenario description

---

### `/logs/`
**Purpose**: Execution logs and debugging information  
**Use**: Track model training, errors, and performance metrics over time

---

## Workflow

### Phase 1: Data Understanding ✅
1. Run `src/data_exploration.py`
2. Review outputs in `/outputs/`
3. Document findings in notes

### Phase 2: Preprocessing (Next)
1. Create `src/preprocessing.py`
2. Handle missing values, outliers
3. Feature engineering
4. Train-test split
5. Save processed data

### Phase 3: Clustering
1. Implement K-Means, DBSCAN in `src/clustering.py`
2. Evaluate with silhouette score, Davies-Bouldin index
3. Visualize clusters
4. Interpret segments

### Phase 4: Classification
1. Implement models in `src/classification.py`
2. Handle class imbalance (SMOTE, class weights)
3. Cross-validation
4. Evaluate performance
5. Feature importance analysis

### Phase 5: Reporting
1. Generate final visualizations
2. Write reflective report in `/reports/`
3. Include code screenshots
4. Add to Git repository

---

## File Naming Conventions

### Python Scripts
- Use snake_case: `data_exploration.py`
- Prefix with number if sequential: `01_preprocessing.py`

### Output Files
- Include number prefix: `01_class_distribution.png`
- Descriptive names: `correlation_heatmap.png`
- Include date if versioned: `results_20260209.csv`

### Models
- Format: `{model_type}_{date}.pkl`
- Example: `kmeans_k5_20260209.pkl`

---

## Best Practices

1. **Never modify raw data** - Always work on copies
2. **Version control** - Commit regularly with clear messages
3. **Document everything** - Add comments and docstrings
4. **Reproducibility** - Set random seeds (e.g., `random_state=42`)
5. **Modular code** - Separate concerns into different files
6. **Test incrementally** - Don't write large blocks without testing

---

## Quick Commands

```bash
# Navigate to project
cd /Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project

# Run data exploration
python src/data_exploration.py

# View outputs
open outputs/01_class_distribution.png

# Check project structure
ls -la

# Git operations
git status
git add .
git commit -m "Descriptive message"
git push origin main
```

---

**Last Updated**: February 9, 2026  
**Maintained by**: Henry Liao
