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
│   ├── raw/                       # Raw data files
│   │   ├── CDC Diabetes Dataset.csv
│   │   └── Dataset Description.txt
│   └── CDC_Diabetes_Cleaned.csv  # Cleaned data
├── doc/                           # Assignment briefs and documentation
├── src/                           # Source code
│   ├── data_exploration.py        # Data understanding & EDA
│   ├── data_cleaning.py           # Data cleaning pipeline
│   ├── clustering/                # Clustering models
│   └── classification.py          # Classification models (TODO)
├── outputs/                       # Generated figures and results
├── models/                        # Saved model files
├── notebooks/                     # Jupyter notebooks (optional)
├── reports/                       # Generated reports
└── logs/                          # Execution logs
```

## 🎯 Key Tasks

### ✅ Completed
- [x] Data understanding and exploration
- [x] Class imbalance analysis (46.15:1 ratio)
- [x] Feature correlation analysis
- [x] Subgroup analysis (age, sex, income)
- [x] Surprising patterns identification

### 🚧 In Progress
- [ ] Data preprocessing
- [ ] Feature engineering
- [ ] Clustering implementation (K-Means, DBSCAN)
- [ ] Classification implementation (handling imbalance)
- [ ] Model evaluation
- [ ] Reflective report writing

## 📊 Key Findings

### Class Distribution
- **No Diabetes**: 213,703 (84.24%)
- **Prediabetes**: 4,631 (1.83%) ⚠️ Severe imbalance
- **Diabetes**: 35,346 (13.93%)

### Top Predictive Features
1. GenHlth (General Health) - r = 0.303
2. HighBP (High Blood Pressure) - r = 0.272
3. BMI (Body Mass Index) - r = 0.224

### Critical Insights
- **Age gradient**: Diabetes prevalence increases from 1.4% (18-24) to 21.8% (70-74)
- **Socioeconomic disparity**: Low income groups show 3x higher diabetes rates (24-26% vs 8%)
- **Surprising**: 63% of diabetics remain physically active, 49% maintain healthy diet

## 🛠️ Technologies

- **Language**: Python 3.x
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Environment**: Conda base environment

## 🚀 Usage

### 1. Data Exploration
```bash
cd /Users/henry/AAI/FAIDM/CDC_DCDC_Diabetes_Project
python src/data_exploration.py
```

Generates 6 analysis figures in `outputs/`:
- Class distribution
- Correlation heatmap
- Top feature correlations
- Age/Sex/Income subgroup analysis

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
