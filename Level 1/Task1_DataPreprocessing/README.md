# Task 1 — Data Preprocessing

**CodVeda ML Internship | Level 1**

---

## 🎯 Objective

Transform raw, messy data into a clean, model-ready format by handling missing values, encoding categorical variables, scaling numerical features, and splitting into train/test sets — all without data leakage.

---

## 📁 Folder Structure

```
Task1_DataPreprocessing/
├── notebooks/
│   └── task1_preprocessing.py     # Full preprocessing script
├── data/
│   ├── train_processed.csv        # Preprocessed training set
│   └── test_processed.csv         # Preprocessed test set
├── outputs/
│   ├── task1_eda_overview.png     # Missing values + distributions
│   ├── task1_scaling_comparison.png  # Before vs. after scaling
│   └── task1_correlation_heatmap.png # Feature correlations
└── README.md
```

---

## 🧠 Theory

### What is Data Preprocessing?

Raw datasets from the real world are almost never model-ready. Common problems:

| Problem | Example | Impact if Ignored |
|---------|---------|-------------------|
| Missing values | Age = NaN | Crashes most ML models |
| Categorical strings | Sex = "male" | Models only accept numbers |
| Different scales | Fare: 5–500 vs SibSp: 0–4 | Distance-based models get dominated by large-scale features |
| No train/test split | Training on all data | Cannot evaluate generalization |

### Preprocessing Decision Map

```
Raw Feature
    │
    ├── Numerical?
    │       ├── Has missing values? → SimpleImputer(strategy="median")
    │       └── Needs scaling?      → StandardScaler()
    │
    └── Categorical?
            ├── Has missing values? → SimpleImputer(strategy="most_frequent")
            ├── Ordinal (has order)? → OrdinalEncoder
            └── Nominal (no order)? → OneHotEncoder  ← Used here
```

### Why Median Imputation for Numerical?

The `Age` and `Fare` columns are **right-skewed** (long tail toward high values). The mean is pulled by outliers:

```
Skewed data: [20, 22, 25, 21, 150]
Mean = 47.6   ← distorted by 150
Median = 22   ← robust, represents typical value
```

### Why OneHotEncoding vs LabelEncoding?

| Encoding | Result for Sex | Problem |
|----------|----------------|---------|
| LabelEncoder | male=1, female=0 | Implies male > female (FALSE ordering) |
| OneHotEncoder | [1,0] or [0,1] | No false ordering, no numeric relationship |

### Why StandardScaler?

StandardScaler transforms each feature: `z = (x - μ) / σ`

This centers each feature at 0 with standard deviation 1.

**Result:** All features contribute equally to distance-based algorithms.

---

## 🚀 Implementation Steps

1. **Create dataset** — Synthetic Titanic-style data with controlled missingness
2. **EDA** — Visualize missing patterns, distributions, class balance
3. **Split** — 80% train / 20% test with stratification
4. **Build Pipeline** — ColumnTransformer with numerical + categorical sub-pipelines
5. **Fit & Transform** — Fit ONLY on training data, transform both sets
6. **Visualize** — Before/after scaling comparison + correlation heatmap
7. **Save** — Export processed CSVs for downstream tasks

---

## ▶️ How to Run

```bash
cd Task1_DataPreprocessing/notebooks
python task1_preprocessing.py
```

---

## 📊 Key Outputs

### Missing Value Report
| Column | Missing Count | Missing % |
|--------|--------------|-----------|
| Age | ~60 | 20.0% |
| Fare | ~15 | 5.0% |
| Embarked | ~6 | 2.0% |

### Processed Feature Space (11 total)
```
Numerical (scaled):   Age, Fare, SibSp
Categorical (OHE):    Sex_female, Sex_male,
                      Embarked_C, Embarked_Q, Embarked_S,
                      Pclass_1, Pclass_2, Pclass_3
```

---

## ⚠️ Common Mistakes to Avoid

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|-----------------|
| Fitting scaler on all data | Leaks test statistics into training | `fit()` on train only, `transform()` on both |
| Using LabelEncoder for nominal features | Creates false ordinal relationships | Use `OneHotEncoder` |
| Imputing before splitting | Test distribution leaks into imputer | Split first, then impute |
| Ignoring the imputation strategy | Mean imputation on skewed data | Use median for skewed distributions |
| Dropping rows with NaN blindly | Loses too much data | Use imputation strategies |

---

## 💡 Optimization Ideas

- **Advanced Imputation:** Use `KNNImputer` (imputes based on similar neighbors) or `IterativeImputer` (models each feature as a function of others) for better results on structured data
- **Feature Engineering:** Create `FamilySize = SibSp + Parch + 1` before encoding
- **Outlier Handling:** Use `RobustScaler` instead of `StandardScaler` if outliers are extreme
- **Target Encoding:** For high-cardinality categoricals, consider target encoding over OHE

---

## 🎤 Interview Q&A

**Q: Why do we split BEFORE preprocessing?**
> A: To prevent data leakage. If we compute the mean of the full dataset and use it to impute, the training process "sees" test set statistics during training, leading to overly optimistic evaluation.

**Q: When would you use OrdinalEncoder instead of OneHotEncoder?**
> A: When the feature has a natural, meaningful order. Example: Education = [High School, Bachelor's, Master's, PhD]. We'd encode these as [1, 2, 3, 4] because the ordering carries real information.

**Q: What's the difference between normalization and standardization?**
> A: Normalization (MinMaxScaler) scales to [0,1] — sensitive to outliers. Standardization (StandardScaler) centers at 0 with unit variance — robust to outliers. Use StandardScaler by default unless you specifically need [0,1] range (e.g., neural networks with sigmoid output).

**Q: What is ColumnTransformer?**
> A: A scikit-learn tool that applies different transformations to different column groups in one step. It takes a list of (name, transformer, columns) tuples and processes them in parallel, then concatenates the results.

---

## 🔗 Resources

- [scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [ColumnTransformer API](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [sklearn Pipeline](https://scikit-learn.org/stable/modules/pipeline.html)
