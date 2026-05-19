# Task 3 — KNN Classification: Iris Dataset

**CodVeda ML Internship | Level 1**

---

## 🎯 Objective

Train a K-Nearest Neighbors classifier on the Iris dataset, systematically compare multiple K values, select the optimal K via cross-validation, and evaluate using the full suite of classification metrics including confusion matrices.

---

## 📁 Folder Structure

```
Task3_KNN_Classification/
├── notebooks/
│   └── task3_knn_classification.py   # Full KNN script
├── data/
│   └── knn_k_comparison.csv          # Accuracy per K value
├── outputs/
│   ├── task3_pairplot.png            # All feature pair combinations
│   ├── task3_boxplots.png            # Feature distributions by class
│   ├── task3_correlation.png         # Feature correlation heatmap
│   ├── task3_k_comparison.png        # Accuracy vs K curve
│   ├── task3_confusion_matrix.png    # Raw + normalized confusion matrix
│   └── task3_decision_boundary.png   # 2D decision boundary (petal features)
└── README.md
```

---

## 🧠 Theory

### How KNN Works

KNN is a **lazy learner** — it stores all training data and makes predictions at query time.

**Prediction algorithm for a new point X:**
```
1. Compute distance from X to every training point
2. Sort by distance → select K nearest neighbors
3. Take majority vote among K neighbors' labels
4. Return that label as prediction
```

**Distance metric (default: Euclidean):**
```
d(p, q) = √[(p₁-q₁)² + (p₂-q₂)² + ... + (pₙ-qₙ)²]
```

### Choosing K: The Bias-Variance Tradeoff

```
K=1:  Every training point is its own neighbor → 100% train accuracy
      But boundaries are jagged → HIGH VARIANCE → overfitting

K=N:  All N training points are neighbors → always predicts majority class
      Completely ignores local structure → HIGH BIAS → underfitting

K_optimal: Somewhere in between — found via cross-validation
```

**Visualization of K effect:**
```
Small K                         Large K
Decision boundary               Decision boundary
   ___                            ___________
  |   |    ← jagged,            |           |  ← smooth,
  | ● |      sensitive to       |     ●     |    ignores noise
  |___|      noise              |___________|
```

### Why Scaling is MANDATORY for KNN

KNN uses raw distance. Without scaling:
```
Feature     Range       Contribution to Distance
──────────────────────────────────────────────
Age         0 – 80     Δ = 20 → distance += √(400) = 20
Salary      20K – 100K  Δ = 5000 → distance += √(25,000,000) = 5000

→ Salary COMPLETELY DOMINATES — Age is irrelevant
```

After StandardScaler: all features contribute equally.

---

## 🌸 Dataset: Iris

| Property | Value |
|----------|-------|
| Samples | 150 |
| Features | 4 (sepal/petal length & width) |
| Classes | 3 (setosa, versicolor, virginica) |
| Class balance | Perfectly balanced (50 per class) |
| Missing values | None |

**Key insight from EDA:**
- **Setosa** is linearly separable from the other two classes (clear gap in petal features)
- **Versicolor vs. Virginica** overlap in petal space → need higher K to smooth the boundary

---

## 🚀 Implementation Steps

1. **Load Iris** — via `sklearn.datasets.load_iris()`
2. **EDA** — Pairplot, boxplots by class, correlation heatmap
3. **Split** — 80/20 stratified (40 per class in train, 10 per class in test)
4. **K experiment** — Evaluate K = 1, 3, 5, ..., 21 (odd values only)
5. **Select best K** — Based on CV accuracy mean
6. **Full evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix
7. **Decision boundary** — 2D visualization using petal features

---

## ▶️ How to Run

```bash
cd Task3_KNN_Classification/notebooks
python task3_knn_classification.py
```

---

## 📊 Results

### K Value Comparison

| K | Test Accuracy | CV Accuracy | CV Std |
|---|--------------|-------------|--------|
| 1 | 0.9667 | 0.9417 | 0.0425 |
| 3 | 0.9333 | 0.9583 | 0.0373 |
| **5** | **0.9333** | **0.9667** | **0.0312** |
| 7 | 0.9667 | 0.9583 | 0.0373 |
| 9 | 0.9667 | 0.9583 | 0.0264 |
| 21 | 0.9667 | 0.9333 | 0.0333 |

**⭐ Best K = 5** (highest CV accuracy with low variance)

### Best Model Metrics (K = 5)

| Metric | Setosa | Versicolor | Virginica | Weighted Avg |
|--------|--------|------------|-----------|-------------|
| Precision | 1.00 | 0.83 | 1.00 | 0.94 |
| Recall | 1.00 | 1.00 | 0.80 | 0.93 |
| F1-Score | 1.00 | 0.91 | 0.89 | 0.93 |

**Interpretation:**
- **Setosa:** Perfect classification — linearly separable from others
- **Versicolor vs. Virginica:** Some confusion (expected — they overlap in feature space)
- Overall **93.3% accuracy** on 30 test samples

---

## 📈 Evaluation Metrics Deep Dive

### Confusion Matrix Reading Guide

```
                  Predicted
                  Setosa  Vrsclr  Vrgnica
Actual  Setosa  [  10       0       0  ]  → Perfect
        Vrsclr  [   0      10       0  ]  → Perfect
        Vrgnica [   0       2       8  ]  → 2 confused with versicolor
```

**Why does Virginica get confused with Versicolor?**
Their petal measurements overlap significantly. KNN with K=5 looks at 5 neighbors — if 3 are versicolor and 2 are virginica, it predicts versicolor even for a true virginica.

### Precision vs Recall: Which Matters More?

| Scenario | Prefer | Why |
|----------|--------|-----|
| Spam filter | Precision | Don't want to mark real emails as spam (false positive costly) |
| Cancer screening | Recall | Don't want to miss actual cancer cases (false negative costly) |
| Iris classification | Balanced → F1 | No asymmetric cost |

---

## ⚠️ Common Mistakes to Avoid

| Mistake | Impact | Fix |
|---------|--------|-----|
| Not scaling features | Large-range features dominate distance | Always StandardScale before KNN |
| Using even K values | Ties possible in binary classification | Prefer odd K |
| Selecting K by test accuracy alone | Overfits to one split | Always use CV |
| Reporting only accuracy | Misleading on imbalanced data | Report Precision, Recall, F1 |
| Using K=1 | Perfect training accuracy, high variance | Never use K=1 for production |

---

## 💡 Optimization Ideas

1. **Grid Search** — Use `GridSearchCV` to tune K and distance metric simultaneously
2. **Weighted KNN** — `weights='distance'` gives closer neighbors more vote weight
3. **Manhattan Distance** — `metric='manhattan'` can outperform Euclidean on high-dimensional data
4. **PCA** — Reduce to 2 principal components before KNN to eliminate correlated features
5. **Feature Selection** — Drop sepal features (less discriminative) — use only petal features

---

## 🎤 Interview Q&A

**Q: Why are odd values of K preferred?**
> A: To avoid tie-breaking issues in binary classification. With K=2, if 1 neighbor is class A and 1 is class B, you have a tie with no clear winner. Odd K always produces a clear majority. For multi-class (Iris), ties can still occur but are less likely.

**Q: What is the time complexity of KNN prediction?**
> A: O(n×d) per prediction, where n = training samples and d = features. KNN stores all training data and computes distances at inference time — making it slow for large datasets. Tree-based optimizations (KD-Tree, Ball-Tree) reduce this to O(log n) on average.

**Q: What's the difference between macro and weighted average in classification_report?**
> A: Macro average: unweighted mean across all classes (treats all classes equally regardless of support). Weighted average: weighted by the number of samples (support) per class. Use weighted when classes are imbalanced; use macro when all classes are equally important.

**Q: What does the decision boundary plot show?**
> A: Colored background regions show what class KNN would predict for any point in that 2D feature space. Points are actual samples. If a point's color doesn't match its background region, it's misclassified. The boundary between regions shows where KNN's decision flips from one class to another.

**Q: How does KNN handle multi-class classification?**
> A: KNN naturally extends to multi-class — it just takes the majority vote among K neighbors. If K=5 and 3 neighbors are virginica, 1 is versicolor, 1 is setosa → predict virginica. No change in algorithm, unlike binary-specific methods that need OvO or OvR wrappers.

---

## 🔗 Resources

- [scikit-learn KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
