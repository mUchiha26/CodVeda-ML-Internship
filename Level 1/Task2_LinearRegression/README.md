# Task 2 — Linear Regression: House Price Prediction

**CodVeda ML Internship | Level 1**

---

## 🎯 Objective

Build, train, evaluate, and interpret a Linear Regression model to predict house prices using structured numerical features. Demonstrate all standard regression evaluation metrics and coefficient interpretation.

---

## 📁 Folder Structure

```
Task2_LinearRegression/
├── notebooks/
│   └── task2_linear_regression.py    # Full regression script
├── data/
│   └── house_prices.csv              # Generated dataset
├── outputs/
│   ├── task2_eda.png                 # Feature vs Price scatter plots
│   ├── task2_price_distribution.png  # Target distribution (raw + log)
│   ├── task2_coefficients.png        # Coefficient bar chart
│   └── task2_diagnostics.png         # 4-panel diagnostic plots
└── README.md
```

---

## 🧠 Theory

### What is Linear Regression?

Linear Regression models the relationship between input features X and a continuous output y as:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

The algorithm finds coefficients β that **minimize Mean Squared Error (MSE)**:

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

This is solved analytically using **Ordinary Least Squares (OLS)**:

```
β = (XᵀX)⁻¹ Xᵀy
```

### Key Assumptions of Linear Regression

| Assumption | What It Means | How to Check |
|-----------|---------------|-------------|
| Linearity | X and y have a linear relationship | Scatter plots |
| Independence | Residuals are independent | Residuals vs. order plot |
| Homoscedasticity | Residual variance is constant | Residuals vs. predicted plot |
| Normality | Residuals are normally distributed | Residual histogram |

### Evaluation Metrics

#### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)

SS_res = Σ(y - ŷ)²     ← How wrong the model is
SS_tot = Σ(y - ȳ)²     ← How spread out the data is

R² = 1.0  → Perfect predictions
R² = 0.0  → Model no better than predicting the mean
R² < 0.0  → Model is WORSE than predicting the mean
```

**Rule of thumb:**
- R² > 0.90 → Excellent
- R² 0.75–0.90 → Good
- R² 0.50–0.75 → Moderate
- R² < 0.50 → Poor (consider other models)

#### RMSE (Root Mean Squared Error)
```
RMSE = √MSE = √[(1/n) × Σ(y - ŷ)²]
```
- **Same unit as target** (dollars in this case) → directly interpretable
- "On average, our price predictions are off by $X"
- Penalizes large errors heavily (squaring effect)

#### MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|y - ŷ|
```
- Also in target units
- **Robust to outliers** (no squaring)
- Preferred metric for stakeholder communication

---

## 🏠 Dataset

**Synthetic House Price Dataset** (500 rows, 6 columns)

| Feature | Type | Effect on Price |
|---------|------|----------------|
| SquareFootage | Numerical | Strong positive (+$80/sqft) |
| Bedrooms | Numerical | Positive (+$5,000/bedroom) |
| HouseAge | Numerical | Negative (-$800/year) |
| DistanceFromCenterKm | Numerical | Negative (-$3,000/km) |
| HasGarage | Binary (0/1) | Positive (+$20,000) |
| **Price** | **Target** | **$30K – $405K** |

Ground truth formula (with noise):
```python
Price = 80×SquareFootage + 5000×Bedrooms - 800×HouseAge
        - 3000×DistanceFromCenter + 20000×HasGarage + 50000 + noise
```

---

## 🚀 Implementation Steps

1. **Generate dataset** — Synthetic data with known ground truth for validation
2. **EDA** — Scatter plots, correlation heatmap, price distribution analysis
3. **Train/test split** — 80/20, no stratification needed for regression
4. **Pipeline** — `StandardScaler → LinearRegression`
5. **Train** — `pipeline.fit(X_train, y_train)`
6. **Evaluate** — R², RMSE, MAE on both train and test sets + 5-fold CV
7. **Interpret coefficients** — Extract and rank feature importance
8. **Diagnostic plots** — Actual vs Predicted, Residuals, CV scores

---

## ▶️ How to Run

```bash
cd Task2_LinearRegression/notebooks
python task2_linear_regression.py
```

---

## 📊 Results

### Evaluation Metrics

| Metric | Value |
|--------|-------|
| Train R² | ~0.971 |
| Test R² | ~0.974 |
| Train RMSE | ~$14,776 |
| Test RMSE | ~$13,363 |
| CV R² (mean ± std) | ~0.969 ± 0.004 |

**Interpretation:**
- R² ≈ 0.97 → Model explains **97% of price variance** → Excellent fit
- RMSE ≈ $13,363 → Average prediction error is ~$13K on houses ranging $30K–$405K
- Train ≈ Test R² → **No overfitting** (linear models rarely overfit on well-constructed data)
- Low CV std (0.004) → Model is **stable** across different data splits

### Coefficient Rankings (Standardized)

| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| SquareFootage | +81,040 | Strongest driver of price |
| DistanceFromCenter | -25,879 | Location matters significantly |
| HouseAge | -10,345 | Older houses cost less |
| HasGarage | +8,731 | Garage adds ~$8.7K (std units) |
| Bedrooms | +7,020 | More bedrooms = higher price |

---

## ⚠️ Common Mistakes to Avoid

| Mistake | Impact | Fix |
|---------|--------|-----|
| Not scaling before interpreting coefficients | Coefficients are incomparable across features | Always StandardScale before fitting for interpretation |
| Ignoring diagnostic plots | Missing assumption violations (heteroscedasticity, non-normality) | Always plot residuals |
| Reporting only R² | Can be misleading (e.g., high R² with terrible RMSE) | Report R², RMSE, AND MAE |
| No cross-validation | Single train/test split can be lucky | Always use CV for reliable estimates |
| Using Linear Regression on highly non-linear data | Poor fit | Check scatter plots first |

---

## 💡 Optimization Ideas

1. **Polynomial Features** — `PolynomialFeatures(degree=2)` captures non-linear effects (e.g., rooms² interactions)
2. **Ridge Regression** — L2 regularization prevents large coefficients; use when features are correlated
3. **Lasso Regression** — L1 regularization forces some coefficients to zero; automatic feature selection
4. **Feature Engineering** — Add `Price_per_sqft`, log-transform `Fare` if right-skewed
5. **Log-transform target** — If price is log-normal, predict `log(price)` and exponentiate back

---

## 🎤 Interview Q&A

**Q: What does R² = 0.97 mean in plain English?**
> A: The model explains 97% of the variation in house prices. If the average house price varies by $86K, our model accounts for 97% of why some houses are more expensive than others.

**Q: Why is RMSE preferred over MSE for reporting?**
> A: RMSE is in the same unit as the target (dollars), making it directly interpretable. MSE is in dollars², which has no intuitive meaning. "We're off by $13,000" is meaningful. "We're off by 178 million dollars²" is not.

**Q: What does a negative coefficient mean?**
> A: As that feature increases by 1 standardized unit, the predicted price DECREASES by the absolute coefficient value (in the model's scale). For DistanceFromCenter, farther from city center → lower price.

**Q: Why use cross-validation when we already have a test set?**
> A: A single 80/20 split can produce lucky or unlucky partitions. 5-fold CV trains and tests 5 times on different partitions, giving a mean ± std estimate that's statistically more reliable.

**Q: What is the difference between Ridge and Lasso?**
> A: Ridge (L2) adds a penalty on β², shrinking all coefficients toward zero. Lasso (L1) adds a penalty on |β|, which can zero out coefficients entirely — performing automatic feature selection. Use Ridge when all features matter; Lasso when you suspect many features are irrelevant.

---

## 🔗 Resources

- [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
