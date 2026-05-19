"""
================================================================================
TASK 2 — LINEAR REGRESSION: HOUSE PRICE PREDICTION
CodVeda ML Internship | Level 1
================================================================================
THEORY:
    Linear Regression models the relationship between features (X) and a
    continuous target (y) as a linear equation:

        ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

    The algorithm finds coefficients (β) that minimize the Mean Squared Error
    (MSE) between predictions ŷ and true values y — solved via Ordinary Least
    Squares (OLS).

EVALUATION METRICS USED:
    - R² Score:  Proportion of variance explained by the model (1.0 = perfect)
    - MSE:       Mean Squared Error (penalizes large errors heavily)
    - RMSE:      Root MSE (same unit as target — interpretable!)
    - MAE:       Mean Absolute Error (robust to outliers)

DATASET CHOICE:
    We generate a synthetic house price dataset with features that have
    real-world intuition: square footage, number of bedrooms, age, and location.
    WHY SYNTHETIC? Gives us controlled ground truth to verify model correctness.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120


# ==============================================================================
# SECTION 1 — DATASET CREATION
# ==============================================================================

def create_house_dataset(n: int = 500) -> pd.DataFrame:
    """
    Generate a synthetic house price dataset with realistic feature relationships.

    Feature Design Decisions:
    ─────────────────────────
    - SquareFootage: Main driver of price. Strong positive correlation.
    - Bedrooms:      Positive effect but diminishing returns.
    - HouseAge:      Negative effect — older houses cost less.
    - DistanceFromCenter: Negative effect — farther = cheaper.
    - HasGarage:     Binary — adds a fixed premium.
    - Noise:         Added to simulate real-world measurement error and
                     unmodeled factors. Without noise, R² would be unrealistically
                     close to 1.0.

    The price formula is KNOWN (unlike real data), so we can judge model quality
    against ground truth.
    """
    sq_ft = np.random.randint(500, 4000, n).astype(float)
    bedrooms = np.random.randint(1, 6, n).astype(float)
    house_age = np.random.randint(0, 50, n).astype(float)
    dist_km = np.random.uniform(1, 30, n)
    has_garage = np.random.choice([0, 1], n, p=[0.4, 0.6]).astype(float)

    # Price formula (ground truth)
    noise = np.random.normal(0, 15000, n)
    price = (
        80 * sq_ft
        + 5000 * bedrooms
        - 800 * house_age
        - 3000 * dist_km
        + 20000 * has_garage
        + 50000
        + noise
    ).clip(30000, None)  # No negative prices

    df = pd.DataFrame({
        "SquareFootage": sq_ft,
        "Bedrooms": bedrooms,
        "HouseAge": house_age,
        "DistanceFromCenterKm": dist_km,
        "HasGarage": has_garage,
        "Price": price,
    })

    return df


# ==============================================================================
# SECTION 2 — EDA & VISUALIZATION
# ==============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Visualize feature distributions and correlations before modeling.

    WHY:
        - Scatter plots reveal the strength/direction of each feature's
          relationship with Price.
        - Correlation heatmap identifies multicollinearity (a potential issue
          for coefficient interpretation in Linear Regression).
        - Distribution plots catch skewness that might warrant log-transform.
    """
    print("\n" + "=" * 70)
    print("SECTION 2 — EDA")
    print("=" * 70)
    print(df.describe().round(2))

    features = ["SquareFootage", "Bedrooms", "HouseAge", "DistanceFromCenterKm", "HasGarage"]

    # ── Scatter plots: each feature vs Price ──────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Task 2 — Feature vs. Price Scatter Plots", fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for i, feat in enumerate(features):
        axes[i].scatter(df[feat], df["Price"], alpha=0.4, s=15, color="steelblue")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Price ($)")
        axes[i].set_title(f"{feat} vs Price")
        # Trend line
        z = np.polyfit(df[feat], df["Price"], 1)
        p = np.poly1d(z)
        x_sorted = np.sort(df[feat])
        axes[i].plot(x_sorted, p(x_sorted), "r--", lw=2, label="Trend")
        axes[i].legend()

    # Correlation heatmap in last subplot
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=axes[5])
    axes[5].set_title("Correlation Heatmap")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "task2_eda.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\n✅ EDA plot saved → {path}")

    # ── Price distribution ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Task 2 — Target (Price) Distribution", fontsize=13, fontweight="bold")

    axes[0].hist(df["Price"], bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Price Distribution (Original)")
    axes[0].set_xlabel("Price ($)")

    axes[1].hist(np.log1p(df["Price"]), bins=40, color="mediumseagreen", edgecolor="white")
    axes[1].set_title("Log(Price) Distribution")
    axes[1].set_xlabel("log(Price)")

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "task2_price_distribution.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"✅ Price distribution plot saved → {path2}")


# ==============================================================================
# SECTION 3 — TRAIN/TEST SPLIT
# ==============================================================================

def split_data(df: pd.DataFrame) -> tuple:
    """
    Split features and target into train/test sets.

    WHY 80/20 FOR REGRESSION?
        Unlike classification, we cannot stratify on a continuous target.
        80/20 with a fixed random_state ensures reproducibility.
        With 500 rows → 400 train, 100 test. Sufficient for linear regression.
    """
    feature_cols = ["SquareFootage", "Bedrooms", "HouseAge",
                    "DistanceFromCenterKm", "HasGarage"]
    X = df[feature_cols]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\n📌 Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


# ==============================================================================
# SECTION 4 — MODEL PIPELINE
# ==============================================================================

def build_model_pipeline() -> Pipeline:
    """
    Build a Linear Regression pipeline with StandardScaler.

    WHY SCALE FOR LINEAR REGRESSION?
        OLS coefficients are SCALE-DEPENDENT. Without scaling, SquareFootage
        (ranging 500-4000) dominates the coefficient space over HasGarage (0-1),
        making coefficient comparison meaningless.
        With StandardScaler, each coefficient reflects that feature's TRUE
        relative impact on price.

    NOTE: Scaling does NOT change predictive performance for OLS Linear
    Regression, but it IS ESSENTIAL for:
        - Coefficient interpretability
        - Regularized regression (Ridge, Lasso)
        - Gradient descent convergence
    """
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    return pipeline


# ==============================================================================
# SECTION 5 — TRAINING & EVALUATION
# ==============================================================================

def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> dict:
    """
    Train the model and compute all evaluation metrics.

    METRICS EXPLANATION:
    ─────────────────────
    R² Score:
        R² = 1 - (SS_res / SS_tot)
        SS_res = sum((y - ŷ)²)   ← residual sum of squares
        SS_tot = sum((y - ȳ)²)   ← total sum of squares
        R² = 0.85 means the model explains 85% of price variance.
        R² close to 1.0 = excellent. R² < 0 = worse than predicting the mean.

    MSE (Mean Squared Error):
        MSE = (1/n) * sum((y - ŷ)²)
        Penalizes large errors quadratically. Bad for outlier-heavy data.

    RMSE (Root MSE):
        RMSE = sqrt(MSE)
        Same unit as target ($). Interpretable: "predictions are off by $X on average."

    MAE (Mean Absolute Error):
        MAE = (1/n) * sum(|y - ŷ|)
        Robust to outliers. Easier to explain to stakeholders.

    Cross-Validation R²:
        Evaluates model stability across 5 different train/test folds.
        High variance in CV scores = model is overfitting or data is noisy.
    """
    # Fit
    pipeline.fit(X_train, y_train)

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Metrics
    metrics = {
        "Train R²": r2_score(y_train, y_train_pred),
        "Test R²": r2_score(y_test, y_test_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
    }

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
    metrics["CV R² Mean"] = cv_scores.mean()
    metrics["CV R² Std"] = cv_scores.std()

    print("\n" + "=" * 70)
    print("SECTION 5 — MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  {'Metric':<25} {'Value':>15}")
    print("  " + "-" * 42)
    for k, v in metrics.items():
        print(f"  {k:<25} {v:>15.4f}")

    # Interpret results
    print("\n📊 INTERPRETATION:")
    r2 = metrics["Test R²"]
    rmse = metrics["Test RMSE"]
    gap = metrics["Train R²"] - metrics["Test R²"]

    if r2 > 0.85:
        print(f"  ✅ R² = {r2:.4f} — Model explains {r2*100:.1f}% of price variance. STRONG fit.")
    elif r2 > 0.70:
        print(f"  ⚠️  R² = {r2:.4f} — Decent fit. Consider feature engineering.")
    else:
        print(f"  ❌ R² = {r2:.4f} — Poor fit. Review features or try nonlinear models.")

    print(f"  📏 RMSE = ${rmse:,.0f} — On average, predictions are off by this amount.")

    if gap > 0.05:
        print(f"  ⚠️  Train-Test gap = {gap:.4f} — Possible overfitting. Consider regularization.")
    else:
        print(f"  ✅ Train-Test gap = {gap:.4f} — No significant overfitting.")

    return metrics, y_test_pred, cv_scores


# ==============================================================================
# SECTION 6 — COEFFICIENT INTERPRETATION
# ==============================================================================

def interpret_coefficients(
    pipeline: Pipeline,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract and interpret model coefficients.

    WHY COEFFICIENTS MATTER:
        In Linear Regression, each coefficient βᵢ tells us:
        "For a 1-unit increase in feature xᵢ (after scaling), the predicted
         price changes by βᵢ dollars, holding all other features constant."

        After StandardScaler, a 1-unit change = 1 standard deviation, so
        larger absolute coefficients = more influential features.

        Positive coefficient → feature increases price.
        Negative coefficient → feature decreases price.

        This makes Linear Regression one of the most INTERPRETABLE ML models —
        a key advantage over black-box models like Random Forests or NNs.
    """
    regressor = pipeline.named_steps["regressor"]
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": regressor.coef_,
        "Abs_Coefficient": np.abs(regressor.coef_),
    }).sort_values("Abs_Coefficient", ascending=False)

    print("\n" + "=" * 70)
    print("SECTION 6 — COEFFICIENT INTERPRETATION")
    print("=" * 70)
    print(f"\n  Intercept (β₀): ${regressor.intercept_:,.2f}")
    print("\n  Coefficients (sorted by absolute impact):")
    print("  " + "-" * 50)
    for _, row in coef_df.iterrows():
        direction = "↑" if row["Coefficient"] > 0 else "↓"
        print(f"  {direction} {row['Feature']:<28} {row['Coefficient']:>12,.2f}")

    # Coefficient bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["steelblue" if c > 0 else "coral" for c in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Task 2 — Linear Regression Coefficients\n(Positive=Price Up | Negative=Price Down)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient Value (Standardized Features)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "task2_coefficients.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Coefficient plot saved → {path}")

    return coef_df


# ==============================================================================
# SECTION 7 — DIAGNOSTIC VISUALIZATIONS
# ==============================================================================

def plot_diagnostics(
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    cv_scores: np.ndarray
) -> None:
    """
    Generate four diagnostic plots for regression model quality assessment.

    PLOTS:
    1. Actual vs Predicted — should lie on y=x diagonal.
    2. Residual Distribution — should be approximately normal (OLS assumption).
    3. Residuals vs Predicted — should be random (no pattern = no heteroscedasticity).
    4. Cross-Validation R² — should show low variance across folds.
    """
    residuals = y_test - y_test_pred

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Task 2 — Model Diagnostic Plots", fontsize=15, fontweight="bold")

    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=25, color="steelblue")
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    axes[0, 0].set_xlabel("Actual Price ($)")
    axes[0, 0].set_ylabel("Predicted Price ($)")
    axes[0, 0].set_title("Actual vs. Predicted Price")
    axes[0, 0].legend()

    # 2. Residual Distribution
    axes[0, 1].hist(residuals, bins=25, color="mediumseagreen", edgecolor="white", density=True)
    # Overlay normal curve
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 200)
    axes[0, 1].plot(x, (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2),
                    "r-", lw=2, label=f"Normal(μ={mu:.0f}, σ={sigma:.0f})")
    axes[0, 1].set_title("Residual Distribution (should be ~Normal)")
    axes[0, 1].set_xlabel("Residual ($)")
    axes[0, 1].legend()

    # 3. Residuals vs Predicted
    axes[1, 0].scatter(y_test_pred, residuals, alpha=0.5, s=25, color="coral")
    axes[1, 0].axhline(0, color="black", linewidth=1.5, linestyle="--")
    axes[1, 0].set_xlabel("Predicted Price ($)")
    axes[1, 0].set_ylabel("Residual ($)")
    axes[1, 0].set_title("Residuals vs. Predicted (should be random)")

    # 4. Cross-validation scores
    fold_labels = [f"Fold {i+1}" for i in range(len(cv_scores))]
    axes[1, 1].bar(fold_labels, cv_scores, color="mediumpurple", edgecolor="white")
    axes[1, 1].axhline(cv_scores.mean(), color="red", linestyle="--",
                        label=f"Mean R² = {cv_scores.mean():.4f}")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("5-Fold Cross-Validation R² Scores")
    axes[1, 1].set_ylabel("R² Score")
    axes[1, 1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "task2_diagnostics.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Diagnostic plots saved → {path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 70)
    print("   TASK 2 — LINEAR REGRESSION: HOUSE PRICE PREDICTION")
    print("   CodVeda ML Internship | Level 1")
    print("=" * 70)

    # Step 1: Data
    print("\n[1/5] Creating dataset...")
    df = create_house_dataset(n=500)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "house_prices.csv"), index=False)
    print(f"   Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # Step 2: EDA
    print("\n[2/5] Running EDA...")
    run_eda(df)

    # Step 3: Split
    print("\n[3/5] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 4: Build model
    print("\n[4/5] Building and training model...")
    feature_cols = ["SquareFootage", "Bedrooms", "HouseAge",
                    "DistanceFromCenterKm", "HasGarage"]
    pipeline = build_model_pipeline()
    metrics, y_test_pred, cv_scores = train_and_evaluate(
        pipeline, X_train, X_test, y_train, y_test
    )

    # Step 5: Interpret & visualize
    print("\n[5/5] Interpreting coefficients & generating diagnostics...")
    interpret_coefficients(pipeline, feature_cols)
    plot_diagnostics(y_test, y_test_pred, cv_scores)

    print("\n" + "=" * 70)
    print("   ✅ TASK 2 COMPLETE — All outputs saved to /outputs")
    print("=" * 70)


if __name__ == "__main__":
    main()
