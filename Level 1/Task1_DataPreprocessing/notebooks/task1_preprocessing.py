"""
================================================================================
TASK 1 — DATA PREPROCESSING
CodVeda ML Internship | Level 1
================================================================================
WHY PREPROCESSING MATTERS:
    Raw data is almost never ready for ML models. Real-world datasets contain
    missing values, categorical strings, features on vastly different scales,
    and unbalanced splits. Feeding raw data directly into a model causes:
        - Numerical errors (NaN propagation)
        - Biased learning (dominant-scale features)
        - Invalid category representations
    This script demonstrates every critical preprocessing step with
    professional justifications for each decision.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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

def create_synthetic_dataset() -> pd.DataFrame:
    """
    Create a realistic synthetic dataset mimicking a Titanic-style passenger
    survival dataset.

    WHY SYNTHETIC?
        We control exactly which columns have missing values, which are
        categorical, and which need scaling — making it ideal for teaching
        all preprocessing concepts in one place.

    Returns
    -------
    pd.DataFrame
        A 300-row dataset with intentional messiness for preprocessing practice.
    """
    n = 300

    data = {
        # Numerical — continuous, will need StandardScaler
        "Age": np.random.normal(loc=35, scale=15, size=n).clip(1, 80),
        "Fare": np.random.exponential(scale=50, size=n).clip(5, 500),
        "SibSp": np.random.randint(0, 5, size=n).astype(float),

        # Categorical — nominal, will need OneHotEncoding
        "Sex": np.random.choice(["male", "female"], size=n),
        "Embarked": np.random.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1]),
        "Pclass": np.random.choice([1, 2, 3], size=n, p=[0.2, 0.3, 0.5]),

        # Target
        "Survived": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
    }

    df = pd.DataFrame(data)

    # ── Inject realistic missing values ────────────────────────────────────────
    # Age: ~20% missing (common in real datasets)
    age_missing_idx = np.random.choice(df.index, size=int(0.20 * n), replace=False)
    df.loc[age_missing_idx, "Age"] = np.nan

    # Embarked: ~2% missing (small amount, common)
    embarked_missing_idx = np.random.choice(df.index, size=int(0.02 * n), replace=False)
    df.loc[embarked_missing_idx, "Embarked"] = np.nan

    # Fare: ~5% missing
    fare_missing_idx = np.random.choice(df.index, size=int(0.05 * n), replace=False)
    df.loc[fare_missing_idx, "Fare"] = np.nan

    return df


# ==============================================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS (Pre-Preprocessing)
# ==============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Perform and visualize initial data quality assessment.

    WHY EDA BEFORE PREPROCESSING?
        You must understand your data before touching it. EDA reveals:
        - Missing value patterns (random vs. systematic)
        - Distribution shapes (normal vs. skewed — affects imputation strategy)
        - Outlier presence
        - Class balance
    """
    print("\n" + "=" * 70)
    print("SECTION 2 — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\n📌 Dataset Shape:", df.shape)
    print("\n📌 Data Types:\n", df.dtypes)
    print("\n📌 First 5 Rows:\n", df.head())
    print("\n📌 Statistical Summary:\n", df.describe(include="all").round(2))

    # Missing value report
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    }).query("`Missing Count` > 0")
    print("\n📌 Missing Values:\n", missing_report)

    # ── Visualization ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Task 1 — Pre-Preprocessing EDA", fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # 1. Missing value heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False, ax=ax1)
    ax1.set_title("Missing Value Map (Yellow = Missing)")

    # 2. Missing value bar chart
    ax2 = fig.add_subplot(gs[0, 2])
    missing_report["Missing %"].plot(kind="barh", ax=ax2, color="salmon")
    ax2.set_title("Missing Value Percentages")
    ax2.set_xlabel("% Missing")

    # 3. Age distribution
    ax3 = fig.add_subplot(gs[1, 0])
    df["Age"].dropna().hist(bins=25, ax=ax3, color="steelblue", edgecolor="white")
    ax3.set_title("Age Distribution (with NaN dropped)")
    ax3.set_xlabel("Age")

    # 4. Fare distribution
    ax4 = fig.add_subplot(gs[1, 1])
    df["Fare"].dropna().hist(bins=30, ax=ax4, color="mediumseagreen", edgecolor="white")
    ax4.set_title("Fare Distribution (Right-Skewed)")
    ax4.set_xlabel("Fare")

    # 5. Target class balance
    ax5 = fig.add_subplot(gs[1, 2])
    df["Survived"].value_counts().plot(kind="bar", ax=ax5, color=["coral", "steelblue"],
                                        edgecolor="white")
    ax5.set_title("Target Class Distribution")
    ax5.set_xticklabels(["Did Not Survive", "Survived"], rotation=0)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "task1_eda_overview.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\n✅ EDA plot saved → {path}")


# ==============================================================================
# SECTION 3 — PREPROCESSING PIPELINE
# ==============================================================================

def build_preprocessing_pipeline(
    numerical_features: list,
    categorical_features: list
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer pipeline for preprocessing.

    WHY PIPELINES?
        Pipelines are the industry standard for ML preprocessing because:
        1. They prevent data leakage — fit() is called ONLY on training data,
           then transform() is applied to test data using training statistics.
        2. They make deployment trivial — the whole pipeline can be serialized.
        3. They are readable, auditable, and reproducible.

    WHY COLUMNTRANSFORMER?
        Different column types need different transformations. ColumnTransformer
        applies the correct pipeline to each column group simultaneously.

    Parameters
    ----------
    numerical_features : list
        Column names of numerical features.
    categorical_features : list
        Column names of categorical features.

    Returns
    -------
    ColumnTransformer
        Assembled preprocessing pipeline (not yet fitted).
    """

    # ── Numerical Pipeline ─────────────────────────────────────────────────────
    # WHY MEDIAN IMPUTATION?
    #   Age and Fare are right-skewed. Mean imputation would be pulled by
    #   outliers, producing unrealistic values. Median is robust to outliers.
    # WHY STANDARDSCALER?
    #   Many ML algorithms (especially distance-based and gradient-based)
    #   are sensitive to feature scale. StandardScaler transforms each feature
    #   to zero mean and unit variance: z = (x - μ) / σ
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # ── Categorical Pipeline ───────────────────────────────────────────────────
    # WHY MOST_FREQUENT IMPUTATION?
    #   For nominal categories like "Embarked", the safest assumption when
    #   the value is missing is the most common port. There is no meaningful
    #   "average" for a categorical variable.
    # WHY ONEHOTENCODING vs LABEL ENCODING?
    #   LabelEncoding assigns arbitrary integers (S=2, C=0, Q=1) implying
    #   ordinal relationships that DON'T EXIST for nominal features.
    #   OneHotEncoding creates a binary column per category — no false ordering.
    #   handle_unknown='ignore' prevents crashes if test set has unseen categories.
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return preprocessor


# ==============================================================================
# SECTION 4 — TRAIN/TEST SPLIT
# ==============================================================================

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2
) -> tuple:
    """
    Split the dataset into training and testing sets.

    WHY 80/20 SPLIT?
        80% training gives the model enough data to learn patterns.
        20% test provides a statistically meaningful held-out evaluation.
        For 300 rows: 240 train, 60 test — reasonable for this dataset size.

    WHY STRATIFY?
        With imbalanced targets (e.g., 60% didn't survive, 40% survived),
        a random split might put all survivors in train and none in test.
        stratify=y ensures each split has the same class proportions.

    WHY random_state?
        Without a fixed seed, each run produces a different split, making
        results irreproducible. random_state=42 is a convention.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    test_size : float

    Returns
    -------
    tuple : X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\n📌 Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    print(f"📌 Train target distribution:\n{y_train.value_counts(normalize=True).round(3)}")
    print(f"📌 Test target distribution:\n{y_test.value_counts(normalize=True).round(3)}")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# SECTION 5 — FIT & TRANSFORM
# ==============================================================================

def fit_and_transform(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_features: list,
    categorical_features: list
) -> tuple:
    """
    Fit the preprocessor on training data ONLY, then transform both sets.

    ⚠️ CRITICAL ML RULE — NO DATA LEAKAGE:
        WRONG:  preprocessor.fit(X_all)   → leaks test statistics into training
        WRONG:  preprocessor.fit(X_test)  → trains on test set
        CORRECT: fit on X_train → transform X_train AND X_test

        If you fit on the full dataset, the model indirectly "sees" the test
        set during training (e.g., the scaler's mean includes test values).
        This leads to optimistically biased evaluation metrics.

    Returns
    -------
    tuple : (X_train_processed, X_test_processed, feature_names)
    """
    # Fit ONLY on training data
    preprocessor.fit(X_train)

    # Transform both
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Reconstruct feature names for interpretability
    ohe_feature_names = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )
    all_feature_names = numerical_features + ohe_feature_names

    print(f"\n📌 Processed train shape: {X_train_processed.shape}")
    print(f"📌 Processed test shape:  {X_test_processed.shape}")
    print(f"📌 Feature names after encoding: {all_feature_names}")

    return X_train_processed, X_test_processed, all_feature_names


# ==============================================================================
# SECTION 6 — POST-PREPROCESSING VISUALIZATION
# ==============================================================================

def visualize_preprocessing_results(
    X_train: pd.DataFrame,
    X_train_processed: np.ndarray,
    numerical_features: list,
    feature_names: list
) -> None:
    """
    Visualize the effect of scaling on numerical features.

    WHY THIS MATTERS:
        Showing before/after scaling helps verify the pipeline worked correctly
        and builds intuition about what StandardScaler actually does.
    """
    fig, axes = plt.subplots(2, len(numerical_features), figsize=(16, 8))
    fig.suptitle("Task 1 — Before vs. After StandardScaler", fontsize=14, fontweight="bold")

    for i, col in enumerate(numerical_features):
        # Before scaling
        axes[0, i].hist(X_train[col].dropna(), bins=25, color="coral", edgecolor="white")
        axes[0, i].set_title(f"{col} — BEFORE")
        axes[0, i].set_xlabel("Original Value")

        # After scaling — grab the column from processed array
        axes[1, i].hist(X_train_processed[:, i], bins=25, color="steelblue", edgecolor="white")
        axes[1, i].set_title(f"{col} — AFTER (z-scored)")
        axes[1, i].set_xlabel("Standardized Value")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "task1_scaling_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ Scaling comparison plot saved → {path}")

    # Correlation heatmap of processed features (first 6)
    processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        processed_df.iloc[:, :8].corr(),
        annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax
    )
    ax.set_title("Correlation Heatmap — Processed Training Features (first 8)", fontsize=13)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "task1_correlation_heatmap.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"✅ Correlation heatmap saved → {path2}")


# ==============================================================================
# SECTION 7 — SAVE PROCESSED DATA
# ==============================================================================

def save_processed_data(
    X_train_processed: np.ndarray,
    X_test_processed: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list
) -> None:
    """
    Save processed arrays as CSVs for use in subsequent tasks.

    WHY SAVE?
        Downstream tasks (Linear Regression, KNN) can import clean, processed
        data without re-running preprocessing. This also makes the pipeline
        auditable — you can inspect what the model actually trained on.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_df["Survived"] = y_train.values
    train_df.to_csv(os.path.join(data_dir, "train_processed.csv"), index=False)

    test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_df["Survived"] = y_test.values
    test_df.to_csv(os.path.join(data_dir, "test_processed.csv"), index=False)

    print(f"\n✅ Processed train data saved → data/train_processed.csv")
    print(f"✅ Processed test data saved  → data/test_processed.csv")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 70)
    print("   TASK 1 — DATA PREPROCESSING")
    print("   CodVeda ML Internship | Level 1")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/6] Creating dataset...")
    df = create_synthetic_dataset()

    # Step 2: EDA
    print("\n[2/6] Running EDA...")
    run_eda(df)

    # Step 3: Define feature groups
    numerical_features = ["Age", "Fare", "SibSp"]
    categorical_features = ["Sex", "Embarked", "Pclass"]
    target = "Survived"

    # Step 4: Split
    print("\n[3/6] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, target_col=target)

    # Step 5: Build pipeline
    print("\n[4/6] Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)

    # Step 6: Fit & transform
    print("\n[5/6] Fitting and transforming...")
    X_train_proc, X_test_proc, feature_names = fit_and_transform(
        preprocessor, X_train, X_test, numerical_features, categorical_features
    )

    # Step 7: Visualize
    print("\n[6/6] Generating visualizations...")
    visualize_preprocessing_results(X_train, X_train_proc, numerical_features, feature_names)

    # Step 8: Save
    save_processed_data(X_train_proc, X_test_proc, y_train, y_test, feature_names)

    print("\n" + "=" * 70)
    print("   ✅ TASK 1 COMPLETE — All outputs saved to /outputs and /data")
    print("=" * 70)


if __name__ == "__main__":
    main()
