"""
================================================================================
TASK 3 — KNN CLASSIFICATION: IRIS DATASET
CodVeda ML Internship | Level 1
================================================================================
THEORY:
    K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning
    algorithm. It makes NO assumptions about the underlying data distribution.

    PREDICTION LOGIC:
        For a new data point X:
        1. Compute distance to every training point (Euclidean by default)
        2. Select the K closest training points
        3. Return the majority class among those K neighbors

    DISTANCE FORMULA (Euclidean):
        d(p, q) = sqrt(Σ(pᵢ - qᵢ)²)

    KEY HYPERPARAMETER: K
        - Small K (K=1): Very flexible, captures noise, HIGH VARIANCE
          → Can overfit (memorizes training data)
        - Large K: Very smooth, ignores local structure, HIGH BIAS
          → Can underfit
        - Optimal K: Found by evaluating multiple values via cross-validation

    WHY SCALING IS MANDATORY FOR KNN:
        KNN uses raw distances. If Age ranges 0-80 and Salary ranges 20000-100000,
        Salary will DOMINATE the distance metric, and Age becomes irrelevant.
        StandardScaler puts all features on equal footing.

DATASET: Iris
    - 150 samples, 3 classes (setosa, versicolor, virginica)
    - 4 features: sepal length, sepal width, petal length, petal width
    - Classic benchmark dataset — well-studied, no missing values
    - Perfect for demonstrating multi-class KNN
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

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
# SECTION 1 — LOAD DATASET
# ==============================================================================

def load_iris_dataset() -> tuple:
    """
    Load the Iris dataset and return as DataFrame + arrays.

    DATASET PITFALLS TO AVOID:
    ─────────────────────────
    1. Class imbalance: Iris is balanced (50 per class) — no SMOTE needed.
    2. Feature scale: Sepal/petal dimensions have similar ranges, but
       we STILL scale because KNN is distance-based. Never skip scaling for KNN.
    3. Feature correlation: Petal length and width are highly correlated.
       KNN handles this implicitly; doesn't need feature selection for a 4-feature dataset.
    4. Tiny dataset (150 rows): Use cross-validation, not just one train/test split.

    Returns
    -------
    tuple : (X, y, class_names, feature_names, df)
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names
    feature_names = iris.feature_names

    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [class_names[i] for i in y]

    print("\n" + "=" * 70)
    print("SECTION 1 — DATASET OVERVIEW")
    print("=" * 70)
    print(f"\n📌 Dataset shape: {X.shape} (samples × features)")
    print(f"📌 Classes: {list(class_names)}")
    print(f"📌 Class distribution:\n{df['species'].value_counts()}")
    print(f"\n📌 First 5 rows:\n{df.head()}")
    print(f"\n📌 Statistical summary:\n{df.describe().round(3)}")

    return X, y, class_names, feature_names, df


# ==============================================================================
# SECTION 2 — EDA
# ==============================================================================

def run_eda(df: pd.DataFrame, feature_names: list) -> None:
    """
    Visualize the Iris dataset structure.

    WHY PAIRPLOT FOR IRIS?
        With 4 features and 3 classes, a pairplot shows EVERY pairwise
        relationship simultaneously. It reveals:
        - Setosa is linearly separable from the other two.
        - Versicolor and Virginica overlap → KNN needs higher K for this boundary.
        - Petal features are more discriminative than sepal features.
    """
    # Pairplot
    print("\n[EDA] Generating pairplot (this may take a moment)...")
    g = sns.pairplot(df, hue="species", diag_kind="hist",
                     plot_kws={"alpha": 0.7, "s": 30},
                     palette={"setosa": "steelblue",
                              "versicolor": "mediumseagreen",
                              "virginica": "coral"})
    g.fig.suptitle("Task 3 — Iris Pairplot (All Feature Combinations)", y=1.02, fontsize=13)
    path = os.path.join(OUTPUT_DIR, "task3_pairplot.png")
    g.fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ Pairplot saved → {path}")

    # Boxplots per feature
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Task 3 — Feature Distributions by Species", fontsize=13, fontweight="bold")
    for i, feat in enumerate(feature_names):
        df.boxplot(column=feat, by="species", ax=axes[i],
                   boxprops=dict(color="steelblue"),
                   medianprops=dict(color="red"))
        axes[i].set_title(feat, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("cm")
    plt.suptitle("")
    fig.suptitle("Task 3 — Feature Distributions by Species", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "task3_boxplots.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"✅ Boxplots saved → {path2}")

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    ax.set_title("Task 3 — Feature Correlation Heatmap")
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "task3_correlation.png")
    plt.savefig(path3, bbox_inches="tight")
    plt.close()
    print(f"✅ Correlation heatmap saved → {path3}")


# ==============================================================================
# SECTION 3 — TRAIN/TEST SPLIT
# ==============================================================================

def split_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Stratified train/test split for multi-class classification.

    WHY STRATIFY FOR IRIS?
        Each class has 50 samples. With 80/20 split = 120 train, 30 test.
        Stratify ensures 10 samples per class in test (not random clumping).
        Critical for balanced evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"\n📌 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"📌 Train class counts: {np.bincount(y_train)}")
    print(f"📌 Test class counts:  {np.bincount(y_test)}")
    return X_train, X_test, y_train, y_test


# ==============================================================================
# SECTION 4 — K VALUE EXPERIMENT
# ==============================================================================

def experiment_with_k_values(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k_range: range
) -> pd.DataFrame:
    """
    Train KNN for multiple K values and compare performance.

    WHY EXPERIMENT WITH K?
        K is KNN's only hyperparameter. The right K balances:
        - Underfitting (K too large → ignores local structure)
        - Overfitting (K too small → memorizes noise)

        Standard practice: test odd K values 1-21 to avoid ties in binary
        classification. For multi-class (Iris), even K can still tie, so odd
        values are still preferred.

    EVALUATION:
        We report both test accuracy AND cross-validation accuracy.
        Test accuracy: single evaluation (can be lucky/unlucky split).
        CV accuracy: average over 5 folds — more RELIABLE estimate.
    """
    results = []

    for k in k_range:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, metric="euclidean"))
        ])

        # Train & test accuracy
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        # Cross-validation (on training data only — no leakage)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

        results.append({
            "K": k,
            "Test Accuracy": test_acc,
            "CV Accuracy Mean": cv_scores.mean(),
            "CV Accuracy Std": cv_scores.std(),
        })

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("SECTION 4 — K VALUE COMPARISON")
    print("=" * 70)
    print(df_results.to_string(index=False, float_format="%.4f"))

    best_row = df_results.loc[df_results["CV Accuracy Mean"].idxmax()]
    print(f"\n⭐ BEST K (by CV Accuracy): K = {int(best_row['K'])} "
          f"| CV Acc = {best_row['CV Accuracy Mean']:.4f} ± {best_row['CV Accuracy Std']:.4f}")

    return df_results


# ==============================================================================
# SECTION 5 — BEST MODEL EVALUATION
# ==============================================================================

def evaluate_best_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    best_k: int,
    class_names: list
) -> tuple:
    """
    Train final model with best K and compute all evaluation metrics.

    METRICS EXPLANATION:
    ─────────────────────
    Accuracy:
        Correct predictions / Total predictions.
        Reliable when classes are BALANCED (Iris: balanced, so OK here).
        ⚠️ MISLEADING on imbalanced data (95% accuracy on 95/5 split = trivial).

    Precision (per class):
        Of all samples predicted as class C, what fraction actually ARE class C?
        High precision = few false positives.
        Formula: TP / (TP + FP)

    Recall (per class):
        Of all actual class C samples, what fraction did we correctly identify?
        High recall = few false negatives.
        Formula: TP / (TP + FN)

    F1-Score (per class):
        Harmonic mean of Precision and Recall.
        F1 = 2 × (Precision × Recall) / (Precision + Recall)
        Useful when you need to balance both concerns.

    Macro Average: Unweighted mean across all classes (treats all equally).
    Weighted Average: Weighted by class support (accounts for class size).

    Confusion Matrix:
        Row = Actual class | Column = Predicted class
        Diagonal = Correct predictions
        Off-diagonal = Confusion between classes
        A model that confuses versicolor with virginica (common!) will show
        off-diagonal values in those rows/columns.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k, metric="euclidean"))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # All metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 70)
    print(f"SECTION 5 — BEST MODEL EVALUATION (K = {best_k})")
    print("=" * 70)
    print(f"\n  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f} (weighted avg)")
    print(f"  Recall    : {rec:.4f} (weighted avg)")
    print(f"  F1-Score  : {f1:.4f} (weighted avg)")
    print("\n  Full Classification Report:")
    print("  " + "-" * 55)
    print(classification_report(y_test, y_pred, target_names=class_names))

    return pipeline, y_pred


# ==============================================================================
# SECTION 6 — VISUALIZATIONS
# ==============================================================================

def plot_k_comparison(df_results: pd.DataFrame) -> None:
    """
    Plot Test Accuracy and CV Accuracy vs K to find the elbow/optimal point.

    WHAT TO LOOK FOR:
        - Both curves should be high. A gap between them indicates variance.
        - The "elbow" where test accuracy peaks (or CV peaks) is optimal K.
        - After the elbow, accuracy typically drops as K → too smooth.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_results["K"], df_results["Test Accuracy"],
            marker="o", color="steelblue", linewidth=2, label="Test Accuracy")
    ax.plot(df_results["K"], df_results["CV Accuracy Mean"],
            marker="s", color="coral", linewidth=2, label="CV Accuracy (mean)")
    ax.fill_between(
        df_results["K"],
        df_results["CV Accuracy Mean"] - df_results["CV Accuracy Std"],
        df_results["CV Accuracy Mean"] + df_results["CV Accuracy Std"],
        alpha=0.2, color="coral", label="CV ±1 Std Dev"
    )

    best_k = df_results.loc[df_results["CV Accuracy Mean"].idxmax(), "K"]
    ax.axvline(best_k, color="green", linestyle="--", linewidth=1.5,
               label=f"Best K = {int(best_k)}")

    ax.set_xlabel("Number of Neighbors (K)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Task 3 — KNN: Accuracy vs. K Value", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_xticks(df_results["K"])
    ax.set_ylim(0.8, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "task3_k_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ K comparison plot saved → {path}")


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    best_k: int
) -> None:
    """
    Plot a normalized and raw confusion matrix side by side.

    WHY BOTH?
        Raw: Shows absolute counts — useful to see HOW MANY samples are confused.
        Normalized: Shows percentages — useful when classes have different sizes.
        For Iris (balanced), both tell the same story — but it's good practice.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Task 3 — Confusion Matrix (K={best_k})", fontsize=14, fontweight="bold")

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    for ax, matrix, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw Counts", "Normalized (Row %)"],
        ["d", ".2f"]
    ):
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "task3_confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved → {path}")


def plot_decision_boundary_2d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_k: int,
    feature_names: list,
    class_names: list
) -> None:
    """
    Plot 2D decision boundaries using the two most discriminative features:
    petal length (index 2) and petal width (index 3).

    WHY THESE TWO FEATURES?
        The pairplot shows petal features provide the clearest class separation.
        We reduce to 2D for visualization; 4D boundaries cannot be shown directly.

    WHAT THE BOUNDARY SHOWS:
        - Colored regions = areas where KNN predicts that class.
        - Points = actual data (shape = train; star = test).
        - Misclassified test points would appear in the WRONG region color.
    """
    # Use petal length and width (features 2, 3)
    feat_idx = [2, 3]
    X_train_2d = X_train[:, feat_idx]
    X_test_2d = X_test[:, feat_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train)

    # Mesh
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_bg = ["#AED6F1", "#A9DFBF", "#F9A7A7"]
    colors_fg = ["steelblue", "mediumseagreen", "coral"]

    for c, color in enumerate(colors_bg):
        ax.contourf(xx, yy, (Z == c).astype(int),
                    levels=[0.5, 1.5], colors=[color], alpha=0.4)

    markers = ["o", "s", "^"]
    for c, (color, marker) in enumerate(zip(colors_fg, markers)):
        mask_train = y_train == c
        mask_test = y_test == c
        ax.scatter(X_train_scaled[mask_train, 0], X_train_scaled[mask_train, 1],
                   color=color, marker=marker, s=60, edgecolors="white",
                   label=f"{class_names[c]} (train)", linewidths=0.5, zorder=3)
        ax.scatter(X_test_scaled[mask_test, 0], X_test_scaled[mask_test, 1],
                   color=color, marker="*", s=200, edgecolors="black",
                   linewidths=0.8, zorder=4)

    ax.set_xlabel(f"{feature_names[feat_idx[0]]} (standardized)", fontsize=11)
    ax.set_ylabel(f"{feature_names[feat_idx[1]]} (standardized)", fontsize=11)
    ax.set_title(f"Task 3 — KNN Decision Boundary (K={best_k})\n"
                 f"2D View: Petal Features Only  ★ = Test Samples", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "task3_decision_boundary.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ Decision boundary plot saved → {path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 70)
    print("   TASK 3 — KNN CLASSIFICATION: IRIS DATASET")
    print("   CodVeda ML Internship | Level 1")
    print("=" * 70)

    # Step 1: Load
    print("\n[1/6] Loading Iris dataset...")
    X, y, class_names, feature_names, df = load_iris_dataset()

    # Step 2: EDA
    print("\n[2/6] Running EDA...")
    run_eda(df, feature_names)

    # Step 3: Split
    print("\n[3/6] Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: K experiment
    print("\n[4/6] Experimenting with K values (1 to 21, odd)...")
    k_range = range(1, 22, 2)  # Odd values only: 1, 3, 5, ..., 21
    df_results = experiment_with_k_values(X_train, X_test, y_train, y_test, k_range)

    # Step 5: Best model
    best_k = int(df_results.loc[df_results["CV Accuracy Mean"].idxmax(), "K"])
    print(f"\n[5/6] Training best model with K = {best_k}...")
    pipeline, y_pred = evaluate_best_model(X_train, X_test, y_train, y_test, best_k, class_names)

    # Step 6: Visualizations
    print("\n[6/6] Generating visualizations...")
    plot_k_comparison(df_results)
    plot_confusion_matrix(y_test, y_pred, class_names, best_k)
    plot_decision_boundary_2d(X_train, y_train, X_test, y_test, best_k, feature_names, class_names)

    # Save K results
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    df_results.to_csv(os.path.join(data_dir, "knn_k_comparison.csv"), index=False)

    print("\n" + "=" * 70)
    print("   ✅ TASK 3 COMPLETE — All outputs saved to /outputs")
    print("=" * 70)


if __name__ == "__main__":
    main()
