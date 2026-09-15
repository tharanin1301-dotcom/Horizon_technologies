# Iris Flower Classification Project
"""
Project: Task-1-Iris-Flower-Classification
Author: <Your Name>
Date: 2026-09-15

This script implements a complete end‑to‑end machine‑learning pipeline for the classic Iris
flower classification problem. It follows the structure required for the Horizon TechX
internship project:

1. Load the dataset (using scikit‑learn’s built‑in Iris data).
2. Perform exploratory data analysis (EDA) and save the figures.
3. Preprocess the data – split, encode, scale.
4. Train a Logistic Regression classifier (primary) and a K‑Nearest Neighbours classifier (optional).
5. Evaluate the models on a held‑out test set and save a confusion‑matrix plot.
6. Demonstrate a reusable prediction function.

All file paths are relative to the repository root so the script can be executed from the
project’s top‑level folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple, Any

# ---------------------------------------------------------------------------
# Helper: safe imports with friendly error messages
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
except ImportError as e:  # pragma: no cover – will be caught during execution
    missing_pkg = e.name
    print(
        f"[ERROR] Missing required package: {missing_pkg}.\n",
        "Please install all dependencies with:\n",
        "    pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Global paths (relative to repository root)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # Horizon-Technologies folder
DATA_DIR = ROOT_DIR / "Task-1-Iris-Flower-Classification" / "data"
OUTPUT_FIG_DIR = (
    ROOT_DIR / "Task-1-Iris-Flower-Classification" / "outputs" / "figures"
)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the Iris dataset and return a tidy DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: sepal length, sepal width, petal length, petal width, target.
    """
    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
def perform_eda(df: pd.DataFrame) -> None:
    """Generate and save EDA visualisations.

    The figures are stored under ``outputs/figures`` so that they can be
    referenced from the README.
    """
    # 2.1 Class distribution bar chart
    plt.figure(figsize=(6, 4))
    sns.countplot(x="species", data=df, palette="viridis")
    plt.title("Iris Species Distribution")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "class_distribution.png")
    plt.close()

    # 2.2 Pairplot – relationships between measurements coloured by species
    pairplot = sns.pairplot(df, hue="species", palette="bright", diag_kind="kde")
    pairplot.fig.suptitle("Pairplot of Iris Measurements", y=1.02)
    pairplot.savefig(OUTPUT_FIG_DIR / "pairplot.png")
    plt.close()

    # 2.3 Correlation heatmap for numeric features
    corr = df.drop(columns="species").corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "correlation_heatmap.png")
    plt.close()

    # 2.4 Boxplot – each feature distribution per species (additional visual)
    melted = df.melt(id_vars="species", var_name="feature", value_name="value")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="feature", y="value", hue="species", data=melted)
    plt.title("Feature Distribution per Species")
    plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "feature_boxplot.png")
    plt.close()

# ---------------------------------------------------------------------------
# 3. Data preprocessing
# ---------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """Split the data, encode the target and scale features.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    X = df.drop(columns="species").values
    y = df["species"].values

    # Encode string labels to integers (required for scikit‑learn models)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train‑test split – reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Feature scaling – only fit on training data to avoid leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

# ---------------------------------------------------------------------------
# 4. Model training
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Fit a Logistic Regression model.

    The ``max_iter`` is increased to ensure convergence for the Iris data.
    """
    # ``multi_class`` defaults to "auto" in recent scikit‑learn versions.
    # It is omitted here for compatibility with older versions.
    model = LogisticRegression(max_iter=200, random_state=42, solver="lbfgs")
    model.fit(X_train, y_train)
    return model

def train_knn(X_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 5) -> KNeighborsClassifier:
    """Fit a simple K‑Nearest Neighbours classifier (optional comparison)."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# ---------------------------------------------------------------------------
# 5. Evaluation utilities
# ---------------------------------------------------------------------------
def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder, model_name: str = "Model"
) -> dict:
    """Generate evaluation metrics and save a confusion‑matrix plot.

    Returns a dictionary with ``accuracy`` and the full classification report string.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.close()

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}

# ---------------------------------------------------------------------------
# 6. Prediction demo
# ---------------------------------------------------------------------------
def predict_species(
    model: Any, scaler: StandardScaler, label_encoder: LabelEncoder, measurements: Tuple[float, float, float, float]
) -> str:
    """Predict the Iris species from raw measurements.

    Parameters
    ----------
    measurements : tuple of four floats
        (sepal_length, sepal_width, petal_length, petal_width)
    """
    arr = np.array(measurements).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred_idx = model.predict(arr_scaled)[0]
    return label_encoder.inverse_transform([pred_idx])[0]

# ---------------------------------------------------------------------------
# 7. Main execution workflow
# ---------------------------------------------------------------------------
def main() -> None:
    print("--- Iris Flower Classification Pipeline ---")
    df = load_data()
    print("Dataset loaded. First few rows:")
    print(df.head())

    # Basic data sanity checks (printed for the user)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values per column:")
    print(df.isnull().sum())
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nDescriptive statistics:")
    print(df.describe())

    perform_eda(df)
    print("\n[INFO] EDA figures saved to outputs/figures.")

    X_train, X_test, y_train, y_test, scaler, le = preprocess_data(df)
    print("\n[INFO] Data preprocessing completed. Training set shape:", X_train.shape)

    # Train primary model
    log_reg = train_logistic_regression(X_train, y_train)
    log_metrics = evaluate_model(log_reg, X_test, y_test, le, model_name="Logistic Regression")
    print("\nLogistic Regression Evaluation:")
    print(f"Accuracy: {log_metrics['accuracy']:.4f}")
    print(log_metrics["report"])

    # Optional comparison model
    knn = train_knn(X_train, y_train)
    knn_metrics = evaluate_model(knn, X_test, y_test, le, model_name="K-Nearest Neighbours")
    print("\nK-Nearest Neighbours Evaluation:")
    print(f"Accuracy: {knn_metrics['accuracy']:.4f}")
    print(knn_metrics["report"])

    # Demonstrate prediction
    example = (5.1, 3.5, 1.4, 0.2)  # typical Setosa sample
    pred_species = predict_species(log_reg, scaler, le, example)
    print("\nPrediction demo (using Logistic Regression):")
    print(f"Input measurements: {example}")
    print(f"Predicted species: {pred_species}")

    print("\n--- Pipeline completed successfully ---")

if __name__ == "__main__":
    main()
