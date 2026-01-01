"""Train a baseline model to predict rejected claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

APPROVED_OUTCOMES = {"approved", "paid"}


def load_claims(path: Path) -> pd.DataFrame:
    """Load claims data and normalize types."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["outcome"] = df["outcome"].astype(str)
    df["rejection_code"] = df["rejection_code"].astype(str)
    df["payer"] = df["payer"].astype(str)
    df["drug_class"] = df["drug_class"].astype(str)
    df["copay_bucket"] = df["copay_bucket"].astype(str)
    df["is_rejected"] = ~df["outcome"].str.lower().isin(APPROVED_OUTCOMES)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create baseline features for modeling."""
    features = df[["payer", "drug_class", "copay_bucket", "store_workload_proxy"]].copy()
    features["claim_month"] = df["date"].dt.month.astype("Int64")
    return features


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    categorical_features = ["payer", "drug_class", "copay_bucket"]
    numeric_features = ["store_workload_proxy", "claim_month"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = LogisticRegression(max_iter=500, class_weight="balanced")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return metrics


def save_metrics(
    metrics: dict[str, float],
    report: str,
    matrix: list[list[int]],
    output_dir: Path,
) -> None:
    payload = {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
    }
    (output_dir / "baseline_model_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = ["# Baseline Model Metrics", "", "## Summary"]
    for key, value in metrics.items():
        lines.append(f"- **{key}**: {value:.3f}")
    lines.extend([
        "",
        "## Confusion Matrix",
        "",
        f"- True Negatives: {matrix[0][0]}",
        f"- False Positives: {matrix[0][1]}",
        f"- False Negatives: {matrix[1][0]}",
        f"- True Positives: {matrix[1][1]}",
        "",
        "## Classification Report",
        "",
        "```",
        report,
        "```",
    ])
    (output_dir / "baseline_model_metrics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def plot_roc_curve(
    y_test: pd.Series,
    y_prob: pd.Series,
    output_dir: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}", color="#4C78A8")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Baseline Model ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_model_roc_curve.png", dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline model to predict rejected claims."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/synthetic_claims.csv"),
        help="Path to the synthetic claims CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/model_v1"),
        help="Directory to write model metrics and plots.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_claims(args.data_path)
    X = build_features(df)
    y = df["is_rejected"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(model, X_test, y_test)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    save_metrics(metrics, report, matrix, output_dir)
    plot_roc_curve(y_test, y_prob, output_dir)

    print("Baseline model metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.3f}")
    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
