"""Run exploratory analysis on the synthetic claims dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

APPROVED_OUTCOMES = {"approved", "paid"}


def load_claims(path: Path) -> pd.DataFrame:
    """Load claims data and apply light normalization."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["outcome"] = df["outcome"].astype(str)
    df["rejection_code"] = df["rejection_code"].astype(str)
    df["payer"] = df["payer"].astype(str)
    df["drug_class"] = df["drug_class"].astype(str)
    df["copay_bucket"] = df["copay_bucket"].astype(str)
    df["is_approved"] = df["outcome"].str.lower().isin(APPROVED_OUTCOMES)
    df["is_rejected"] = ~df["is_approved"]
    df["is_abandoned"] = df["outcome"].str.lower().eq("abandoned")
    df["is_pa_required"] = df["rejection_code"].str.contains(
        "pa required", case=False, na=False
    )
    return df


def plot_rejection_rate_by_payer(df: pd.DataFrame, output_dir: Path) -> None:
    payer_rates = (
        df.groupby("payer")["is_rejected"].mean().sort_values(ascending=False)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x=payer_rates.index, y=payer_rates.values, color="#4C78A8")
    plt.title("Rejection Rate by Payer")
    plt.ylabel("Rejection Rate")
    plt.xlabel("Payer")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "rejection_rate_by_payer.png", dpi=300)
    plt.close()


def plot_top_rejection_codes(df: pd.DataFrame, output_dir: Path) -> None:
    top_codes = df["rejection_code"].value_counts().head(8)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_codes.values, y=top_codes.index, color="#F58518")
    plt.title("Top Rejection Codes")
    plt.xlabel("Claim Count")
    plt.ylabel("Rejection Code")
    plt.tight_layout()
    plt.savefig(output_dir / "top_rejection_codes.png", dpi=300)
    plt.close()


def plot_days_to_resolve_by_outcome(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="outcome",
        y="days_to_resolve",
        color="#54A24B",
    )
    plt.title("Days to Resolve by Outcome")
    plt.xlabel("Outcome")
    plt.ylabel("Days to Resolve")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "days_to_resolve_by_outcome.png", dpi=300)
    plt.close()


def build_insights(df: pd.DataFrame) -> list[str]:
    total_claims = len(df)
    overall_reject_rate = df["is_rejected"].mean()
    top_rejection_code = df["rejection_code"].value_counts().idxmax()
    top_rejection_share = df["rejection_code"].value_counts().max() / total_claims

    payer_rates = df.groupby("payer")["is_rejected"].mean()
    highest_reject_payer = payer_rates.idxmax()
    highest_reject_rate = payer_rates.max()

    pa_median_days = df.loc[df["is_pa_required"], "days_to_resolve"].median()

    copay_abandon = df.groupby("copay_bucket")["is_abandoned"].mean()
    highest_abandon_bucket = copay_abandon.idxmax()
    highest_abandon_rate = copay_abandon.max()

    touches_summary = df.groupby("is_rejected")["touches_proxy"].mean()
    rejected_touches = touches_summary.get(True, float("nan"))
    approved_touches = touches_summary.get(False, float("nan"))

    insights = [
        (
            f"Overall reject rate is {overall_reject_rate:.1%} across "
            f"{total_claims:,} claims."
        ),
        (
            f"{top_rejection_code} is the most common rejection code, "
            f"representing {top_rejection_share:.1%} of all claims."
        ),
        (
            f"{highest_reject_payer} has the highest rejection rate at "
            f"{highest_reject_rate:.1%}."
        ),
        (
            "PA-required claims resolve in a median of "
            f"{pa_median_days:.1f} days."
        ),
        (
            f"{highest_abandon_bucket} copays show the highest abandonment rate "
            f"at {highest_abandon_rate:.1%}; rejected claims average "
            f"{rejected_touches:.1f} touches vs {approved_touches:.1f} for "
            "approved claims."
        ),
    ]
    return insights


def save_insights(insights: list[str], output_dir: Path) -> None:
    insight_path = output_dir / "insights.md"
    lines = ["# EDA Insights", ""]
    lines.extend([f"- {insight}" for insight in insights])
    insight_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA and generate plots + insights."
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
        default=Path("outputs/eda"),
        help="Directory to write charts + insights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_claims(args.data_path)

    plot_rejection_rate_by_payer(df, output_dir)
    plot_top_rejection_codes(df, output_dir)
    plot_days_to_resolve_by_outcome(df, output_dir)

    insights = build_insights(df)
    save_insights(insights, output_dir)

    print("EDA insights:")
    for insight in insights:
        print(f"- {insight}")
    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
