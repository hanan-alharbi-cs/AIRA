"""
Analyze engineered features by injury target.

This analysis is diagnostic only.
It does not train a model and does not modify project data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "injury_player_dataset.csv"
)

FEATURES = [
    "session_load_mean",
    "session_load_max",
    "acute_load_mean",
    "acute_load_max",
    "chronic_load_mean",
    "acwr_mean",
    "acwr_max",
    "sleep_deviation_mean",
    "sleep_deviation_min",
    "reaction_time_deviation_mean",
    "reaction_time_deviation_max",
    "recovery_drop_mean",
    "recovery_drop_min",
    "warning_points_mean",
    "warning_points_max",
]


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    df = pd.read_csv(DATASET)

    print("Dataset shape:", df.shape)
    print()

    print("Target distribution:")
    print(
        df["injury_target"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()

    grouped = (
        df.groupby("injury_target")[FEATURES]
        .agg(["mean", "median", "std"])
        .round(3)
    )

    print("Feature statistics by injury target:")
    print(grouped.to_string())
    print()

    comparison_rows = []

    for feature in FEATURES:
        injured = df.loc[
            df["injury_target"] == 1,
            feature,
        ]

        non_injured = df.loc[
            df["injury_target"] == 0,
            feature,
        ]

        injured_mean = injured.mean()
        non_injured_mean = non_injured.mean()

        pooled_std = (
            (injured.std() + non_injured.std()) / 2
        )

        standardized_difference = (
            (injured_mean - non_injured_mean)
            / pooled_std
            if pooled_std > 0
            else 0.0
        )

        comparison_rows.append(
            {
                "feature": feature,
                "injured_mean": injured_mean,
                "non_injured_mean": non_injured_mean,
                "mean_difference": (
                    injured_mean - non_injured_mean
                ),
                "standardized_difference": (
                    standardized_difference
                ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        "standardized_difference",
        key=lambda series: series.abs(),
        ascending=False,
    )

    print("Feature separation ranking:")
    print(
        comparison.round(3).to_string(index=False)
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "modeling"
        / "feature_analysis.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved analysis: {output_path}"
    )


if __name__ == "__main__":
    main()