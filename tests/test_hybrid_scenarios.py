"""
Hybrid AI scenario test.

Uses real rows from the current synthetic AthleteGuard dataset
to inspect low / moderate / high-risk behavior.

This is a diagnostic test, not clinical validation.
"""

from __future__ import annotations

import pandas as pd

from ai.hybrid.hybrid_risk_engine import (
    calculate_hybrid_risk,
)


FINAL_DATASET = (
    "data/processed/final_dataset.csv"
)

TEMPORAL_PREDICTIONS = (
    "data/modeling/temporal_baseline_predictions.csv"
)


def build_player_data(row: pd.Series) -> dict:
    """Convert one dataset row to the Hybrid Engine contract."""

    return {
        # Readiness inputs
        "player_id": row["player_id"],
        "training_duration_min": float(
            row["minutes_played"]
        ),
        "rpe": float(
            row["rpe"]
        ),
        "sleep_duration": float(
            row["sleep_duration"]
        ),
        "sleep_quality": 80.0,
        "reaction_time_ms": float(
            row["reaction_time_ms"]
        ),
        "recovery_score": float(
            row["recovery_score"]
        ),

        "baseline_sleep": float(
            row["baseline_sleep"]
        ),

        "baseline_reaction_time_ms": float(
            row["baseline_reaction_time"]
        ),

        "baseline_training_load": float(
            row["baseline_training_load"]
        ),

        "acute_load": float(
            row["acute_load"]
        ),

        "chronic_load": float(
            row["chronic_load"]
        ),

        "injury_context": False,

        # ML features
        "session_load": float(
            row["session_load"]
        ),

        "acwr": float(
            row["acwr"]
        ),

        "sleep_deviation": float(
            row["sleep_deviation"]
        ),

        "reaction_time_deviation": float(
            row["reaction_time_deviation"]
        ),

        "recovery_drop": float(
            row["recovery_drop"]
        ),

        "warning_points": float(
            row["warning_points"]
        ),
    }


def main() -> None:
    final_df = pd.read_csv(
        FINAL_DATASET
    )

    predictions_df = pd.read_csv(
        TEMPORAL_PREDICTIONS
    )

    merged = final_df.merge(
        predictions_df[
            [
                "player_id",
                "match_id",
                "predicted_probability",
                "injury_soon_target",
                "matches_until_injury",
            ]
        ],
        on=[
            "player_id",
            "match_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Low-risk candidate
    # --------------------------------------------------------

    low_candidates = (
        merged[
            merged["injury_soon_target"] == 0
        ]
        .sort_values(
            "predicted_probability"
        )
    )

    low_row = low_candidates.iloc[0]

    # --------------------------------------------------------
    # High-risk candidate
    # --------------------------------------------------------

    high_candidates = (
        merged[
            merged["injury_soon_target"] == 1
        ]
        .sort_values(
            "predicted_probability",
            ascending=False,
        )
    )

    high_row = high_candidates.iloc[0]

    # --------------------------------------------------------
    # Moderate-risk candidate
    #
    # Select a positive example around the middle of the
    # model probability distribution.
    # --------------------------------------------------------

    positive_candidates = (
        merged[
            merged["injury_soon_target"] == 1
        ]
        .sort_values(
            "predicted_probability"
        )
        .reset_index(drop=True)
    )

    middle_index = len(
        positive_candidates
    ) // 2

    moderate_row = positive_candidates.iloc[
        middle_index
    ]

    scenarios = [
        (
            "LOW CANDIDATE",
            low_row,
        ),
        (
            "MODERATE CANDIDATE",
            moderate_row,
        ),
        (
            "HIGH CANDIDATE",
            high_row,
        ),
    ]

    print()
    print(
        "=" * 70
    )
    print(
        "ATHLETEGUARD HYBRID SCENARIO TEST"
    )
    print(
        "=" * 70
    )

    for name, row in scenarios:

        player_data = build_player_data(
            row
        )

        result = calculate_hybrid_risk(
            player_data
        )

        print()
        print(
            f"[{name}]"
        )
        print(
            "-" * 70
        )

        print(
            f"Player:                 {row['player_id']}"
        )

        print(
            f"Match:                  {row['match_id']}"
        )

        print(
            f"Matches until injury:   "
            f"{row['matches_until_injury']}"
        )

        print(
            f"Target:                 "
            f"{int(row['injury_soon_target'])}"
        )

        print(
            f"Model probability:      "
            f"{result['ml_injury_probability']:.4f}"
        )

        print(
            f"ML risk:                "
            f"{result['ml_risk_level']}"
        )

        print(
            f"Readiness score:        "
            f"{result['readiness_score']}"
        )

        print(
            f"Readiness risk:         "
            f"{result['readiness_risk_level']}"
        )

        print(
            f"FINAL HYBRID RISK:      "
            f"{result['final_risk_level']}"
        )

        print(
            f"Recommendation:         "
            f"{result['recommendation']}"
        )

        print(
            f"Factors:                "
            f"{result['factors']}"
        )

    print()
    print(
        "=" * 70
    )
    print(
        "Scenario test completed."
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()