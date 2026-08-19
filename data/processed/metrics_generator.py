"""
Generate match-level AthleteGuard risk features.

Unified workload definition:

    session_load = minutes_played * rpe

The value is generated upstream in match_stats.csv and reused here.

Historical baselines and ACWR use previous matches only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCH_STATS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_stats.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metrics.csv"
)


REQUIRED_COLUMNS = {
    "player_id",
    "match_id",
    "minutes_played",
    "rpe",
    "session_load",
    "reaction_time_ms",
    "sleep_duration",
    "recovery_score",
}


def generate_metrics() -> None:
    """Generate match-level workload and recovery metrics."""

    if not MATCH_STATS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {MATCH_STATS_FILE}"
        )

    df = pd.read_csv(
        MATCH_STATS_FILE
    )

    missing_columns = (
        REQUIRED_COLUMNS - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "match_stats.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Validate one row per player + match
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[
            "player_id",
            "match_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            df.loc[
                duplicate_mask,
                [
                    "player_id",
                    "match_id",
                ],
            ]
            .drop_duplicates()
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate player/match rows found: "
            f"{duplicates[:10]}"
        )

    # --------------------------------------------------------
    # Validate workload definition
    # --------------------------------------------------------

    expected_session_load = (
        df["minutes_played"]
        * df["rpe"]
    )

    workload_difference = (
        df["session_load"]
        - expected_session_load
    ).abs()

    if (
        workload_difference
        > 1e-6
    ).any():
        raise ValueError(
            "session_load is inconsistent with "
            "minutes_played × rpe."
        )

    # --------------------------------------------------------
    # Deterministic ordering
    # --------------------------------------------------------

    df["match_number"] = (
        df["match_id"]
        .astype(str)
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .astype(int)
    )

    df = (
        df.sort_values(
            [
                "player_id",
                "match_number",
            ]
        )
        .reset_index(drop=True)
    )

    grouped = df.groupby(
        "player_id",
        group_keys=False,
    )

    # --------------------------------------------------------
    # Historical baselines
    #
    # Previous observations only.
    # --------------------------------------------------------

    df["baseline_training_load"] = (
        grouped["session_load"]
        .transform(
            lambda s: (
                s.shift(1)
                .expanding(
                    min_periods=1
                )
                .mean()
            )
        )
    )

    df["baseline_reaction_time"] = (
        grouped["reaction_time_ms"]
        .transform(
            lambda s: (
                s.shift(1)
                .expanding(
                    min_periods=1
                )
                .mean()
            )
        )
    )

    df["baseline_sleep"] = (
        grouped["sleep_duration"]
        .transform(
            lambda s: (
                s.shift(1)
                .expanding(
                    min_periods=1
                )
                .mean()
            )
        )
    )

    df["baseline_recovery"] = (
        grouped["recovery_score"]
        .transform(
            lambda s: (
                s.shift(1)
                .expanding(
                    min_periods=1
                )
                .mean()
            )
        )
    )

    # --------------------------------------------------------
    # First match initialization
    # --------------------------------------------------------

    first_match = (
        grouped.cumcount() == 0
    )

    df.loc[
        first_match,
        "baseline_training_load",
    ] = df.loc[
        first_match,
        "session_load",
    ]

    df.loc[
        first_match,
        "baseline_reaction_time",
    ] = df.loc[
        first_match,
        "reaction_time_ms",
    ]

    df.loc[
        first_match,
        "baseline_sleep",
    ] = df.loc[
        first_match,
        "sleep_duration",
    ]

    df.loc[
        first_match,
        "baseline_recovery",
    ] = df.loc[
        first_match,
        "recovery_score",
    ]

    # --------------------------------------------------------
    # Historical acute/chronic workload
    #
    # Current match is excluded.
    # --------------------------------------------------------

    df["acute_load"] = (
        grouped["session_load"]
        .transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    window=3,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    df["chronic_load"] = (
        grouped["session_load"]
        .transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    window=10,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    # First match has no previous history.
    df.loc[
        first_match,
        "acute_load",
    ] = df.loc[
        first_match,
        "session_load",
    ]

    df.loc[
        first_match,
        "chronic_load",
    ] = df.loc[
        first_match,
        "session_load",
    ]

    # --------------------------------------------------------
    # ACWR
    # --------------------------------------------------------

    df["acwr"] = (
        df["acute_load"]
        / df["chronic_load"].replace(
            0,
            pd.NA,
        )
    )

    df["acwr"] = (
        df["acwr"]
        .fillna(1.0)
        .astype(float)
    )

    # --------------------------------------------------------
    # Deviations
    # --------------------------------------------------------

    df["sleep_deviation"] = (
        df["sleep_duration"]
        - df["baseline_sleep"]
    )

    df["reaction_time_deviation"] = (
        df["reaction_time_ms"]
        - df["baseline_reaction_time"]
    )

    df["recovery_drop"] = (
        df["baseline_recovery"]
        - df["recovery_score"]
    )

    # First match = neutral deviations.
    df.loc[
        first_match,
        [
            "sleep_deviation",
            "reaction_time_deviation",
            "recovery_drop",
        ],
    ] = 0.0

    # --------------------------------------------------------
    # Warning points
    # --------------------------------------------------------

    df["warning_points"] = (
        (
            df["acwr"] > 1.5
        ).astype(int)
        + (
            df["reaction_time_deviation"]
            > 20.0
        ).astype(int)
        + (
            df["sleep_deviation"]
            < -1.0
        ).astype(int)
        + (
            df["recovery_drop"]
            > 10.0
        ).astype(int)
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_columns = [
        "player_id",
        "match_id",
        "minutes_played",
        "rpe",
        "session_load",
        "acute_load",
        "chronic_load",
        "acwr",
        "sleep_deviation",
        "reaction_time_deviation",
        "recovery_drop",
        "warning_points",
        "baseline_training_load",
        "baseline_reaction_time",
        "baseline_sleep",
        "baseline_recovery",
        "reaction_time_ms",
        "sleep_duration",
        "recovery_score",
    ]

    output = df[
        output_columns
    ].copy()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "metrics.csv generated successfully."
    )

    print(
        f"Rows: {len(output)}"
    )

    print(
        f"Players: "
        f"{output['player_id'].nunique()}"
    )

    print(
        f"Matches: "
        f"{output['match_id'].nunique()}"
    )

    print()
    print(
        "Mean workload:"
    )

    print(
        f"session_load: "
        f"{output['session_load'].mean():.2f}"
    )

    print()
    print(
        "Unique feature values:"
    )

    for column in [
        "session_load",
        "acute_load",
        "chronic_load",
        "acwr",
        "sleep_deviation",
        "reaction_time_deviation",
        "recovery_drop",
        "warning_points",
    ]:
        print(
            f"{column}: "
            f"{output[column].nunique()}"
        )


if __name__ == "__main__":
    generate_metrics()