"""
Build the final match-level AthleteGuard dataset.

The final dataset is one row per player + match.

This version joins:
- player information
- static performance statistics
- match-level metrics

using both player_id and match_id whenever match-level data is involved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYERS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "players_clean.csv"
)

STATS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stats_clean.csv"
)

MATCH_STATS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_stats.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metrics.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_dataset.csv"
)


def generate_final_dataset() -> None:
    """Generate one row per player and match."""

    players = pd.read_csv(
        PLAYERS_FILE
    )

    stats = pd.read_csv(
        STATS_FILE
    )

    match_stats = pd.read_csv(
        MATCH_STATS_FILE
    )

    metrics = pd.read_csv(
        METRICS_FILE
    )

    # --------------------------------------------------------
    # Validate required identifiers
    # --------------------------------------------------------

    for name, frame, required in [
        (
            "players",
            players,
            {"player_id"},
        ),
        (
            "stats",
            stats,
            {"player_id"},
        ),
        (
            "match_stats",
            match_stats,
            {
                "player_id",
                "match_id",
            },
        ),
        (
            "metrics",
            metrics,
            {
                "player_id",
                "match_id",
            },
        ),
    ]:
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: "
                f"{sorted(missing)}"
            )

    # --------------------------------------------------------
    # Keep one player-level record from stats.
    # --------------------------------------------------------

    stats_player = (
        stats
        .drop_duplicates("player_id")
        .copy()
    )

    # Avoid accidental duplicate column names after merging.
    overlapping_stats = (
        set(stats_player.columns)
        & set(players.columns)
    ) - {"player_id"}

    stats_player = stats_player.drop(
        columns=sorted(overlapping_stats),
        errors="ignore",
    )

    # --------------------------------------------------------
    # Base: player + match rows
    # --------------------------------------------------------

    df = match_stats.copy()

    # Add player information.
    df = df.merge(
        players,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    # Add player-level static stats.
    df = df.merge(
        stats_player,
        on="player_id",
        how="left",
        validate="many_to_one",
        suffixes=("_match", "_player"),
    )

    # Add match-level engineered metrics.
    df = df.merge(
        metrics,
        on=[
            "player_id",
            "match_id",
        ],
        how="left",
        validate="one_to_one",
        suffixes=("", "_metric"),
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if df.empty:
        raise ValueError(
            "Final dataset is empty."
        )

    duplicate_rows = df.duplicated(
        subset=[
            "player_id",
            "match_id",
        ]
    )

    if duplicate_rows.any():
        raise ValueError(
            "Final dataset contains duplicate "
            "player_id + match_id rows."
        )

    if df["acwr"].isna().any():
        raise ValueError(
            "ACWR contains missing values."
        )

    if df["sleep_deviation"].isna().any():
        raise ValueError(
            "sleep_deviation contains missing values."
        )

    if df["reaction_time_deviation"].isna().any():
        raise ValueError(
            "reaction_time_deviation contains missing values."
        )

    if df["recovery_drop"].isna().any():
        raise ValueError(
            "recovery_drop contains missing values."
        )

    if df["warning_points"].isna().any():
        raise ValueError(
            "warning_points contains missing values."
        )

    # --------------------------------------------------------
    # Stable ordering
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
        .drop(
            columns=["match_number"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "final_dataset.csv generated successfully."
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Players: {df['player_id'].nunique()}"
    )

    print(
        f"Matches: {df['match_id'].nunique()}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print()
    print(
        "Key feature uniqueness:"
    )

    for column in [
        "acwr",
        "sleep_deviation",
        "reaction_time_deviation",
        "recovery_drop",
        "warning_points",
    ]:
        print(
            f"{column}: {df[column].nunique()}"
        )


if __name__ == "__main__":
    generate_final_dataset()