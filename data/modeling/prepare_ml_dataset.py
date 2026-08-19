"""
Prepare the AthleteGuard injury-classification dataset.

Current modeling target:
    injury_target = 1 if the player has a recorded injury
    injury_target = 0 if the player has "No Injury"

Important:
The current injuries.csv has no injury date or match_id.
Therefore this dataset supports a player-level classification baseline,
not a verified future injury prediction task.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_dataset.csv"
)

INJURIES_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "injuries.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
)

ROW_LEVEL_OUTPUT = (
    OUTPUT_DIR
    / "injury_training_rows.csv"
)

PLAYER_LEVEL_OUTPUT = (
    OUTPUT_DIR
    / "injury_player_dataset.csv"
)


MODEL_FEATURES = [
    "session_load",
    "acute_load",
    "chronic_load",
    "acwr",
    "sleep_deviation",
    "reaction_time_deviation",
    "recovery_drop",
    "warning_points",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the source datasets."""

    if not FINAL_DATASET.exists():
        raise FileNotFoundError(
            f"Final dataset not found: {FINAL_DATASET}"
        )

    if not INJURIES_DATASET.exists():
        raise FileNotFoundError(
            f"Injury dataset not found: {INJURIES_DATASET}"
        )

    features_df = pd.read_csv(FINAL_DATASET)
    injuries_df = pd.read_csv(INJURIES_DATASET)

    required_feature_columns = {
        "player_id",
        *MODEL_FEATURES,
    }

    missing_features = (
        required_feature_columns
        - set(features_df.columns)
    )

    if missing_features:
        raise ValueError(
            "Missing required feature columns: "
            f"{sorted(missing_features)}"
        )

    required_injury_columns = {
        "player_id",
        "injury",
        "recovery_days",
    }

    missing_injury_columns = (
        required_injury_columns
        - set(injuries_df.columns)
    )

    if missing_injury_columns:
        raise ValueError(
            "Missing required injury columns: "
            f"{sorted(missing_injury_columns)}"
        )

    return features_df, injuries_df


def build_player_target(
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one injury target per player."""

    target_df = injuries_df[
        [
            "player_id",
            "injury",
            "recovery_days",
        ]
    ].copy()

    target_df["injury_target"] = (
        target_df["injury"]
        .astype(str)
        .str.strip()
        .str.lower()
        .ne("no injury")
        .astype(int)
    )

    duplicate_players = (
        target_df["player_id"]
        .duplicated(keep=False)
    )

    if duplicate_players.any():
        duplicates = sorted(
            target_df.loc[
                duplicate_players,
                "player_id",
            ].astype(str).unique()
        )

        raise ValueError(
            "Multiple injury records found for the same player: "
            f"{duplicates}"
        )

    return target_df


def build_row_level_dataset(
    features_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the player-level injury target to every row."""

    selected_columns = [
        "player_id",
        *MODEL_FEATURES,
    ]

    rows = features_df[selected_columns].copy()

    merged = rows.merge(
        target_df,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    if merged["injury_target"].isna().any():
        missing_players = sorted(
            merged.loc[
                merged["injury_target"].isna(),
                "player_id",
            ].astype(str).unique()
        )

        raise ValueError(
            "Some players in final_dataset.csv have no injury label: "
            f"{missing_players}"
        )

    for column in MODEL_FEATURES:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="raise",
        )

    merged["injury_target"] = merged[
        "injury_target"
    ].astype(int)

    return merged


def build_player_level_dataset(
    row_level_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate match-level observations to one row per player.

    The current target exists at player level, so the player-level
    dataset is the safer primary dataset for baseline modeling.
    """

    aggregation = {
        "session_load": ["mean", "max"],
        "acute_load": ["mean", "max"],
        "chronic_load": ["mean"],
        "acwr": ["mean", "max"],
        "sleep_deviation": ["mean", "min"],
        "reaction_time_deviation": ["mean", "max"],
        "recovery_drop": ["mean", "min"],
        "warning_points": ["mean", "max"],
    }

    aggregated = (
        row_level_df
        .groupby("player_id", as_index=False)
        .agg(aggregation)
    )

    flattened_columns = ["player_id"]

    for feature, stat in aggregated.columns.tolist()[1:]:
        flattened_columns.append(
            f"{feature}_{stat}"
        )

    aggregated.columns = flattened_columns

    target_columns = (
        row_level_df[
            [
                "player_id",
                "injury",
                "recovery_days",
                "injury_target",
            ]
        ]
        .drop_duplicates("player_id")
    )

    player_level = aggregated.merge(
        target_columns,
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    return player_level


def validate_target(
    player_level_df: pd.DataFrame,
) -> None:
    """Validate that both target classes exist."""

    counts = (
        player_level_df["injury_target"]
        .value_counts()
        .sort_index()
    )

    print("Target distribution:")
    print(counts.to_string())

    if len(counts) != 2:
        raise ValueError(
            "The target contains fewer than two classes. "
            "A binary classifier cannot be trained."
        )

    if (counts < 2).any():
        raise ValueError(
            "One target class has fewer than two players. "
            "The dataset is too small for a meaningful baseline."
        )


def main() -> None:
    """Prepare and save row-level and player-level datasets."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_df, injuries_df = load_inputs()

    target_df = build_player_target(
        injuries_df
    )

    row_level_df = build_row_level_dataset(
        features_df,
        target_df,
    )

    player_level_df = build_player_level_dataset(
        row_level_df
    )

    validate_target(
        player_level_df
    )

    row_level_df.to_csv(
        ROW_LEVEL_OUTPUT,
        index=False,
    )

    player_level_df.to_csv(
        PLAYER_LEVEL_OUTPUT,
        index=False,
    )

    print()
    print("Preparation completed successfully.")
    print(f"Row-level dataset: {ROW_LEVEL_OUTPUT}")
    print(f"Row-level shape: {row_level_df.shape}")
    print()
    print(f"Player-level dataset: {PLAYER_LEVEL_OUTPUT}")
    print(f"Player-level shape: {player_level_df.shape}")
    print()
    print(
        "Important: the current target is player-level injury "
        "association, not verified future injury prediction."
    )


if __name__ == "__main__":
    main()