"""
Prepare a temporal injury-prediction dataset for AthleteGuard.

Target definition
-----------------
For an injured player:
    target = 1 for observations in the pre-injury prediction window
    target = 0 for earlier observations

For a non-injured player:
    target = 0 for all observations.

Important:
- The actual injury-match row is excluded from prediction rows.
- Features from the current row are used to predict the upcoming injury.
- Injury metadata such as injury type and recovery_days are never features.
- This is synthetic development data, not clinical validation.
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

INJURY_EVENTS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "injury_events.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "temporal_injury_dataset.csv"
)


FEATURES = [
    "session_load",
    "acute_load",
    "chronic_load",
    "acwr",
    "sleep_deviation",
    "reaction_time_deviation",
    "recovery_drop",
    "warning_points",
]


# Number of matches before the injury event that count as
# positive prediction examples.
PREDICTION_WINDOW_MATCHES = 3


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate final features and injury events."""

    if not FINAL_DATASET.exists():
        raise FileNotFoundError(
            f"Missing final dataset: {FINAL_DATASET}"
        )

    if not INJURY_EVENTS.exists():
        raise FileNotFoundError(
            f"Missing injury events: {INJURY_EVENTS}"
        )

    df = pd.read_csv(FINAL_DATASET)
    events = pd.read_csv(INJURY_EVENTS)

    required_features = {
        "player_id",
        "match_id",
        *FEATURES,
    }

    missing_features = (
        required_features - set(df.columns)
    )

    if missing_features:
        raise ValueError(
            "final_dataset.csv is missing columns: "
            f"{sorted(missing_features)}"
        )

    required_events = {
        "player_id",
        "injury_match_number",
        "has_injury_event",
    }

    missing_events = (
        required_events - set(events.columns)
    )

    if missing_events:
        raise ValueError(
            "injury_events.csv is missing columns: "
            f"{sorted(missing_events)}"
        )

    return df, events


def prepare_dataset(
    df: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Create one temporal prediction row per player/match."""

    work = df.copy()

    work["match_number"] = (
        work["match_id"]
        .astype(str)
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .astype(int)
    )

    work = (
        work.sort_values(
            [
                "player_id",
                "match_number",
            ]
        )
        .reset_index(drop=True)
    )

    event_map = {
        str(row.player_id): (
            int(row.injury_match_number)
            if pd.notna(row.injury_match_number)
            and int(row.has_injury_event) == 1
            else None
        )
        for row in events.itertuples(index=False)
    }

    target_values: list[int] = []
    event_distance_values: list[int | None] = []

    for row in work.itertuples(index=False):
        player_id = str(row.player_id)
        match_number = int(row.match_number)

        injury_match = event_map.get(player_id)

        # ----------------------------------------------------
        # No injury event
        # ----------------------------------------------------

        if injury_match is None:
            target_values.append(0)
            event_distance_values.append(None)
            continue

        distance = (
            injury_match - match_number
        )

        # Actual injury row is not a prediction observation.
        if distance == 0:
            target_values.append(0)
            event_distance_values.append(0)
            continue

        # Positive prediction window:
        # the 3 matches immediately before injury.
        if 1 <= distance <= PREDICTION_WINDOW_MATCHES:
            target_values.append(1)
        else:
            target_values.append(0)

        event_distance_values.append(distance)

    work["injury_soon_target"] = target_values
    work["matches_until_injury"] = event_distance_values

    # Remove the actual injury-event row from the modeling dataset.
    work = work.loc[
        work["matches_until_injury"] != 0
    ].copy()

    # Only modeling columns + traceability columns.
    output_columns = [
        "player_id",
        "match_id",
        "match_number",
        "matches_until_injury",
        *FEATURES,
        "injury_soon_target",
    ]

    output = work[
        output_columns
    ].copy()

    # Validate feature types.
    for feature in FEATURES:
        output[feature] = pd.to_numeric(
            output[feature],
            errors="raise",
        )

    output["injury_soon_target"] = (
        output["injury_soon_target"]
        .astype(int)
    )

    return output


def validate_dataset(
    dataset: pd.DataFrame,
) -> None:
    """Run structural checks before saving."""

    if dataset.empty:
        raise ValueError(
            "Temporal dataset is empty."
        )

    if dataset[FEATURES].isna().any().any():
        raise ValueError(
            "Missing values found in model features."
        )

    if len(
        dataset["injury_soon_target"].unique()
    ) != 2:
        raise ValueError(
            "Temporal target must contain both 0 and 1 classes."
        )

    # Verify no injury event itself remains.
    if (
        dataset["matches_until_injury"]
        == 0
    ).any():
        raise ValueError(
            "Actual injury-match rows must be excluded."
        )

    # Every positive row must be inside the configured window.
    positive_rows = dataset.loc[
        dataset["injury_soon_target"] == 1
    ]

    if not (
        positive_rows["matches_until_injury"]
        .between(
            1,
            PREDICTION_WINDOW_MATCHES,
        )
    ).all():
        raise ValueError(
            "Positive rows are outside the prediction window."
        )


def main() -> None:
    """Build and save the temporal injury dataset."""

    df, events = load_inputs()

    temporal = prepare_dataset(
        df,
        events,
    )

    validate_dataset(
        temporal
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "Temporal injury dataset generated successfully."
    )

    print(
        f"Rows: {len(temporal)}"
    )

    print(
        f"Players: "
        f"{temporal['player_id'].nunique()}"
    )

    print(
        f"Positive rows: "
        f"{int((temporal['injury_soon_target'] == 1).sum())}"
    )

    print(
        f"Negative rows: "
        f"{int((temporal['injury_soon_target'] == 0).sum())}"
    )

    print()
    print(
        "Target distribution:"
    )

    print(
        temporal["injury_soon_target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Positive rows by matches until injury:"
    )

    print(
        temporal.loc[
            temporal["injury_soon_target"] == 1,
            "matches_until_injury",
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()