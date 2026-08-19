"""
Generate deterministic synthetic injury events.

This creates a match-linked injury timeline for development
and model evaluation.

Important:
These are synthetic research labels, not real medical records.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INJURIES_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "injuries.csv"
)

PLAYER_MATCHES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_matches.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "injury_events.csv"
)


# Match positions are chosen so every synthetic injury has
# multiple previous matches available for temporal features.
INJURY_MATCH_NUMBERS = {
    "P002": 22,
    "P004": 18,
    "P006": 24,
    "P008": 20,
    "P009": 16,
    "P010": 25,
    "P012": 21,
    "P018": 19,
    "P020": 23,
}


def main() -> None:
    if not INJURIES_FILE.exists():
        raise FileNotFoundError(
            f"Missing injuries file: {INJURIES_FILE}"
        )

    if not PLAYER_MATCHES_FILE.exists():
        raise FileNotFoundError(
            f"Missing player matches file: {PLAYER_MATCHES_FILE}"
        )

    injuries = pd.read_csv(
        INJURIES_FILE
    )

    player_matches = pd.read_csv(
        PLAYER_MATCHES_FILE
    )

    required_injury_columns = {
        "player_id",
        "injury",
        "recovery_days",
    }

    required_match_columns = {
        "player_id",
        "match_id",
    }

    if not required_injury_columns.issubset(
        injuries.columns
    ):
        raise ValueError(
            "injuries.csv is missing required columns."
        )

    if not required_match_columns.issubset(
        player_matches.columns
    ):
        raise ValueError(
            "player_matches.csv is missing required columns."
        )

    match_number = (
        player_matches[
            ["player_id", "match_id"]
        ]
        .drop_duplicates()
        .assign(
            match_number=lambda df: (
                df["match_id"]
                .astype(str)
                .str.extract(
                    r"(\d+)",
                    expand=False,
                )
                .astype(int)
            )
        )
    )

    rows: list[dict[str, object]] = []

    for record in injuries.itertuples(
        index=False
    ):
        player_id = str(record.player_id)
        injury = str(record.injury)
        recovery_days = int(record.recovery_days)

        if injury.strip().lower() == "no injury":
            rows.append(
                {
                    "player_id": player_id,
                    "injury": "No Injury",
                    "injury_match_id": None,
                    "injury_match_number": None,
                    "recovery_days": 0,
                    "has_injury_event": 0,
                }
            )
            continue

        if player_id not in INJURY_MATCH_NUMBERS:
            raise ValueError(
                f"No synthetic injury match assigned to {player_id}."
            )

        target_match_number = (
            INJURY_MATCH_NUMBERS[player_id]
        )

        player_rows = match_number[
            match_number["player_id"] == player_id
        ]

        matching_match = player_rows[
            player_rows["match_number"]
            == target_match_number
        ]

        if matching_match.empty:
            raise ValueError(
                f"Match M{target_match_number:03d} "
                f"not found for {player_id}."
            )

        injury_match_id = str(
            matching_match.iloc[0]["match_id"]
        )

        rows.append(
            {
                "player_id": player_id,
                "injury": injury,
                "injury_match_id": injury_match_id,
                "injury_match_number": target_match_number,
                "recovery_days": recovery_days,
                "has_injury_event": 1,
            }
        )

    output = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "Synthetic injury events generated successfully."
    )

    print(
        f"Rows: {len(output)}"
    )

    print(
        f"Injury events: "
        f"{int(output['has_injury_event'].sum())}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()