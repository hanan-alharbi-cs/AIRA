"""
Generate deterministic synthetic player-match workload data.

Each row represents one player in one match.

Fields:
- player_id
- match_id
- minutes_played
- rpe

Session load will later be calculated consistently as:

    session_load = minutes_played * rpe

Important:
This is synthetic development data only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYERS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "players_clean.csv"
)

MATCHES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches_clean.csv"
)

INJURY_EVENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "injury_events.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_matches.csv"
)

GLOBAL_SEED = 20260818


def stable_seed(
    player_id: str,
    match_id: str,
) -> int:
    """Create a deterministic seed."""

    key = (
        f"{GLOBAL_SEED}|"
        f"{player_id}|"
        f"{match_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(key).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    ) % (2**32)


def load_injury_matches() -> dict[str, int]:
    """Load the synthetic injury match number for each injured player."""

    if not INJURY_EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing injury events file: {INJURY_EVENTS_FILE}"
        )

    events = pd.read_csv(
        INJURY_EVENTS_FILE
    )

    required = {
        "player_id",
        "injury_match_number",
        "has_injury_event",
    }

    missing = required - set(events.columns)

    if missing:
        raise ValueError(
            "injury_events.csv is missing columns: "
            f"{sorted(missing)}"
        )

    injury_map: dict[str, int] = {}

    for row in events.itertuples(index=False):
        if int(row.has_injury_event) != 1:
            continue

        if pd.isna(row.injury_match_number):
            raise ValueError(
                f"Missing injury match for {row.player_id}"
            )

        injury_map[
            str(row.player_id)
        ] = int(row.injury_match_number)

    return injury_map


def pre_injury_factor(
    match_number: int,
    injury_match_number: int | None,
) -> float:
    """
    Return workload stress factor before a synthetic injury.

    0.0 -> normal
    0.5 -> 3 matches before injury
    1.0 -> 2 matches before injury
    1.5 -> 1 match before injury
    2.0 -> injury match
    """

    if injury_match_number is None:
        return 0.0

    distance = (
        injury_match_number
        - match_number
    )

    if distance == 3:
        return 0.5

    if distance == 2:
        return 1.0

    if distance == 1:
        return 1.5

    if distance == 0:
        return 2.0

    return 0.0


def generate_player_matches() -> None:
    """Generate one deterministic workload row per player/match."""

    if not PLAYERS_FILE.exists():
        raise FileNotFoundError(
            f"Missing players file: {PLAYERS_FILE}"
        )

    if not MATCHES_FILE.exists():
        raise FileNotFoundError(
            f"Missing matches file: {MATCHES_FILE}"
        )

    players = pd.read_csv(
        PLAYERS_FILE
    )

    matches = pd.read_csv(
        MATCHES_FILE
    )

    injury_map = load_injury_matches()

    if "player_id" not in players.columns:
        raise ValueError(
            "players_clean.csv must contain player_id."
        )

    if "match_id" not in matches.columns:
        raise ValueError(
            "matches_clean.csv must contain match_id."
        )

    rows: list[dict[str, object]] = []

    for player_id in players[
        "player_id"
    ].astype(str):

        injury_match_number = injury_map.get(
            player_id
        )

        for match_id in matches[
            "match_id"
        ].astype(str):

            digits = "".join(
                character
                for character in match_id
                if character.isdigit()
            )

            if not digits:
                raise ValueError(
                    f"Unable to determine match number from {match_id}"
                )

            match_number = int(digits)

            rng = np.random.default_rng(
                stable_seed(
                    player_id,
                    match_id,
                )
            )

            stress_factor = pre_injury_factor(
                match_number,
                injury_match_number,
            )

            # ------------------------------------------------
            # Playing time
            # ------------------------------------------------

            base_minutes = rng.normal(
                loc=68.0,
                scale=9.0,
            )

            minutes = (
                base_minutes
                + (
                    4.0
                    * stress_factor
                )
                + rng.normal(
                    0.0,
                    1.5,
                )
            )

            minutes = int(
                np.clip(
                    round(minutes),
                    40,
                    95,
                )
            )

            # ------------------------------------------------
            # RPE
            # ------------------------------------------------

            base_rpe = rng.normal(
                loc=6.0,
                scale=0.55,
            )

            rpe = (
                base_rpe
                + (
                    0.75
                    * stress_factor
                )
                + rng.normal(
                    0.0,
                    0.15,
                )
            )

            rpe = float(
                np.clip(
                    rpe,
                    3.0,
                    9.5,
                )
            )

            rows.append(
                {
                    "player_id": player_id,
                    "match_id": match_id,
                    "minutes_played": minutes,
                    "rpe": round(
                        rpe,
                        2,
                    ),
                    "synthetic_workload_stress": round(
                        stress_factor,
                        2,
                    ),
                }
            )

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if df.duplicated(
        [
            "player_id",
            "match_id",
        ]
    ).any():
        raise ValueError(
            "Duplicate player/match rows were generated."
        )

    expected_rows = (
        players["player_id"].nunique()
        * matches["match_id"].nunique()
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"generated {len(df)}."
        )

    if df["rpe"].isna().any():
        raise ValueError(
            "RPE contains missing values."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "player_matches.csv generated successfully."
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
        f"Mean minutes: "
        f"{df['minutes_played'].mean():.2f}"
    )

    print(
        f"Mean RPE: "
        f"{df['rpe'].mean():.2f}"
    )

    print(
        f"Min RPE: "
        f"{df['rpe'].min():.2f}"
    )

    print(
        f"Max RPE: "
        f"{df['rpe'].max():.2f}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    generate_player_matches()