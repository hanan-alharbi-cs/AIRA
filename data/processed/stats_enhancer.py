"""
Generate deterministic synthetic match-level monitoring data.

AthleteGuard workload definition:

    session_load = minutes_played * rpe

Generated monitoring signals:
- minutes_played
- rpe
- session_load
- sleep_duration
- sleep_quality
- reaction_time_ms
- recovery_score
- stress_level

Important:
This is synthetic development data for an MVP/prototype.
It is not clinical data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_STATS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stats_clean.csv"
)

PLAYER_MATCHES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_matches.csv"
)

INJURY_EVENTS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "injury_events.csv"
)

OUTPUT_STATS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_stats.csv"
)

GLOBAL_SEED = 20260818


# ============================================================
# Reproducible random seed
# ============================================================

def stable_seed(
    player_id: str,
    match_id: str,
) -> int:
    """Create a deterministic seed for a player/match pair."""

    key = (
        f"{GLOBAL_SEED}|"
        f"{player_id}|"
        f"{match_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        key
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )
        % (2**32)
    )


# ============================================================
# Bounded normal helper
# ============================================================

def clipped_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    minimum: float,
    maximum: float,
) -> float:
    """Sample a bounded normal value."""

    value = rng.normal(
        loc=mean,
        scale=std,
    )

    return float(
        np.clip(
            value,
            minimum,
            maximum,
        )
    )


# ============================================================
# Injury stress
# ============================================================

def injury_stress_level(
    match_number: int,
    injury_match_number: int | None,
) -> float:
    """
    Synthetic stress intensity relative to an injury event.

    0.00 = normal
    0.35 = mild pre-injury stress
    0.60 = moderate pre-injury stress
    0.85 = severe pre-injury stress
    1.00 = injury-match stress
    """

    if injury_match_number is None:
        return 0.0

    distance = (
        injury_match_number
        - match_number
    )

    if distance == 3:
        return 0.35

    if distance == 2:
        return 0.60

    if distance == 1:
        return 0.85

    if distance == 0:
        return 1.00

    return 0.0


# ============================================================
# Injury events
# ============================================================

def load_injury_events() -> dict[str, int]:
    """Load synthetic injury match numbers."""

    if not INJURY_EVENTS.exists():
        raise FileNotFoundError(
            f"Injury events file not found: {INJURY_EVENTS}"
        )

    events = pd.read_csv(
        INJURY_EVENTS
    )

    required = {
        "player_id",
        "injury_match_number",
        "has_injury_event",
    }

    missing = (
        required
        - set(events.columns)
    )

    if missing:
        raise ValueError(
            "injury_events.csv is missing columns: "
            f"{sorted(missing)}"
        )

    injury_map: dict[str, int] = {}

    for row in events.itertuples(
        index=False
    ):

        if int(
            row.has_injury_event
        ) != 1:
            continue

        if pd.isna(
            row.injury_match_number
        ):
            raise ValueError(
                f"Missing injury match for "
                f"{row.player_id}"
            )

        injury_map[
            str(row.player_id)
        ] = int(
            row.injury_match_number
        )

    return injury_map


# ============================================================
# Main generator
# ============================================================

def generate_match_stats() -> None:
    """Generate deterministic match-level monitoring data."""

    if not INPUT_STATS.exists():
        raise FileNotFoundError(
            f"Input stats file not found: {INPUT_STATS}"
        )

    if not PLAYER_MATCHES.exists():
        raise FileNotFoundError(
            f"Player matches file not found: {PLAYER_MATCHES}"
        )

    base_stats = pd.read_csv(
        INPUT_STATS
    )

    player_matches = pd.read_csv(
        PLAYER_MATCHES
    )

    injury_map = load_injury_events()

    # --------------------------------------------------------
    # Required workload fields
    # --------------------------------------------------------

    required_match_columns = {
        "player_id",
        "match_id",
        "minutes_played",
        "rpe",
    }

    missing = (
        required_match_columns
        - set(player_matches.columns)
    )

    if missing:
        raise ValueError(
            "player_matches.csv is missing columns: "
            f"{sorted(missing)}"
        )

    if player_matches["rpe"].isna().any():
        raise ValueError(
            "player_matches.csv contains missing RPE values."
        )

    if not player_matches["rpe"].between(
        0,
        10,
    ).all():
        raise ValueError(
            "RPE values must be between 0 and 10."
        )

    # --------------------------------------------------------
    # Player baseline values
    # --------------------------------------------------------

    baseline_columns = [
        "player_id"
    ]

    for column in [
        "reaction_time_ms",
        "sleep_duration",
        "recovery_score",
    ]:
        if column in base_stats.columns:
            baseline_columns.append(
                column
            )

    baseline = (
        base_stats[
            baseline_columns
        ]
        .drop_duplicates(
            "player_id"
        )
        .copy()
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = player_matches.merge(
        baseline,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Match number
    # --------------------------------------------------------

    merged["match_number"] = (
        merged["match_id"]
        .astype(str)
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .astype(int)
    )

    merged = (
        merged
        .sort_values(
            [
                "player_id",
                "match_number",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Generate observations
    # --------------------------------------------------------

    output_rows: list[
        dict[str, object]
    ] = []

    for row in merged.itertuples(
        index=False
    ):

        player_id = str(
            row.player_id
        )

        match_id = str(
            row.match_id
        )

        match_number = int(
            row.match_number
        )

        rng = np.random.default_rng(
            stable_seed(
                player_id,
                match_id,
            )
        )

        minutes = float(
            row.minutes_played
        )

        rpe = float(
            row.rpe
        )

        # ----------------------------------------------------
        # Baseline physiological profile
        # ----------------------------------------------------

        base_reaction = (
            float(
                row.reaction_time_ms
            )
            if hasattr(
                row,
                "reaction_time_ms",
            )
            and pd.notna(
                row.reaction_time_ms
            )
            else clipped_normal(
                rng,
                mean=220.0,
                std=12.0,
                minimum=185.0,
                maximum=250.0,
            )
        )

        base_sleep = (
            float(
                row.sleep_duration
            )
            if hasattr(
                row,
                "sleep_duration",
            )
            and pd.notna(
                row.sleep_duration
            )
            else clipped_normal(
                rng,
                mean=7.8,
                std=0.35,
                minimum=6.8,
                maximum=9.0,
            )
        )

        base_recovery = (
            float(
                row.recovery_score
            )
            if hasattr(
                row,
                "recovery_score",
            )
            and pd.notna(
                row.recovery_score
            )
            else clipped_normal(
                rng,
                mean=80.0,
                std=4.0,
                minimum=68.0,
                maximum=92.0,
            )
        )

        # ----------------------------------------------------
        # Injury-linked stress
        # ----------------------------------------------------

        stress = injury_stress_level(
            match_number=match_number,
            injury_match_number=injury_map.get(
                player_id
            ),
        )

        # ----------------------------------------------------
        # Unified session load
        # ----------------------------------------------------

        session_load = (
            minutes * rpe
        )

        # ----------------------------------------------------
        # Workload pressure
        # ----------------------------------------------------

        workload_pressure = np.clip(
            (
                0.35 * max(
                    (minutes - 60.0) / 35.0,
                    0.0,
                )
                + 0.25 * max(
                    (rpe - 6.0) / 3.5,
                    0.0,
                )
                + 0.40 * stress
                + rng.normal(
                    0.0,
                    0.08,
                )
            ),
            0.0,
            1.25,
        )

        # ----------------------------------------------------
        # Sleep duration
        # ----------------------------------------------------

        sleep_drop = (
            1.55 * stress
            + 0.30 * max(
                workload_pressure - 0.35,
                0.0,
            )
        )

        sleep_duration = (
            base_sleep
            - sleep_drop
            + rng.normal(
                0.0,
                0.15,
            )
        )

        sleep_duration = float(
            np.clip(
                sleep_duration,
                5.0,
                9.5,
            )
        )

        # ----------------------------------------------------
        # NEW: Sleep quality
        # ----------------------------------------------------

        sleep_deficit = max(
            0.0,
            8.0 - sleep_duration,
        )

        sleep_quality = (
            88.0
            - (
                12.0
                * sleep_deficit
            )
            - (
                20.0
                * stress
            )
            - (
                5.0
                * max(
                    workload_pressure - 0.50,
                    0.0,
                )
            )
            + rng.normal(
                0.0,
                2.5,
            )
        )

        sleep_quality = float(
            np.clip(
                sleep_quality,
                35.0,
                100.0,
            )
        )

        # ----------------------------------------------------
        # Reaction time
        # ----------------------------------------------------

        reaction_increase = (
            32.0 * stress
            + 10.0 * max(
                workload_pressure - 0.30,
                0.0,
            )
        )

        reaction_time = (
            base_reaction
            + reaction_increase
            + rng.normal(
                0.0,
                3.5,
            )
        )

        reaction_time = float(
            np.clip(
                reaction_time,
                175.0,
                330.0,
            )
        )

        # ----------------------------------------------------
        # Recovery
        # ----------------------------------------------------

        recovery_drop = (
            23.0 * stress
            + 7.0 * max(
                workload_pressure - 0.35,
                0.0,
            )
        )

        recovery = (
            base_recovery
            - recovery_drop
            + rng.normal(
                0.0,
                1.8,
            )
        )

        recovery = float(
            np.clip(
                recovery,
                35.0,
                95.0,
            )
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_rows.append(
            {
                "player_id": player_id,
                "match_id": match_id,
                "minutes_played": int(
                    minutes
                ),
                "rpe": round(
                    rpe,
                    2,
                ),
                "session_load": round(
                    session_load,
                    2,
                ),
                "stress_level": round(
                    stress,
                    3,
                ),
                "reaction_time_ms": round(
                    reaction_time,
                    2,
                ),
                "sleep_duration": round(
                    sleep_duration,
                    2,
                ),
                "sleep_quality": round(
                    sleep_quality,
                    2,
                ),
                "recovery_score": round(
                    recovery,
                    2,
                ),
            }
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    output = pd.DataFrame(
        output_rows
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if output.empty:
        raise ValueError(
            "No match-level statistics were generated."
        )

    if output.duplicated(
        [
            "player_id",
            "match_id",
        ]
    ).any():
        raise ValueError(
            "Duplicate player/match rows were generated."
        )

    expected_rows = (
        player_matches[
            "player_id"
        ].nunique()
        * player_matches[
            "match_id"
        ].nunique()
    )

    if len(output) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"generated {len(output)}."
        )

    # Validate unified workload definition.
    expected_load = (
        output["minutes_played"]
        * output["rpe"]
    )

    if not (
        (
            output["session_load"]
            - expected_load
        ).abs()
        < 1e-6
    ).all():
        raise ValueError(
            "session_load is inconsistent with "
            "minutes_played × rpe."
        )

    # Validate sleep quality.
    if not output[
        "sleep_quality"
    ].between(
        0,
        100,
    ).all():
        raise ValueError(
            "sleep_quality must remain between 0 and 100."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_STATS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_STATS,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "Synthetic match-level monitoring data generated."
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

    print(
        f"Injury-linked players: "
        f"{len(injury_map)}"
    )

    print(
        f"Mean minutes: "
        f"{output['minutes_played'].mean():.2f}"
    )

    print(
        f"Mean RPE: "
        f"{output['rpe'].mean():.2f}"
    )

    print(
        f"Mean session load: "
        f"{output['session_load'].mean():.2f}"
    )

    print(
        f"Mean sleep quality: "
        f"{output['sleep_quality'].mean():.2f}"
    )

    print(
        f"Min sleep quality: "
        f"{output['sleep_quality'].min():.2f}"
    )

    print(
        f"Max sleep quality: "
        f"{output['sleep_quality'].max():.2f}"
    )

    print(
        f"Output: {OUTPUT_STATS}"
    )

    print()
    print(
        "Stress distribution:"
    )

    print(
        output[
            "stress_level"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    generate_match_stats()