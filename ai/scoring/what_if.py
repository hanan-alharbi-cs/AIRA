"""
AthleteGuard Hybrid What-If Engine.

Compares the athlete's current state with a temporary scenario.

The scenario can modify:
- training duration
- RPE
- sleep duration
- sleep quality
- recovery score
- acute/chronic load
- temporal ML features

The original player data is never modified or persisted.

Important:
This is a prototype decision-support tool using synthetic
development data. It is not medical clearance.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ai.hybrid.hybrid_risk_engine import (
    calculate_hybrid_risk,
)
from ai.scoring.readiness_service import (
    calculate_readiness_output,
)


# ============================================================
# Helpers
# ============================================================

def _build_readiness_result(
    player_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the unified Readiness Engine."""

    return calculate_readiness_output(
        player_data
    )


def _build_hybrid_result(
    player_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the real Hybrid AI engine."""

    return calculate_hybrid_risk(
        player_data
    )


def _risk_rank(
    risk_level: str,
) -> int:
    """
    Convert risk level into an ordered rank.

    low       = 0
    moderate  = 1
    elevated  = 2
    """

    mapping = {
        "low": 0,
        "moderate": 1,
        "elevated": 2,
    }

    return mapping.get(
        str(risk_level),
        -1,
    )


def _risk_delta(
    current_risk: str,
    scenario_risk: str,
) -> str:
    """Describe the direction of risk change."""

    current_rank = _risk_rank(
        current_risk
    )

    scenario_rank = _risk_rank(
        scenario_risk
    )

    if scenario_rank < current_rank:
        return "improved"

    if scenario_rank > current_rank:
        return "worsened"

    return "unchanged"


# ============================================================
# Main What-If function
# ============================================================

def calculate_what_if(
    player_data: Mapping[str, Any],
    changes: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Compare current athlete state with a temporary scenario.

    Example changes:

        {
            "training_duration_min": 60,
            "rpe": 5.0,
            "sleep_duration": 8.5,
            "sleep_quality": 90,
            "recovery_score": 85
        }

    The original player_data is never modified.
    """

    # --------------------------------------------------------
    # Copy original data
    # --------------------------------------------------------

    original_data = dict(
        player_data
    )

    scenario_data = deepcopy(
        original_data
    )

    scenario_data.update(
        dict(changes)
    )

    # --------------------------------------------------------
    # Current state
    # --------------------------------------------------------

    current_readiness = (
        _build_readiness_result(
            original_data
        )
    )

    current_hybrid = (
        _build_hybrid_result(
            original_data
        )
    )

    # --------------------------------------------------------
    # Scenario state
    # --------------------------------------------------------

    scenario_readiness = (
        _build_readiness_result(
            scenario_data
        )
    )

    scenario_hybrid = (
        _build_hybrid_result(
            scenario_data
        )
    )

    # --------------------------------------------------------
    # Readiness values
    # --------------------------------------------------------

    current_readiness_score = float(
        current_readiness[
            "readiness_score"
        ]
    )

    scenario_readiness_score = float(
        scenario_readiness[
            "readiness_score"
        ]
    )

    readiness_delta = round(
        scenario_readiness_score
        - current_readiness_score,
        1,
    )

    # --------------------------------------------------------
    # ML probability
    # --------------------------------------------------------

    current_ml_probability = float(
        current_hybrid[
            "ml_injury_probability"
        ]
    )

    scenario_ml_probability = float(
        scenario_hybrid[
            "ml_injury_probability"
        ]
    )

    ml_probability_delta = round(
        scenario_ml_probability
        - current_ml_probability,
        4,
    )

    # --------------------------------------------------------
    # Hybrid risk
    # --------------------------------------------------------

    current_final_risk = str(
        current_hybrid[
            "final_risk_level"
        ]
    )

    scenario_final_risk = str(
        scenario_hybrid[
            "final_risk_level"
        ]
    )

    final_risk_change = _risk_delta(
        current_final_risk,
        scenario_final_risk,
    )

    # --------------------------------------------------------
    # Readiness risk
    # --------------------------------------------------------

    current_readiness_risk = str(
        current_hybrid[
            "readiness_risk_level"
        ]
    )

    scenario_readiness_risk = str(
        scenario_hybrid[
            "readiness_risk_level"
        ]
    )

    readiness_risk_change = _risk_delta(
        current_readiness_risk,
        scenario_readiness_risk,
    )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    return {
        "player_id": scenario_hybrid[
            "player_id"
        ],

        # ----------------------------------------------------
        # Current state
        # ----------------------------------------------------

        "current": {
            "readiness_score": round(
                current_readiness_score,
                1,
            ),
            "readiness_risk": (
                current_readiness_risk
            ),
            "ml_injury_probability": round(
                current_ml_probability,
                4,
            ),
            "ml_risk": str(
                current_hybrid[
                    "ml_risk_level"
                ]
            ),
            "final_risk": (
                current_final_risk
            ),
            "recommendation": str(
                current_hybrid[
                    "recommendation"
                ]
            ),
        },

        # ----------------------------------------------------
        # Scenario state
        # ----------------------------------------------------

        "scenario": {
            "readiness_score": round(
                scenario_readiness_score,
                1,
            ),
            "readiness_risk": (
                scenario_readiness_risk
            ),
            "ml_injury_probability": round(
                scenario_ml_probability,
                4,
            ),
            "ml_risk": str(
                scenario_hybrid[
                    "ml_risk_level"
                ]
            ),
            "final_risk": (
                scenario_final_risk
            ),
            "recommendation": str(
                scenario_hybrid[
                    "recommendation"
                ]
            ),
            "factors": (
                scenario_hybrid[
                    "factors"
                ]
            ),
        },

        # ----------------------------------------------------
        # Changes
        # ----------------------------------------------------

        "changes": dict(
            changes
        ),

        # ----------------------------------------------------
        # Deltas
        # ----------------------------------------------------

        "deltas": {
            "readiness_score": readiness_delta,
            "ml_injury_probability": (
                ml_probability_delta
            ),
            "readiness_risk_change": (
                readiness_risk_change
            ),
            "final_risk_change": (
                final_risk_change
            ),
        },

        # ----------------------------------------------------
        # Simple interpretation
        # ----------------------------------------------------

        "interpretation": (
            "Scenario improves the athlete's overall "
            "risk profile."
            if final_risk_change == "improved"
            else (
                "Scenario worsens the athlete's overall "
                "risk profile."
                if final_risk_change == "worsened"
                else (
                    "Scenario does not change the "
                    "final hybrid risk level."
                )
            )
        ),
    }


# ============================================================
# Manual development test
# ============================================================

if __name__ == "__main__":

    sample_player = {
        "player_id": 12,

        "training_duration_min": 76.0,
        "rpe": 5.58,

        "sleep_duration": 6.8,
        "sleep_quality": 70.0,

        "reaction_time_ms": 210.0,
        "recovery_score": 65.0,

        "baseline_sleep": 8.012,
        "baseline_reaction_time_ms": 197.47,
        "baseline_training_load": 363.25,

        "acute_load": 410.0,
        "chronic_load": 363.25,

        "injury_context": False,

        "session_load": 424.08,
        "acwr": 1.13,
        "sleep_deviation": -1.212,
        "reaction_time_deviation": 12.53,
        "recovery_drop": -13.0,
        "warning_points": 2.0,
    }

    scenario_changes = {
        "training_duration_min": 60.0,
        "rpe": 5.0,
        "sleep_duration": 8.5,
        "sleep_quality": 90.0,
        "recovery_score": 85.0,
        "session_load": 300.0,
        "acute_load": 350.0,
        "acwr": 0.96,
        "sleep_deviation": 0.488,
        "recovery_drop": 0.0,
        "warning_points": 0.0,
    }

    result = calculate_what_if(
        sample_player,
        scenario_changes,
    )

    print(
        "================================================"
    )

    print(
        "ATHLETEGUARD HYBRID WHAT-IF TEST"
    )

    print(
        "================================================"
    )

    print(
        "Current readiness:",
        result[
            "current"
        ][
            "readiness_score"
        ],
    )

    print(
        "Scenario readiness:",
        result[
            "scenario"
        ][
            "readiness_score"
        ],
    )

    print(
        "Readiness delta:",
        result[
            "deltas"
        ][
            "readiness_score"
        ],
    )

    print(
        "Current ML probability:",
        result[
            "current"
        ][
            "ml_injury_probability"
        ],
    )

    print(
        "Scenario ML probability:",
        result[
            "scenario"
        ][
            "ml_injury_probability"
        ],
    )

    print(
        "Current final risk:",
        result[
            "current"
        ][
            "final_risk"
        ],
    )

    print(
        "Scenario final risk:",
        result[
            "scenario"
        ][
            "final_risk"
        ],
    )

    print(
        "Risk change:",
        result[
            "deltas"
        ][
            "final_risk_change"
        ],
    )

    print(
        "Interpretation:",
        result[
            "interpretation"
        ],
    )

    print(
        "================================================"
    )