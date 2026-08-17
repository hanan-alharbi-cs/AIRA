from __future__ import annotations

from typing import Any, Mapping

from ai.features.load_features import (
    calculate_acwr,
    calculate_session_load,
)
from ai.features.baseline_features import (
    calculate_sleep_deviation,
    calculate_reaction_time_deviation,
    calculate_training_load_deviation,
)
from ai.features.recovery_features import (
    calculate_recovery_subscore,
    calculate_sleep_subscore,
    calculate_reaction_subscore,
)
from ai.scoring.risk import get_risk_level


WEIGHTS = {
    "recovery": 0.30,
    "sleep": 0.25,
    "workload": 0.20,
    "reaction": 0.15,
    "context": 0.10,
}


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    """Keep a score between 0 and 100."""
    return max(minimum, min(maximum, value))


def calculate_workload_balance_subscore(
    current_load: float,
    baseline_load: float,
) -> float:
    """
    Convert workload deviation from baseline to a 0-100 balance score.

    Positive deviation above baseline reduces the score.
    """
    deviation = calculate_training_load_deviation(
        current_load=current_load,
        baseline_training_load=baseline_load,
    )

    if deviation <= 0:
        return 100.0

    return clamp(100.0 - (deviation * 100.0))


def calculate_context_subscore(
    injury_context: bool,
) -> float:
    """
    Conservative prototype treatment for recent injury context.

    This is a prototype decision-support rule, not a medical rule.
    """
    return 70.0 if injury_context else 100.0


def calculate_data_quality(
    player_data: Mapping[str, Any],
) -> float:
    """Calculate completeness of required input fields."""

    required_fields = [
        "player_id",
        "training_duration_min",
        "rpe",
        "sleep_duration",
        "sleep_quality",
        "reaction_time_ms",
        "recovery_score",
    ]

    available = sum(
        1
        for field in required_fields
        if player_data.get(field) is not None
    )

    return round(
        available / len(required_fields),
        2,
    )


def build_recommendation(
    readiness_score: float,
    sleep_deviation: float,
    workload_deviation: float,
    reaction_deviation: float,
    recovery_score: float,
) -> str:
    """
    Generate a transparent rule-based recommendation.
    """

    reasons = []

    if sleep_deviation < -0.15:
        reasons.append("reduced sleep")

    if workload_deviation > 0.20:
        reasons.append("elevated recent workload")

    if reaction_deviation > 0.10:
        reasons.append("slower reaction time")

    if recovery_score < 60:
        reasons.append("low recovery")

    if readiness_score < 40:
        return (
            "Reduce training intensity and prioritize recovery."
        )

    if readiness_score < 70:
        return (
            "Consider moderating training intensity and "
            "prioritizing recovery."
        )

    return (
        "Continue the planned training program while "
        "monitoring readiness."
    )


def calculate_readiness(
    player_data: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Calculate the AthleteGuard prototype readiness score.

    Rule-based scoring is the primary source of truth for the MVP.
    """

    player_id = str(player_data["player_id"])

    training_duration_min = float(
        player_data["training_duration_min"]
    )

    rpe = float(player_data["rpe"])

    sleep_duration = float(
        player_data["sleep_duration"]
    )

    sleep_quality = float(
        player_data["sleep_quality"]
    )

    reaction_time_ms = float(
        player_data["reaction_time_ms"]
    )

    recovery_score = float(
        player_data["recovery_score"]
    )

    baseline_sleep = float(
        player_data.get("baseline_sleep", 8.0)
    )

    baseline_reaction = float(
        player_data.get(
            "baseline_reaction_time_ms",
            reaction_time_ms,
        )
    )

    baseline_training_load = float(
        player_data.get(
            "baseline_training_load",
            training_duration_min * rpe,
        )
    )

    acute_load = float(
        player_data.get("acute_load", 0.0)
    )

    chronic_load = float(
        player_data.get("chronic_load", 0.0)
    )

    injury_context = bool(
        player_data.get("injury_context", False)
    )

    # -------------------------
    # 1. Feature Engineering
    # -------------------------

    session_load = calculate_session_load(
        training_duration_min,
        rpe,
    )

    acwr = calculate_acwr(
        acute_load,
        chronic_load,
    )

    sleep_deviation = calculate_sleep_deviation(
        sleep_duration,
        baseline_sleep,
    )

    reaction_deviation = calculate_reaction_time_deviation(
        reaction_time_ms,
        baseline_reaction,
    )

    workload_deviation = calculate_training_load_deviation(
        session_load,
        baseline_training_load,
    )

    # -------------------------
    # 2. Subscores
    # -------------------------

    recovery_subscore = calculate_recovery_subscore(
        recovery_score
    )

    sleep_subscore = calculate_sleep_subscore(
        sleep_duration=sleep_duration,
        baseline_sleep=baseline_sleep,
        sleep_quality=sleep_quality,
    )

    reaction_subscore = calculate_reaction_subscore(
        reaction_time_ms=reaction_time_ms,
        baseline_reaction_time_ms=baseline_reaction,
    )

    workload_balance_subscore = (
        calculate_workload_balance_subscore(
            current_load=session_load,
            baseline_load=baseline_training_load,
        )
    )

    context_subscore = calculate_context_subscore(
        injury_context
    )

    # -------------------------
    # 3. Readiness Score
    # -------------------------

    readiness_score = clamp(
        WEIGHTS["recovery"] * recovery_subscore
        + WEIGHTS["sleep"] * sleep_subscore
        + WEIGHTS["workload"] * workload_balance_subscore
        + WEIGHTS["reaction"] * reaction_subscore
        + WEIGHTS["context"] * context_subscore
    )

    readiness_score = round(
        readiness_score,
        1,
    )

    # -------------------------
    # 4. Risk / Status
    # -------------------------

    risk_level = get_risk_level(
        readiness_score
    )

    # -------------------------
    # 5. Recommendation
    # -------------------------

    recommendation = build_recommendation(
        readiness_score=readiness_score,
        sleep_deviation=sleep_deviation,
        workload_deviation=workload_deviation,
        reaction_deviation=reaction_deviation,
        recovery_score=recovery_score,
    )

    return {
        "player_id": player_id,
        "readiness_score": readiness_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "data_quality": calculate_data_quality(
            player_data
        ),
        "metrics": {
            "session_load": round(
                session_load,
                2,
            ),
            "acute_load": round(
                acute_load,
                2,
            ),
            "chronic_load": round(
                chronic_load,
                2,
            ),
            "acwr": (
                round(acwr, 3)
                if acwr is not None
                else None
            ),
        },
        "subscores": {
            "recovery": round(
                recovery_subscore,
                2,
            ),
            "sleep": round(
                sleep_subscore,
                2,
            ),
            "workload_balance": round(
                workload_balance_subscore,
                2,
            ),
            "reaction": round(
                reaction_subscore,
                2,
            ),
            "context": round(
                context_subscore,
                2,
            ),
        },
        "deviations": {
            "sleep_deviation": round(
                sleep_deviation,
                3,
            ),
            "reaction_time_deviation": round(
                reaction_deviation,
                3,
            ),
            "training_load_deviation": round(
                workload_deviation,
                3,
            ),
        },
    }