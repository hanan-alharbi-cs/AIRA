"""
AthleteGuard Event Readiness Engine.

Calculates a prototype readiness assessment for an
upcoming training session, match, or competition.

Inputs:
- Current Hybrid Risk
- Current Readiness Score
- Days until event
- Event importance

Important:
This is a decision-support prototype built on synthetic
development data. It is not a medical clearance system.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping


# ============================================================
# Constants
# ============================================================

MIN_EVENT_READINESS = 0.0
MAX_EVENT_READINESS = 100.0

VALID_RISKS = {
    "low",
    "moderate",
    "elevated",
}

VALID_IMPORTANCE = {
    1,
    2,
    3,
    4,
    5,
}


# ============================================================
# Helpers
# ============================================================

def clamp(
    value: float,
    minimum: float = MIN_EVENT_READINESS,
    maximum: float = MAX_EVENT_READINESS,
) -> float:
    """Keep a numeric value inside a bounded range."""

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def parse_event_date(
    event_date: str | date | datetime,
) -> date:
    """Convert supported event-date inputs to date."""

    if isinstance(
        event_date,
        datetime,
    ):
        return event_date.date()

    if isinstance(
        event_date,
        date,
    ):
        return event_date

    if isinstance(
        event_date,
        str,
    ):

        try:
            return datetime.strptime(
                event_date,
                "%Y-%m-%d",
            ).date()

        except ValueError as exc:

            raise ValueError(
                "event_date must use YYYY-MM-DD format."
            ) from exc

    raise ValueError(
        "Unsupported event_date type."
    )


def calculate_days_until_event(
    event_date: str | date | datetime,
    reference_date: date | None = None,
) -> int:
    """Calculate calendar days remaining until the event."""

    event_day = parse_event_date(
        event_date
    )

    if reference_date is None:
        reference_date = date.today()

    return (
        event_day
        - reference_date
    ).days


def normalize_importance(
    importance: int | float,
) -> int:
    """Validate and normalize event importance."""

    try:
        normalized = int(
            importance
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "Event importance must be an integer from 1 to 5."
        ) from exc

    if normalized not in VALID_IMPORTANCE:
        raise ValueError(
            "Event importance must be between 1 and 5."
        )

    return normalized


# ============================================================
# Readiness score
# ============================================================

def calculate_event_readiness_score(
    readiness_score: float,
    final_risk_level: str,
    days_until_event: int,
    event_importance: int,
) -> float:
    """
    Calculate a prototype Event Readiness score.

    Logic:
    - Start from current readiness.
    - Penalize elevated/moderate hybrid risk.
    - Add a small urgency adjustment for near events.
    - Use event importance only as context.

    Event importance does NOT override player health/risk.
    """

    if not 0.0 <= readiness_score <= 100.0:
        raise ValueError(
            "readiness_score must be between 0 and 100."
        )

    if final_risk_level not in VALID_RISKS:
        raise ValueError(
            f"Unsupported final risk level: "
            f"{final_risk_level}"
        )

    event_importance = normalize_importance(
        event_importance
    )

    score = float(
        readiness_score
    )

    # --------------------------------------------------------
    # Risk penalty
    # --------------------------------------------------------

    risk_penalty = {
        "low": 0.0,
        "moderate": 12.0,
        "elevated": 28.0,
    }[final_risk_level]

    score -= risk_penalty

    # --------------------------------------------------------
    # Near-event pressure
    #
    # This reflects decision urgency, not medical severity.
    # --------------------------------------------------------

    if days_until_event <= 0:
        urgency_penalty = 8.0

    elif days_until_event == 1:
        urgency_penalty = 6.0

    elif days_until_event <= 3:
        urgency_penalty = 3.0

    elif days_until_event <= 7:
        urgency_penalty = 1.0

    else:
        urgency_penalty = 0.0

    score -= urgency_penalty

    # --------------------------------------------------------
    # Event importance
    #
    # Important events do not make a player healthier.
    # We only use importance as a very small planning factor.
    # --------------------------------------------------------

    importance_context = (
        event_importance - 3
    ) * 0.5

    score += importance_context

    return round(
        clamp(score),
        1,
    )


# ============================================================
# Status
# ============================================================

def classify_event_readiness(
    event_readiness_score: float,
    final_risk_level: str,
) -> str:
    """
    Classify Event Readiness for decision support.

    Important:
    This is not medical clearance.
    """

    if final_risk_level == "elevated":
        return "high_risk"

    if event_readiness_score < 50:
        return "high_risk"

    if (
        final_risk_level == "moderate"
        or event_readiness_score < 70
    ):
        return "needs_reassessment"

    return "ready_with_monitoring"


# ============================================================
# Recommendation
# ============================================================

def build_event_recommendation(
    *,
    event_readiness_score: float,
    final_risk_level: str,
    days_until_event: int,
    event_name: str,
) -> str:
    """
    Build a conservative event-specific recommendation.

    The system does not provide medical clearance or diagnosis.
    """

    if final_risk_level == "elevated":

        return (
            f"High-risk indicators are present before "
            f"{event_name}. "
            "Full-intensity participation is not recommended "
            "without reassessment and appropriate professional review."
        )

    if event_readiness_score < 50:

        return (
            f"Event readiness is low for {event_name}. "
            "Prioritize recovery and reassess before the event."
        )

    if final_risk_level == "moderate":

        return (
            f"Moderate-risk indicators are present before "
            f"{event_name}. "
            "Consider reducing training intensity and "
            "reassessing readiness before participation."
        )

    if days_until_event <= 2:

        return (
            f"Readiness is currently favorable for "
            f"{event_name}, with close monitoring recommended "
            "because the event is approaching."
        )

    return (
        f"Readiness is currently favorable for "
        f"{event_name}. "
        "Continue planned preparation while monitoring "
        "readiness and recovery."
    )


# ============================================================
# Main function
# ============================================================

def calculate_event_readiness(
    *,
    event_name: str,
    event_date: str | date | datetime,
    event_importance: int,
    hybrid_result: Mapping[str, Any],
    reference_date: date | None = None,
) -> dict[str, Any]:
    """
    Calculate the athlete's readiness for an upcoming event.

    hybrid_result must contain:
    - readiness_score
    - final_risk_level
    """

    if not event_name or not str(
        event_name
    ).strip():

        raise ValueError(
            "event_name cannot be empty."
        )

    event_day = parse_event_date(
        event_date
    )

    days_until_event = calculate_days_until_event(
        event_day,
        reference_date=reference_date,
    )

    if days_until_event < 0:
        raise ValueError(
            "The selected event is in the past."
        )

    if "readiness_score" not in hybrid_result:
        raise ValueError(
            "hybrid_result is missing readiness_score."
        )

    if "final_risk_level" not in hybrid_result:
        raise ValueError(
            "hybrid_result is missing final_risk_level."
        )

    readiness_score = float(
        hybrid_result[
            "readiness_score"
        ]
    )

    final_risk_level = str(
        hybrid_result[
            "final_risk_level"
        ]
    )

    normalized_importance = normalize_importance(
        event_importance
    )

    event_readiness_score = (
        calculate_event_readiness_score(
            readiness_score=readiness_score,
            final_risk_level=final_risk_level,
            days_until_event=days_until_event,
            event_importance=normalized_importance,
        )
    )

    status = classify_event_readiness(
        event_readiness_score=event_readiness_score,
        final_risk_level=final_risk_level,
    )

    recommendation = build_event_recommendation(
        event_readiness_score=event_readiness_score,
        final_risk_level=final_risk_level,
        days_until_event=days_until_event,
        event_name=str(
            event_name
        ),
    )

    return {
        "event_name": str(
            event_name
        ),
        "event_date": event_day.isoformat(),
        "days_until_event": days_until_event,
        "event_importance": normalized_importance,
        "event_readiness_score": event_readiness_score,
        "status": status,
        "final_risk_level": final_risk_level,
        "current_readiness_score": round(
            readiness_score,
            1,
        ),
        "recommendation": recommendation,
    }


# ============================================================
# Manual development test
# ============================================================

if __name__ == "__main__":

    example_hybrid_result = {
        "readiness_score": 72.0,
        "final_risk_level": "moderate",
    }

    result = calculate_event_readiness(
        event_name="League Match",
        event_date="2026-08-20",
        event_importance=3,
        hybrid_result=example_hybrid_result,
        reference_date=date(
            2026,
            8,
            18,
        ),
    )

    print(
        "Event readiness test:"
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )