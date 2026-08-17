from __future__ import annotations

from typing import Any


def calculate_early_warning(
    *,
    workload_spike: bool,
    sleep_deviation_bad: bool,
    recovery_drop: bool,
    reaction_deviation_bad: bool,
    injury_context: bool,
) -> dict[str, Any]:
    """
    Calculate the AthleteGuard composite early-warning index.

    The MVP combines multiple signals instead of relying on ACWR alone.
    This is a decision-support prototype, not a medical diagnosis.
    """

    warning_points = 0
    signals: list[str] = []

    if workload_spike:
        warning_points += 1
        signals.append("workload_spike")

    if sleep_deviation_bad:
        warning_points += 1
        signals.append("sleep_deviation")

    if recovery_drop:
        warning_points += 1
        signals.append("recovery_drop")

    if reaction_deviation_bad:
        warning_points += 1
        signals.append("reaction_time_deviation")

    if injury_context:
        warning_points += 1
        signals.append("injury_context")

    if warning_points >= 4:
        risk_level = "elevated"
    elif warning_points >= 2:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "warning_points": warning_points,
        "risk_level": risk_level,
        "signals": signals,
    }