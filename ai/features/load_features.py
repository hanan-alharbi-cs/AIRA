from __future__ import annotations

from collections.abc import Sequence


def calculate_session_load(
    training_duration_min: float,
    rpe: float,
) -> float:
    """
    Calculate session training load using:

        session_load = training_duration_min × RPE

    RPE must be between 1 and 10.
    """

    if training_duration_min < 0:
        raise ValueError("training_duration_min cannot be negative.")

    if not 1 <= rpe <= 10:
        raise ValueError("RPE must be between 1 and 10.")

    return training_duration_min * rpe


def calculate_acute_load(
    daily_loads: Sequence[float],
) -> float:
    """
    Calculate acute workload using the most recent 7 days.
    """

    loads = [float(value) for value in daily_loads]

    if not loads:
        return 0.0

    return sum(loads[-7:])


def calculate_chronic_load(
    daily_loads: Sequence[float],
) -> float:
    """
    Calculate chronic workload as the average of the most recent
    28 days.
    """

    loads = [float(value) for value in daily_loads]

    if not loads:
        return 0.0

    recent_loads = loads[-28:]

    return sum(recent_loads) / len(recent_loads)


def calculate_acwr(
    acute_load: float,
    chronic_load: float,
) -> float | None:
    """
    Calculate ACWR = acute load / chronic load.

    Returns None when chronic load is zero because the ratio
    is undefined.
    """

    if chronic_load <= 0:
        return None

    return acute_load / chronic_load


def calculate_load_deviation(
    current_load: float,
    baseline_load: float,
) -> float:
    """
    Calculate relative deviation from the athlete's baseline load.

    Example:
        current = 120
        baseline = 100
        result = 0.20  -> +20%
    """

    if baseline_load <= 0:
        return 0.0

    return (current_load - baseline_load) / baseline_load