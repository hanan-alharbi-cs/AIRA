from __future__ import annotations


def calculate_baseline_deviation(
    current_value: float,
    baseline_value: float,
) -> float:
    """
    Calculate relative deviation from an athlete-specific baseline.

    Example:
        current = 90
        baseline = 100
        result = -0.10  -> 10% below baseline
    """

    if baseline_value <= 0:
        return 0.0

    return (current_value - baseline_value) / baseline_value


def calculate_sleep_deviation(
    sleep_duration: float,
    baseline_sleep: float,
) -> float:
    """Calculate sleep deviation from the athlete's baseline."""
    return calculate_baseline_deviation(
        current_value=sleep_duration,
        baseline_value=baseline_sleep,
    )


def calculate_reaction_time_deviation(
    reaction_time_ms: float,
    baseline_reaction_time_ms: float,
) -> float:
    """Calculate reaction-time deviation from the athlete's baseline."""
    return calculate_baseline_deviation(
        current_value=reaction_time_ms,
        baseline_value=baseline_reaction_time_ms,
    )


def calculate_training_load_deviation(
    current_load: float,
    baseline_training_load: float,
) -> float:
    """Calculate training-load deviation from the athlete's baseline."""
    return calculate_baseline_deviation(
        current_value=current_load,
        baseline_value=baseline_training_load,
    )