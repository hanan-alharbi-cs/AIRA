from __future__ import annotations


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    """Keep a numeric score within the 0-100 range."""
    return max(minimum, min(maximum, value))


def calculate_recovery_subscore(
    recovery_score: float,
) -> float:
    """
    Recovery score is already represented on a 0-100 scale.
    """
    return clamp(recovery_score)


def calculate_sleep_subscore(
    sleep_duration: float,
    baseline_sleep: float,
    sleep_quality: float,
) -> float:
    """
    Convert sleep duration and sleep quality into a 0-100 score.

    MVP implementation:
    - 60% sleep duration relative to athlete baseline
    - 40% subjective sleep quality
    """

    if baseline_sleep <= 0:
        duration_score = 50.0
    else:
        duration_ratio = sleep_duration / baseline_sleep
        duration_score = clamp(duration_ratio * 100.0)

    quality_score = clamp(
        (sleep_quality / 5.0) * 100.0
    )

    return clamp(
        0.60 * duration_score
        + 0.40 * quality_score
    )


def calculate_reaction_subscore(
    reaction_time_ms: float,
    baseline_reaction_time_ms: float,
) -> float:
    """
    Convert reaction-time deviation into a 0-100 score.

    A slower reaction time produces a lower score.
    """

    if baseline_reaction_time_ms <= 0:
        return 50.0

    deviation = (
        reaction_time_ms - baseline_reaction_time_ms
    ) / baseline_reaction_time_ms

    score = 100.0 - (deviation * 100.0)

    return clamp(score)