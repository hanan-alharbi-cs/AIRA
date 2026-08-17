from __future__ import annotations

from typing import Any, Mapping


def explain_readiness(
    player_data: Mapping[str, Any],
    readiness_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Generate the top factors affecting the athlete's readiness.

    This is the mandatory rule-based fallback explanation
    for the MVP. It does not depend on SHAP.
    """

    factors: list[dict[str, Any]] = []

    deviations = readiness_result.get("deviations", {})

    sleep_deviation = float(
        deviations.get("sleep_deviation", 0.0)
    )

    workload_deviation = float(
        deviations.get("training_load_deviation", 0.0)
    )

    reaction_deviation = float(
        deviations.get("reaction_time_deviation", 0.0)
    )

    recovery_score = float(
        player_data.get("recovery_score", 100.0)
    )

    injury_context = bool(
        player_data.get("injury_context", False)
    )

    # 1. Sleep
    if sleep_deviation < -0.15:
        factors.append(
            {
                "feature": "sleep_duration",
                "label": "قلة النوم عن خط الأساس",
                "impact": -12,
                "direction": "negative",
            }
        )

    # 2. Recent workload
    if workload_deviation > 0.20:
        factors.append(
            {
                "feature": "acute_load",
                "label": "ارتفاع الحمل الحديث",
                "impact": -9,
                "direction": "negative",
            }
        )

    # 3. Reaction time
    if reaction_deviation > 0.10:
        factors.append(
            {
                "feature": "reaction_time_deviation",
                "label": "زمن استجابة أبطأ من المعتاد",
                "impact": -7,
                "direction": "negative",
            }
        )

    # 4. Recovery
    if recovery_score < 60:
        factors.append(
            {
                "feature": "recovery_score",
                "label": "انخفاض مستوى التعافي",
                "impact": -6,
                "direction": "negative",
            }
        )

    # 5. Injury context
    if injury_context:
        factors.append(
            {
                "feature": "injury_context",
                "label": "وجود سياق إصابة حديث",
                "impact": -5,
                "direction": "negative",
            }
        )

    # Keep the strongest three factors for the UI/demo.
    factors.sort(
        key=lambda factor: abs(float(factor["impact"])),
        reverse=True,
    )

    return factors[:3]