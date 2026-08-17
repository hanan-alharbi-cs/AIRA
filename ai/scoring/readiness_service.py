from __future__ import annotations

from typing import Any, Mapping

from ai.explainability.rules_explainer import explain_readiness
from ai.scoring.early_warning import calculate_early_warning
from ai.scoring.readiness_engine import calculate_readiness
from ai.models.readiness_contract import ReadinessOutput


def calculate_readiness_output(
    player_data: Mapping[str, Any],
) -> ReadinessOutput:
    """
    Build the complete AI output contract.

    This is the single function that the backend should call.
    """

    readiness_result = calculate_readiness(
        player_data
    )

    factors = explain_readiness(
        player_data,
        readiness_result,
    )

    deviations = readiness_result["deviations"]

    sleep_deviation_bad = (
        deviations["sleep_deviation"] < -0.15
    )

    workload_spike = (
        deviations["training_load_deviation"] > 0.20
    )

    recovery_drop = (
        float(player_data.get("recovery_score", 100.0))
        < 60
    )

    reaction_deviation_bad = (
        deviations["reaction_time_deviation"] > 0.10
    )

    injury_context = bool(
        player_data.get("injury_context", False)
    )

    early_warning = calculate_early_warning(
        workload_spike=workload_spike,
        sleep_deviation_bad=sleep_deviation_bad,
        recovery_drop=recovery_drop,
        reaction_deviation_bad=reaction_deviation_bad,
        injury_context=injury_context,
    )

    output: ReadinessOutput = {
        "player_id": readiness_result["player_id"],
        "readiness_score": readiness_result["readiness_score"],
        "risk_level": readiness_result["risk_level"],
        "recommendation": readiness_result["recommendation"],
        "data_quality": readiness_result["data_quality"],
        "factors": factors,
        "early_warning": early_warning,
        "metrics": readiness_result["metrics"],
        "subscores": readiness_result["subscores"],
        "deviations": readiness_result["deviations"],
    }

    return output