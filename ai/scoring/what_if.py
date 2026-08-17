from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ai.scoring.readiness_service import calculate_readiness_output


def calculate_what_if(
    player_data: Mapping[str, Any],
    changes: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Calculate a temporary What-If readiness scenario.

    The original player data is never modified.
    Scenario values are temporary and are not persisted.
    """

    original_data = dict(player_data)

    scenario_data = deepcopy(original_data)
    scenario_data.update(changes)

    current_result = calculate_readiness_output(
        original_data
    )

    scenario_result = calculate_readiness_output(
        scenario_data
    )

    current_score = float(
        current_result["readiness_score"]
    )

    scenario_score = float(
        scenario_result["readiness_score"]
    )

    delta = round(
        scenario_score - current_score,
        1,
    )

    return {
        "player_id": scenario_result["player_id"],
        "current_score": current_score,
        "scenario_score": scenario_score,
        "delta": delta,
        "scenario_risk_level": scenario_result[
            "risk_level"
        ],
        "scenario_recommendation": scenario_result[
            "recommendation"
        ],
        "scenario_factors": scenario_result[
            "factors"
        ],
        "changes": dict(changes),
    }