"""
Hybrid AthleteGuard risk engine.

Combines:
1. Rule-based Readiness Engine.
2. Temporal ML injury-risk model.

The two engines intentionally receive different representations
of the same athlete observation:

- Readiness Engine -> raw athlete measurements + historical baselines.
- ML Engine        -> engineered temporal features.

This keeps responsibilities separated and makes the integration
easier to test and maintain.

Important:
This is a synthetic-development prototype, not a clinical system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from ai.scoring.readiness_service import calculate_readiness_output


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "temporal_injury_risk_baseline.joblib"
)


# ============================================================
# ML feature contract
# ============================================================

ML_FEATURES = [
    "session_load",
    "acute_load",
    "chronic_load",
    "acwr",
    "sleep_deviation",
    "reaction_time_deviation",
    "recovery_drop",
    "warning_points",
]


# ============================================================
# Readiness Engine input contract
# ============================================================

READINESS_REQUIRED_FIELDS = [
    "player_id",
    "training_duration_min",
    "rpe",
    "sleep_duration",
    "sleep_quality",
    "reaction_time_ms",
    "recovery_score",
]


# ============================================================
# Model loading
# ============================================================

def load_model() -> Any:
    """Load the persisted temporal injury-risk model."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Temporal ML model not found: {MODEL_FILE}"
        )

    return joblib.load(
        MODEL_FILE
    )


# ============================================================
# Readiness input preparation
# ============================================================

def build_readiness_input(
    player_data: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build the raw-data contract expected by Readiness Engine.

    ML-engineered fields are intentionally ignored here because
    the Readiness Engine calculates its own derived metrics.
    """

    missing = [
        field
        for field in READINESS_REQUIRED_FIELDS
        if player_data.get(field) is None
    ]

    if missing:
        raise ValueError(
            "Missing Readiness Engine fields: "
            f"{missing}"
        )

    readiness_input: dict[str, Any] = {
        field: player_data[field]
        for field in READINESS_REQUIRED_FIELDS
    }

    # Optional historical/context fields.
    for field in [
        "baseline_sleep",
        "baseline_reaction_time_ms",
        "baseline_training_load",
        "acute_load",
        "chronic_load",
        "injury_context",
    ]:
        if player_data.get(field) is not None:
            readiness_input[field] = player_data[field]

    return readiness_input


# ============================================================
# ML input preparation
# ============================================================

def build_ml_input(
    player_data: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Build a one-row DataFrame using the exact feature names
    expected by the trained scikit-learn pipeline.
    """

    missing = [
        feature
        for feature in ML_FEATURES
        if player_data.get(feature) is None
    ]

    if missing:
        raise ValueError(
            "Missing ML features: "
            f"{missing}"
        )

    row = {
        feature: float(
            player_data[feature]
        )
        for feature in ML_FEATURES
    }

    return pd.DataFrame(
        [row],
        columns=ML_FEATURES,
    )


# ============================================================
# ML prediction
# ============================================================

def predict_injury_probability(
    player_data: Mapping[str, Any],
    model: Any | None = None,
) -> float:
    """Predict injury probability for the current temporal observation."""

    if model is None:
        model = load_model()

    ml_input = build_ml_input(
        player_data
    )

    probability = model.predict_proba(
        ml_input
    )[0, 1]

    return round(
        float(probability),
        4,
    )


# ============================================================
# ML risk classification
# ============================================================

def classify_ml_risk(
    probability: float,
) -> str:
    """
    Convert ML probability to a prototype risk band.

    Prototype thresholds:
        >= 0.70 -> elevated
        >= 0.40 -> moderate
        <  0.40 -> low

    These thresholds must be validated later using
    independent validation data.
    """

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "ML probability must be between 0 and 1."
        )

    if probability >= 0.70:
        return "elevated"

    if probability >= 0.40:
        return "moderate"

    return "low"


# ============================================================
# Hybrid risk combination
# ============================================================

def combine_risk_levels(
    readiness_risk: str,
    ml_risk: str,
    ml_probability: float,
) -> str:
    """
    Combine Readiness Engine risk and ML risk.

    Hybrid policy:
    - If either engine reports elevated risk,
      final risk = elevated.
    - Otherwise, if either engine reports moderate risk,
      final risk = moderate.
    - Only when both engines report low risk,
      final risk = low.

    This is intentionally conservative for the prototype.
    """

    valid_levels = {
        "low",
        "moderate",
        "elevated",
    }

    if readiness_risk not in valid_levels:
        raise ValueError(
            f"Unsupported readiness risk: {readiness_risk}"
        )

    if ml_risk not in valid_levels:
        raise ValueError(
            f"Unsupported ML risk: {ml_risk}"
        )

    if not 0.0 <= ml_probability <= 1.0:
        raise ValueError(
            "ML probability must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Strongest signal wins.
    # --------------------------------------------------------

    if (
        readiness_risk == "elevated"
        or ml_risk == "elevated"
    ):
        return "elevated"

    # --------------------------------------------------------
    # If neither is elevated but one is moderate.
    # --------------------------------------------------------

    if (
        readiness_risk == "moderate"
        or ml_risk == "moderate"
    ):
        return "moderate"

    # --------------------------------------------------------
    # Both are low.
    # --------------------------------------------------------

    return "low"


# ============================================================
# Hybrid recommendation
# ============================================================

def build_hybrid_recommendation(
    *,
    final_risk: str,
    readiness_result: Mapping[str, Any],
    ml_probability: float,
) -> str:
    """Build a transparent combined recommendation."""

    if final_risk == "elevated":
        return (
            "High-risk indicators detected. "
            "Reduce training intensity, prioritize recovery, "
            "and review the athlete before the next session."
        )

    if final_risk == "moderate":
        return (
            "Moderate-risk indicators detected. "
            "Consider moderating training intensity and "
            "monitoring recovery closely."
        )

    base_recommendation = str(
        readiness_result["recommendation"]
    )

    if ml_probability >= 0.40:
        return (
            f"{base_recommendation} "
            "The ML model also indicates increased injury risk; "
            "continue closer monitoring."
        )

    return base_recommendation


# ============================================================
# Main Hybrid calculation
# ============================================================

def calculate_hybrid_risk(
    player_data: Mapping[str, Any],
    model: Any | None = None,
) -> dict[str, Any]:
    """
    Run both AI layers and return one unified Hybrid result.
    """

    # --------------------------------------------------------
    # 1. Rule-based Readiness Engine
    # --------------------------------------------------------

    readiness_input = build_readiness_input(
        player_data
    )

    readiness_result = calculate_readiness_output(
        readiness_input
    )

    # --------------------------------------------------------
    # 2. Temporal ML Engine
    # --------------------------------------------------------

    ml_probability = predict_injury_probability(
        player_data,
        model=model,
    )

    ml_risk = classify_ml_risk(
        ml_probability
    )

    # --------------------------------------------------------
    # 3. Readiness risk
    # --------------------------------------------------------

    readiness_risk = str(
        readiness_result["risk_level"]
    )

    # --------------------------------------------------------
    # 4. Hybrid decision
    # --------------------------------------------------------

    final_risk = combine_risk_levels(
        readiness_risk=readiness_risk,
        ml_risk=ml_risk,
        ml_probability=ml_probability,
    )

    # --------------------------------------------------------
    # 5. Recommendation
    # --------------------------------------------------------

    recommendation = build_hybrid_recommendation(
        final_risk=final_risk,
        readiness_result=readiness_result,
        ml_probability=ml_probability,
    )

    # --------------------------------------------------------
    # 6. Unified response
    # --------------------------------------------------------

    return {
        "player_id": readiness_result[
            "player_id"
        ],
        "final_risk_level": final_risk,
        "readiness_score": readiness_result[
            "readiness_score"
        ],
        "readiness_risk_level": readiness_risk,
        "ml_injury_probability": ml_probability,
        "ml_risk_level": ml_risk,
        "recommendation": recommendation,
        "data_quality": readiness_result[
            "data_quality"
        ],
        "factors": readiness_result[
            "factors"
        ],
        "early_warning": readiness_result[
            "early_warning"
        ],
        "metrics": readiness_result[
            "metrics"
        ],
        "subscores": readiness_result[
            "subscores"
        ],
        "deviations": readiness_result[
            "deviations"
        ],
    }