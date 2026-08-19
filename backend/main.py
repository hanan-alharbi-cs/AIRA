"""
AthleteGuard AI Backend
-----------------------
FastAPI service for athlete readiness, hybrid injury-risk,
and What-If decision-support scenarios.

Current scope:
- Health endpoints
- Unified Readiness Engine
- Temporal ML injury-risk prediction
- Hybrid Readiness + ML risk assessment
- Hybrid What-If scenario simulation

Architecture:

    Request
       |
       +--> Readiness Engine
       |
       +--> Temporal ML
       |
       +--> Hybrid Risk
       |
       +--> What-If Scenario
       |
       --> API Response

Important:
This is a synthetic-development prototype and not a
clinical diagnosis or medical clearance system.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from ai.hybrid.hybrid_risk_engine import (
    calculate_hybrid_risk,
)
from ai.scoring.readiness_service import (
    calculate_readiness_output,
)
from ai.scoring.what_if import (
    calculate_what_if,
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="AthleteGuard AI API",
    description=(
        "Backend API for athlete readiness, temporal "
        "injury-risk, hybrid risk assessment, and "
        "temporary What-If training scenarios."
    ),
    version="1.3.0",
)


# ============================================================
# Readiness request / response models
# ============================================================

class ReadinessRequest(BaseModel):
    """
    Request for the unified Readiness Engine.

    This endpoint uses the same readiness logic that powers
    the Hybrid AI.
    """

    player_id: int = Field(
        ...,
        ge=1,
        description="Unique athlete/player identifier.",
        examples=[12],
    )

    training_duration_min: float = Field(
        ...,
        gt=0,
        description="Training duration in minutes.",
        examples=[76.0],
    )

    rpe: float = Field(
        ...,
        ge=0,
        le=10,
        description="Rate of perceived exertion.",
        examples=[5.58],
    )

    sleep_duration: float = Field(
        ...,
        ge=0,
        le=24,
        description="Sleep duration in hours.",
        examples=[8.49],
    )

    sleep_quality: float = Field(
        ...,
        ge=0,
        le=100,
        description="Sleep quality score.",
        examples=[80.0],
    )

    reaction_time_ms: float = Field(
        ...,
        gt=0,
        description="Reaction time in milliseconds.",
        examples=[194.54],
    )

    recovery_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Recovery score.",
        examples=[78.38],
    )

    baseline_sleep: float = Field(
        default=8.0,
        gt=0,
        le=24,
        description="Historical baseline sleep duration.",
        examples=[8.012],
    )

    baseline_reaction_time_ms: float | None = Field(
        default=None,
        gt=0,
        description="Historical baseline reaction time.",
        examples=[197.47],
    )

    baseline_training_load: float | None = Field(
        default=None,
        gt=0,
        description="Historical baseline training load.",
        examples=[363.25],
    )

    acute_load: float = Field(
        ...,
        ge=0,
        description="Current acute workload.",
        examples=[375.34],
    )

    chronic_load: float = Field(
        ...,
        gt=0,
        description="Current chronic workload.",
        examples=[363.25],
    )

    injury_context: bool = Field(
        default=False,
        description="Recent injury context flag.",
    )


class ReadinessResponse(BaseModel):
    """Unified Readiness Engine response."""

    player_id: str

    readiness_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    risk_level: str

    recommendation: str

    data_quality: float = Field(
        ...,
        ge=0,
        le=1,
    )

    factors: list[Any]

    early_warning: Any

    metrics: dict[str, Any]

    subscores: dict[str, Any]

    deviations: dict[str, Any]


# ============================================================
# Hybrid request / response models
# ============================================================

class HybridRiskRequest(BaseModel):
    """
    Input contract for the combined Readiness + Temporal ML engine.

    Contains:
    1. Raw readiness measurements
    2. Historical baselines
    3. Temporal ML engineered features
    """

    player_id: int = Field(
        ...,
        ge=1,
        description="Unique athlete/player identifier.",
        examples=[12],
    )

    # --------------------------------------------------------
    # Raw Readiness Engine inputs
    # --------------------------------------------------------

    training_duration_min: float = Field(
        ...,
        gt=0,
        description="Training duration in minutes.",
        examples=[76.0],
    )

    rpe: float = Field(
        ...,
        ge=0,
        le=10,
        description="Rate of perceived exertion.",
        examples=[5.58],
    )

    sleep_duration: float = Field(
        ...,
        ge=0,
        le=24,
        description="Sleep duration in hours.",
        examples=[8.49],
    )

    sleep_quality: float = Field(
        ...,
        ge=0,
        le=100,
        description="Sleep quality score.",
        examples=[80.0],
    )

    reaction_time_ms: float = Field(
        ...,
        gt=0,
        description="Reaction time in milliseconds.",
        examples=[194.54],
    )

    recovery_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Recovery score.",
        examples=[78.38],
    )

    baseline_sleep: float = Field(
        default=8.0,
        gt=0,
        le=24,
        description="Historical baseline sleep duration.",
        examples=[8.012],
    )

    baseline_reaction_time_ms: float | None = Field(
        default=None,
        gt=0,
        description="Historical baseline reaction time.",
        examples=[197.47],
    )

    baseline_training_load: float | None = Field(
        default=None,
        gt=0,
        description="Historical baseline training load.",
        examples=[363.25],
    )

    acute_load: float = Field(
        ...,
        ge=0,
        description="Current acute workload.",
        examples=[375.34],
    )

    chronic_load: float = Field(
        ...,
        gt=0,
        description="Current chronic workload.",
        examples=[363.25],
    )

    injury_context: bool = Field(
        default=False,
        description="Recent injury context flag.",
    )

    # --------------------------------------------------------
    # Temporal ML engineered features
    # --------------------------------------------------------

    session_load: float = Field(
        ...,
        ge=0,
        description="Current session load.",
        examples=[424.08],
    )

    acwr: float = Field(
        ...,
        ge=0,
        description="Acute:Chronic Workload Ratio.",
        examples=[1.033],
    )

    sleep_deviation: float = Field(
        ...,
        description="Sleep deviation from baseline.",
        examples=[0.478],
    )

    reaction_time_deviation: float = Field(
        ...,
        description="Reaction-time deviation from baseline.",
        examples=[-2.93],
    )

    recovery_drop: float = Field(
        ...,
        description="Drop in recovery from baseline.",
        examples=[-3.792],
    )

    warning_points: float = Field(
        ...,
        ge=0,
        description="Aggregated warning indicator count.",
        examples=[0.0],
    )


class HybridRiskResponse(BaseModel):
    """Combined Readiness + Temporal ML result."""

    player_id: str

    final_risk_level: str

    readiness_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    readiness_risk_level: str

    ml_injury_probability: float = Field(
        ...,
        ge=0,
        le=1,
    )

    ml_risk_level: str

    recommendation: str

    data_quality: float = Field(
        ...,
        ge=0,
        le=1,
    )

    factors: list[Any]

    early_warning: Any

    metrics: dict[str, Any]

    subscores: dict[str, Any]

    deviations: dict[str, Any]


# ============================================================
# What-If request model
# ============================================================

class WhatIfRequest(HybridRiskRequest):
    """
    Request for a temporary What-If scenario.

    The base athlete data is identical to HybridRiskRequest.

    'changes' contains temporary values that should be
    simulated without modifying or persisting the original
    athlete data.
    """

    changes: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Temporary scenario changes. "
            "Examples include training_duration_min, rpe, "
            "sleep_duration, sleep_quality, recovery_score, "
            "session_load, acute_load, acwr, recovery_drop, "
            "and warning_points."
        ),
        examples=[
            {
                "training_duration_min": 60.0,
                "rpe": 5.0,
                "sleep_duration": 8.5,
                "sleep_quality": 90.0,
                "recovery_score": 85.0,
                "session_load": 300.0,
            }
        ],
    )


# ============================================================
# Health response
# ============================================================

class HealthResponse(BaseModel):
    """API health response."""

    status: str
    service: str
    version: str


# ============================================================
# Health endpoints
# ============================================================

@app.get(
    "/",
    response_model=HealthResponse,
    tags=["Health"],
)
def root() -> HealthResponse:
    """Basic API health check."""

    return HealthResponse(
        status="ok",
        service="AthleteGuard AI API",
        version=app.version,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health_check() -> HealthResponse:
    """Dedicated health endpoint for monitoring."""

    return HealthResponse(
        status="healthy",
        service="AthleteGuard AI API",
        version=app.version,
    )


# ============================================================
# Unified Readiness endpoint
# ============================================================

@app.post(
    "/get_player_readiness",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    tags=["Readiness"],
)
def get_player_readiness(
    data: ReadinessRequest,
) -> ReadinessResponse:
    """
    Run the unified Readiness Engine.

    Uses the same readiness logic that powers the Hybrid AI.
    """

    try:

        player_data = data.model_dump()

        result = calculate_readiness_output(
            player_data
        )

        return ReadinessResponse(
            **result
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# ============================================================
# Hybrid Risk endpoint
# ============================================================

@app.post(
    "/get_hybrid_risk",
    response_model=HybridRiskResponse,
    status_code=status.HTTP_200_OK,
    tags=["Hybrid Risk"],
)
def get_hybrid_risk(
    data: HybridRiskRequest,
) -> HybridRiskResponse:
    """
    Run the combined Readiness + Temporal ML engine.
    """

    try:

        player_data = data.model_dump()

        result = calculate_hybrid_risk(
            player_data
        )

        return HybridRiskResponse(
            **result
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# ============================================================
# What-If endpoint
# ============================================================

@app.post(
    "/get_what_if",
    status_code=status.HTTP_200_OK,
    tags=["What-If"],
)
def get_what_if(
    data: WhatIfRequest,
) -> dict[str, Any]:
    """
    Compare the athlete's current state with a temporary
    intervention scenario.

    The scenario is not persisted.
    """

    try:

        request_data = data.model_dump()

        changes = request_data.pop(
            "changes",
            {},
        )

        result = calculate_what_if(
            player_data=request_data,
            changes=changes,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc