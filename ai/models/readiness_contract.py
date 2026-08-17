from __future__ import annotations

from typing import Literal, TypedDict


RiskLevel = Literal["low", "moderate", "elevated"]
Direction = Literal["positive", "negative"]


class ReadinessFactor(TypedDict):
    feature: str
    label: str
    impact: float
    direction: Direction


class EarlyWarningOutput(TypedDict):
    warning_points: int
    risk_level: RiskLevel
    signals: list[str]


class ReadinessMetrics(TypedDict):
    session_load: float
    acute_load: float
    chronic_load: float
    acwr: float | None


class ReadinessSubscores(TypedDict):
    recovery: float
    sleep: float
    workload_balance: float
    reaction: float
    context: float


class ReadinessDeviations(TypedDict):
    sleep_deviation: float
    reaction_time_deviation: float
    training_load_deviation: float


class ReadinessOutput(TypedDict):
    player_id: str
    readiness_score: float
    risk_level: RiskLevel
    recommendation: str
    data_quality: float
    factors: list[ReadinessFactor]
    early_warning: EarlyWarningOutput
    metrics: ReadinessMetrics
    subscores: ReadinessSubscores
    deviations: ReadinessDeviations