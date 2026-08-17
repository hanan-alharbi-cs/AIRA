from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class ReadinessModel(ABC):
    """
    Interface for an optional ML readiness model.

    The rule-based readiness engine remains the primary
    source of truth for the MVP.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the ML model is ready for inference."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        features: Mapping[str, Any],
    ) -> float:
        """Return an optional ML readiness prediction."""
        raise NotImplementedError


class RuleBasedFallbackModel(ReadinessModel):
    """
    Fallback used when an ML model is unavailable.

    The main rule-based Readiness Engine remains responsible
    for the actual readiness score.
    """

    def is_available(self) -> bool:
        return False

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> float:
        raise RuntimeError(
            "ML model is not available. "
            "Use the rule-based readiness engine."
        )


class MLModelAdapter(ReadinessModel):
    """
    Adapter for an optional trained ML model.

    The actual trained model will be connected only after
    the synthetic dataset has been validated.
    """

    def __init__(self, model: Any = None) -> None:
        self.model = model

    def is_available(self) -> bool:
        return self.model is not None

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> float:
        if not self.is_available():
            raise RuntimeError(
                "ML model is not loaded."
            )

        prediction = self.model.predict(
            [dict(features)]
        )

        return float(prediction[0])