from __future__ import annotations

import unittest

from ai.hybrid.hybrid_risk_engine import (
    classify_ml_risk,
    combine_risk_levels,
)


class HybridRiskTests(unittest.TestCase):

    def test_low_ml_probability(self) -> None:
        self.assertEqual(
            classify_ml_risk(0.20),
            "low",
        )

    def test_moderate_ml_probability(self) -> None:
        self.assertEqual(
            classify_ml_risk(0.50),
            "moderate",
        )

    def test_elevated_ml_probability(self) -> None:
        self.assertEqual(
            classify_ml_risk(0.80),
            "elevated",
        )

    def test_two_low_signals_remain_low(self) -> None:
        self.assertEqual(
            combine_risk_levels(
                readiness_risk="low",
                ml_risk="low",
                ml_probability=0.10,
            ),
            "low",
        )

    def test_moderate_readiness_plus_elevated_ml(self) -> None:
        self.assertEqual(
            combine_risk_levels(
                readiness_risk="moderate",
                ml_risk="elevated",
                ml_probability=0.80,
            ),
            "elevated",
        )

    def test_elevated_readiness_plus_moderate_ml(self) -> None:
        self.assertEqual(
            combine_risk_levels(
                readiness_risk="elevated",
                ml_risk="moderate",
                ml_probability=0.50,
            ),
            "elevated",
        )


if __name__ == "__main__":
    unittest.main()