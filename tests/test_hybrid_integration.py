from __future__ import annotations

import unittest

import pandas as pd

from ai.hybrid.hybrid_risk_engine import (
    calculate_hybrid_risk,
)


class HybridIntegrationTests(unittest.TestCase):

    def test_real_model_and_readiness_engine_work_together(
        self,
    ) -> None:

        dataset = pd.read_csv(
            "data/modeling/final_dataset.csv"
            if False
            else "data/processed/final_dataset.csv"
        )

        row = dataset.iloc[0]

        # Build raw inputs required by Readiness Engine.
        # The current synthetic dataset has no explicit RPE or
        # sleep-quality fields, so use transparent prototype defaults
        # for integration testing only.
        player_data = {
            "player_id": row["player_id"],

            # Readiness Engine inputs
            "training_duration_min": float(
                row["minutes_played"]
            ),
            "rpe": 6.0,
            "sleep_duration": float(
                row["sleep_duration"]
            ),
            "sleep_quality": 80.0,
            "reaction_time_ms": float(
                row["reaction_time_ms"]
            ),
            "recovery_score": float(
                row["recovery_score"]
            ),

            # Historical/context values
            "baseline_sleep": 8.0,
            "baseline_reaction_time_ms": float(
                row["baseline_reaction_time"]
            ),
            "baseline_training_load": float(
                row["baseline_training_load"]
            ),
            "acute_load": float(
                row["acute_load"]
            ),
            "chronic_load": float(
                row["chronic_load"]
            ),
            "injury_context": False,

            # ML features
            "session_load": float(
                row["session_load"]
            ),
            "acwr": float(
                row["acwr"]
            ),
            "sleep_deviation": float(
                row["sleep_deviation"]
            ),
            "reaction_time_deviation": float(
                row["reaction_time_deviation"]
            ),
            "recovery_drop": float(
                row["recovery_drop"]
            ),
            "warning_points": float(
                row["warning_points"]
            ),
        }

        result = calculate_hybrid_risk(
            player_data
        )

        self.assertIn(
            result["final_risk_level"],
            {
                "low",
                "moderate",
                "elevated",
            },
        )

        self.assertGreaterEqual(
            result["ml_injury_probability"],
            0.0,
        )

        self.assertLessEqual(
            result["ml_injury_probability"],
            1.0,
        )

        self.assertGreaterEqual(
            result["readiness_score"],
            0.0,
        )

        self.assertLessEqual(
            result["readiness_score"],
            100.0,
        )

        self.assertIsInstance(
            result["recommendation"],
            str,
        )

        self.assertIn(
            "factors",
            result,
        )

        self.assertIn(
            "early_warning",
            result,
        )


if __name__ == "__main__":
    unittest.main()