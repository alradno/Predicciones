from __future__ import annotations

import unittest

from predicciones.football.model_challenge import (
    choose_probability_source,
    get_model_variant_spec,
    validate_model_candidate_contract,
)


class ModelChallengeTests(unittest.TestCase):
    def test_calibrated_source_wins_only_when_log_loss_and_brier_are_safe(self) -> None:
        self.assertEqual(
            choose_probability_source(
                {"log_loss": 0.70, "brier_score": 0.22},
                {"log_loss": 0.68, "brier_score": 0.21},
                min_log_loss_delta=0.005,
            ),
            "calibrated",
        )
        self.assertEqual(
            choose_probability_source(
                {"log_loss": 0.70, "brier_score": 0.22},
                {"log_loss": 0.68, "brier_score": 0.23},
                min_log_loss_delta=0.005,
            ),
            "raw",
        )

    def test_model_candidate_contract_blocks_locked_holdout_policy_tuning(self) -> None:
        ok, blockers = validate_model_candidate_contract(
            {
                "lane_id": "football_1x2_global",
                "locked_holdout_used_for_training": True,
                "policy_reoptimized": True,
            }
        )

        self.assertFalse(ok)
        self.assertIn("locked_holdout_used_for_training", blockers)
        self.assertIn("policy_reoptimized", blockers)

    def test_model_variants_are_scoped_to_active_football_lanes(self) -> None:
        spec = get_model_variant_spec("challenger_hgb_poisson_calibrated")

        self.assertEqual(spec.role.value, "challenger")
        self.assertEqual(set(spec.lane_ids), {"football_1x2_global", "football_goals_core"})


if __name__ == "__main__":
    unittest.main()
