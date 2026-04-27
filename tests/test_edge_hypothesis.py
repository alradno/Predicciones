from __future__ import annotations

import unittest

from predicciones.core.edge_hypothesis import (
    EdgeHypothesisInputs,
    EdgeHypothesisStatus,
    classify_edge_hypothesis,
)
from predicciones.core.policy import EdgePolicySpec
from predicciones.core.promotion_state import PromotionStatus


class EdgeHypothesisTests(unittest.TestCase):
    def test_collecting_until_forward_sample_is_large_enough(self) -> None:
        decision = classify_edge_hypothesis(
            EdgeHypothesisInputs(
                lane_id="football_1x2_global",
                valid_forward_decisions=150,
                settled_decisions=149,
                fresh_book_rate=0.95,
                observed_roi=0.80,
                roi_lower_bound_95=0.10,
                avg_clv=0.01,
                rolling_roi_min=0.05,
            )
        )

        self.assertEqual(decision.status, EdgeHypothesisStatus.collecting)
        self.assertIn("settled_decisions_below_roi45_minimum:149<300", decision.blockers)
        self.assertFalse(decision.to_dict()["actionable_as_capital"])

    def test_rejects_roi_45_when_sample_is_large_but_target_fails(self) -> None:
        decision = classify_edge_hypothesis(
            EdgeHypothesisInputs(
                lane_id="football_1x2_global",
                valid_forward_decisions=400,
                settled_decisions=300,
                fresh_book_rate=0.95,
                observed_roi=0.44,
                roi_lower_bound_95=0.01,
                avg_clv=0.01,
                rolling_roi_min=0.00,
            )
        )

        self.assertEqual(decision.status, EdgeHypothesisStatus.rejected)
        self.assertIn("observed_roi_below_target:0.4400<0.4500", decision.blockers)

    def test_supports_roi_45_without_live_capital_permission(self) -> None:
        decision = classify_edge_hypothesis(
            EdgeHypothesisInputs(
                lane_id="football_goals_core",
                valid_forward_decisions=400,
                settled_decisions=300,
                fresh_book_rate=0.95,
                observed_roi=0.45,
                roi_lower_bound_95=0.01,
                avg_clv=0.01,
                rolling_roi_min=0.00,
                promotion_status=PromotionStatus.capital_promotable,
            )
        )

        self.assertEqual(decision.status, EdgeHypothesisStatus.supported)
        self.assertFalse(decision.to_dict()["actionable_as_capital"])

    def test_policy_spec_blocks_locked_holdout_training(self) -> None:
        policy = EdgePolicySpec(
            policy_id="bad",
            lane_id="football_1x2_global",
            edge_threshold=0.02,
            ev_threshold=0.0,
            min_odds=1.2,
            max_odds=6.0,
            locked_holdout_used_for_training=True,
        )

        with self.assertRaisesRegex(ValueError, "locked_holdout"):
            policy.validate()


if __name__ == "__main__":
    unittest.main()
