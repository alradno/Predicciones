from __future__ import annotations

import unittest

from predicciones.core.promotion_state import (
    ForwardSampleInputs,
    PromotionInputs,
    PromotionStatus,
    SampleStatus,
    classify_forward_sample,
    classify_promotion_status,
)


def _promotion_inputs(**overrides: object) -> PromotionInputs:
    values = {
        "valid_forward_decisions": 400,
        "settled_decisions": 300,
        "fresh_book_rate": 0.95,
        "roi_lower_bound_95": 0.05,
        "avg_clv": 0.01,
        "max_drawdown_units": 10.0,
    }
    values.update(overrides)
    return PromotionInputs(**values)


class PromotionStateTests(unittest.TestCase):
    def test_coverage_blocked_when_fresh_book_rate_low(self) -> None:
        decision = classify_forward_sample(
            ForwardSampleInputs(
                valid_forward_decisions=120,
                settled_decisions=50,
                fresh_book_rate=0.79,
            )
        )

        self.assertEqual(decision.sample_status, SampleStatus.coverage_blocked)
        self.assertFalse(decision.actionable_roi)
        self.assertFalse(decision.can_reopen_decision_region_analysis)
        self.assertIn("fresh_book_rate_below_minimum", decision.sample_blockers[0])

    def test_collecting_when_valid_decisions_below_minimum(self) -> None:
        decision = classify_forward_sample(
            ForwardSampleInputs(
                valid_forward_decisions=99,
                settled_decisions=50,
                fresh_book_rate=0.80,
            )
        )

        self.assertEqual(decision.sample_status, SampleStatus.collecting_forward_sample)
        self.assertFalse(decision.actionable_roi)
        self.assertFalse(decision.can_reopen_decision_region_analysis)
        self.assertIn("valid_forward_decisions_below_minimum", decision.sample_blockers[0])

    def test_settlement_pending_when_valid_enough_but_not_settled(self) -> None:
        decision = classify_forward_sample(
            ForwardSampleInputs(
                valid_forward_decisions=100,
                settled_decisions=39,
                fresh_book_rate=0.80,
            )
        )

        self.assertEqual(decision.sample_status, SampleStatus.settlement_pending)
        self.assertFalse(decision.actionable_roi)
        self.assertFalse(decision.can_reopen_decision_region_analysis)
        self.assertIn("settled_decisions_below_minimum", decision.sample_blockers[0])

    def test_sample_ready_when_valid_settled_and_fresh(self) -> None:
        decision = classify_forward_sample(
            ForwardSampleInputs(
                valid_forward_decisions=100,
                settled_decisions=40,
                fresh_book_rate=0.80,
            )
        )

        self.assertEqual(decision.sample_status, SampleStatus.sample_ready)
        self.assertEqual(decision.sample_blockers, ())

    def test_sample_ready_is_not_actionable_roi(self) -> None:
        decision = classify_forward_sample(
            ForwardSampleInputs(
                valid_forward_decisions=150,
                settled_decisions=75,
                fresh_book_rate=0.95,
            )
        )

        self.assertEqual(decision.sample_status, SampleStatus.sample_ready)
        self.assertFalse(decision.actionable_roi)

    def test_sample_ready_allows_analysis_but_not_actionable_roi(self) -> None:
        decision = classify_forward_sample(
            ForwardSampleInputs(
                valid_forward_decisions=150,
                settled_decisions=75,
                fresh_book_rate=0.95,
            )
        )

        self.assertTrue(decision.can_reopen_decision_region_analysis)
        self.assertFalse(decision.actionable_roi)

    def test_analysis_ready_not_equal_paper_promotable(self) -> None:
        decision = classify_promotion_status(
            _promotion_inputs(settled_decisions=149)
        )

        self.assertEqual(decision.promotion_status, PromotionStatus.analysis_ready)
        self.assertFalse(decision.actionable_roi)
        self.assertFalse(decision.paper_stake_allowed)
        self.assertFalse(decision.capital_stake_allowed)
        self.assertIn("settled_decisions_below_paper_minimum", decision.promotion_blockers[0])

    def test_paper_promotable_requires_roi_lower_bound_positive(self) -> None:
        decision = classify_promotion_status(
            _promotion_inputs(settled_decisions=150, roi_lower_bound_95=0.0)
        )

        self.assertEqual(decision.promotion_status, PromotionStatus.analysis_ready)
        self.assertFalse(decision.actionable_roi)
        self.assertFalse(decision.paper_stake_allowed)
        self.assertFalse(decision.capital_stake_allowed)
        self.assertIn("roi_lower_bound_95_below_paper_minimum", decision.promotion_blockers[0])

    def test_paper_promotable_requires_clv_non_negative(self) -> None:
        decision = classify_promotion_status(
            _promotion_inputs(settled_decisions=150, avg_clv=-0.001)
        )

        self.assertEqual(decision.promotion_status, PromotionStatus.analysis_ready)
        self.assertFalse(decision.actionable_roi)
        self.assertFalse(decision.paper_stake_allowed)
        self.assertFalse(decision.capital_stake_allowed)
        self.assertIn("avg_clv_below_paper_minimum", decision.promotion_blockers[0])

    def test_capital_promotable_requires_larger_sample(self) -> None:
        decision = classify_promotion_status(
            _promotion_inputs(settled_decisions=299)
        )

        self.assertEqual(decision.promotion_status, PromotionStatus.paper_promotable)
        self.assertTrue(decision.actionable_roi)
        self.assertTrue(decision.paper_stake_allowed)
        self.assertFalse(decision.capital_stake_allowed)
        self.assertIn("settled_decisions_below_capital_minimum", decision.promotion_blockers[0])

    def test_capital_promotable_requires_drawdown_within_limit(self) -> None:
        decision = classify_promotion_status(
            _promotion_inputs(max_drawdown_units=50.01)
        )

        self.assertEqual(decision.promotion_status, PromotionStatus.paper_promotable)
        self.assertTrue(decision.actionable_roi)
        self.assertTrue(decision.paper_stake_allowed)
        self.assertFalse(decision.capital_stake_allowed)
        self.assertIn("max_drawdown_units_above_capital_maximum", decision.promotion_blockers[0])

    def test_not_actionable_when_sample_not_ready(self) -> None:
        decision = classify_promotion_status(
            _promotion_inputs(valid_forward_decisions=99)
        )

        self.assertEqual(decision.promotion_status, PromotionStatus.not_actionable)
        self.assertFalse(decision.actionable_roi)
        self.assertFalse(decision.paper_stake_allowed)
        self.assertFalse(decision.capital_stake_allowed)
        self.assertIn("valid_forward_decisions_below_minimum", decision.promotion_blockers[0])

    def test_missing_metrics_block_promotion(self) -> None:
        missing_all = classify_promotion_status(
            _promotion_inputs(
                roi_lower_bound_95=None,
                avg_clv=None,
                max_drawdown_units=None,
            )
        )
        missing_roi = classify_promotion_status(
            _promotion_inputs(roi_lower_bound_95=None)
        )
        missing_clv = classify_promotion_status(
            _promotion_inputs(avg_clv=None)
        )
        missing_drawdown = classify_promotion_status(
            _promotion_inputs(max_drawdown_units=None)
        )

        self.assertEqual(missing_all.promotion_status, PromotionStatus.analysis_ready)
        self.assertFalse(missing_all.paper_stake_allowed)
        self.assertFalse(missing_all.capital_stake_allowed)
        self.assertIn("roi_lower_bound_95_missing", missing_all.promotion_blockers)
        self.assertIn("avg_clv_missing", missing_all.promotion_blockers)
        self.assertIn("max_drawdown_units_missing", missing_all.promotion_blockers)

        self.assertEqual(missing_roi.promotion_status, PromotionStatus.analysis_ready)
        self.assertFalse(missing_roi.paper_stake_allowed)
        self.assertIn("roi_lower_bound_95_missing", missing_roi.promotion_blockers)

        self.assertEqual(missing_clv.promotion_status, PromotionStatus.analysis_ready)
        self.assertFalse(missing_clv.paper_stake_allowed)
        self.assertIn("avg_clv_missing", missing_clv.promotion_blockers)

        self.assertEqual(missing_drawdown.promotion_status, PromotionStatus.paper_promotable)
        self.assertFalse(missing_drawdown.capital_stake_allowed)
        self.assertIn("max_drawdown_units_missing", missing_drawdown.promotion_blockers)

    def test_capital_promotable_sets_capital_stake_allowed_true(self) -> None:
        decision = classify_promotion_status(_promotion_inputs())

        self.assertEqual(decision.promotion_status, PromotionStatus.capital_promotable)
        self.assertTrue(decision.actionable_roi)
        self.assertTrue(decision.paper_stake_allowed)
        self.assertTrue(decision.capital_stake_allowed)
        self.assertEqual(decision.promotion_blockers, ())

    def test_paper_promotable_does_not_allow_capital_stake(self) -> None:
        decision = classify_promotion_status(
            _promotion_inputs(settled_decisions=150)
        )

        self.assertEqual(decision.promotion_status, PromotionStatus.paper_promotable)
        self.assertTrue(decision.actionable_roi)
        self.assertTrue(decision.paper_stake_allowed)
        self.assertFalse(decision.capital_stake_allowed)


if __name__ == "__main__":
    unittest.main()
