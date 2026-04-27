from __future__ import annotations

import unittest

from predicciones.core.forward_metrics import (
    ForwardDecisionResult,
    bootstrap_roi_lower_bound,
    build_ev_bucket_report,
    build_forward_metrics_report,
    build_odds_bucket_report,
    calculate_clv_decimal_odds,
    calculate_clv_probability,
    calculate_max_drawdown_units,
)


def _decision(
    *,
    ev: float | None = None,
    odds: float | None = None,
    profit_units: float | None = None,
    settled: bool = False,
    lane_id: str = "lane_a",
    index: int = 1,
    decision_implied_probability: float | None = None,
    closing_implied_probability: float | None = None,
    decision_decimal_odds: float | None = None,
    closing_decimal_odds: float | None = None,
) -> ForwardDecisionResult:
    return ForwardDecisionResult(
        lane_id=lane_id,
        event_id=f"event_{index}",
        selection_id=f"selection_{index}",
        ev=ev,
        odds=odds,
        stake=1.0,
        profit_units=profit_units,
        settled=settled,
        decision_implied_probability=decision_implied_probability,
        closing_implied_probability=closing_implied_probability,
        decision_decimal_odds=decision_decimal_odds,
        closing_decimal_odds=closing_decimal_odds,
    )


class ForwardMetricsTests(unittest.TestCase):
    def test_clv_probability_sign(self) -> None:
        self.assertAlmostEqual(calculate_clv_probability(0.40, 0.45), 0.05, places=6)
        self.assertAlmostEqual(calculate_clv_probability(0.45, 0.40), -0.05, places=6)
        self.assertIsNone(calculate_clv_probability(None, 0.45))

    def test_clv_decimal_odds_sign(self) -> None:
        self.assertAlmostEqual(calculate_clv_decimal_odds(2.20, 2.00), 0.20, places=6)
        self.assertAlmostEqual(calculate_clv_decimal_odds(1.80, 2.00), -0.20, places=6)
        self.assertIsNone(calculate_clv_decimal_odds(2.20, None))

    def test_drawdown_units(self) -> None:
        self.assertIsNone(calculate_max_drawdown_units([]))
        self.assertAlmostEqual(calculate_max_drawdown_units([1.0, -0.5, 2.0, -3.0, 1.0]), 3.0, places=6)
        self.assertAlmostEqual(calculate_max_drawdown_units([-1.0, 0.5]), 1.0, places=6)

    def test_bootstrap_lower_bound_none_when_too_few_settled(self) -> None:
        self.assertIsNone(bootstrap_roi_lower_bound([]))
        self.assertIsNone(bootstrap_roi_lower_bound([0.10] * 39))

    def test_bootstrap_lower_bound_is_deterministic(self) -> None:
        returns = [0.25 if index % 3 else -1.0 for index in range(60)]

        first = bootstrap_roi_lower_bound(returns, n_bootstrap=300, seed=7)
        second = bootstrap_roi_lower_bound(returns, n_bootstrap=300, seed=7)

        self.assertEqual(first, second)
        self.assertIsInstance(first, float)

    def test_ev_bucket_report(self) -> None:
        report = build_ev_bucket_report(
            [
                _decision(ev=-0.01, settled=True, profit_units=-1.0, index=1),
                _decision(ev=0.01, settled=True, profit_units=0.5, index=2),
                _decision(ev=0.03, settled=False, profit_units=None, index=3),
                _decision(ev=0.06, settled=True, profit_units=2.0, index=4),
                _decision(ev=None, settled=False, profit_units=None, index=5),
            ]
        )

        self.assertEqual([row["bucket"] for row in report], ["ev < 0", "0 <= ev < 0.02", "0.02 <= ev < 0.05", "ev >= 0.05", "unknown"])
        self.assertEqual([row["decisions"] for row in report], [1, 1, 1, 1, 1])
        self.assertEqual([row["settled_decisions"] for row in report], [1, 1, 0, 1, 0])
        self.assertAlmostEqual(report[0]["profit_units"], -1.0, places=6)
        self.assertAlmostEqual(report[1]["roi"], 0.5, places=6)
        self.assertIsNone(report[2]["roi"])

    def test_odds_bucket_report(self) -> None:
        report = build_odds_bucket_report(
            [
                _decision(odds=1.19, settled=True, profit_units=-1.0, index=1),
                _decision(odds=1.20, settled=True, profit_units=0.2, index=2),
                _decision(odds=1.50, settled=True, profit_units=0.5, index=3),
                _decision(odds=2.00, settled=True, profit_units=1.0, index=4),
                _decision(odds=3.00, settled=True, profit_units=-1.0, index=5),
                _decision(odds=6.00, settled=True, profit_units=1.5, index=6),
                _decision(odds=None, settled=False, profit_units=None, index=7),
            ]
        )

        self.assertEqual(
            [row["bucket"] for row in report],
            [
                "odds < 1.2",
                "1.2 <= odds < 1.5",
                "1.5 <= odds < 2.0",
                "2.0 <= odds < 3.0",
                "3.0 <= odds < 6.0",
                "odds >= 6.0",
                "unknown",
            ],
        )
        self.assertEqual([row["decisions"] for row in report], [1, 1, 1, 1, 1, 1, 1])
        self.assertAlmostEqual(report[5]["roi"], 1.5, places=6)
        self.assertIsNone(report[6]["roi"])

    def test_forward_metrics_report_shape(self) -> None:
        decisions = [
            _decision(
                ev=0.03,
                odds=1.80,
                settled=True,
                profit_units=0.25 if index % 2 else -1.0,
                lane_id="lane_a" if index < 20 else "lane_b",
                index=index,
                decision_implied_probability=0.50,
                closing_implied_probability=0.52,
                decision_decimal_odds=2.00,
                closing_decimal_odds=1.90,
            )
            for index in range(40)
        ]

        report = build_forward_metrics_report(decisions)

        self.assertEqual(
            set(report.keys()),
            {
                "settled_decisions",
                "flat_stake_roi",
                "flat_stake_profit_units",
                "max_drawdown_units",
                "avg_clv_probability",
                "median_clv_probability",
                "avg_clv_decimal_odds",
                "median_clv_decimal_odds",
                "roi_lower_bound_95",
                "ev_bucket_report",
                "odds_bucket_report",
            },
        )
        self.assertEqual(report["settled_decisions"], 40)
        self.assertAlmostEqual(report["flat_stake_roi"], -0.375, places=6)
        self.assertAlmostEqual(report["avg_clv_probability"], 0.02, places=6)
        self.assertAlmostEqual(report["median_clv_decimal_odds"], 0.10, places=6)
        self.assertIsInstance(report["roi_lower_bound_95"], float)
        self.assertEqual(len(report["ev_bucket_report"]), 5)
        self.assertEqual(len(report["odds_bucket_report"]), 7)


if __name__ == "__main__":
    unittest.main()
