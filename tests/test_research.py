from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from predicciones.config import (
    BacktestConfig,
    ExecutionConfig,
    ProjectPaths,
    ResearchConfig,
    Settings,
    SnapshotConfig,
)
from predicciones.research import (
    build_market_snapshots,
    build_promotion_report,
    build_candidate_rows,
    format_net_summary,
    load_research_run,
    select_candidate_bets,
    simulate_execution,
)
from predicciones.reporting import build_backtest_truth_summary, build_research_truth_summary, format_summary
from predicciones.strategy import BetPolicy


def _settings(tmpdir: str) -> Settings:
    root = Path(tmpdir)
    paths = ProjectPaths(
        root=root,
        data_dir=root / "data",
        outputs_dir=root / "outputs",
        runs_dir=root / "outputs" / "runs",
        models_dir=root / "outputs" / "models",
        benchmarks_dir=root / "benchmarks",
    )
    for path in (paths.data_dir, paths.outputs_dir, paths.runs_dir, paths.models_dir, paths.benchmarks_dir):
        path.mkdir(parents=True, exist_ok=True)
    return Settings(
        paths=paths,
        anthropic_api_key=None,
        claude_model="none",
        default_leagues=("E0",),
        default_seasons=("2425",),
        benchmark_dir_name="legacy_v1",
        snapshot=SnapshotConfig(default_kickoff_hour=15, closing_proxy_capture_minutes=45),
        execution=ExecutionConfig(decision_minutes_before_kickoff=45, slippage_rate=0.02, commission_rate=0.05),
        research=ResearchConfig(stage1_min_bets=5, stage2_min_bets=3, min_segment_bets=2, min_segment_folds=1),
        backtest=BacktestConfig(),
    )


class ResearchPrediccionesTests(unittest.TestCase):
    def test_build_market_snapshots_creates_proxy_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            market = pd.DataFrame(
                {
                    "match_id": [1],
                    "Date": pd.to_datetime(["2024-08-01"]),
                    "league_code": ["E0"],
                    "season": ["2425"],
                    "HomeTeam": ["A"],
                    "AwayTeam": ["B"],
                    "odds_home": [2.0],
                    "odds_draw": [3.5],
                    "odds_away": [3.8],
                }
            )
            snapshots = build_market_snapshots(market, settings=settings)
            self.assertEqual(str(snapshots.iloc[0]["source_type"]), "closing_proxy")
            self.assertEqual(int(snapshots.iloc[0]["time_to_kickoff_minutes"]), 45)

    def test_candidate_selection_and_execution_use_best_quote(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            predictions = pd.DataFrame(
                {
                    "match_id": [1],
                    "fold_id": [1],
                    "fold_segment": ["train"],
                    "Date": pd.to_datetime(["2024-08-01"]),
                    "kickoff_time": pd.to_datetime(["2024-08-01 15:00:00"]),
                    "league_code": ["E0"],
                    "league_name": ["Premier League"],
                    "season": ["2425"],
                    "HomeTeam": ["A"],
                    "AwayTeam": ["B"],
                    "actual_outcome": ["home"],
                    "actual_target": [2],
                    "baseline_prediction": ["home"],
                    "raw_prediction": ["home"],
                    "calibrated_prediction": ["home"],
                    "expected_goals_home": [1.6],
                    "expected_goals_away": [0.9],
                    "market_prob_away": [0.24],
                    "market_prob_draw": [0.28],
                    "market_prob_home": [0.48],
                    "prob_away_raw": [0.18],
                    "prob_draw_raw": [0.24],
                    "prob_home_raw": [0.58],
                    "prob_away_calibrated": [0.17],
                    "prob_draw_calibrated": [0.23],
                    "prob_home_calibrated": [0.60],
                }
            )
            snapshots = pd.DataFrame(
                {
                    "match_id": [1, 1],
                    "Date": pd.to_datetime(["2024-08-01", "2024-08-01"]),
                    "league_code": ["E0", "E0"],
                    "league_name": ["Premier League", "Premier League"],
                    "season": ["2425", "2425"],
                    "HomeTeam": ["A", "A"],
                    "AwayTeam": ["B", "B"],
                    "kickoff_time": pd.to_datetime(["2024-08-01 15:00:00", "2024-08-01 15:00:00"]),
                    "snapshot_time": pd.to_datetime(["2024-08-01 14:15:00", "2024-08-01 14:00:00"]),
                    "source_name": ["book_a", "book_b"],
                    "source_type": ["user_quote", "user_quote"],
                    "liquidity": [100.0, 100.0],
                    "odds_home": [2.20, 2.35],
                    "odds_draw": [3.50, 3.45],
                    "odds_away": [3.80, 3.75],
                }
            )
            candidate_rows = build_candidate_rows(predictions, snapshots, settings)
            bets = select_candidate_bets(
                candidate_rows,
                policy=BetPolicy(edge_threshold=0.02, ev_threshold=0.05, min_odds=1.2, max_odds=5.0),
                probability_source="raw",
            )
            self.assertEqual(len(bets), 1)
            self.assertAlmostEqual(float(bets.iloc[0]["quoted_odds"]), 2.35, places=6)

            execution = simulate_execution(
                bets,
                execution=settings.execution,
                flat_stake=1.0,
                allow_proxy=True,
            )
            self.assertEqual(str(execution.iloc[0]["execution_status"]), "executed")
            expected_odds = 2.35 * (1.0 - settings.execution.slippage_rate)
            expected_profit = (expected_odds - 1.0) * (1.0 - settings.execution.commission_rate)
            self.assertAlmostEqual(float(execution.iloc[0]["net_profit"]), expected_profit, places=6)

    def test_promotion_report_blocks_proxy_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            rows = pd.DataFrame(
                {
                    "source_type": ["closing_proxy"] * 6,
                    "fold_id": [1, 1, 2, 2, 3, 3],
                    "execution_status": ["executed"] * 6,
                    "net_profit": [1.0, 0.8, 1.2, 0.5, 1.1, 0.7],
                    "accepted_stake": [1.0] * 6,
                }
            )
            report = build_promotion_report(rows, rows.iloc[:3].copy(), settings)
            self.assertFalse(report["stage_2_holdout"]["passed"])
            self.assertTrue(report["stage_2_holdout"]["proxy_blocked"])

    def test_backtest_truth_summary_flags_market_and_policy_degradation(self) -> None:
        summary = {
            "baseline": {"accuracy": 0.55, "log_loss": 1.01, "brier_score": 0.62, "roi": 0.04, "profit": 4.0},
            "model_raw": {"accuracy": 0.51, "log_loss": 1.07, "brier_score": 0.66, "roi": -0.01, "profit": -1.0},
            "model_calibrated": {"accuracy": 0.49, "log_loss": 1.09, "brier_score": 0.67},
            "strategy": {"bets": 12, "roi": -0.08, "profit": -3.0},
        }
        truth = build_backtest_truth_summary(summary)
        text = format_summary({**summary, "honest_diagnostics": truth})
        self.assertEqual(truth["overall_verdict"], "not_ready")
        self.assertIn("market_accuracy_still_ahead", truth["flags"])
        self.assertIn("policy_roi_non_positive", truth["flags"])
        self.assertIn("Veredicto:", text)

    def test_research_truth_summary_surfaces_proxy_blocker(self) -> None:
        summary = {
            "selected_probability_source": "raw",
            "probability_decision": {
                "raw": {"log_loss": 1.02, "net_strategy": {"roi": 0.03}},
                "calibrated": {"log_loss": 1.05, "net_strategy": {"roi": 0.01}},
            },
            "baseline": {"accuracy": 0.54, "log_loss": 1.03, "brier_score": 0.63},
            "model_raw": {"accuracy": 0.53, "log_loss": 1.01, "brier_score": 0.62},
            "model_calibrated": {"accuracy": 0.52, "log_loss": 1.04, "brier_score": 0.64},
            "holdout_overall": {"executed": 4, "roi": 0.12},
            "holdout_niche": {"executed": 3, "roi": 0.18},
            "training_niche": {"executed": 9, "roi": 0.20},
            "promotion_report": {
                "stage_1_research": {"passed": True},
                "stage_2_holdout": {"passed": False, "proxy_blocked": True},
                "stage_3_shadow": {"passed": False},
                "blockers": ["Se estan usando closing proxies."],
            },
        }
        truth = build_research_truth_summary(summary)
        text = format_net_summary({**summary, "experimental_truth": truth})
        self.assertEqual(truth["overall_verdict"], "blocked_by_proxies")
        self.assertTrue(truth["promotion_readiness"]["proxy_blocked"])
        self.assertIn("Veredicto:", text)

    def test_promotion_report_consumes_shadow_and_live_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            training_rows = pd.DataFrame(
                {
                    "source_type": ["user_quote"] * 6,
                    "fold_id": [1, 1, 2, 2, 3, 3],
                    "execution_status": ["executed"] * 6,
                    "net_profit": [0.8, 0.7, 0.9, 0.8, 0.7, 0.9],
                    "accepted_stake": [1.0] * 6,
                }
            )
            holdout_rows = pd.DataFrame(
                {
                    "source_type": ["user_quote"] * 3,
                    "fold_id": [4, 4, 5],
                    "execution_status": ["executed"] * 3,
                    "net_profit": [0.7, 0.8, 0.9],
                    "accepted_stake": [1.0] * 3,
                }
            )
            report = build_promotion_report(
                training_rows,
                holdout_rows,
                settings,
                shadow_summary={"passed": True, "validation_stage": "shadow"},
                live_summary={"passed": True, "validation_stage": "live"},
            )
            self.assertTrue(report["stage_3_shadow"]["passed"])
            self.assertTrue(report["stage_4_limited_live"]["passed"])
            self.assertTrue(report["automation_ready"])
            self.assertEqual(report["gate"]["stage"], "automation_ready")

    def test_backtest_truth_summary_prefers_nested_signal_and_policy_blocks(self) -> None:
        summary = {
            "signal_summary": {
                "selected_probability_source": "raw",
                "baseline_metrics": {"accuracy": 0.50, "log_loss": 1.10, "brier_score": 0.68},
                "raw_metrics": {"accuracy": 0.56, "log_loss": 1.01, "brier_score": 0.61},
                "calibrated_metrics": {"accuracy": 0.55, "log_loss": 1.00, "brier_score": 0.60},
            },
            "policy_summary": {
                "holdout_metrics": {"bets": 10, "roi": 0.12, "profit": 2.5},
            },
        }
        truth = build_backtest_truth_summary(summary)
        self.assertTrue(truth["signal_quality"]["beats_market"])
        self.assertTrue(truth["policy_effect"]["positive_roi"])
        self.assertEqual(truth["selected_probability_source"], "raw")

    def test_research_truth_summary_reports_execution_viability_block(self) -> None:
        summary = {
            "signal_summary": {
                "selected_probability_source": "calibrated",
                "baseline_metrics": {"accuracy": 0.50, "log_loss": 1.10, "brier_score": 0.68},
                "raw_metrics": {"accuracy": 0.56, "log_loss": 1.01, "brier_score": 0.61},
                "calibrated_metrics": {"accuracy": 0.57, "log_loss": 0.99, "brier_score": 0.59},
                "probability_decision": {"decision": "calibrated"},
            },
            "policy_summary": {
                "holdout_metrics": {"executed": 4, "roi": 0.14},
                "holdout_niche_metrics": {"executed": 3, "roi": 0.20},
                "training_niche_metrics": {"executed": 8, "roi": 0.18},
            },
            "execution_summary": {
                "coverage": {"proxy_blocked_for_promotion": False},
            },
            "promotion_report": {
                "stage_1_research": {"passed": True},
                "stage_2_holdout": {"passed": True, "proxy_blocked": False},
                "stage_3_shadow": {"passed": True},
                "blockers": [],
            },
        }
        truth = build_research_truth_summary(summary)
        self.assertTrue(truth["signal_quality"]["beats_market"])
        self.assertTrue(truth["policy_effect"]["positive_holdout_roi"])
        self.assertTrue(truth["execution_viability"]["forward_ready"])

    def test_load_research_run_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "summary.json").write_text('{"selected_probability_source": "raw"}', encoding="utf-8")
            pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2024-08-01"]),
                    "kickoff_time": pd.to_datetime(["2024-08-01 15:00:00"]),
                    "match_id": [1],
                }
            ).to_csv(root / "prediction_rows.csv", index=False)
            pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2024-08-01"]),
                    "kickoff_time": pd.to_datetime(["2024-08-01 15:00:00"]),
                    "decision_time": pd.to_datetime(["2024-08-01 14:15:00"]),
                    "snapshot_time": pd.to_datetime(["2024-08-01 14:10:00"]),
                    "match_id": [1],
                }
            ).to_csv(root / "candidate_rows.csv", index=False)

            summary, predictions, candidate_rows = load_research_run(root)
            self.assertEqual(summary["selected_probability_source"], "raw")
            self.assertEqual(len(predictions), 1)
            self.assertEqual(len(candidate_rows), 1)


if __name__ == "__main__":
    unittest.main()
