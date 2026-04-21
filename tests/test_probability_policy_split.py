from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from predicciones.config import (
    BacktestConfig,
    ExecutionConfig,
    PolicySearchSpace,
    ProjectPaths,
    ResearchConfig,
    Settings,
    SnapshotConfig,
)
from predicciones.research_candidates import choose_probability_source, optimize_research_policy
from predicciones.strategy import score_policy_metrics


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
        research=ResearchConfig(
            min_segment_bets=1,
            min_segment_folds=1,
            positive_fold_ratio=0.5,
            provisional_edge_threshold=0.01,
            provisional_ev_threshold=0.0,
            flat_stake=1.0,
        ),
        backtest=BacktestConfig(
            policy_search=PolicySearchSpace(
                edge_thresholds=(0.01, 0.02),
                ev_thresholds=(0.0,),
                min_odds_options=(1.2,),
                max_odds_options=(5.0,),
                min_bets=1,
                max_kelly_fraction=0.25,
            )
        ),
    )


class ProbabilityPolicySplitTests(unittest.TestCase):
    def test_choose_probability_source_uses_oof_quality_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            predictions = pd.DataFrame(
                {
                    "fold_segment": ["train"],
                    "actual_target": [2],
                    "prob_away_raw": [0.20],
                    "prob_draw_raw": [0.20],
                    "prob_home_raw": [0.60],
                    "prob_away_calibrated": [0.10],
                    "prob_draw_calibrated": [0.20],
                    "prob_home_calibrated": [0.70],
                }
            )
            candidate_rows = pd.DataFrame({"fold_segment": ["train"]})

            raw_metrics = {"roi": 0.25, "profit": 2.5, "wins": 1, "bets": 1, "stake": 1.0, "max_drawdown": 0.0}
            calibrated_metrics = {"roi": -0.10, "profit": -1.0, "wins": 0, "bets": 1, "stake": 1.0, "max_drawdown": 0.2}
            with mock.patch(
                "predicciones.research_candidates._evaluate_policy",
                side_effect=[
                    (pd.DataFrame(), pd.DataFrame(), raw_metrics),
                    (pd.DataFrame(), pd.DataFrame(), calibrated_metrics),
                ],
            ):
                selected_source, decision = choose_probability_source(predictions, candidate_rows, settings)

            self.assertEqual(selected_source, "calibrated")
            self.assertEqual(decision["criterion"], "oof_probability_quality")
            self.assertLess(decision["raw"]["log_loss"], float("inf"))
            self.assertLess(decision["calibrated"]["log_loss"], decision["raw"]["log_loss"])
            self.assertGreater(decision["raw"]["net_strategy"]["roi"], decision["calibrated"]["net_strategy"]["roi"])
            self.assertGreater(decision["comparison"]["log_loss_delta"], 0.0)

    def test_optimize_research_policy_prefers_economic_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            candidate_rows = pd.DataFrame({"fold_segment": ["train"], "match_id": [1]})

            def fake_evaluate_policy(
                candidate_rows: pd.DataFrame,
                policy,
                probability_source: str,
                settings: Settings,
                allow_proxy: bool,
                segment_filter: dict[str, object] | None = None,
            ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
                if policy.edge_threshold == 0.01:
                    execution_rows = pd.DataFrame(
                        {
                            "execution_status": ["executed", "executed"],
                            "fold_id": [1, 2],
                            "net_profit": [1.0, -0.5],
                            "accepted_stake": [1.0, 1.0],
                        }
                    )
                    metrics = {"executed": 2, "roi": 0.30, "profit": 0.50, "stake": 2.0, "max_drawdown": 2.0}
                else:
                    execution_rows = pd.DataFrame(
                        {
                            "execution_status": ["executed", "executed"],
                            "fold_id": [1, 2],
                            "net_profit": [0.2, 0.1],
                            "accepted_stake": [1.0, 1.0],
                        }
                    )
                    metrics = {"executed": 2, "roi": 0.18, "profit": 0.30, "stake": 2.0, "max_drawdown": 0.0}
                return pd.DataFrame(), execution_rows, metrics

            with mock.patch("predicciones.research_candidates._evaluate_policy", side_effect=fake_evaluate_policy):
                tuned_policy = optimize_research_policy(candidate_rows, probability_source="raw", settings=settings)

            self.assertEqual(tuned_policy.edge_threshold, 0.02)
            self.assertGreater(
                score_policy_metrics({"roi": 0.18, "max_drawdown": 0.0, "stake": 2.0}, positive_fold_target=0.5, drawdown_weight=0.2),
                score_policy_metrics({"roi": 0.30, "max_drawdown": 2.0, "stake": 2.0}, positive_fold_target=0.5, drawdown_weight=0.2),
            )


if __name__ == "__main__":
    unittest.main()
