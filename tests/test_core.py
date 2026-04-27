from __future__ import annotations

import warnings
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from predicciones.backtest import rolling_origin_windows
from predicciones.config import BacktestConfig
from predicciones.contracts import OUTCOME_ORDER
from predicciones.data_sources import FootballDataClient
from predicciones.football.dataset import model_feature_columns
from predicciones.ingestion import build_market_odds, canonicalize_matches
from predicciones.models import build_goal_model, fit_goal_model, outcome_probabilities_from_lambdas, score_matrix_from_lambdas
from predicciones.strategy import BetPolicy, add_edge_columns, select_bets


class CorePrediccionesTests(unittest.TestCase):
    def test_build_market_odds_removes_margin(self) -> None:
        matches = pd.DataFrame(
            {
                "match_id": [0],
                "Date": pd.to_datetime(["2024-08-01"]),
                "league_code": ["E0"],
                "season": ["2425"],
                "HomeTeam": ["A"],
                "AwayTeam": ["B"],
                "B365H": [1.9],
                "B365D": [3.8],
                "B365A": [4.4],
            }
        )
        market = build_market_odds(matches)
        total = market.loc[0, ["market_prob_home", "market_prob_draw", "market_prob_away"]].sum()
        self.assertAlmostEqual(float(total), 1.0, places=8)
        self.assertGreater(float(market.loc[0, "bookmaker_margin"]), 0.0)

    def test_football_data_extra_league_schema_is_normalized(self) -> None:
        raw = pd.DataFrame(
            {
                "Season": ["2025/2026", "2024/2025"],
                "Date": ["21/04/2026", "21/04/2025"],
                "Home": ["UNAM Pumas", "Atlas"],
                "Away": ["Juarez", "Puebla"],
                "HG": [2, 1],
                "AG": [1, 1],
                "Res": ["H", "D"],
                "B365CH": [2.1, 2.2],
                "B365CD": [3.4, 3.2],
                "B365CA": [3.8, 3.5],
            }
        )
        with patch("predicciones.data_sources.pd.read_csv", return_value=raw):
            frame = FootballDataClient().load_one("MEX", "2526")
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "HomeTeam"], "UNAM Pumas")
        self.assertEqual(frame.loc[0, "AwayTeam"], "Juarez")
        self.assertEqual(frame.loc[0, "FTHG"], 2)
        self.assertEqual(frame.loc[0, "FTR"], "H")
        self.assertEqual(frame.loc[0, "B365H"], 2.1)
        self.assertEqual(frame.loc[0, "league_code"], "MEX")

    def test_poisson_outcome_probabilities_sum_to_one(self) -> None:
        matrix = score_matrix_from_lambdas(1.4, 1.1, rho=-0.05, max_goals=10)
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=8)
        probs = outcome_probabilities_from_lambdas(np.array([1.4]), np.array([1.1]), rho=-0.05, max_goals=10)
        self.assertAlmostEqual(float(probs[0].sum()), 1.0, places=8)

    def test_select_bets_uses_edge_thresholds(self) -> None:
        frame = pd.DataFrame(
            {
                "match_id": [1],
                "Date": pd.to_datetime(["2024-08-01"]),
                "league_code": ["E0"],
                "league_name": ["Premier League"],
                "season": ["2425"],
                "HomeTeam": ["A"],
                "AwayTeam": ["B"],
                "actual_outcome": ["home"],
                "odds_home": [2.5],
                "odds_draw": [3.0],
                "odds_away": [3.2],
                "market_prob_home": [0.35],
                "market_prob_draw": [0.30],
                "market_prob_away": [0.35],
                "prob_away_calibrated": [0.20],
                "prob_draw_calibrated": [0.28],
                "prob_home_calibrated": [0.52],
            }
        )
        frame = add_edge_columns(frame)
        bets = select_bets(frame, BetPolicy(edge_threshold=0.05, ev_threshold=0.05, min_odds=1.2, max_odds=5.0))
        self.assertEqual(len(bets), 1)
        self.assertEqual(bets.iloc[0]["selection"], "home")

    def test_rolling_origin_split_has_no_same_day_leakage(self) -> None:
        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        dataset = pd.DataFrame(
            {
                "match_id": np.arange(500),
                "Date": dates,
                "league_code": ["E0"] * 500,
            }
        )
        windows = rolling_origin_windows(
            dataset,
            BacktestConfig(min_train_matches=100, min_train_days=100, test_window_days=28),
        )
        self.assertGreater(len(windows), 0)
        for window in windows:
            self.assertLess(window.train_end, window.test_start)

    def test_model_feature_columns_excludes_targets(self) -> None:
        frame = pd.DataFrame(
            {
                "match_id": [1],
                "Date": pd.to_datetime(["2024-08-01"]),
                "league_code": ["E0"],
                "league_name": ["Premier League"],
                "season": ["2425"],
                "HomeTeam": ["A"],
                "AwayTeam": ["B"],
                "home_goals": [1.0],
                "away_goals": [0.0],
                "target": [2],
                "elo_diff": [10.0],
            }
        )
        self.assertEqual(model_feature_columns(frame), ["league_code", "elo_diff"])

    def test_goal_model_sanitizes_fully_empty_columns_without_warnings(self) -> None:
        train_rows = pd.DataFrame(
            {
                "league_code": ["E0"] * 60,
                "usable_feature": np.linspace(0.0, 1.0, 60),
                "empty_feature": [np.nan] * 60,
            }
        )
        home_goals = pd.Series(np.linspace(0.0, 2.0, 60))
        away_goals = pd.Series(np.linspace(1.0, 0.0, 60))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bundle = build_goal_model(train_rows)
            fit_goal_model(bundle, train_rows, home_goals, away_goals)

        self.assertEqual(caught, [])
        self.assertEqual(bundle.feature_columns, ["league_code", "usable_feature"])
        self.assertEqual(bundle.dropped_feature_columns, ["empty_feature"])
        self.assertEqual(bundle.dropped_feature_reasons["empty_feature"], "all_missing")

        prediction_rows = pd.DataFrame(
            {
                "league_code": ["E0", "E0"],
                "usable_feature": [0.25, 0.75],
                "empty_feature": [123.0, -999.0],
            }
        )
        baseline_rows = prediction_rows.copy()
        baseline_rows["empty_feature"] = np.nan

        prediction_home, prediction_away = bundle.predict_lambdas(prediction_rows)
        baseline_home, baseline_away = bundle.predict_lambdas(baseline_rows)

        np.testing.assert_allclose(prediction_home, baseline_home)
        np.testing.assert_allclose(prediction_away, baseline_away)

    def test_canonicalize_matches_infers_season_without_result(self) -> None:
        fixtures = pd.DataFrame(
            {
                "Date": ["2025-08-15"],
                "league_code": ["E0"],
                "HomeTeam": ["Team A"],
                "AwayTeam": ["Team B"],
            }
        )
        canonical = canonicalize_matches(fixtures, require_results=False)
        self.assertEqual(canonical.iloc[0]["season"], "2526")

if __name__ == "__main__":
    unittest.main()
