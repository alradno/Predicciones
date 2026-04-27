from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from predicciones.backtest import run_backtest
from predicciones.config import BacktestConfig, ProjectPaths, Settings
from predicciones.contracts import RunContext
from predicciones.football.dataset import build_feature_rows
from predicciones.ingestion import build_market_odds, canonicalize_matches


class IntegrationPrediccionesTests(unittest.TestCase):
    def test_small_backtest_runs_end_to_end(self) -> None:
        rows = []
        outcomes = ["H", "A", "D"]
        for idx, date in enumerate(pd.date_range("2023-01-01", periods=180, freq="D")):
            rows.append(
                {
                    "Date": date.strftime("%d/%m/%Y"),
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "season": "2324" if date.year == 2023 else "2425",
                    "HomeTeam": f"Home{idx % 10}",
                    "AwayTeam": f"Away{(idx + 3) % 10}",
                    "FTHG": float(idx % 4),
                    "FTAG": float((idx + 1) % 3),
                    "FTR": outcomes[idx % len(outcomes)],
                    "HS": float(10 + (idx % 8)),
                    "AS": float(8 + (idx % 7)),
                    "HST": float(4 + (idx % 4)),
                    "AST": float(3 + (idx % 3)),
                    "HC": float(5 + (idx % 5)),
                    "AC": float(4 + (idx % 4)),
                    "HY": float(idx % 4),
                    "AY": float((idx + 1) % 4),
                    "HR": 0.0,
                    "AR": 0.0,
                    "B365H": 2.1 + ((idx % 5) * 0.1),
                    "B365D": 3.2 + ((idx % 4) * 0.1),
                    "B365A": 2.4 + ((idx % 6) * 0.1),
                }
            )

        canonical = canonicalize_matches(pd.DataFrame(rows), require_results=True)
        feature_rows = build_feature_rows(canonical, rolling_window=5)
        market = build_market_odds(canonical)
        dataset = feature_rows.merge(market, on=["match_id", "Date", "league_code", "season", "HomeTeam", "AwayTeam"], how="left")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = RunContext(run_id="test_run", run_dir=root)
            result = run_backtest(
                dataset=dataset,
                config=BacktestConfig(
                    rolling_window=5,
                    min_train_matches=60,
                    min_train_days=60,
                    calibration_matches=30,
                    policy_matches=30,
                    min_subtrain_matches=60,
                    test_window_days=14,
                ),
                run=run,
            )
        self.assertGreater(len(result.predictions), 0)
        self.assertIn("strategy", result.summary)


if __name__ == "__main__":
    unittest.main()
