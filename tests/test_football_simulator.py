from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predicciones.config import ProjectPaths, Settings
from predicciones.football.simulator import train_football_simulator


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
        claude_model="test",
        default_leagues=("E0",),
        default_seasons=("2324",),
        benchmark_dir_name="legacy",
    )


def _synthetic_training_dataset(path: Path, include_market_reference: bool = False) -> Path:
    rows: list[dict[str, object]] = []
    for index in range(75):
        split = "train" if index < 45 else "dev" if index < 60 else "locked_holdout"
        rating_diff = float(((index % 15) - 7) * 35)
        home_goals = max(0, int(round(1.35 + rating_diff / 350.0 + (index % 4 == 0))))
        away_goals = max(0, int(round(1.15 - rating_diff / 420.0 + (index % 5 == 0))))
        if home_goals > away_goals:
            outcome = "home"
        elif home_goals < away_goals:
            outcome = "away"
        else:
            outcome = "draw"
        row = {
            "match_id": f"match_{index}",
            "match_date": f"2023-{1 + index // 28:02d}-{1 + index % 28:02d}",
            "split": split,
            "league_code": "E0" if index % 2 == 0 else "SP1",
            "home_team_name": f"Home {index}",
            "away_team_name": f"Away {index}",
            "known_before_match": 1,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "outcome": outcome,
            "total_goals": home_goals + away_goals,
            "btts": int(home_goals > 0 and away_goals > 0),
            "home_matches_played_pre": index % 20,
            "away_matches_played_pre": (index + 3) % 20,
            "home_points_per_match_last_5": 1.4 + rating_diff / 700.0,
            "away_points_per_match_last_5": 1.2 - rating_diff / 750.0,
            "points_form_diff_5": rating_diff / 200.0,
            "home_internal_elo_pre": 1500.0 + rating_diff / 2.0,
            "away_internal_elo_pre": 1500.0 - rating_diff / 2.0,
            "internal_elo_diff": rating_diff,
            "rest_advantage": float((index % 5) - 2),
            "home_clubelo_pre": 1500.0 + rating_diff / 2.2,
            "away_clubelo_pre": 1500.0 - rating_diff / 2.2,
            "clubelo_diff": rating_diff / 1.1,
        }
        if include_market_reference:
            row.update({"odds_home": 2.0, "market_prob_home": 0.5, "market_reference_present": 1})
        rows.append(row)

    dataset = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    feature_columns = [
        "home_matches_played_pre",
        "away_matches_played_pre",
        "home_points_per_match_last_5",
        "away_points_per_match_last_5",
        "points_form_diff_5",
        "home_internal_elo_pre",
        "away_internal_elo_pre",
        "internal_elo_diff",
        "rest_advantage",
        "home_clubelo_pre",
        "away_clubelo_pre",
        "clubelo_diff",
    ]
    if include_market_reference:
        feature_columns.extend(["odds_home", "market_prob_home", "market_reference_present"])
    path.with_name("simulation_training_manifest.json").write_text(
        json.dumps(
            {
                "rows": len(dataset),
                "exclude_market_reference": not include_market_reference,
                "target_columns": ["home_goals", "away_goals", "outcome", "total_goals", "btts"],
                "feature_columns": feature_columns,
                "global_roi_actionable": False,
                "picks_emitidos": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


class FootballSimulatorTests(unittest.TestCase):
    def test_trains_poisson_simulator_and_writes_auditable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            dataset_path = _synthetic_training_dataset(settings.paths.outputs_dir / "sim_data" / "simulation_training_dataset.csv")

            result = train_football_simulator(settings=settings, dataset_path=dataset_path)

            self.assertEqual(result.summary["model_name"], "football_sim_poisson_v1")
            self.assertTrue(result.artifacts["model_bundle"].exists())
            self.assertTrue(result.artifacts["simulation_predictions"].exists())
            self.assertTrue(result.artifacts["simulation_model_report"].exists())
            self.assertTrue((settings.paths.outputs_dir / "latest_football_sim_model.txt").exists())
            predictions = pd.read_csv(result.artifacts["simulation_predictions"])
            raw_sum = predictions[["prob_away_raw", "prob_draw_raw", "prob_home_raw"]].sum(axis=1)
            btts_sum = predictions["prob_btts_yes_raw"] + predictions["prob_btts_no_raw"]
            self.assertTrue(np.allclose(raw_sum, 1.0, atol=1e-6))
            self.assertTrue(np.allclose(btts_sum, 1.0, atol=1e-6))
            bundle = joblib.load(result.artifacts["model_bundle"])
            self.assertEqual(bundle["model_name"], "football_sim_poisson_v1")
            self.assertFalse(bundle["global_roi_actionable"])
            self.assertFalse((settings.paths.data_dir / "polymarket_shadow.sqlite").exists())
            self.assertFalse((settings.paths.data_dir / "polymarket_multi_market.sqlite").exists())

    def test_rejects_market_reference_columns_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            dataset_path = _synthetic_training_dataset(
                settings.paths.outputs_dir / "sim_data" / "simulation_training_dataset.csv",
                include_market_reference=True,
            )

            with self.assertRaisesRegex(ValueError, "market_reference"):
                train_football_simulator(settings=settings, dataset_path=dataset_path)

    def test_missing_dataset_error_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            with self.assertRaisesRegex(FileNotFoundError, "export-sim-training-dataset"):
                train_football_simulator(settings=settings, dataset_path=settings.paths.outputs_dir / "missing.csv")


if __name__ == "__main__":
    unittest.main()
