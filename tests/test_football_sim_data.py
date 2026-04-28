from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from predicciones.config import ProjectPaths, Settings
from predicciones.football.sim_data import (
    FOOTBALL_SIM_FEATURE_FAMILIES,
    FOOTBALL_SIM_GOLD_VERSION,
    SIM_QUARANTINE_SOURCE_IDS,
    _clubelo_candidate_names,
    build_sim_features,
    collect_sim_data_source,
    default_football_sim_db_path,
    export_sim_training_dataset,
    get_sim_data_source_spec,
    normalize_sim_data,
    report_sim_data,
)


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


def _fake_football_data(_league: str, _season: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "01/08/2023",
                "HomeTeam": "Alpha FC",
                "AwayTeam": "Beta FC",
                "FTHG": 2,
                "FTAG": 1,
                "FTR": "H",
                "HS": 11,
                "AS": 8,
                "HST": 5,
                "AST": 3,
                "HC": 6,
                "AC": 4,
                "HY": 1,
                "AY": 2,
                "HR": 0,
                "AR": 0,
                "B365H": 2.0,
                "B365D": 3.2,
                "B365A": 3.8,
                "league_code": "E0",
                "league_name": "Premier League",
                "season": "2324",
                "source_url": "https://example.test/e0.csv",
            },
            {
                "Date": "08/08/2023",
                "HomeTeam": "Beta FC",
                "AwayTeam": "Alpha FC",
                "FTHG": 0,
                "FTAG": 0,
                "FTR": "D",
                "HS": 7,
                "AS": 10,
                "HST": 2,
                "AST": 4,
                "HC": 3,
                "AC": 5,
                "HY": 1,
                "AY": 1,
                "HR": 0,
                "AR": 0,
                "B365H": 2.8,
                "B365D": 3.1,
                "B365A": 2.5,
                "league_code": "E0",
                "league_name": "Premier League",
                "season": "2324",
                "source_url": "https://example.test/e0.csv",
            },
            {
                "Date": "15/08/2023",
                "HomeTeam": "Alpha FC",
                "AwayTeam": "Gamma FC",
                "FTHG": 1,
                "FTAG": 3,
                "FTR": "A",
                "HS": 9,
                "AS": 13,
                "HST": 4,
                "AST": 6,
                "HC": 4,
                "AC": 8,
                "HY": 2,
                "AY": 1,
                "HR": 0,
                "AR": 0,
                "B365H": 1.9,
                "B365D": 3.4,
                "B365A": 4.0,
                "league_code": "E0",
                "league_name": "Premier League",
                "season": "2324",
                "source_url": "https://example.test/e0.csv",
            },
        ]
    )


def _fake_statsbomb_json(path: str):
    if path == "competitions.json":
        return [
            {
                "competition_id": 11,
                "season_id": 90,
                "country_name": "Spain",
                "competition_name": "La Liga",
                "competition_gender": "male",
                "season_name": "2020/2021",
                "match_available": "2025-01-01T00:00:00",
                "match_available_360": None,
            }
        ]
    if path == "matches/11/90.json":
        return [
            {
                "match_id": 1001,
                "match_date": "2023-08-01",
                "home_team": {"home_team_name": "Alpha FC"},
                "away_team": {"away_team_name": "Beta FC"},
                "home_score": 2,
                "away_score": 1,
            }
        ]
    if path == "lineups/1001.json":
        return [
            {
                "team_name": "Alpha FC",
                "lineup": [{"player_id": 10, "player_name": "Alpha Striker"}],
            }
        ]
    if path == "events/1001.json":
        return [
            {"id": "e1", "team": {"name": "Alpha FC"}, "player": {"id": 10, "name": "Alpha Striker"}, "type": {"name": "Shot"}, "minute": 5},
            {"id": "e2", "team": {"name": "Beta FC"}, "type": {"name": "Pass"}, "minute": 8},
        ]
    raise FileNotFoundError(path)


def _fake_clubelo(team_name: str) -> pd.DataFrame:
    elo = {"Alpha FC": 1600.0, "Beta FC": 1500.0, "Gamma FC": 1525.0}.get(team_name, 1450.0)
    return pd.DataFrame(
        [
            {"From": "2023-07-01", "To": "2023-08-31", "Club": team_name, "Elo": elo},
            {"From": "2023-09-01", "To": "2023-10-01", "Club": team_name, "Elo": elo + 10},
        ]
    )


def _fake_openfootball_json(_path: str):
    return {
        "name": "Premier League",
        "matches": [
            {"date": "2023-08-01", "team1": "Alpha FC", "team2": "Beta FC", "score": {"ft": [2, 1]}},
            {"date": "2023-08-10", "team1": "Delta FC", "team2": "Gamma FC", "score": {"ft": [1, 1]}},
        ],
    }


class FootballSimDataTests(unittest.TestCase):
    def test_registry_rejects_unknown_source(self) -> None:
        self.assertEqual(get_sim_data_source_spec("football_data").source_id, "football_data")
        with self.assertRaises(ValueError):
            get_sim_data_source_spec("not_declared")

    def test_clubelo_candidates_handle_football_data_aliases(self) -> None:
        self.assertEqual(_clubelo_candidate_names("Nott'm Forest")[0], "Forest")
        self.assertIn("NottmForest", _clubelo_candidate_names("Nott'm Forest"))
        self.assertIn("NottinghamForest", _clubelo_candidate_names("Nott'm Forest"))
        self.assertEqual(_clubelo_candidate_names("Man United")[0], "ManUnited")
        self.assertEqual(_clubelo_candidate_names("Bayern Munich")[0], "Bayern")
        self.assertEqual(_clubelo_candidate_names("Ath Madrid")[0], "Atletico")
        self.assertEqual(_clubelo_candidate_names("AZ Alkmaar")[0], "Alkmaar")
        self.assertEqual(_clubelo_candidate_names("For Sittard")[0], "Sittard")
        self.assertEqual(_clubelo_candidate_names("St Etienne")[0], "Saint-Etienne")
        self.assertEqual(_clubelo_candidate_names("NAC Breda")[0], "Breda")
        self.assertEqual(_clubelo_candidate_names("VVV Venlo")[0], "Venlo")

    def test_collect_stores_raw_payload_hash_source_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_football_sim_db_path(settings)
            summary, artifacts = collect_sim_data_source(
                settings=settings,
                source_id="football_data",
                leagues=("E0",),
                seasons=("2324",),
                db_path=db_path,
                football_data_loader=_fake_football_data,
            )

            self.assertEqual(summary["raw_payloads_inserted"], 1)
            self.assertTrue(artifacts["source_coverage_report"].exists())
            connection = sqlite3.connect(db_path)
            row = connection.execute(
                "SELECT content_hash, source_id, fetched_at, row_count FROM sim_raw_payloads"
            ).fetchone()
            connection.close()
            self.assertEqual(row[1], "football_data")
            self.assertEqual(row[3], 3)
            self.assertGreater(len(row[0]), 20)
            self.assertIn("T", row[2])

            connection = sqlite3.connect(db_path)
            snapshots = connection.execute(
                "SELECT source_id, payload_count, row_count, license_status, quarantine FROM sim_source_snapshots"
            ).fetchall()
            connection.close()
            by_source = {row[0]: row for row in snapshots}
            self.assertEqual(by_source["football_data"][1], 1)
            self.assertEqual(by_source["football_data"][2], 3)
            self.assertIn("registered_free_public", by_source["football_data"][3])
            self.assertEqual(by_source["fbref"][4], 1)
            self.assertEqual(by_source["understat"][4], 1)

    def test_normalize_builds_stable_canonical_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_football_sim_db_path(settings)
            collect_sim_data_source(
                settings=settings,
                source_id="football_data",
                leagues=("E0",),
                seasons=("2324",),
                db_path=db_path,
                football_data_loader=_fake_football_data,
            )
            normalize_sim_data(settings=settings, db_path=db_path)
            first = report_sim_data(settings=settings, db_path=db_path)[0]
            normalize_sim_data(settings=settings, db_path=db_path)
            second = report_sim_data(settings=settings, db_path=db_path)[0]

            self.assertEqual(first["teams"], 3)
            self.assertEqual(first["matches"], 3)
            self.assertEqual(first["teams"], second["teams"])
            self.assertEqual(first["matches"], second["matches"])

    def test_build_features_is_pre_match_and_blocks_unknown_lineups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_football_sim_db_path(settings)
            collect_sim_data_source(
                settings=settings,
                source_id="football_data",
                leagues=("E0",),
                seasons=("2324",),
                db_path=db_path,
                football_data_loader=_fake_football_data,
            )
            normalize_sim_data(settings=settings, db_path=db_path)
            connection = sqlite3.connect(db_path)
            match_id, team_id = connection.execute(
                "SELECT match_id, home_team_id FROM sim_matches ORDER BY match_date LIMIT 1"
            ).fetchone()
            connection.execute(
                """
                INSERT INTO sim_lineups (
                    lineup_id, match_id, team_id, player_id, player_name, known_before_match, source_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("lineup_1", match_id, team_id, "player_1", "Hidden Player", 0, "test"),
            )
            connection.commit()
            connection.close()

            summary, artifacts = build_sim_features(settings=settings, db_path=db_path)
            self.assertEqual(summary["gold_feature_rows"], 3)
            self.assertEqual(summary["gold_version"], FOOTBALL_SIM_GOLD_VERSION)
            self.assertEqual(summary["team_season_rows"], 3)
            self.assertEqual(summary["match_state_rows"], 3)

            leakage = json.loads(artifacts["leakage_audit_report"].read_text(encoding="utf-8"))
            manifest = json.loads(artifacts["simulation_feature_manifest"].read_text(encoding="utf-8"))
            gold_manifest = json.loads(artifacts["football_sim_gold_v1_manifest"].read_text(encoding="utf-8"))
            self.assertEqual(leakage["leakage_violations"], 0)
            self.assertEqual(leakage["match_state_leakage_violations"], 0)
            self.assertEqual(manifest["gold_version"], FOOTBALL_SIM_GOLD_VERSION)
            self.assertEqual(manifest["required_feature_families"], list(FOOTBALL_SIM_FEATURE_FAMILIES))
            self.assertEqual(manifest["families"]["player_availability"]["status"], "blocked_lineup_coverage_low")
            self.assertFalse(gold_manifest["contracts"]["market_reference_training_enabled"])
            self.assertFalse(gold_manifest["contracts"]["synthetic_data_counts_as_roi_evidence"])
            self.assertFalse((settings.paths.data_dir / "polymarket_shadow.sqlite").exists())
            self.assertFalse((settings.paths.data_dir / "polymarket_multi_market.sqlite").exists())

            connection = sqlite3.connect(db_path)
            state_columns = [row[1] for row in connection.execute("PRAGMA table_info(sim_match_state_features)").fetchall()]
            state = connection.execute(
                """
                SELECT feature_family_set, home_goals, away_goals, total_goals, btts
                FROM sim_match_state_features ORDER BY match_start_time LIMIT 1
                """
            ).fetchone()
            team_season = connection.execute(
                """
                SELECT matches_played, home_matches, away_matches, dataset_role
                FROM sim_team_season_features
                WHERE lower(team_name) = 'alpha fc'
                """
            ).fetchone()
            connection.close()
            self.assertNotIn("odds_home", state_columns)
            self.assertIn("team_form", json.loads(state[0]))
            self.assertEqual(state[3], state[1] + state[2])
            self.assertEqual(state[4], 1)
            self.assertEqual(team_season[0], 3)
            self.assertEqual(team_season[1], 2)
            self.assertEqual(team_season[2], 1)
            self.assertEqual(team_season[3], "observed_team_season_summary_not_pre_match_feature")

    def test_max_free_profile_records_failures_without_aborting(self) -> None:
        def loader(league: str, season: str) -> pd.DataFrame:
            if league == "E0":
                return _fake_football_data(league, season)
            raise RuntimeError("not_available")

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            summary, _ = collect_sim_data_source(
                settings=settings,
                source_id="football_data",
                leagues=(),
                seasons=("2324",),
                db_path=default_football_sim_db_path(settings),
                football_data_loader=loader,
                profile="max_free_v1",
                probe_missing=True,
            )

            self.assertGreaterEqual(summary["raw_payloads_inserted"], 1)
            self.assertGreater(len(summary["failures"]), 0)

    def test_statsbomb_normalizes_competitions_lineups_and_events_without_known_lineup_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_football_sim_db_path(settings)
            collect_sim_data_source(
                settings=settings,
                source_id="statsbomb_open_data",
                leagues=(),
                seasons=(),
                db_path=db_path,
                statsbomb_json_loader=_fake_statsbomb_json,
                max_statsbomb_matches=1,
            )
            normalize_sim_data(settings=settings, db_path=db_path)
            summary = report_sim_data(settings=settings, db_path=db_path)[0]

            self.assertEqual(summary["lineups"], 1)
            self.assertEqual(summary["events"], 2)
            connection = sqlite3.connect(db_path)
            known = connection.execute("SELECT known_before_match FROM sim_lineups").fetchone()[0]
            connection.close()
            self.assertEqual(known, 0)

    def test_openfootball_is_auxiliary_alias_source_not_gold_match_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_football_sim_db_path(settings)
            collect_sim_data_source(
                settings=settings,
                source_id="openfootball",
                leagues=("E0",),
                seasons=("2324",),
                db_path=db_path,
                openfootball_json_loader=_fake_openfootball_json,
            )
            normalize_sim_data(settings=settings, db_path=db_path)

            connection = sqlite3.connect(db_path)
            aliases = connection.execute(
                "SELECT COUNT(*) FROM sim_team_aliases WHERE source_id = 'openfootball'"
            ).fetchone()[0]
            matches = connection.execute(
                "SELECT COUNT(*) FROM sim_matches WHERE source_id = 'openfootball'"
            ).fetchone()[0]
            snapshot = connection.execute(
                "SELECT license_status, payload_count FROM sim_source_snapshots WHERE source_id = 'openfootball'"
            ).fetchone()
            connection.close()
            self.assertEqual(aliases, 4)
            self.assertEqual(matches, 0)
            self.assertIn("registered_cc0_public", snapshot[0])
            self.assertEqual(snapshot[1], 1)

    def test_clubelo_rating_is_pre_match_and_falls_back_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_football_sim_db_path(settings)
            collect_sim_data_source(
                settings=settings,
                source_id="football_data",
                leagues=("E0",),
                seasons=("2324",),
                db_path=db_path,
                football_data_loader=_fake_football_data,
            )
            normalize_sim_data(settings=settings, db_path=db_path)
            collect_sim_data_source(
                settings=settings,
                source_id="clubelo",
                leagues=(),
                seasons=(),
                db_path=db_path,
                clubelo_loader=_fake_clubelo,
            )
            normalize_sim_data(settings=settings, db_path=db_path)
            second_collect, _ = collect_sim_data_source(
                settings=settings,
                source_id="clubelo",
                leagues=(),
                seasons=(),
                db_path=db_path,
                clubelo_loader=_fake_clubelo,
            )
            _, artifacts = build_sim_features(settings=settings, db_path=db_path)

            manifest = json.loads(artifacts["simulation_feature_manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["families"]["clubelo"]["status"], "ready")
            connection = sqlite3.connect(db_path)
            first = connection.execute(
                """
                SELECT home_clubelo_pre, away_clubelo_pre, clubelo_mapping_status
                FROM sim_gold_features ORDER BY match_start_time LIMIT 1
                """
            ).fetchone()
            connection.close()
            self.assertEqual(first[0], 1600.0)
            self.assertEqual(first[1], 1500.0)
            self.assertEqual(first[2], "active")
            self.assertEqual(second_collect["raw_payloads_inserted"], 0)

    def test_export_training_dataset_excludes_market_reference_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_football_sim_db_path(settings)
            collect_sim_data_source(
                settings=settings,
                source_id="football_data",
                leagues=("E0",),
                seasons=("2324",),
                db_path=db_path,
                football_data_loader=_fake_football_data,
            )
            normalize_sim_data(settings=settings, db_path=db_path)
            build_sim_features(settings=settings, db_path=db_path)
            summary, artifacts = export_sim_training_dataset(settings=settings, db_path=db_path)

            dataset = pd.read_csv(artifacts["simulation_training_dataset"])
            team_seasons = pd.read_csv(artifacts["football_sim_gold_v1_team_seasons"])
            match_state = pd.read_csv(artifacts["football_sim_gold_v1_match_state"])
            manifest = json.loads(artifacts["simulation_training_manifest"].read_text(encoding="utf-8"))
            self.assertEqual(summary["training_rows"], 3)
            self.assertIn("total_goals", dataset.columns)
            self.assertIn("split", dataset.columns)
            self.assertNotIn("odds_home", dataset.columns)
            self.assertNotIn("market_prob_home", dataset.columns)
            self.assertEqual(manifest["gold_version"], FOOTBALL_SIM_GOLD_VERSION)
            self.assertEqual(manifest["required_feature_families"], list(FOOTBALL_SIM_FEATURE_FAMILIES))
            self.assertFalse(manifest["synthetic_data_counts_as_roi_evidence"])
            self.assertEqual(len(team_seasons), 3)
            self.assertEqual(len(match_state), 3)

    def test_quarantine_sources_are_manifested_but_excluded_from_gold_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_football_sim_db_path(settings)
            summary, artifacts = collect_sim_data_source(
                settings=settings,
                source_id="football_data",
                leagues=("E0",),
                seasons=("2324",),
                db_path=db_path,
                football_data_loader=_fake_football_data,
            )

            self.assertGreaterEqual(summary["source_snapshots"], len(SIM_QUARANTINE_SOURCE_IDS))
            manifest = json.loads(artifacts["football_sim_gold_v1_manifest"].read_text(encoding="utf-8"))
            source_manifest = json.loads(artifacts["source_license_manifest"].read_text(encoding="utf-8"))
            quarantined = {
                row["source_id"]
                for row in source_manifest["snapshots"]
                if row["quarantine"]
            }
            self.assertEqual(quarantined, set(SIM_QUARANTINE_SOURCE_IDS))
            self.assertTrue(manifest["contracts"]["quarantine_sources_excluded_from_gold"])
            self.assertTrue(set(SIM_QUARANTINE_SOURCE_IDS).isdisjoint(set(source_manifest["gold_allowed_sources"])))

    def test_clubelo_out_of_scope_leagues_are_reported_as_source_scope(self) -> None:
        def mex_loader(league: str, season: str) -> pd.DataFrame:
            frame = _fake_football_data(league, season).copy()
            frame["league_code"] = "MEX"
            frame["league_name"] = "Liga MX"
            return frame

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_football_sim_db_path(settings)
            collect_sim_data_source(
                settings=settings,
                source_id="football_data",
                leagues=("MEX",),
                seasons=("2324",),
                db_path=db_path,
                football_data_loader=mex_loader,
            )
            normalize_sim_data(settings=settings, db_path=db_path)
            _, artifacts = build_sim_features(settings=settings, db_path=db_path)

            quality = json.loads(artifacts["simulation_quality_report"].read_text(encoding="utf-8"))
            self.assertEqual(quality["counts_by_type"], {"external_rating_source_scope": 1})
            self.assertEqual(quality["counts_by_severity"], {"info": 1})
            self.assertTrue(quality["issues"][0]["raw"]["source_scope_gap"])


if __name__ == "__main__":
    unittest.main()
