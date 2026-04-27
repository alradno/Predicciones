from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predicciones.config import BacktestConfig, ExecutionConfig, PolymarketConfig, ProjectPaths, ResearchConfig, Settings, SnapshotConfig
from predicciones.lanes.runtime import (
    ACTIVE_FAMILIES,
    GLOBAL_FOOTBALL_1X2_LEAGUES,
    MULTI_MARKET_DATABASE_FILENAME,
    _build_raw_capture_plan,
    _fixture_rows_from_template,
    _lane_sample_status,
    _league_code_from_market_row,
    build_market_lane_predictions,
    classify_market,
    capture_multi_market,
    create_market_lane_policy,
    default_multi_market_db_path,
    discover_multi_market,
    get_market_family_spec,
    get_market_lane_spec,
    market_family_manifest,
    market_family_registry,
    market_lane_registry,
    report_multi_market,
    report_market_lane,
    report_sport_merge,
    run_market_lane,
)


class _FakeLaneGoalModel:
    feature_columns: list[str] = []

    def predict_lambdas(self, feature_rows: pd.DataFrame):
        return np.full(len(feature_rows), 2.2), np.full(len(feature_rows), 0.7)


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
        default_seasons=("2526",),
        benchmark_dir_name="legacy_v1",
        snapshot=SnapshotConfig(),
        execution=ExecutionConfig(),
        research=ResearchConfig(),
        polymarket=PolymarketConfig(checkpoint_interval_seconds=1),
        backtest=BacktestConfig(),
    )


def _lane_history_matches() -> pd.DataFrame:
    rows = []
    teams = ("Arsenal", "Chelsea")
    for index in range(12):
        home = teams[index % 2]
        away = teams[(index + 1) % 2]
        home_goals = 2 if home == "Arsenal" else 1
        away_goals = 0 if home == "Arsenal" else 1
        rows.append(
            {
                "match_id": index + 1,
                "Date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=index * 7),
                "league_code": "E0",
                "league_name": "Premier League",
                "season": "2425",
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": home_goals,
                "FTAG": away_goals,
                "outcome": "home" if home_goals > away_goals else "draw" if home_goals == away_goals else "away",
            }
        )
    return pd.DataFrame(rows)


def _write_fake_lane_model(settings: Settings) -> Path:
    path = settings.paths.models_dir / "fake_lane_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": _FakeLaneGoalModel(),
            "history_matches": _lane_history_matches(),
            "feature_columns": [],
            "rolling_window": 3,
            "rho": 0.0,
            "max_poisson_goals": 8,
            "model_variant": "fake_lane_goal_model",
        },
        path,
    )
    return path


def _write_frozen_football_policy(settings: Settings) -> Path:
    source_dir = settings.paths.runs_dir / "frozen_policy"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_bundle = source_dir / "policy_bundle.json"
    source_bundle.write_text(
        json.dumps(
            {
                "probability_source": "raw",
                "bundle_status": "frozen",
                "policy": {
                    "edge_threshold": 0.02,
                    "ev_threshold": 0.0,
                    "min_odds": 1.2,
                    "max_odds": 4.0,
                    "kelly_fraction": 0.25,
                    "family": "edge_ev_threshold",
                    "allowed_outcomes": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (settings.paths.outputs_dir / "latest_polymarket_policy.txt").write_text(str(source_bundle), encoding="utf-8")
    return source_bundle


def _write_football_1x2_lane_predictions(settings: Settings) -> None:
    lane_dir = settings.paths.outputs_dir / "lanes" / "football_1x2_global"
    lane_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "event_slug": "premier-league-arsenal-chelsea-2026-04-21",
                "selection": "home",
                "model_prob": 0.60,
                "probability_source": "raw",
            },
            {
                "event_slug": "premier-league-arsenal-chelsea-2026-04-21",
                "selection": "draw",
                "model_prob": 0.20,
                "probability_source": "raw",
            },
            {
                "event_slug": "premier-league-arsenal-chelsea-2026-04-21",
                "selection": "away",
                "model_prob": 0.30,
                "probability_source": "raw",
            },
        ]
    ).to_csv(lane_dir / "model_predictions.csv", index=False)


class _FakeGamma:
    def __init__(self) -> None:
        self.events_by_query = {
            "Premier League": [
                {
                    "id": "ev-football",
                    "slug": "premier-league-arsenal-chelsea-2026-04-21",
                    "title": "Premier League: Arsenal vs Chelsea",
                    "markets": [
                        {
                            "id": "m-football-home",
                            "slug": "arsenal-win",
                            "question": "Will Arsenal win against Chelsea?",
                            "sportsMarketType": "moneyline",
                            "gameStartTime": "2026-04-21T18:00:00Z",
                            "outcomes": '["Yes", "No"]',
                            "outcomePrices": '["0.55", "0.45"]',
                            "clobTokenIds": '["101", "102"]',
                            "active": True,
                            "closed": False,
                            "acceptingOrders": True,
                        },
                        {
                            "id": "m-football-away",
                            "slug": "chelsea-win",
                            "question": "Will Chelsea win against Arsenal?",
                            "sportsMarketType": "moneyline",
                            "gameStartTime": "2026-04-21T18:00:00Z",
                            "outcomes": '["Yes", "No"]',
                            "outcomePrices": '["0.30", "0.70"]',
                            "clobTokenIds": '["103", "104"]',
                            "active": True,
                            "closed": False,
                            "acceptingOrders": True,
                        },
                        {
                            "id": "m-football-draw",
                            "slug": "arsenal-chelsea-draw",
                            "question": "Will Arsenal vs Chelsea end in a draw?",
                            "sportsMarketType": "moneyline",
                            "gameStartTime": "2026-04-21T18:00:00Z",
                            "outcomes": '["Yes", "No"]',
                            "outcomePrices": '["0.25", "0.75"]',
                            "clobTokenIds": '["105", "106"]',
                            "active": True,
                            "closed": False,
                            "acceptingOrders": True,
                        },
                        {
                            "id": "m-football-goals",
                            "slug": "arsenal-chelsea-over-25",
                            "question": "Will Arsenal vs Chelsea have over 2.5 total goals?",
                            "sportsMarketType": "total",
                            "gameStartTime": "2026-04-21T18:00:00Z",
                            "outcomes": '["Yes", "No"]',
                            "outcomePrices": '["0.48", "0.52"]',
                            "clobTokenIds": '["201", "202"]',
                            "active": True,
                            "closed": False,
                            "acceptingOrders": True,
                        },
                        {
                            "id": "m-football-cards",
                            "slug": "arsenal-chelsea-cards",
                            "question": "Will there be over 4.5 yellow cards?",
                            "sportsMarketType": "total",
                            "gameStartTime": "2026-04-21T18:00:00Z",
                            "outcomes": '["Yes", "No"]',
                            "clobTokenIds": '["301", "302"]',
                            "active": True,
                            "closed": False,
                            "acceptingOrders": True,
                        },
                    ],
                }
            ],
            "Tennis": [
                {
                    "id": "ev-tennis",
                    "slug": "atp-tennis-alcaraz-sinner-2026-04-21",
                    "title": "ATP Tennis: Alcaraz vs Sinner",
                    "markets": [
                        {
                            "id": "m-tennis",
                            "slug": "alcaraz-win",
                            "question": "Will Carlos Alcaraz win the tennis match?",
                            "sportsMarketType": "moneyline",
                            "gameStartTime": "2026-04-21T16:00:00Z",
                            "outcomes": '["Yes", "No"]',
                            "outcomePrices": '["0.51", "0.49"]',
                            "clobTokenIds": '["401", "402"]',
                            "active": True,
                            "closed": False,
                            "acceptingOrders": True,
                        }
                    ],
                }
            ],
            "NBA": [
                {
                    "id": "ev-nba",
                    "slug": "nba-lakers-celtics-2026-04-21",
                    "title": "NBA: Lakers vs Celtics",
                    "markets": [
                        {
                            "id": "m-nba-total",
                            "slug": "lakers-celtics-total-points",
                            "question": "Will Lakers vs Celtics have over 220.5 total points?",
                            "sportsMarketType": "total",
                            "gameStartTime": "2026-04-21T22:00:00Z",
                            "outcomes": '["Yes", "No"]',
                            "clobTokenIds": '["501", "502"]',
                            "active": True,
                            "closed": False,
                            "acceptingOrders": True,
                        }
                    ],
                }
            ],
        }

    def search_events(self, query: str, limit: int = 50):
        return self.events_by_query.get(query, [])[:limit]


class _FakeClob:
    def get_order_book(self, token_id: str):
        return {
            "asset_id": str(token_id),
            "asks": [{"price": "0.55", "size": "10"}, {"price": "0.57", "size": "5"}],
            "bids": [{"price": "0.53", "size": "9"}],
        }


class _HistoricalOnlyGamma:
    def search_events(self, query: str, limit: int = 50):
        if query != "Premier League":
            return []
        return [
            {
                "id": "ev-old-football",
                "slug": "premier-league-old-2023",
                "title": "Premier League: Old Team vs Past Team",
                "markets": [
                    {
                        "id": "m-old-football-home",
                        "slug": "old-team-win",
                        "question": "Will Old Team win against Past Team?",
                        "sportsMarketType": "moneyline",
                        "gameStartTime": "2023-08-27T18:00:00Z",
                        "outcomes": '["Yes", "No"]',
                        "outcomePrices": '["0.55", "0.45"]',
                        "clobTokenIds": '["901", "902"]',
                        "active": True,
                        "closed": False,
                        "acceptingOrders": True,
                    }
                ],
            }
        ][:limit]


class _SportsSeriesGamma:
    def search_events(self, query: str, limit: int = 50):
        return []

    def list_sports(self):
        return [
            {"sport": "epl", "series": "10188"},
            {"sport": "nba", "series": "10345"},
        ]

    def list_events(self, limit: int = 50, **params):
        if str(params.get("series_id")) == "10188":
            return [
                {
                    "id": "ev-epl-future",
                    "slug": "epl-mancity-palace-2026-05-22",
                    "title": "Manchester City FC vs. Crystal Palace FC",
                    "startTime": "2026-05-22T14:00:00Z",
                    "markets": [
                        {
                            "id": "m-epl-future-home",
                            "slug": "epl-mancity-palace-home",
                            "question": "Will Manchester City FC win on 2026-05-22?",
                            "sportsMarketType": "moneyline",
                            "gameStartTime": "2026-05-22 14:00:00+00",
                            "outcomes": '["Yes", "No"]',
                            "outcomePrices": '["0.55", "0.45"]',
                            "clobTokenIds": '["1001", "1002"]',
                            "active": True,
                            "closed": False,
                            "acceptingOrders": True,
                        },
                        {
                            "id": "m-epl-future-draw",
                            "slug": "epl-mancity-palace-draw",
                            "question": "Will Manchester City FC vs. Crystal Palace FC end in a draw?",
                            "sportsMarketType": "moneyline",
                            "gameStartTime": "2026-05-22 14:00:00+00",
                            "outcomes": '["Yes", "No"]',
                            "outcomePrices": '["0.25", "0.75"]',
                            "clobTokenIds": '["1003", "1004"]',
                            "active": True,
                            "closed": False,
                            "acceptingOrders": True,
                        },
                    ],
                }
            ][:limit]
        return []


class MultiMarketTests(unittest.TestCase):
    def test_registry_declares_active_and_deferred_families(self) -> None:
        registry = market_family_registry()
        self.assertTrue(set(ACTIVE_FAMILIES).issubset(registry))
        self.assertEqual(get_market_family_spec("football_1x2_global").outcome_schema, ("home", "draw", "away"))
        self.assertEqual(get_market_family_spec("football_cards").initial_state, "deferred")
        with self.assertRaises(KeyError):
            get_market_family_spec("unregistered_market_family")

        manifest = market_family_manifest()
        self.assertFalse(manifest["global_roi_actionable"])
        self.assertEqual(manifest["database_filename"], MULTI_MARKET_DATABASE_FILENAME)

    def test_lane_registry_includes_legacy_reference_and_global_legacy_slice(self) -> None:
        registry = market_lane_registry()
        self.assertIn("football_1x2_canonical", registry)
        self.assertTrue(get_market_lane_spec("football_1x2_canonical").reference_only)
        self.assertEqual(get_market_lane_spec("football_1x2_canonical").status, "reference_only")
        self.assertEqual(get_market_lane_spec("football_1x2_global").legacy_included, ("E0", "SP1", "D1"))
        self.assertIn("I1", GLOBAL_FOOTBALL_1X2_LEAGUES)
        self.assertIn("F1", GLOBAL_FOOTBALL_1X2_LEAGUES)
        self.assertIn("N1", GLOBAL_FOOTBALL_1X2_LEAGUES)
        self.assertIn("P1", GLOBAL_FOOTBALL_1X2_LEAGUES)
        self.assertIn("MEX", GLOBAL_FOOTBALL_1X2_LEAGUES)
        self.assertIn("USA", GLOBAL_FOOTBALL_1X2_LEAGUES)
        with self.assertRaises(KeyError):
            get_market_lane_spec("not_a_lane")

    def test_parsers_separate_market_families_and_deferred_markets(self) -> None:
        football_event = {"title": "Premier League: Arsenal vs Chelsea", "slug": "premier-league-ars-che"}
        home = classify_market(
            football_event,
            {"question": "Will Arsenal win?", "outcomes": '["Yes", "No"]', "sportsMarketType": "moneyline"},
        )
        goals = classify_market(
            football_event,
            {"question": "Will Arsenal vs Chelsea have over 2.5 total goals?", "outcomes": '["Yes", "No"]'},
        )
        cards = classify_market(
            football_event,
            {"question": "Will Arsenal vs Chelsea have over 4.5 yellow cards?", "outcomes": '["Yes", "No"]'},
        )
        tennis = classify_market(
            {"title": "ATP Tennis: Alcaraz vs Sinner", "slug": "atp-tennis-alcaraz-sinner"},
            {"question": "Will Carlos Alcaraz win the match?", "outcomes": '["Yes", "No"]', "sportsMarketType": "moneyline"},
        )
        nba_total = classify_market(
            {"title": "NBA: Lakers vs Celtics", "slug": "nba-lakers-celtics"},
            {"question": "Will the game have over 220.5 total points?", "outcomes": '["Yes", "No"]'},
        )
        football_outright = classify_market(
            {"title": "La Liga Winner", "slug": "la-liga-winner-114"},
            {"question": "Will Real Madrid win the 2025-26 La Liga?", "outcomes": '["Yes", "No"]'},
        )

        self.assertEqual(home.market_family, "football_1x2_global")
        self.assertEqual(goals.market_family, "football_goals_core")
        self.assertEqual(cards.market_family, "football_cards")
        self.assertEqual(cards.status, "deferred")
        self.assertEqual(tennis.market_family, "tennis_match_winner")
        self.assertEqual(nba_total.market_family, "non_football_totals")
        self.assertEqual(nba_total.status, "deferred")
        self.assertIsNone(football_outright)

    def test_discovery_writes_isolated_database_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            result = discover_multi_market(
                settings=settings,
                db_path=default_multi_market_db_path(settings),
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )

            self.assertTrue(result.database_path.name.endswith(MULTI_MARKET_DATABASE_FILENAME))
            self.assertFalse((settings.paths.data_dir / "polymarket_shadow.sqlite").exists())
            self.assertTrue(result.artifacts["market_family_manifest"].exists())
            self.assertTrue(result.artifacts["market_family_coverage_report"].exists())
            self.assertTrue(result.artifacts["raw_inventory_quality_report"].exists())
            self.assertFalse(json.loads(result.artifacts["market_family_coverage_report"].read_text())["global_roi_actionable"])
            inventory = json.loads(result.artifacts["raw_inventory_quality_report"].read_text(encoding="utf-8"))
            self.assertFalse(inventory["global_roi_actionable"])
            self.assertIn("lanes", inventory)

            connection = sqlite3.connect(result.database_path)
            try:
                count = connection.execute("SELECT COUNT(*) FROM mm_market_catalog").fetchone()[0]
                families = {row[0] for row in connection.execute("SELECT DISTINCT market_family FROM mm_market_catalog")}
                raw_markets = connection.execute("SELECT COUNT(*) FROM mm_raw_markets").fetchone()[0]
                lane_links = connection.execute("SELECT COUNT(*) FROM mm_lane_market_links").fetchone()[0]
            finally:
                connection.close()
            self.assertGreaterEqual(count, 4)
            self.assertGreaterEqual(raw_markets, count)
            self.assertEqual(lane_links, count)
            self.assertIn("football_1x2_global", families)
            self.assertIn("football_goals_core", families)
            self.assertIn("tennis_match_winner", families)
            self.assertIn("football_cards", families)

    def test_capture_creates_ledger_without_emitting_picks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            result = capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))

            self.assertTrue(result.artifacts["multi_market_forward_sample_report"].exists())
            self.assertTrue(result.artifacts["multi_market_forward_ledger"].exists())
            self.assertTrue(result.artifacts["raw_capture_plan_csv"].exists())
            self.assertTrue(result.artifacts["raw_capture_plan_json"].exists())
            self.assertFalse(result.summary["global_roi_actionable"])
            self.assertIsNone(result.summary["global_roi"])
            self.assertGreater(result.summary["book_checkpoints"], 0)
            self.assertGreater(result.summary["raw_capture_plan"]["planned_markets"], 0)
            self.assertEqual(sum(family["valid_decisions"] for family in result.summary["families"]), 0)

            ledger = pd.read_csv(result.artifacts["multi_market_forward_ledger"])
            capture_plan = pd.read_csv(result.artifacts["raw_capture_plan_csv"])
            self.assertIn("lane_id", ledger.columns)
            self.assertIn("previous_book_status", capture_plan.columns)
            self.assertIn("selected_for_capture", capture_plan.columns)
            self.assertTrue((ledger["selected_outcome"].fillna("") == "").all())
            self.assertTrue(ledger["decision_status"].isin(["capture_only", "shadow_collect_only"]).all())

            connection = sqlite3.connect(db_path)
            try:
                raw_orderbooks = connection.execute("SELECT COUNT(*) FROM mm_raw_orderbooks").fetchone()[0]
                compat_checkpoints = connection.execute("SELECT COUNT(*) FROM mm_book_checkpoints").fetchone()[0]
            finally:
                connection.close()
            self.assertGreater(raw_orderbooks, 0)
            self.assertEqual(compat_checkpoints, 0)

    def test_report_multi_market_does_not_expose_actionable_global_roi(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            capture = capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))
            payload, text = report_multi_market(capture.run.run_dir)
            self.assertFalse(payload["global_roi_actionable"])
            self.assertIn("global_roi_actionable: false", text)
            self.assertIn("portfolio_readiness", text)

    def test_run_market_lane_generates_isolated_lane_artifacts_and_legacy_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            capture_multi_market(
                settings=settings,
                db_path=db_path,
                clob=_FakeClob(),
                now=pd.Timestamp("2026-04-20T10:05:00Z"),
            )

            summary, artifacts = run_market_lane(
                settings=settings,
                lane_id="football_1x2_global",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )
            self.assertEqual(summary["lane_id"], "football_1x2_global")
            self.assertEqual(summary["legacy_included"], ["E0", "SP1", "D1"])
            self.assertGreater(summary["legacy_slice_markets"], 0)
            self.assertTrue(artifacts["lane_manifest"].exists())
            self.assertTrue(artifacts["lane_readiness_report"].exists())
            self.assertTrue(artifacts["lane_decision_inventory_report"].exists())
            self.assertTrue(artifacts["lane_candidate_report"].exists())
            self.assertTrue(artifacts["lane_candidate_rows"].exists())
            self.assertTrue(artifacts["model_prediction_template"].exists())
            self.assertTrue(artifacts["lane_blocker_audit_csv"].exists())
            self.assertTrue(artifacts["lane_blocker_audit_json"].exists())
            self.assertTrue(artifacts["raw_capture_health_report"].exists())
            self.assertTrue(artifacts["forward_ledger"].exists())
            self.assertTrue(artifacts["legacy_parity_report"].exists())
            blocker_audit = json.loads(artifacts["lane_blocker_audit_json"].read_text(encoding="utf-8"))
            self.assertIn("final_blocker_counts", blocker_audit)
            self.assertTrue(blocker_audit["model_probability_missing_actionable"])
            health = json.loads(artifacts["raw_capture_health_report"].read_text(encoding="utf-8"))
            self.assertIn("capture_status_counts", health)
            self.assertLessEqual(health["book_market_coverage_rate"], 1.0)
            self.assertLessEqual(health["fresh_book_rate"], 1.0)
            self.assertFalse(health["global_roi_actionable"])
            self.assertEqual(summary["readiness_status"], "model_ready_policy_pending")
            self.assertTrue(summary["lane_readiness"]["coverage_ready"])
            self.assertTrue(summary["lane_readiness"]["model_ready"])
            self.assertFalse(summary["lane_readiness"]["policy_ready"])
            self.assertFalse(summary["can_emit_picks"])
            parity = json.loads(artifacts["legacy_parity_report"].read_text(encoding="utf-8"))
            self.assertTrue(parity["structural_parity_ready"])
            self.assertFalse(parity["can_replace_legacy"])

            payload, text = report_market_lane(settings=settings, lane_id="football_1x2_global")
            self.assertFalse(payload["global_roi_actionable"])
            self.assertIn("global_roi_actionable: false", text)
            self.assertIn("readiness_status: model_ready_policy_pending", text)

    def test_football_1x2_global_can_activate_frozen_legacy_policy_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            source_dir = settings.paths.runs_dir / "frozen_policy"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_bundle = source_dir / "policy_bundle.json"
            source_bundle.write_text(
                json.dumps(
                    {
                        "source_mode": "retro_approx",
                        "probability_source": "raw",
                        "bundle_status": "frozen",
                        "policy": {
                            "edge_threshold": 0.02,
                            "ev_threshold": 0.0,
                            "min_odds": 1.2,
                            "max_odds": 4.0,
                            "kelly_fraction": 0.25,
                            "family": "edge_ev_threshold",
                            "allowed_outcomes": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (settings.paths.outputs_dir / "latest_polymarket_policy.txt").write_text(str(source_bundle), encoding="utf-8")
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))

            summary, artifacts = run_market_lane(
                settings=settings,
                lane_id="football_1x2_global",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )

            self.assertEqual(summary["readiness_status"], "ready_to_emit_shadow_candidates")
            self.assertTrue(summary["can_emit_picks"])
            self.assertEqual(summary["policy_transfer_mode"], "legacy_frozen_policy_bridge")
            self.assertTrue(artifacts["policy_bundle"].exists())
            lane_policy = json.loads(artifacts["policy_bundle"].read_text(encoding="utf-8"))
            self.assertFalse(lane_policy["policy_reoptimized"])
            self.assertFalse(lane_policy["thresholds_changed"])
            self.assertEqual(lane_policy["source_policy_bundle_path"], str(source_bundle))
            self.assertEqual(lane_policy["policy"]["edge_threshold"], 0.02)
            ledger = pd.read_csv(artifacts["forward_ledger"])
            self.assertTrue((ledger["decision_status"] == "candidate_ready").all())
            self.assertTrue((ledger["policy_mode"] == "policy_ready").all())
            candidate_report = json.loads(artifacts["lane_candidate_report"].read_text(encoding="utf-8"))
            self.assertEqual(candidate_report["candidate_scoring_status"], "model_probability_missing")
            self.assertEqual(candidate_report["selected_candidates"], 0)

            payload, text = report_market_lane(settings=settings, lane_id="football_1x2_global")
            self.assertTrue(payload["can_emit_picks"])
            self.assertIn("policy_transfer_mode: legacy_frozen_policy_bridge", text)
            self.assertIn("candidate_scoring_status: model_probability_missing", text)

    def test_lane_candidate_rows_apply_frozen_policy_when_predictions_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            source_dir = settings.paths.runs_dir / "frozen_policy"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_bundle = source_dir / "policy_bundle.json"
            source_bundle.write_text(
                json.dumps(
                    {
                        "probability_source": "raw",
                        "bundle_status": "frozen",
                        "policy": {
                            "edge_threshold": 0.02,
                            "ev_threshold": 0.0,
                            "min_odds": 1.2,
                            "max_odds": 4.0,
                            "kelly_fraction": 0.25,
                            "family": "edge_ev_threshold",
                            "allowed_outcomes": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (settings.paths.outputs_dir / "latest_polymarket_policy.txt").write_text(str(source_bundle), encoding="utf-8")
            lane_dir = settings.paths.outputs_dir / "lanes" / "football_1x2_global"
            lane_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "event_slug": "premier-league-arsenal-chelsea-2026-04-21",
                        "selection": "home",
                        "model_prob": 0.60,
                        "probability_source": "raw",
                    },
                    {
                        "event_slug": "premier-league-arsenal-chelsea-2026-04-21",
                        "selection": "draw",
                        "model_prob": 0.20,
                        "probability_source": "raw",
                    },
                    {
                        "event_slug": "premier-league-arsenal-chelsea-2026-04-21",
                        "selection": "away",
                        "model_prob": 0.30,
                        "probability_source": "raw",
                    },
                ]
            ).to_csv(lane_dir / "model_predictions.csv", index=False)

            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))

            summary, artifacts = run_market_lane(
                settings=settings,
                lane_id="football_1x2_global",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )

            candidate_report = json.loads(artifacts["lane_candidate_report"].read_text(encoding="utf-8"))
            candidates = pd.read_csv(artifacts["lane_candidate_rows"])
            template = pd.read_csv(artifacts["model_prediction_template"])
            selected = candidates[candidates["candidate_status"] == "selected_candidate"]
            self.assertEqual(candidate_report["candidate_scoring_status"], "selected_candidates_ready")
            self.assertEqual(candidate_report["selected_candidates"], 1)
            self.assertEqual(summary["valid_decisions"], 1)
            self.assertEqual(set(selected["selection"]), {"home"})
            self.assertGreater(float(selected.iloc[0]["edge"]), 0.02)
            self.assertGreater(float(selected.iloc[0]["ev"]), 0.0)
            self.assertIn("model_prob", template.columns)
            self.assertEqual(set(template["selection"]), {"home", "draw", "away"})

    def test_sample_report_marks_roi_hidden_until_sample_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            _write_frozen_football_policy(settings)
            _write_football_1x2_lane_predictions(settings)

            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))
            summary, artifacts = run_market_lane(
                settings=settings,
                lane_id="football_1x2_global",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )

            sample_report = json.loads(artifacts["sample_report"].read_text(encoding="utf-8"))
            required_fields = {
                "sample_status",
                "sample_blockers",
                "valid_forward_decisions",
                "settled_decisions",
                "fresh_book_rate",
                "actionable_roi",
                "can_reopen_decision_region_analysis",
                "roi_display_mode",
            }
            self.assertTrue(required_fields.issubset(sample_report))
            self.assertEqual(summary["sample_status"], "collecting_forward_sample")
            self.assertEqual(sample_report["settled_decisions"], 0)
            self.assertFalse(sample_report["actionable_roi"])
            self.assertFalse(sample_report["can_reopen_decision_region_analysis"])
            self.assertEqual(sample_report["roi_display_mode"], "hidden_until_sample_ready")
            self.assertEqual(sample_report["diagnostic_only"]["reason"], "ROI is not actionable until sample_ready")

    def test_sample_report_uses_promotion_state_classifier(self) -> None:
        spec = get_market_lane_spec("football_1x2_global")

        coverage_blocked = _lane_sample_status(
            spec=spec,
            valid_decisions=100,
            settled_decisions=40,
            fresh_book_rate=0.79,
        )
        self.assertEqual(coverage_blocked["sample_status"], "coverage_blocked")
        self.assertFalse(coverage_blocked["actionable_roi"])
        self.assertEqual(coverage_blocked["roi_display_mode"], "hidden_until_sample_ready")
        self.assertTrue(coverage_blocked["sample_blockers"][0].startswith("fresh_book_rate_below_minimum"))

        settlement_pending = _lane_sample_status(
            spec=spec,
            valid_decisions=100,
            settled_decisions=39,
            fresh_book_rate=1.0,
        )
        self.assertEqual(settlement_pending["sample_status"], "settlement_pending")
        self.assertFalse(settlement_pending["actionable_roi"])
        self.assertFalse(settlement_pending["can_reopen_decision_region_analysis"])

    def test_build_market_lane_predictions_generates_legacy_1x2_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            model_path = _write_fake_lane_model(settings)
            source_dir = settings.paths.runs_dir / "frozen_policy"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_bundle = source_dir / "policy_bundle.json"
            source_bundle.write_text(
                json.dumps(
                    {
                        "probability_source": "raw",
                        "policy": {
                            "edge_threshold": 0.02,
                            "ev_threshold": 0.0,
                            "min_odds": 1.2,
                            "max_odds": 4.0,
                            "allowed_outcomes": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (settings.paths.outputs_dir / "latest_polymarket_policy.txt").write_text(str(source_bundle), encoding="utf-8")
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))
            run_market_lane(
                settings=settings,
                lane_id="football_1x2_global",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )

            summary, artifacts = build_market_lane_predictions(
                settings=settings,
                lane_id="football_1x2_global",
                db_path=db_path,
                model_path=model_path,
                now=pd.Timestamp("2026-04-20T10:15:00Z"),
            )

            predictions = pd.read_csv(artifacts["model_predictions"])
            self.assertEqual(summary["predicted_rows"], 3)
            self.assertEqual(set(predictions["selection"]), {"home", "draw", "away"})
            self.assertAlmostEqual(float(predictions["model_prob"].sum()), 1.0, places=6)
            lane_latest = settings.paths.outputs_dir / "lanes" / "football_1x2_global" / "latest_model.txt"
            lane_latest.write_text(str(model_path), encoding="utf-8")
            default_summary, _ = build_market_lane_predictions(
                settings=settings,
                lane_id="football_1x2_global",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:17:00Z"),
            )
            self.assertEqual(Path(default_summary["model_path"]), model_path)
            rerun_summary, rerun_artifacts = run_market_lane(
                settings=settings,
                lane_id="football_1x2_global",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:20:00Z"),
            )
            candidate_report = json.loads(rerun_artifacts["lane_candidate_report"].read_text(encoding="utf-8"))
            blocker_audit = pd.read_csv(rerun_artifacts["lane_blocker_audit_csv"])
            blocker_report = json.loads(rerun_artifacts["lane_blocker_audit_json"].read_text(encoding="utf-8"))
            self.assertGreater(candidate_report["scorable_candidates"], 0)
            self.assertEqual(candidate_report["candidate_scoring_status"], "selected_candidates_ready")
            self.assertGreaterEqual(rerun_summary["valid_decisions"], 1)
            self.assertIn("final_blocker", blocker_audit.columns)
            self.assertTrue(blocker_report["model_probability_missing_actionable"])
            self.assertIn("predicted", set(blocker_audit["final_blocker"]))

    def test_global_football_league_codes_are_recognized_but_require_history(self) -> None:
        serie_a_row = pd.Series(
            {
                "event_slug": "sea-juventus-ac-milan-2026-04-21",
                "event_title": "Serie A: Juventus vs AC Milan",
                "game_start_time": "2026-04-21T18:00:00Z",
            }
        )
        ligue_1_row = pd.Series(
            {
                "event_slug": "fl1-psg-lyon-2026-04-21",
                "event_title": "Ligue 1: PSG vs Lyon",
                "game_start_time": "2026-04-21T20:00:00Z",
            }
        )
        self.assertEqual(_league_code_from_market_row(serie_a_row), "I1")
        self.assertEqual(_league_code_from_market_row(ligue_1_row), "F1")
        self.assertEqual(
            _league_code_from_market_row(pd.Series({"event_slug": "ere-aja-psv-2026-05-02", "event_title": "Eredivisie: Ajax vs PSV"})),
            "N1",
        )
        self.assertEqual(
            _league_code_from_market_row(pd.Series({"event_slug": "mex-pum-jua-2026-04-21", "event_title": "Liga MX: Pumas vs Juarez"})),
            "MEX",
        )
        self.assertEqual(
            _league_code_from_market_row(pd.Series({"event_slug": "mls-nyc-fcc-2026-04-22", "event_title": "MLS: New York City vs FC Cincinnati"})),
            "USA",
        )

        template = pd.DataFrame(
            [
                {"event_slug": "sea-juventus-ac-milan-2026-04-21", "selection": "home"},
                {"event_slug": "sea-juventus-ac-milan-2026-04-21", "selection": "draw"},
                {"event_slug": "sea-juventus-ac-milan-2026-04-21", "selection": "away"},
            ]
        )
        catalog = pd.DataFrame([serie_a_row])
        fixtures, contexts, blockers = _fixture_rows_from_template(template, catalog, _lane_history_matches())

        self.assertTrue(fixtures.empty)
        self.assertEqual(blockers.get("unsupported_league"), 3)
        self.assertEqual(contexts["sea-juventus-ac-milan-2026-04-21"]["league_code"], "I1")
        self.assertEqual(contexts["sea-juventus-ac-milan-2026-04-21"]["blocker"], "unsupported_league")

    def test_build_market_lane_predictions_generates_goals_probabilities_without_policy_picks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            model_path = _write_fake_lane_model(settings)
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))
            _, artifacts = run_market_lane(
                settings=settings,
                lane_id="football_goals_core",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )
            template = pd.read_csv(artifacts["model_prediction_template"])
            btts_rows = template.head(1).copy()
            btts_rows["market_id"] = "manual-btts"
            btts_rows["market_subtype"] = "btts"
            btts_rows["selection"] = "btts_yes"
            template = pd.concat([template, btts_rows], ignore_index=True)
            template.to_csv(settings.paths.outputs_dir / "lanes" / "football_goals_core" / "model_prediction_template.csv", index=False)

            summary, output_artifacts = build_market_lane_predictions(
                settings=settings,
                lane_id="football_goals_core",
                db_path=db_path,
                model_path=model_path,
                now=pd.Timestamp("2026-04-20T10:15:00Z"),
            )

            predictions = pd.read_csv(output_artifacts["model_predictions"])
            self.assertGreater(summary["predicted_rows"], 0)
            total_market = predictions[predictions["market_subtype"] == "total_goals"]
            self.assertAlmostEqual(float(total_market["model_prob"].sum()), 1.0, places=6)
            self.assertIn("btts_yes", set(predictions["selection"]))
            rerun_summary, rerun_artifacts = run_market_lane(
                settings=settings,
                lane_id="football_goals_core",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:20:00Z"),
            )
            candidate_report = json.loads(rerun_artifacts["lane_candidate_report"].read_text(encoding="utf-8"))
            research_report = json.loads(rerun_artifacts["football_goals_policy_research_report"].read_text(encoding="utf-8"))
            research_rows = pd.read_csv(rerun_artifacts["football_goals_policy_research_rows"])
            self.assertGreater(candidate_report["scorable_candidates"], 0)
            self.assertEqual(candidate_report["candidate_scoring_status"], "policy_not_ready")
            self.assertEqual(rerun_summary["readiness_status"], "model_ready_policy_pending")
            self.assertEqual(rerun_summary["valid_decisions"], 0)
            self.assertFalse(research_report["can_emit_picks"])
            self.assertFalse(research_report["global_roi_actionable"])
            self.assertGreater(len(research_rows), 0)
            self.assertTrue((research_rows["can_emit_picks"] == False).all())  # noqa: E712

    def test_football_goals_core_native_policy_activates_its_own_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            model_path = _write_fake_lane_model(settings)
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))
            run_market_lane(
                settings=settings,
                lane_id="football_goals_core",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )
            build_market_lane_predictions(
                settings=settings,
                lane_id="football_goals_core",
                db_path=db_path,
                model_path=model_path,
                now=pd.Timestamp("2026-04-20T10:15:00Z"),
            )
            pending_summary, _ = run_market_lane(
                settings=settings,
                lane_id="football_goals_core",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:20:00Z"),
            )
            self.assertEqual(pending_summary["readiness_status"], "model_ready_policy_pending")

            policy_summary, policy_artifacts = create_market_lane_policy(
                settings=settings,
                lane_id="football_goals_core",
                now=pd.Timestamp("2026-04-20T10:25:00Z"),
            )
            self.assertTrue(policy_summary["policy_ready"])
            self.assertTrue(policy_artifacts["policy_bundle"].exists())
            self.assertTrue(policy_artifacts["lane_policy_benchmark_report"].exists())

            ready_summary, ready_artifacts = run_market_lane(
                settings=settings,
                lane_id="football_goals_core",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:30:00Z"),
            )
            candidate_report = json.loads(ready_artifacts["lane_candidate_report"].read_text(encoding="utf-8"))
            bundle = json.loads(ready_artifacts["policy_bundle"].read_text(encoding="utf-8"))
            research_report = json.loads(ready_artifacts["football_goals_policy_research_report"].read_text(encoding="utf-8"))
            research_rows = pd.read_csv(ready_artifacts["football_goals_policy_research_rows"])
            self.assertEqual(ready_summary["readiness_status"], "ready_to_emit_shadow_candidates")
            self.assertTrue(ready_summary["can_emit_picks"])
            self.assertEqual(bundle["policy_transfer_mode"], "native_lane_policy")
            self.assertEqual(bundle["source_benchmark_id"], "football_goals_core_v1")
            self.assertFalse(bundle["global_roi_actionable"])
            self.assertIn("allowed_market_subtypes", bundle["policy"])
            self.assertNotEqual(candidate_report["candidate_scoring_status"], "policy_not_ready")
            self.assertTrue(research_report["policy_ready"])
            self.assertTrue(research_report["can_emit_picks"])
            self.assertEqual(research_report["roi_status"], "not_actionable_until_forward_settled")
            self.assertTrue((research_rows["policy_ready"] == True).all())  # noqa: E712

    def test_build_market_lane_predictions_rejects_undeclared_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            with self.assertRaises(KeyError):
                build_market_lane_predictions(settings=settings, lane_id="unknown_lane")

    def test_football_goals_lane_reports_subtypes_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))
            summary, _ = run_market_lane(
                settings=settings,
                lane_id="football_goals_core",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )
            self.assertIn("total_goals", summary["market_subtype_counts"])
            self.assertEqual(summary["lane_readiness"]["subtype_status"], "partial_coverage")
            self.assertIn("btts", summary["lane_readiness"]["missing_subtypes"])
            self.assertTrue(summary["lane_readiness"]["model_ready"])
            self.assertFalse(summary["lane_readiness"]["policy_ready"])
            self.assertFalse(summary["global_roi_actionable"])

    def test_non_football_moneyline_lane_stays_capture_only_until_model_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))
            summary, _ = run_market_lane(
                settings=settings,
                lane_id="tennis_match_winner",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )
            self.assertEqual(summary["readiness_status"], "capture_only_model_pending")
            self.assertFalse(summary["lane_readiness"]["model_ready"])
            self.assertFalse(summary["can_emit_picks"])
            self.assertIn("model_not_ready", summary["readiness_blockers"])

    def test_historical_active_inventory_does_not_unlock_forward_readiness_or_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_multi_market_db_path(settings)
            discovery = discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_HistoricalOnlyGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            coverage = json.loads(discovery.artifacts["market_family_coverage_report"].read_text(encoding="utf-8"))
            football = next(item for item in coverage["families"] if item["market_family"] == "football_1x2_global")
            self.assertEqual(football["coverage_status"], "inventory_not_forward_ready")
            self.assertEqual(football["future_active_markets"], 0)
            self.assertGreater(football["active_historical_markets"], 0)

            capture = capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))
            self.assertEqual(capture.summary["markets_considered"], 0)

            summary, artifacts = run_market_lane(
                settings=settings,
                lane_id="football_1x2_global",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )
            self.assertEqual(summary["readiness_status"], "inventory_not_forward_ready")
            self.assertIn("inventory_not_forward_ready", summary["readiness_blockers"])
            self.assertFalse(summary["lane_readiness"]["coverage_ready"])
            ledger = pd.read_csv(artifacts["forward_ledger"])
            self.assertEqual(len(ledger), 0)

    def test_sports_series_discovery_finds_forward_markets_when_public_search_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_multi_market_db_path(settings)
            discovery = discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_SportsSeriesGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            self.assertEqual(discovery.summary["public_search_events_seen"], 0)
            self.assertEqual(discovery.summary["sports_series_events_seen"], 1)
            coverage = json.loads(discovery.artifacts["market_family_coverage_report"].read_text(encoding="utf-8"))
            football = next(item for item in coverage["families"] if item["market_family"] == "football_1x2_global")
            self.assertEqual(football["coverage_status"], "capture_ready")
            self.assertEqual(football["future_active_markets"], 2)

            capture_multi_market(settings=settings, db_path=db_path, clob=_FakeClob(), now=pd.Timestamp("2026-04-20T10:05:00Z"))
            summary, artifacts = run_market_lane(
                settings=settings,
                lane_id="football_1x2_global",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:10:00Z"),
            )
            self.assertEqual(summary["readiness_status"], "model_ready_policy_pending")
            self.assertTrue(summary["lane_readiness"]["coverage_ready"])
            ledger = pd.read_csv(artifacts["forward_ledger"])
            self.assertEqual(len(ledger), 2)

    def test_capture_can_filter_lane_and_limit_markets_for_safe_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_SportsSeriesGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            result = capture_multi_market(
                settings=settings,
                db_path=db_path,
                clob=_FakeClob(),
                lane_id="football_1x2_global",
                max_markets=1,
                now=pd.Timestamp("2026-04-20T10:05:00Z"),
            )
            self.assertEqual(result.summary["capture_lane_filter"], "football_1x2_global")
            self.assertEqual(result.summary["capture_max_markets"], 1)
            self.assertEqual(result.summary["markets_considered"], 1)
            ledger = pd.read_csv(result.artifacts["multi_market_forward_ledger"])
            self.assertEqual(set(ledger["lane_id"]), {"football_1x2_global"})
            self.assertEqual(len(ledger), 1)

    def test_raw_capture_plan_prioritizes_missing_before_stale_and_fresh(self) -> None:
        now = pd.Timestamp("2026-04-20T12:00:00Z")
        catalog = pd.DataFrame(
            [
                {
                    "market_id": "fresh-market",
                    "event_slug": "event-fresh",
                    "lane_id": "football_1x2_global",
                    "market_family": "football_1x2_global",
                    "market_subtype": "1x2_home",
                    "game_start_time": "2026-04-21T18:00:00Z",
                    "clob_token_ids_json": '["1"]',
                },
                {
                    "market_id": "stale-market",
                    "event_slug": "event-stale",
                    "lane_id": "football_1x2_global",
                    "market_family": "football_1x2_global",
                    "market_subtype": "1x2_home",
                    "game_start_time": "2026-04-21T17:00:00Z",
                    "clob_token_ids_json": '["2"]',
                },
                {
                    "market_id": "missing-market",
                    "event_slug": "event-missing",
                    "lane_id": "football_1x2_global",
                    "market_family": "football_1x2_global",
                    "market_subtype": "1x2_home",
                    "game_start_time": "2026-04-21T16:00:00Z",
                    "clob_token_ids_json": '["3"]',
                },
                {
                    "market_id": "no-token-market",
                    "event_slug": "event-no-token",
                    "lane_id": "football_1x2_global",
                    "market_family": "football_1x2_global",
                    "market_subtype": "1x2_home",
                    "game_start_time": "2026-04-21T15:00:00Z",
                    "clob_token_ids_json": "[]",
                },
            ]
        )
        existing_books = pd.DataFrame(
            [
                {"market_id": "fresh-market", "timestamp": "2026-04-20T11:45:00Z"},
                {"market_id": "stale-market", "timestamp": "2026-04-20T09:00:00Z"},
            ]
        )

        plan = _build_raw_capture_plan(
            catalog=catalog,
            existing_books=existing_books,
            now=now,
            capture_priority="missing_or_stale",
            freshness_seconds=3600,
            max_markets=2,
        )

        selected = plan[plan["selected_for_capture"].astype(bool)]
        self.assertEqual(selected["previous_book_status"].tolist(), ["capture_not_attempted", "stale_book"])
        self.assertEqual(set(selected["market_id"]), {"missing-market", "stale-market"})
        self.assertEqual(
            plan.set_index("market_id").loc["fresh-market", "previous_book_status"],
            "fresh_book",
        )
        self.assertFalse(bool(plan.set_index("market_id").loc["no-token-market", "eligible_by_priority"]))

    def test_capture_include_policy_ready_only_excludes_goals_research_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            source_dir = settings.paths.runs_dir / "frozen_policy"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_bundle = source_dir / "policy_bundle.json"
            source_bundle.write_text(
                json.dumps(
                    {
                        "probability_source": "raw",
                        "policy": {
                            "edge_threshold": 0.02,
                            "ev_threshold": 0.0,
                            "min_odds": 1.2,
                            "max_odds": 4.0,
                            "allowed_outcomes": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (settings.paths.outputs_dir / "latest_polymarket_policy.txt").write_text(str(source_bundle), encoding="utf-8")
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )

            result = capture_multi_market(
                settings=settings,
                db_path=db_path,
                clob=_FakeClob(),
                include_policy_ready_only=True,
                now=pd.Timestamp("2026-04-20T10:05:00Z"),
            )

            capture_plan = pd.read_csv(result.artifacts["raw_capture_plan_csv"])
            selected = capture_plan[capture_plan["selected_for_capture"].astype(bool)]
            self.assertTrue(result.summary["include_policy_ready_only"])
            self.assertGreater(len(selected), 0)
            self.assertEqual(set(selected["lane_id"]), {"football_1x2_global"})
            self.assertNotIn("football_goals_core", set(selected["lane_id"]))

    def test_capture_can_limit_by_events_without_splitting_lane_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_SportsSeriesGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            result = capture_multi_market(
                settings=settings,
                db_path=db_path,
                clob=_FakeClob(),
                lane_id="football_1x2_global",
                max_events=1,
                now=pd.Timestamp("2026-04-20T10:05:00Z"),
            )
            self.assertEqual(result.summary["capture_lane_filter"], "football_1x2_global")
            self.assertEqual(result.summary["capture_max_events"], 1)
            self.assertEqual(result.summary["markets_considered"], 2)
            capture_plan = pd.read_csv(result.artifacts["raw_capture_plan_csv"])
            self.assertEqual(int(capture_plan["selected_for_capture"].sum()), 2)
            ledger = pd.read_csv(result.artifacts["multi_market_forward_ledger"])
            self.assertEqual(ledger["event_slug"].nunique(), 1)
            self.assertEqual(len(ledger), 2)

    def test_sport_merge_creates_feature_store_without_roi_or_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = default_multi_market_db_path(settings)
            discover_multi_market(
                settings=settings,
                db_path=db_path,
                gamma=_FakeGamma(),
                now=pd.Timestamp("2026-04-20T10:00:00Z"),
            )
            report, artifacts = report_sport_merge(
                settings=settings,
                sport="football",
                db_path=db_path,
                now=pd.Timestamp("2026-04-20T10:20:00Z"),
            )
            self.assertTrue(artifacts["sport_context_features"].exists())
            self.assertFalse(report["global_roi_actionable"])
            self.assertFalse(report["promotion_allowed"])
            features = pd.read_csv(artifacts["sport_context_features"])
            self.assertIn("has_football_1x2_global", features.columns)
            self.assertIn("has_football_goals_core", features.columns)
            self.assertTrue(artifacts["stable_sport_context_features"].exists())
            self.assertTrue(artifacts["stable_sport_merge_report"].exists())
            self.assertIn("fixture_key", features.columns)
            self.assertFalse(features["global_roi_actionable"].any())

    def test_multi_market_lane_cycle_script_has_safe_dry_run_mode(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_multi_market_lane_cycle.ps1"
        self.assertTrue(script_path.exists())
        content = script_path.read_text(encoding="utf-8")
        self.assertIn("DryRun", content)
        self.assertIn("ModelPath", content)
        self.assertIn("--model-path", content)
        self.assertIn('@("lane", "capture-raw")', content)
        self.assertIn('@("lane", "build-predictions")', content)
        self.assertIn('@("lane", "run-shadow")', content)
        self.assertIn('@("lane", "report")', content)
        self.assertIn("football_goals_core", content)


if __name__ == "__main__":
    unittest.main()
