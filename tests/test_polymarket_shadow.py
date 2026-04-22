from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from predicciones.config import (
    BacktestConfig,
    ExecutionConfig,
    PolymarketConfig,
    ProjectPaths,
    ResearchConfig,
    Settings,
    SnapshotConfig,
)
from predicciones.polymarket_shadow import (
    _build_forward_sample_manifest,
    _build_forward_sample_report,
    _decision_from_prediction,
    _decision_window_diagnostics,
    _derive_fixtures_from_groups,
    _forward_sample_flags,
    _summarize_shadow,
    _stream_polymarket_updates,
    _upsert_rows,
    capture_rest_books,
    discover_market_catalog,
    init_polymarket_db,
    report_polymarket,
    simulate_taker_yes_fill,
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
        claude_model="none",
        default_leagues=("E0",),
        default_seasons=("2526",),
        benchmark_dir_name="legacy_v1",
        snapshot=SnapshotConfig(default_kickoff_hour=15, closing_proxy_capture_minutes=45),
        execution=ExecutionConfig(),
        research=ResearchConfig(),
        polymarket=PolymarketConfig(book_freshness_seconds=5),
        backtest=BacktestConfig(),
    )


class _FakeGammaClient:
    def __init__(self, event: dict, query: str = "Premier League") -> None:
        self.event = event
        self.query = query

    def search_events(self, query: str, limit: int = 50):
        if query != self.query:
            return []
        return [self.event]

    def event_by_slug(self, slug: str):
        return [self.event] if slug == self.event["slug"] else []


class PolymarketShadowTests(unittest.TestCase):
    def test_discover_market_catalog_builds_complete_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            event = {
                "id": "100",
                "slug": "epl-mac-ars-2026-04-19",
                "title": "Manchester City FC vs. Arsenal FC",
                "markets": [
                    {
                        "id": "1",
                        "question": "Will Manchester City FC win on 2026-04-19?",
                        "sportsMarketType": "moneyline",
                        "gameStartTime": "2026-04-19T15:30:00Z",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["11", "12"]',
                        "feesEnabled": True,
                        "feeSchedule": {"rate": 0.03},
                        "active": True,
                        "closed": False,
                        "acceptingOrders": True,
                    },
                    {
                        "id": "2",
                        "question": "Will Manchester City FC vs. Arsenal FC end in a draw?",
                        "sportsMarketType": "moneyline",
                        "gameStartTime": "2026-04-19T15:30:00Z",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["21", "22"]',
                        "feesEnabled": True,
                        "feeSchedule": {"rate": 0.03},
                        "active": True,
                        "closed": False,
                        "acceptingOrders": True,
                    },
                    {
                        "id": "3",
                        "question": "Will Arsenal FC win on 2026-04-19?",
                        "sportsMarketType": "moneyline",
                        "gameStartTime": "2026-04-19T15:30:00Z",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["31", "32"]',
                        "feesEnabled": True,
                        "feeSchedule": {"rate": 0.03},
                        "active": True,
                        "closed": False,
                        "acceptingOrders": True,
                    },
                ],
            }
            catalog, groups = discover_market_catalog(settings, _FakeGammaClient(event), now=pd.Timestamp("2026-04-17T00:00:00Z"))
            self.assertEqual(len(catalog), 3)
            self.assertEqual(len(groups), 1)
            self.assertEqual(str(groups.iloc[0]["mapping_status"]), "complete")
            self.assertEqual(str(groups.iloc[0]["league_code"]), "E0")

    def test_discover_market_catalog_marks_group_unmapped_when_draw_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            event = {
                "id": "100",
                "slug": "epl-mac-ars-2026-04-19",
                "title": "Manchester City FC vs. Arsenal FC",
                "markets": [
                    {
                        "id": "1",
                        "question": "Will Manchester City FC win on 2026-04-19?",
                        "sportsMarketType": "moneyline",
                        "gameStartTime": "2026-04-19T15:30:00Z",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["11", "12"]',
                        "active": True,
                        "closed": False,
                        "acceptingOrders": True,
                    },
                    {
                        "id": "3",
                        "question": "Will Arsenal FC win on 2026-04-19?",
                        "sportsMarketType": "moneyline",
                        "gameStartTime": "2026-04-19T15:30:00Z",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["31", "32"]',
                        "active": True,
                        "closed": False,
                        "acceptingOrders": True,
                    },
                ],
            }
            _, groups = discover_market_catalog(settings, _FakeGammaClient(event), now=pd.Timestamp("2026-04-17T00:00:00Z"))
            self.assertEqual(str(groups.iloc[0]["mapping_status"]), "unmapped")

    def test_simulate_taker_yes_fill_handles_partial_and_fee(self) -> None:
        fill = simulate_taker_yes_fill(
            asks=[{"price": 0.50, "size": 20}, {"price": 0.60, "size": 10}],
            notional=25.0,
            fee_rate=0.03,
            slippage_cushion=0.01,
        )
        self.assertTrue(fill["partial_fill"])
        self.assertLess(fill["fill_rate"], 1.0)
        self.assertGreater(fill["fee_paid"], 0.0)
        self.assertGreater(fill["cost_basis"], fill["raw_spend"])

    def test_decision_skips_when_book_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            row = pd.Series(
                {
                    "match_id": "m1",
                    "Date": pd.Timestamp("2026-04-19"),
                    "kickoff_time": pd.Timestamp("2026-04-19T15:30:00"),
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "HomeTeam": "Man City",
                    "AwayTeam": "Arsenal",
                    "group_key": "epl-mac-ars-2026-04-19",
                    "prob_home_raw": 0.60,
                    "prob_draw_raw": 0.20,
                    "prob_away_raw": 0.20,
                }
            )
            groups = pd.DataFrame(
                [
                    {
                        "group_key": "epl-mac-ars-2026-04-19",
                        "mapping_status": "complete",
                        "home_market_id": "1",
                        "draw_market_id": "2",
                        "away_market_id": "3",
                    }
                ]
            )
            catalog = pd.DataFrame(
                [
                    {"market_id": "1", "fees_enabled": 1, "fee_rate": 0.03},
                    {"market_id": "2", "fees_enabled": 1, "fee_rate": 0.03},
                    {"market_id": "3", "fees_enabled": 1, "fee_rate": 0.03},
                ]
            )
            checkpoints = pd.DataFrame(
                [
                    {
                        "market_id": "1",
                        "asset_id": "11",
                        "timestamp": pd.Timestamp("2026-04-19T14:44:40Z"),
                        "asks_json": '[{"price": 0.53, "size": 100}]',
                        "event_type": "decision_checkpoint",
                    },
                    {
                        "market_id": "2",
                        "asset_id": "21",
                        "timestamp": pd.Timestamp("2026-04-19T14:44:40Z"),
                        "asks_json": '[{"price": 0.25, "size": 100}]',
                        "event_type": "decision_checkpoint",
                    },
                    {
                        "market_id": "3",
                        "asset_id": "31",
                        "timestamp": pd.Timestamp("2026-04-19T14:44:40Z"),
                        "asks_json": '[{"price": 0.22, "size": 100}]',
                        "event_type": "decision_checkpoint",
                    },
                ]
            )
            decision, fills = _decision_from_prediction(
                row=row,
                groups=groups,
                catalog=catalog,
                checkpoints=checkpoints,
                settings=settings,
                probability_source="raw",
            )
            self.assertEqual(decision["skip_reason"], "stale_book")
            self.assertEqual(fills, [])

    def test_decision_uses_decision_freshness_for_t45m_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            row = pd.Series(
                {
                    "match_id": "m1",
                    "Date": pd.Timestamp("2026-04-19"),
                    "kickoff_time": pd.Timestamp("2026-04-19T15:30:00"),
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "HomeTeam": "Man City",
                    "AwayTeam": "Arsenal",
                    "group_key": "epl-mac-ars-2026-04-19",
                    "prob_home_raw": 0.70,
                    "prob_draw_raw": 0.15,
                    "prob_away_raw": 0.15,
                }
            )
            groups = pd.DataFrame(
                [
                    {
                        "group_key": "epl-mac-ars-2026-04-19",
                        "mapping_status": "complete",
                        "home_market_id": "1",
                        "draw_market_id": "2",
                        "away_market_id": "3",
                    }
                ]
            )
            catalog = pd.DataFrame(
                [
                    {"market_id": "1", "fees_enabled": 1, "fee_rate": 0.03},
                    {"market_id": "2", "fees_enabled": 1, "fee_rate": 0.03},
                    {"market_id": "3", "fees_enabled": 1, "fee_rate": 0.03},
                ]
            )
            checkpoints = pd.DataFrame(
                [
                    {
                        "market_id": "1",
                        "asset_id": "11",
                        "timestamp": pd.Timestamp("2026-04-19T14:44:50Z"),
                        "asks_json": '[{"price": 0.53, "size": 100}]',
                        "event_type": "decision_checkpoint",
                    },
                    {
                        "market_id": "2",
                        "asset_id": "21",
                        "timestamp": pd.Timestamp("2026-04-19T14:44:50Z"),
                        "asks_json": '[{"price": 0.25, "size": 100}]',
                        "event_type": "decision_checkpoint",
                    },
                    {
                        "market_id": "3",
                        "asset_id": "31",
                        "timestamp": pd.Timestamp("2026-04-19T14:44:50Z"),
                        "asks_json": '[{"price": 0.22, "size": 100}]',
                        "event_type": "decision_checkpoint",
                    },
                ]
            )
            decision, fills = _decision_from_prediction(
                row=row,
                groups=groups,
                catalog=catalog,
                checkpoints=checkpoints,
                settings=settings,
                probability_source="raw",
            )
            self.assertEqual(decision["skip_reason"], "")
            self.assertEqual(decision["selection"], "home")
            self.assertAlmostEqual(float(decision["book_age_seconds"]), 10.0, places=6)
            self.assertGreater(len(fills), 0)

    def test_sqlite_upsert_deduplicates_checkpoint_primary_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "pm.sqlite"
            connection = init_polymarket_db(db_path)
            row = {
                "asset_id": "11",
                "market_id": "1",
                "group_key": "g1",
                "timestamp": "2026-04-19T14:45:00+00:00",
                "event_type": "decision_checkpoint",
                "top_ask": 0.53,
                "asks_json": "[]",
                "bids_json": "[]",
                "source": "test",
                "raw_json": "{}",
            }
            _upsert_rows(connection, "pm_book_checkpoints", [row])
            changed = dict(row)
            changed["top_ask"] = 0.55
            _upsert_rows(connection, "pm_book_checkpoints", [changed])
            result = pd.read_sql_query("SELECT COUNT(*) AS rows, MAX(top_ask) AS top_ask FROM pm_book_checkpoints", connection)
            self.assertEqual(int(result.iloc[0]["rows"]), 1)
            self.assertAlmostEqual(float(result.iloc[0]["top_ask"]), 0.55, places=6)
            decision_columns = {row[1] for row in connection.execute("PRAGMA table_info(pm_shadow_decisions)").fetchall()}
            fill_columns = {row[1] for row in connection.execute("PRAGMA table_info(pm_shadow_fills)").fetchall()}
            self.assertIn("price_provenance", decision_columns)
            self.assertIn("expected_fill_probability", decision_columns)
            self.assertIn("price_provenance", fill_columns)
            self.assertIn("closing_reference_prob", fill_columns)
            connection.close()

    def test_forward_sample_report_tracks_valid_cumulative_decisions(self) -> None:
        decisions = pd.DataFrame(
            [
                {
                    "decision_id": "d1",
                    "run_id": "r1",
                    "match_id": "m1",
                    "group_key": "g1",
                    "league_code": "E0",
                    "kickoff_time": "2026-04-19T15:30:00Z",
                    "mapping_status": "complete",
                    "selection": "home",
                    "skip_reason": "",
                    "price_provenance": "exact",
                    "validation_stage": "shadow",
                    "model_prob": 0.60,
                    "top_ask": 0.50,
                    "expected_edge": 0.10,
                    "expected_ev": 0.20,
                    "snapshot_time": "2026-04-19T14:44:59Z",
                    "book_age_seconds": 1.0,
                },
                {
                    "decision_id": "d2",
                    "run_id": "r1",
                    "match_id": "m2",
                    "group_key": "g2",
                    "league_code": "E0",
                    "kickoff_time": "2026-04-20T15:30:00Z",
                    "mapping_status": "complete",
                    "selection": "",
                    "skip_reason": "stale_book",
                    "price_provenance": "exact",
                    "validation_stage": "shadow",
                    "model_prob": np.nan,
                    "top_ask": np.nan,
                    "snapshot_time": "2026-04-20T14:40:00Z",
                    "book_age_seconds": 300.0,
                },
                {
                    "decision_id": "d3",
                    "run_id": "r1",
                    "match_id": "m3",
                    "group_key": "g3",
                    "league_code": "SP1",
                    "kickoff_time": "2026-04-21T15:30:00Z",
                    "mapping_status": "complete",
                    "selection": "",
                    "skip_reason": "no_book_checkpoint",
                    "price_provenance": "exact",
                    "validation_stage": "shadow",
                    "model_prob": np.nan,
                    "top_ask": np.nan,
                    "snapshot_time": "",
                    "book_age_seconds": np.nan,
                },
                {
                    "decision_id": "d4",
                    "run_id": "r1",
                    "match_id": "m4",
                    "group_key": "g4",
                    "league_code": "D1",
                    "kickoff_time": "2026-04-22T15:30:00Z",
                    "mapping_status": "complete",
                    "selection": "",
                    "skip_reason": "policy_rejected",
                    "price_provenance": "exact",
                    "validation_stage": "shadow",
                    "model_prob": 0.45,
                    "top_ask": 0.50,
                    "snapshot_time": "2026-04-22T14:44:59Z",
                    "book_age_seconds": 1.0,
                },
                {
                    "decision_id": "d5",
                    "run_id": "r1",
                    "match_id": "m5",
                    "group_key": "g5",
                    "league_code": "SP1",
                    "kickoff_time": "2026-04-23T15:30:00Z",
                    "mapping_status": "complete",
                    "selection": "away",
                    "skip_reason": "",
                    "price_provenance": "exact",
                    "validation_stage": "shadow",
                    "model_prob": 0.55,
                    "top_ask": 0.49,
                    "expected_edge": 0.06,
                    "expected_ev": 0.12,
                    "snapshot_time": "2026-04-23T14:44:58Z",
                    "book_age_seconds": 2.0,
                },
            ]
        )
        fills = pd.DataFrame(
            [
                {"fill_id": "f1", "decision_id": "d1", "status": "settled", "cost_basis": 10.0, "net_profit": 1.0},
                {"fill_id": "f2", "decision_id": "d1", "status": "settled", "cost_basis": 10.0, "net_profit": 1.0},
                {"fill_id": "f3", "decision_id": "d5", "status": "open", "cost_basis": 10.0, "net_profit": 0.0},
            ]
        )

        report, ledger = _build_forward_sample_report(
            decisions,
            fills,
            decisions,
            fills,
            decision_book_freshness_seconds=5,
        )
        flagged = _forward_sample_flags(decisions, decision_book_freshness_seconds=5)

        self.assertEqual(int(flagged["valid_forward_sample"].sum()), 2)
        self.assertEqual(report["sample_status"], "coverage_blocked")
        self.assertEqual(report["cumulative"]["valid_forward_decisions"], 2)
        self.assertEqual(report["cumulative"]["settled_unique_decisions"], 1)
        self.assertEqual(report["cumulative"]["settled_fill_rows"], 2)
        self.assertAlmostEqual(report["cumulative"]["net_roi"], 0.10, places=6)
        self.assertFalse(report["actionable_roi"])
        self.assertEqual(report["roi_display_mode"], "hidden_until_sample_ready")
        self.assertIn("fresh_book_rate_below_minimum", report["sample_blockers"][0])
        blockers = {item["forward_sample_blocker"]: item["count"] for item in report["cumulative"]["blockers"]}
        self.assertEqual(blockers["coverage_stale_book"], 1)
        self.assertEqual(blockers["coverage_missing_book"], 1)
        self.assertEqual(blockers["policy_not_in_zone"], 1)
        self.assertIn("valid_forward_sample", ledger.columns)
        self.assertEqual(int(ledger["policy_selected"].sum()), 2)

    def test_forward_sample_classifies_missing_book_as_coverage_not_mapping(self) -> None:
        decisions = pd.DataFrame(
            [
                {
                    "decision_id": "d1",
                    "match_id": "m1",
                    "group_key": "g1",
                    "league_code": "E0",
                    "kickoff_time": "2026-04-19T15:30:00Z",
                    "mapping_status": "missing_book",
                    "selection": "",
                    "skip_reason": "no_book_checkpoint",
                    "price_provenance": "resolution_only",
                    "model_prob": np.nan,
                    "top_ask": np.nan,
                    "snapshot_time": "",
                    "book_age_seconds": np.nan,
                }
            ]
        )

        flagged = _forward_sample_flags(decisions, decision_book_freshness_seconds=15)

        self.assertEqual(int(flagged.iloc[0]["complete_mapping"]), 1)
        self.assertEqual(str(flagged.iloc[0]["forward_sample_blocker"]), "coverage_missing_book")

    def test_derive_fixtures_from_groups_keeps_only_forward_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "pm.sqlite"
            connection = init_polymarket_db(db_path)
            now = pd.Timestamp.now(tz="UTC")
            rows = []
            for suffix, kickoff in (
                ("past", now - pd.Timedelta(days=1)),
                ("future", now + pd.Timedelta(days=1)),
            ):
                rows.append(
                    {
                        "group_key": f"g-{suffix}",
                        "event_slug": f"event-{suffix}",
                        "event_title": f"Home {suffix} vs Away {suffix}",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "sport_code": "epl",
                        "home_team": f"Home {suffix}",
                        "away_team": f"Away {suffix}",
                        "game_start_time": kickoff.isoformat(),
                        "home_market_id": f"h-{suffix}",
                        "draw_market_id": f"d-{suffix}",
                        "away_market_id": f"a-{suffix}",
                        "mapping_status": "complete",
                        "mapping_reason": "",
                        "raw_market_ids_json": "{}",
                        "updated_at": now.isoformat(),
                    }
                )
            _upsert_rows(connection, "pm_market_groups", rows)

            history = pd.DataFrame(
                [
                    {"league_code": "E0", "HomeTeam": "Home future", "AwayTeam": "Away future"},
                    {"league_code": "E0", "HomeTeam": "Home past", "AwayTeam": "Away past"},
                ]
            )
            fixtures = _derive_fixtures_from_groups(connection, history)

            self.assertEqual(fixtures["group_key"].astype(str).tolist(), ["g-future"])
            connection.close()

    def test_forward_sample_manifest_locks_policy_and_t45m_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            manifest = _build_forward_sample_manifest(
                run_id="shadow_polymarket_test",
                db_path=Path(tmpdir) / "pm.sqlite",
                model_path=Path(tmpdir) / "model.joblib",
                policy_bundle_path=Path(tmpdir) / "policy_bundle.json",
                policy_bundle_resolution="latest_polymarket_policy",
                payload={"model_variant": "v3r"},
                probability_source="raw",
                policy_source_mode="retro",
                policy_payload={"edge_threshold": 0.03, "ev_threshold": 0.05},
                policy_coverage_status="coverage_ready",
                policy_bundle_status="promotable_for_forward",
                policy_minimum_quality="history_proxy",
                settings=settings,
                forward_sample_report={"sample_status": "collecting_forward_sample"},
            )

            self.assertFalse(manifest["policy_reoptimized"])
            self.assertFalse(manifest["t45m_policy_touched"])
            self.assertFalse(manifest["live_money_used"])
            self.assertEqual(manifest["decision_offset_minutes"], 45)
            self.assertEqual(manifest["policy_bundle_resolution"], "latest_polymarket_policy")
            self.assertEqual(manifest["forward_sample_targets"]["valid_forward_decisions"], 100)

    def test_decision_window_diagnostics_reports_next_t45m_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            now = pd.Timestamp("2026-04-19T12:00:00Z")
            groups = pd.DataFrame(
                [
                    {
                        "group_key": "g1",
                        "league_code": "E0",
                        "mapping_status": "complete",
                        "game_start_time": "2026-04-19T14:00:00Z",
                    }
                ]
            )

            diagnostics = _decision_window_diagnostics(groups, settings=settings, now=now)

            self.assertEqual(diagnostics["status"], "ready_to_capture")
            self.assertEqual(diagnostics["upcoming_complete_groups"], 1)
            self.assertEqual(diagnostics["seconds_until_next_decision"], 4500.0)
            self.assertGreaterEqual(diagnostics["recommended_stream_seconds"], 5100)

    def test_forward_sample_cycle_script_supports_dry_run(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_forward_sample_cycle.ps1"
        text = script.read_text(encoding="utf-8")
        self.assertIn("[switch]$DryRun", text)
        self.assertIn("$HeartbeatSeconds", text)
        self.assertIn("$CaptureChunkSeconds", text)
        self.assertIn("$MaxCollectRetries", text)
        self.assertIn("Get-ForwardDbSnapshot", text)
        self.assertIn("Invoke-ForwardCommandWithHeartbeat", text)
        self.assertIn("Invoke-CollectPolymarketSafely", text)
        self.assertIn("collect_chunks_planned", text)
        self.assertIn("[heartbeat] phase=", text)
        self.assertIn("collect-polymarket", text)
        self.assertIn("shadow-polymarket", text)

    def test_stream_polymarket_keeps_rest_checkpoints_when_websocket_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = _settings(tmpdir)
            connection = init_polymarket_db(root / "pm.sqlite")
            _upsert_rows(
                connection,
                "pm_market_catalog",
                [
                    {
                        "market_id": "1",
                        "event_id": "e1",
                        "event_slug": "epl-ars-che",
                        "event_title": "Arsenal vs Chelsea",
                        "market_slug": "arsenal-vs-chelsea-home",
                        "question": "Will Arsenal beat Chelsea?",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "sport_code": "epl",
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "market_role": "home",
                        "game_start_time": "2026-04-20T20:00:00Z",
                        "yes_token_id": "11",
                        "no_token_id": "12",
                        "fees_enabled": 1,
                        "fee_rate": 0.03,
                        "status": "open",
                        "active": 1,
                        "closed": 0,
                        "accepting_orders": 1,
                        "raw_json": "{}",
                        "updated_at": "2026-04-17T00:00:00Z",
                    },
                    {
                        "market_id": "2",
                        "event_id": "e1",
                        "event_slug": "epl-ars-che",
                        "event_title": "Arsenal vs Chelsea",
                        "market_slug": "arsenal-vs-chelsea-draw",
                        "question": "Will Arsenal vs Chelsea end in a draw?",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "sport_code": "epl",
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "market_role": "draw",
                        "game_start_time": "2026-04-20T20:00:00Z",
                        "yes_token_id": "21",
                        "no_token_id": "22",
                        "fees_enabled": 1,
                        "fee_rate": 0.03,
                        "status": "open",
                        "active": 1,
                        "closed": 0,
                        "accepting_orders": 1,
                        "raw_json": "{}",
                        "updated_at": "2026-04-17T00:00:00Z",
                    },
                    {
                        "market_id": "3",
                        "event_id": "e1",
                        "event_slug": "epl-ars-che",
                        "event_title": "Arsenal vs Chelsea",
                        "market_slug": "arsenal-vs-chelsea-away",
                        "question": "Will Chelsea beat Arsenal?",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "sport_code": "epl",
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "market_role": "away",
                        "game_start_time": "2026-04-20T20:00:00Z",
                        "yes_token_id": "31",
                        "no_token_id": "32",
                        "fees_enabled": 1,
                        "fee_rate": 0.03,
                        "status": "open",
                        "active": 1,
                        "closed": 0,
                        "accepting_orders": 1,
                        "raw_json": "{}",
                        "updated_at": "2026-04-17T00:00:00Z",
                    },
                ],
            )
            _upsert_rows(
                connection,
                "pm_market_groups",
                [
                    {
                        "group_key": "epl-ars-che-2026-04-20",
                        "event_slug": "epl-ars-che",
                        "event_title": "Arsenal vs Chelsea",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "sport_code": "epl",
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "game_start_time": "2026-04-20T20:00:00Z",
                        "home_market_id": "1",
                        "draw_market_id": "2",
                        "away_market_id": "3",
                        "mapping_status": "complete",
                        "mapping_reason": "",
                        "raw_market_ids_json": "{}",
                        "updated_at": "2026-04-17T00:00:00Z",
                    }
                ],
            )

            class _TimeoutClob:
                async def stream_market(self, **kwargs):
                    raise TimeoutError("timed out during opening handshake")

                async def stream_sports(self, **kwargs):
                    return None

                def get_order_book(self, token_id: str):
                    return {
                        "timestamp": 1_000_000,
                        "asset_id": str(token_id),
                        "bids": [{"price": 0.49, "size": 10}],
                        "asks": [{"price": 0.51, "size": 10}],
                    }

                def get_last_trade_price(self, token_id: str):
                    return {"price": 0.50, "side": "buy"}

            diagnostics = asyncio.run(
                _stream_polymarket_updates(connection, settings=settings, clob=_TimeoutClob(), stream_seconds=1)
            )

            self.assertEqual(diagnostics["market_stream_errors"], 1)
            self.assertIn("TimeoutError", diagnostics["market_stream_last_error"])
            self.assertEqual(diagnostics["sports_stream_completed"], 1)
            checkpoint_rows = connection.execute(
                "SELECT COUNT(*) FROM pm_book_checkpoints WHERE event_type = 'periodic_checkpoint'"
            ).fetchone()[0]
            self.assertEqual(checkpoint_rows, 3)
            connection.close()

    def test_shadow_summary_breaks_down_blockers_and_coverage(self) -> None:
        decisions = pd.DataFrame(
            [
                {
                    "decision_id": "d1",
                    "group_key": "",
                    "skip_reason": "no_market_group",
                    "mapping_status": "missing",
                    "selection": "",
                    "snapshot_time": "",
                    "book_age_seconds": np.nan,
                    "created_at": "2026-04-19T14:00:00Z",
                },
                {
                    "decision_id": "d2",
                    "group_key": "g2",
                    "skip_reason": "no_book_checkpoint",
                    "mapping_status": "missing_book",
                    "selection": "",
                    "snapshot_time": "",
                    "book_age_seconds": np.nan,
                    "created_at": "2026-04-19T14:01:00Z",
                },
                {
                    "decision_id": "d3",
                    "group_key": "g3",
                    "skip_reason": "stale_book",
                    "mapping_status": "stale_book",
                    "selection": "",
                    "snapshot_time": "2026-04-19T14:44:40Z",
                    "book_age_seconds": 600.0,
                    "created_at": "2026-04-19T14:02:00Z",
                },
                {
                    "decision_id": "d4",
                    "group_key": "g4",
                    "skip_reason": "policy_rejected",
                    "mapping_status": "complete",
                    "selection": "",
                    "snapshot_time": "2026-04-19T14:44:58Z",
                    "book_age_seconds": 2.0,
                    "created_at": "2026-04-19T14:03:00Z",
                },
                {
                    "decision_id": "d5",
                    "group_key": "g5",
                    "skip_reason": "",
                    "mapping_status": "complete",
                    "selection": "home",
                    "snapshot_time": "2026-04-19T14:44:59Z",
                    "book_age_seconds": 1.0,
                    "created_at": "2026-04-19T14:04:00Z",
                },
            ]
        )
        fills = pd.DataFrame(
            [
                {
                    "fill_id": "f1",
                    "decision_id": "d5",
                    "status": "settled",
                    "notional": 10.0,
                    "cost_basis": 9.0,
                    "net_profit": 1.0,
                    "fill_rate": 1.0,
                    "partial_fill": 0,
                    "fee_paid": 0.05,
                    "raw_vwap": 0.51,
                    "effective_vwap": 0.52,
                    "top_ask": 0.50,
                    "created_at": "2026-04-19T14:05:00Z",
                }
            ]
        )
        mappings = pd.DataFrame(
            [
                {"mapping_status": "complete"},
                {"mapping_status": "complete"},
                {"mapping_status": "unmapped"},
            ]
        )

        summary = _summarize_shadow(
            decisions,
            fills,
            mappings,
            decision_window_minutes=45,
            decision_book_freshness_seconds=5,
        )

        self.assertEqual(summary["primary_blocker"], "coverage")
        self.assertEqual(summary["blocker_totals"]["coverage"], 2)
        self.assertEqual(summary["blocker_totals"]["mapping"], 1)
        self.assertEqual(summary["blocker_totals"]["policy"], 1)
        self.assertEqual(summary["decision_coverage"]["total_decisions"], 5)
        self.assertEqual(summary["decision_coverage"]["decisions_with_book_snapshot"], 3)
        self.assertEqual(summary["decision_coverage"]["decisions_with_fresh_book"], 2)
        self.assertEqual(summary["decision_coverage"]["decisions_with_stale_book"], 1)
        self.assertEqual(summary["decision_coverage"]["decisions_with_missing_book"], 2)
        self.assertEqual(summary["decision_coverage"]["book_age_buckets"][0]["bucket"], "0-5m")
        self.assertEqual(summary["decision_coverage"]["book_age_buckets"][0]["count"], 2)
        self.assertEqual(summary["decision_coverage"]["timing_blockers"], {"no_book_checkpoint": 1, "stale_book": 1, "no_ask_ladder": 0})
        self.assertAlmostEqual(summary["decision_coverage"]["timing_failure_rate"], 2.0 / 5.0, places=6)
        self.assertTrue(any(item["skip_reason"] == "no_book_checkpoint" for item in summary["skip_reasons"]))
        self.assertTrue(any(item["blocker"] == "coverage" for item in summary["skip_reason_blockers"]))

    def test_capture_rest_books_distinguishes_missing_from_failed_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = _settings(tmpdir)
            connection = init_polymarket_db(root / "pm.sqlite")
            _upsert_rows(
                connection,
                "pm_market_catalog",
                [
                    {
                        "market_id": "1",
                        "event_id": "e1",
                        "event_slug": "epl-ars-che",
                        "event_title": "Arsenal vs Chelsea",
                        "market_slug": "arsenal-vs-chelsea-home",
                        "question": "Will Arsenal beat Chelsea?",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "sport_code": "epl",
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "market_role": "home",
                        "game_start_time": "2026-04-19T15:30:00Z",
                        "yes_token_id": "11",
                        "no_token_id": "12",
                        "fees_enabled": 1,
                        "fee_rate": 0.03,
                        "status": "open",
                        "active": 1,
                        "closed": 0,
                        "accepting_orders": 1,
                        "raw_json": "{}",
                        "updated_at": "2026-04-17T00:00:00Z",
                    },
                    {
                        "market_id": "2",
                        "event_id": "e1",
                        "event_slug": "epl-ars-che",
                        "event_title": "Arsenal vs Chelsea",
                        "market_slug": "arsenal-vs-chelsea-draw",
                        "question": "Will Arsenal vs Chelsea end in a draw?",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "sport_code": "epl",
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "market_role": "draw",
                        "game_start_time": "2026-04-19T15:30:00Z",
                        "yes_token_id": "21",
                        "no_token_id": "22",
                        "fees_enabled": 1,
                        "fee_rate": 0.03,
                        "status": "open",
                        "active": 1,
                        "closed": 0,
                        "accepting_orders": 1,
                        "raw_json": "{}",
                        "updated_at": "2026-04-17T00:00:00Z",
                    },
                    {
                        "market_id": "3",
                        "event_id": "e1",
                        "event_slug": "epl-ars-che",
                        "event_title": "Arsenal vs Chelsea",
                        "market_slug": "arsenal-vs-chelsea-away",
                        "question": "Will Chelsea beat Arsenal?",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "sport_code": "epl",
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "market_role": "away",
                        "game_start_time": "2026-04-19T15:30:00Z",
                        "yes_token_id": "31",
                        "no_token_id": "32",
                        "fees_enabled": 1,
                        "fee_rate": 0.03,
                        "status": "open",
                        "active": 1,
                        "closed": 0,
                        "accepting_orders": 1,
                        "raw_json": "{}",
                        "updated_at": "2026-04-17T00:00:00Z",
                    },
                ],
            )
            _upsert_rows(
                connection,
                "pm_market_groups",
                [
                    {
                        "group_key": "epl-ars-che-2026-04-19",
                        "event_slug": "epl-ars-che",
                        "event_title": "Arsenal vs Chelsea",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "sport_code": "epl",
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "game_start_time": "2026-04-19T15:30:00Z",
                        "home_market_id": "1",
                        "draw_market_id": "2",
                        "away_market_id": "3",
                        "mapping_status": "complete",
                        "mapping_reason": "",
                        "raw_market_ids_json": "{}",
                        "updated_at": "2026-04-17T00:00:00Z",
                    }
                ],
            )

            class _FakeClob:
                def get_order_book(self, token_id: str):
                    if str(token_id) == "11":
                        return {"timestamp": 1_000_000, "asset_id": "11", "bids": [{"price": 0.49, "size": 10}], "asks": [{"price": 0.51, "size": 10}]}
                    if str(token_id) == "21":
                        response = requests.Response()
                        response.status_code = 404
                        raise requests.HTTPError(response=response)
                    raise requests.RequestException("boom")

                def get_last_trade_price(self, token_id: str):
                    return {"price": 0.50, "side": "buy"}

            result = capture_rest_books(connection, settings=settings, clob=_FakeClob(), event_type="decision_checkpoint")
            capture = result["capture"]
            self.assertEqual(capture["book_fetch_attempts"], 3)
            self.assertEqual(capture["book_fetch_successes"], 1)
            self.assertEqual(capture["book_fetch_missing"], 1)
            self.assertEqual(capture["book_fetch_failed"], 2)
            self.assertEqual(result["checkpoint_rows"], 1)
            self.assertEqual(result["trade_rows"], 1)
            connection.close()

    def test_report_polymarket_surfaces_primary_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "source_mode": "forward_t45m",
                "settled_bets": 0,
                "net_roi": 0.0,
                "net_pnl": 0.0,
                "fill_rate": 0.0,
                "partial_fill_rate": 0.0,
                "mapping_precision": 0.5,
                "primary_blocker": "coverage",
                "primary_blocker_count": 4,
                "blocker_totals": {"coverage": 4, "mapping": 2, "policy": 1},
                "decision_coverage": {
                    "decisions_with_book_snapshot": 1,
                    "total_decisions": 6,
                    "decisions_with_fresh_book": 1,
                    "decisions_with_stale_book": 0,
                    "decisions_with_missing_book": 5,
                },
                "skip_reasons": [
                    {"skip_reason": "no_book_checkpoint", "count": 4},
                    {"skip_reason": "no_market_group", "count": 2},
                    {"skip_reason": "policy_rejected", "count": 1},
                ],
            }
            (root / "shadow_summary.json").write_text(json.dumps(summary), encoding="utf-8")

            _, text = report_polymarket(root)

            self.assertIn("Primary blocker: coverage (4)", text)
            self.assertIn("Decision-window coverage: 1/6 snapshots", text)
            self.assertIn("Skip reasons: no_book_checkpoint=4", text)


if __name__ == "__main__":
    unittest.main()
