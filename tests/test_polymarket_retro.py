from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from predicciones.config import (
    BacktestConfig,
    ExecutionConfig,
    PolymarketConfig,
    ProjectPaths,
    ResearchConfig,
    Settings,
    SnapshotConfig,
)
from predicciones.dataset import (
    build_feature_rows,
    feature_family_columns,
    model_feature_columns_for_variant,
    variant_feature_manifest,
)
from predicciones.polymarket_retro import (
    _blend_probabilities,
    _build_fixture_groups_v2,
    _classify_market_shape,
    _confidence_backoff_report,
    _confidence_scores,
    _coverage_summary,
    _mapping_audit_rows,
    _oof_frozen_policy_report,
    _decision_region_report,
    _pre_holdout_windows,
    _prune_feature_columns_for_fold,
    _parse_beat_question,
    _parse_draw_question,
    _parse_slug_fixture,
    _parse_who_will_win_game,
    _model_status,
    _needs_backfill,
    _rank_policies,
    _retro_price_payload,
    _select_probability_source,
)
from predicciones.decision_region_adjustment import (
    REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK,
    REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK,
    apply_regional_adjustment,
    fit_regional_adjustment_model,
)
from predicciones.decision_region_model import (
    DECISION_SCORER_LOGIT_PROB,
    DECISION_SCORER_RELIABILITY_PROB,
    TRAINING_SCOPE_ELIGIBLE,
    crossfit_decision_region_scores,
)
from predicciones.candidate_stability_gate import (
    STABILITY_GATE_EXPANDING_OUTCOME,
    STABILITY_GATE_HIERARCHICAL,
    STABILITY_GATE_OUTCOME_PARENT,
    STABILITY_GATE_SOFT_PARENT,
    STABILITY_GATE_VALIDATED_OUTCOME,
    STABILITY_GATE_VALIDATED_OUTCOME_SOFT,
    apply_stability_gate,
    crossfit_stability_gate,
    fit_stability_gate_model,
)
from predicciones.polymarket_shadow import (
    _build_catalog_from_events,
    _decision_from_prediction,
    _infer_league_from_event,
    _resolve_team_alias,
    _team_match_score,
    _upsert_rows,
    init_polymarket_db,
)
from predicciones.strategy import BetPolicy, candidate_score_columns, select_candidate_rows


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
        polymarket=PolymarketConfig(
            book_freshness_seconds=5,
            historical_proxy_haircut=0.03,
            retro_min_mapped_matches=2,
            retro_min_selected_predictions=2,
            mapping_score_threshold=0.8,
            mapping_score_gap=0.02,
        ),
        backtest=BacktestConfig(),
    )


class _UnusedClob:
    def get_prices_history(self, *args, **kwargs):  # pragma: no cover - no debe llamarse en el test local-first
        raise AssertionError("No deberia consultar prices-history remoto en este test.")


class PolymarketRetroTests(unittest.TestCase):
    def test_polymarket_backfill_refreshes_when_cached_groups_do_not_cover_new_history_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            connection = init_polymarket_db(Path(tmpdir) / "pm.sqlite")
            matches = pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2025-08-15", "2026-04-01"]),
                    "league_code": ["E0", "E0"],
                    "HomeTeam": ["Arsenal", "Chelsea"],
                    "AwayTeam": ["Chelsea", "Arsenal"],
                }
            )
            _upsert_rows(
                connection,
                "pm_market_groups",
                [
                    {
                        "group_key": "old",
                        "game_start_time": "2025-05-25T00:00:00+00:00",
                        "mapping_status": "complete",
                    }
                ],
            )
            self.assertTrue(_needs_backfill(connection, matches, settings))
            connection.execute("DELETE FROM pm_market_groups")
            _upsert_rows(
                connection,
                "pm_market_groups",
                [
                    {
                        "group_key": "fresh",
                        "game_start_time": "2026-03-31T00:00:00+00:00",
                        "mapping_status": "complete",
                    }
                ],
            )
            self.assertFalse(_needs_backfill(connection, matches, settings))
            connection.close()

    def test_build_catalog_from_legacy_closed_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            events = {
                "bundesliga-bayern-vs-dortmund": {
                    "slug": "bundesliga-bayern-vs-dortmund",
                    "title": "Bundesliga: Bayern vs. Dortmund",
                    "closed": True,
                    "startTime": "2025-04-12T16:30:00Z",
                    "markets": [
                        {"id": "1", "question": "Will Bayern beat Dortmund?", "outcomes": '["Yes", "No"]', "clobTokenIds": '["11", "12"]', "active": True, "closed": True, "acceptingOrders": False},
                        {"id": "2", "question": "Will Bayern vs. Dortmund end in a draw?", "outcomes": '["Yes", "No"]', "clobTokenIds": '["21", "22"]', "active": True, "closed": True, "acceptingOrders": False},
                        {"id": "3", "question": "Will Dortmund beat Bayern?", "outcomes": '["Yes", "No"]', "clobTokenIds": '["31", "32"]', "active": True, "closed": True, "acceptingOrders": False},
                    ],
                }
            }
            team_lookup = {"D1": ["FC Bayern Munchen", "Borussia Dortmund"]}
            catalog, groups = _build_catalog_from_events(
                settings,
                events,
                "2026-04-17T00:00:00+00:00",
                team_lookup=team_lookup,
            )
            self.assertEqual(len(catalog), 3)
            self.assertEqual(len(groups), 1)
            self.assertEqual(str(groups.iloc[0]["mapping_status"]), "complete")
            self.assertEqual(str(groups.iloc[0]["league_code"]), "D1")

    def test_mapping_audit_uses_exact_alias_and_time_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            predictions = pd.DataFrame(
                [
                    {
                        "match_id": "m1",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "HomeTeam": "Manchester City",
                        "AwayTeam": "Arsenal",
                        "kickoff_time": pd.Timestamp("2026-04-19T15:30:00Z"),
                    }
                ]
            )
            groups = pd.DataFrame(
                [
                    {
                        "group_key": "epl-mci-ars",
                        "event_slug": "epl-mci-ars",
                        "league_code": "E0",
                        "home_team": "Manchester City FC",
                        "away_team": "Arsenal FC",
                        "game_start_time": pd.Timestamp("2026-04-19T15:30:00Z"),
                        "mapping_status": "complete",
                    }
                ]
            )
            alias_frame = pd.DataFrame(
                [
                    {"league_code": "E0", "canonical_team": "Manchester City FC", "alias_key": "manchester city", "alias_text": "Manchester City", "alias_id": "1", "source": "history_matches", "score": 1.0, "updated_at": "2026-04-17T00:00:00+00:00"},
                    {"league_code": "E0", "canonical_team": "Arsenal FC", "alias_key": "arsenal", "alias_text": "Arsenal", "alias_id": "2", "source": "history_matches", "score": 1.0, "updated_at": "2026-04-17T00:00:00+00:00"},
                ]
            )
            audit = _mapping_audit_rows(settings, predictions, groups, alias_frame)
            self.assertEqual(str(audit.iloc[0]["mapping_status"]), "complete")
            self.assertEqual(str(audit.iloc[0]["mapping_stage"]), "exact")
            self.assertEqual(str(audit.iloc[0]["group_key"]), "epl-mci-ars")

    def test_mapping_audit_uses_direct_match_id_for_reprogrammed_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            predictions = pd.DataFrame(
                [
                    {
                        "match_id": "2790",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "HomeTeam": "Everton",
                        "AwayTeam": "Liverpool",
                        "kickoff_time": pd.Timestamp("2025-02-12T19:30:00Z"),
                    }
                ]
            )
            groups = pd.DataFrame(
                [
                    {
                        "group_key": "E0|2025-02-12|everton|liverpool",
                        "event_slug": "epl-eve-liv-2024-12-07",
                        "league_code": "E0",
                        "match_id": "2790",
                        "home_team": "Everton",
                        "away_team": "Liverpool",
                        "game_start_time": pd.Timestamp("2024-12-07T16:14:00Z"),
                        "mapping_status": "complete",
                    }
                ]
            )
            alias_frame = pd.DataFrame(
                [
                    {"league_code": "E0", "canonical_team": "Everton", "alias_key": "everton", "alias_text": "Everton", "alias_id": "1", "source": "history_matches", "score": 1.0, "updated_at": "2026-04-17T00:00:00+00:00"},
                    {"league_code": "E0", "canonical_team": "Liverpool", "alias_key": "liverpool", "alias_text": "Liverpool", "alias_id": "2", "source": "history_matches", "score": 1.0, "updated_at": "2026-04-17T00:00:00+00:00"},
                ]
            )
            audit = _mapping_audit_rows(settings, predictions, groups, alias_frame)
            self.assertEqual(str(audit.iloc[0]["mapping_status"]), "complete")
            self.assertEqual(str(audit.iloc[0]["mapping_stage"]), "match_id")
            self.assertEqual(str(audit.iloc[0]["group_key"]), "E0|2025-02-12|everton|liverpool")
            self.assertEqual(str(audit.iloc[0]["audit_reason"]), "direct_match_id_reprogrammed")

    def test_mapping_audit_rejects_stale_direct_match_id_with_wrong_teams(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            predictions = pd.DataFrame(
                [
                    {
                        "match_id": "2134",
                        "league_code": "E0",
                        "league_name": "Premier League",
                        "HomeTeam": "Burnley",
                        "AwayTeam": "Southampton",
                        "kickoff_time": pd.Timestamp("2019-08-10T15:00:00Z"),
                    }
                ]
            )
            groups = pd.DataFrame(
                [
                    {
                        "group_key": "E0|2024-08-16|man united|fulham",
                        "event_slug": "epl-man-united-fulham-2024-08-16",
                        "league_code": "E0",
                        "match_id": "2134",
                        "home_team": "Man United",
                        "away_team": "Fulham",
                        "game_start_time": pd.Timestamp("2024-08-16T15:00:00Z"),
                        "mapping_status": "complete",
                    }
                ]
            )
            audit = _mapping_audit_rows(settings, predictions, groups, pd.DataFrame())
            self.assertNotEqual(str(audit.iloc[0]["mapping_status"]), "complete")
            self.assertEqual(str(audit.iloc[0]["mapping_status"]), "out_of_window")
            self.assertEqual(str(audit.iloc[0]["audit_reason"]), "direct_match_id_rejected_team_or_time_mismatch")

    def test_team_alias_logic_handles_common_league_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _ = _settings(tmpdir)
            self.assertEqual(
                _resolve_team_alias("Manchester City", ["Man City", "Arsenal"], league_code="E0"),
                "Man City",
            )
            self.assertEqual(
                _resolve_team_alias("Celta Vigo", ["Celta", "Valencia"], league_code="SP1"),
                "Celta",
            )
            self.assertEqual(
                _resolve_team_alias("Wolverhampton", ["Wolves", "Chelsea"], league_code="E0"),
                "Wolves",
            )
            self.assertEqual(_team_match_score("Ath Madrid", "Atletico Madrid", "SP1"), 1.0)
            self.assertEqual(_team_match_score("FC Koln", "1. FC Koln", "D1"), 1.0)

    def test_market_parsers_cover_draw_beat_and_slug(self) -> None:
        self.assertEqual(
            _parse_draw_question("Will Real Madrid vs. Atletico Madrid end in a draw?"),
            ("Real Madrid", "Atletico Madrid"),
        )
        self.assertEqual(
            _parse_draw_question("Will Real Madrid vs. Atletico Madrid end in a draw"),
            ("Real Madrid", "Atletico Madrid"),
        )
        self.assertEqual(
            _parse_draw_question("Will the Germany vs. Scotland match be a Draw?"),
            ("Germany", "Scotland"),
        )
        self.assertEqual(
            _parse_beat_question("Will Real Madrid beat Atletico Madrid?"),
            ("Real Madrid", "Atletico Madrid"),
        )
        self.assertEqual(
            _parse_who_will_win_game("Premier League: Who will win the Crystal Palace vs. Liverpool game on August 15?"),
            ("Crystal Palace", "Liverpool"),
        )
        self.assertEqual(
            _parse_slug_fixture("la-liga-real-madrid-vs-athletico-madrid-2025-02-08"),
            ("real madrid", "athletico madrid"),
        )

    def test_league_inference_ignores_out_of_scope_events_with_epl_tag(self) -> None:
        event = {
            "title": "Euro 2024: Germany vs. Scotland",
            "seriesSlug": "",
            "tags": [
                {"label": "EPL", "slug": "EPL"},
                {"label": "Euro 2024", "slug": "euro-2024"},
            ],
        }
        self.assertEqual(
            _infer_league_from_event("euro-2024-germany-vs-scotland", event),
            (None, None, None),
        )

    def test_market_shape_distinguishes_true_1x2_and_binary_daily_markets(self) -> None:
        event = {"slug": "epl-dailies-2024-09-14", "title": "EPL"}
        true_market = {
            "question": "Will Arsenal beat Chelsea?",
            "outcomes": '["Yes", "No"]',
        }
        binary_market = {
            "question": "Chelsea vs. Brentford",
            "outcomes": '["CHE", "BRE/Draw"]',
        }
        self.assertEqual(
            _classify_market_shape(event, true_market, "E0", "Arsenal", "Chelsea"),
            ("true_1x2_leg_home", "home"),
        )
        self.assertEqual(
            _classify_market_shape(event, binary_market, "E0", "Chelsea", "Brentford"),
            ("binary_home_vs_away_draw", "home"),
        )

    def test_build_fixture_groups_v2_marks_complete_and_duplicate_roles(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "event_slug": "ars-che",
                    "event_title": "Arsenal vs Chelsea",
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "sport_code": "epl",
                    "match_id": "match-1",
                    "match_date": pd.Timestamp("2025-02-08T00:00:00Z"),
                    "game_start_time": pd.Timestamp("2025-02-08T15:00:00Z"),
                    "fixture_key": "E0|2025-02-08|arsenal|chelsea",
                    "undirected_fixture_key": "E0|2025-02-08|arsenal|chelsea",
                    "home_team_canonical": "Arsenal",
                    "away_team_canonical": "Chelsea",
                    "question": "Will Arsenal beat Chelsea?",
                    "group_item_title": "",
                    "market_slug": "arsenal-vs-chelsea-home",
                    "market_shape": "true_1x2_leg_home",
                    "market_role": "home",
                    "market_quality_rank": 10.0,
                    "classification_status": "candidate_true_1x2",
                    "classification_reason": "matched_true_1x2_candidate",
                    "updated_at": "2026-04-17T00:00:00+00:00",
                },
                {
                    "market_id": "m2",
                    "event_slug": "ars-che",
                    "event_title": "Arsenal vs Chelsea",
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "sport_code": "epl",
                    "match_id": "match-1",
                    "match_date": pd.Timestamp("2025-02-08T00:00:00Z"),
                    "game_start_time": pd.Timestamp("2025-02-08T15:00:00Z"),
                    "fixture_key": "E0|2025-02-08|arsenal|chelsea",
                    "undirected_fixture_key": "E0|2025-02-08|arsenal|chelsea",
                    "home_team_canonical": "Arsenal",
                    "away_team_canonical": "Chelsea",
                    "question": "Will Arsenal vs Chelsea end in a draw?",
                    "group_item_title": "",
                    "market_slug": "arsenal-vs-chelsea-draw",
                    "market_shape": "true_1x2_leg_draw",
                    "market_role": "draw",
                    "market_quality_rank": 10.0,
                    "classification_status": "candidate_true_1x2",
                    "classification_reason": "matched_true_1x2_candidate",
                    "updated_at": "2026-04-17T00:00:00+00:00",
                },
                {
                    "market_id": "m3",
                    "event_slug": "ars-che",
                    "event_title": "Arsenal vs Chelsea",
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "sport_code": "epl",
                    "match_id": "match-1",
                    "match_date": pd.Timestamp("2025-02-08T00:00:00Z"),
                    "game_start_time": pd.Timestamp("2025-02-08T15:00:00Z"),
                    "fixture_key": "E0|2025-02-08|arsenal|chelsea",
                    "undirected_fixture_key": "E0|2025-02-08|arsenal|chelsea",
                    "home_team_canonical": "Arsenal",
                    "away_team_canonical": "Chelsea",
                    "question": "Will Chelsea beat Arsenal?",
                    "group_item_title": "",
                    "market_slug": "arsenal-vs-chelsea-away",
                    "market_shape": "true_1x2_leg_away",
                    "market_role": "away",
                    "market_quality_rank": 10.0,
                    "classification_status": "candidate_true_1x2",
                    "classification_reason": "matched_true_1x2_candidate",
                    "updated_at": "2026-04-17T00:00:00+00:00",
                },
                {
                    "market_id": "m4",
                    "event_slug": "ars-che",
                    "event_title": "Arsenal vs Chelsea",
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "sport_code": "epl",
                    "match_id": "match-1",
                    "match_date": pd.Timestamp("2025-02-08T00:00:00Z"),
                    "game_start_time": pd.Timestamp("2025-02-08T15:00:00Z"),
                    "fixture_key": "E0|2025-02-08|arsenal|chelsea",
                    "undirected_fixture_key": "E0|2025-02-08|arsenal|chelsea",
                    "home_team_canonical": "Arsenal",
                    "away_team_canonical": "Chelsea",
                    "question": "Will Arsenal beat Chelsea? duplicate",
                    "group_item_title": "",
                    "market_slug": "arsenal-vs-chelsea-home-dup",
                    "market_shape": "true_1x2_leg_home",
                    "market_role": "home",
                    "market_quality_rank": 5.0,
                    "classification_status": "candidate_true_1x2",
                    "classification_reason": "matched_true_1x2_candidate",
                    "updated_at": "2026-04-17T00:00:00+00:00",
                },
            ]
        )
        groups, audit = _build_fixture_groups_v2(candidates)
        self.assertEqual(int(groups["group_status"].eq("complete_group").sum()), 1)
        self.assertEqual(int(audit["classification_status"].eq("1x2_counted").sum()), 3)
        self.assertEqual(int(audit["classification_status"].eq("duplicate").sum()), 1)

    def test_retro_price_payload_prefers_local_history_over_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            market_row = pd.Series({"market_id": "1", "fee_rate": 0.03, "fees_enabled": 1})
            price_history = pd.DataFrame(
                [
                    {
                        "market_id": "1",
                        "timestamp": pd.Timestamp("2026-04-19T14:00:00Z"),
                        "price": 0.44,
                        "decision_time": pd.Timestamp("2026-04-19T14:45:00Z"),
                        "lag_seconds": 2700.0,
                    }
                ]
            )
            payload = _retro_price_payload(
                settings=settings,
                checkpoints=pd.DataFrame(columns=["market_id", "timestamp", "asks_json", "asset_id", "event_type"]),
                price_history=price_history,
                clob=_UnusedClob(),
                market_row=market_row,
                decision_time=pd.Timestamp("2026-04-19T14:45:00Z"),
                market_probability=0.41,
            )
            self.assertEqual(payload["quality_tier"], "history_exact")
            self.assertEqual(payload["source_type"], "polymarket_history_local")
            self.assertAlmostEqual(float(payload["top_ask"]), 0.44, places=6)

    def test_forward_decision_applies_policy_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            row = pd.Series(
                {
                    "match_id": "m1",
                    "Date": pd.Timestamp("2026-04-19"),
                    "kickoff_time": pd.Timestamp("2026-04-19T15:30:00Z"),
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
            groups = pd.DataFrame([{"group_key": "epl-mac-ars-2026-04-19", "mapping_status": "complete", "home_market_id": "1", "draw_market_id": "2", "away_market_id": "3"}])
            catalog = pd.DataFrame(
                [
                    {"market_id": "1", "fees_enabled": 1, "fee_rate": 0.03},
                    {"market_id": "2", "fees_enabled": 1, "fee_rate": 0.03},
                    {"market_id": "3", "fees_enabled": 1, "fee_rate": 0.03},
                ]
            )
            checkpoints = pd.DataFrame(
                [
                    {"market_id": "1", "asset_id": "11", "timestamp": pd.Timestamp("2026-04-19T14:44:58Z"), "asks_json": '[{"price": 0.53, "size": 100}]', "event_type": "decision_checkpoint"},
                    {"market_id": "2", "asset_id": "21", "timestamp": pd.Timestamp("2026-04-19T14:44:58Z"), "asks_json": '[{"price": 0.25, "size": 100}]', "event_type": "decision_checkpoint"},
                    {"market_id": "3", "asset_id": "31", "timestamp": pd.Timestamp("2026-04-19T14:44:58Z"), "asks_json": '[{"price": 0.22, "size": 100}]', "event_type": "decision_checkpoint"},
                ]
            )
            strict_policy = BetPolicy(edge_threshold=0.20, ev_threshold=0.10, min_odds=1.2, max_odds=6.0)
            decision, fills = _decision_from_prediction(
                row,
                groups,
                catalog,
                checkpoints,
                settings,
                probability_source="raw",
                policy=strict_policy,
            )
            self.assertEqual(decision["skip_reason"], "policy_rejected")
            self.assertEqual(fills, [])

    def test_coverage_summary_marks_bundle_provisional_when_sample_is_small(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            predictions = pd.DataFrame([{"match_id": "1"}, {"match_id": "2"}, {"match_id": "3"}])
            mappings = pd.DataFrame([{"match_id": "1", "group_key": "g1"}, {"match_id": "2", "group_key": ""}])
            mapping_audit = pd.DataFrame(
                [
                    {"mapping_status": "complete"},
                    {"mapping_status": "missing"},
                    {"mapping_status": "ambiguous_mapping"},
                ]
            )
            candidates = pd.DataFrame([{"quality_tier": "history_proxy", "quote_status": "eligible"}])
            selected = pd.DataFrame([{"quality_tier": "history_proxy"}])
            summary = _coverage_summary(settings, predictions, mappings, mapping_audit, candidates, selected)
            self.assertEqual(summary["coverage_status"], "coverage_limited")
            self.assertEqual(summary["bundle_status"], "provisional")
            self.assertTrue(summary["insufficient_sample"])

    def test_feature_builder_emits_shrunk_columns_and_variant_feature_sets(self) -> None:
        matches = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "Date": pd.Timestamp("2025-01-01"),
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "season": "2425",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea",
                    "FTHG": 2,
                    "FTAG": 0,
                    "outcome": "home",
                },
                {
                    "match_id": 2,
                    "Date": pd.Timestamp("2025-01-08"),
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "season": "2425",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Liverpool",
                    "FTHG": 1,
                    "FTAG": 1,
                    "outcome": "draw",
                },
            ]
        )
        features = build_feature_rows(matches)
        second = features.loc[features["match_id"].eq(2)].iloc[0]
        self.assertEqual(float(second["home_overall_sample_size"]), 1.0)
        self.assertEqual(float(second["home_side_sample_size"]), 1.0)
        self.assertAlmostEqual(float(second["home_exp_attack_overall_shrunk"]), 10.0 / 9.0, places=6)
        self.assertAlmostEqual(float(second["home_exp_attack_home_shrunk"]), 34.0 / 27.0, places=6)
        families = feature_family_columns(features)
        manifest = variant_feature_manifest(
            features,
            variants=["v1", "v2r", "v3r", "v4", "v5", "v6", "v7", "v4b", "v5b", "v6b", "v7b"],
        )
        v1_columns = model_feature_columns_for_variant(features, variant="v1")
        v2r_columns = model_feature_columns_for_variant(features, variant="v2r")
        v3r_columns = model_feature_columns_for_variant(features, variant="v3r")
        v4_columns = model_feature_columns_for_variant(features, variant="v4")
        v5_columns = model_feature_columns_for_variant(features, variant="v5")
        v6_columns = model_feature_columns_for_variant(features, variant="v6")
        v7_columns = model_feature_columns_for_variant(features, variant="v7")
        v4b_columns = model_feature_columns_for_variant(features, variant="v4b")
        v5b_columns = model_feature_columns_for_variant(features, variant="v5b")
        v6b_columns = model_feature_columns_for_variant(features, variant="v6b")
        v7b_columns = model_feature_columns_for_variant(features, variant="v7b")
        self.assertNotIn("home_exp_attack_overall_shrunk", v1_columns)
        self.assertIn("home_exp_attack_overall_shrunk", v2r_columns)
        self.assertNotIn("home_attack_minus_away_defense_shrunk", v2r_columns)
        self.assertIn("home_attack_minus_away_defense_shrunk", v3r_columns)
        self.assertIn("opponent_strength_diff_5", v4_columns)
        self.assertIn("matchup_balance_interaction_shrunk", v5_columns)
        self.assertIn("home_long_sample_ratio_overall", v6_columns)
        self.assertIn("confidence_score_v2", v6_columns)
        self.assertIn("rest_advantage_x_season_progress_pct", v7_columns)
        self.assertIn("home_attack_minus_away_defense_shrunk_x_confidence_score_v2", v7_columns)
        self.assertFalse(set(v1_columns) & set(families["rival_context"]))
        self.assertFalse(set(v1_columns) & set(families["uncertainty"]))
        self.assertFalse(set(v1_columns) & set(families["interactions"]))
        self.assertFalse(set(v1_columns) & set(families["season_state"]))
        self.assertTrue(set(v4_columns) - set(v2r_columns))
        self.assertTrue(set(v5_columns) - set(v3r_columns))
        self.assertTrue(set(v6_columns) - set(v5_columns))
        self.assertTrue(set(v7_columns) - set(v6_columns))
        self.assertTrue(set(v5_columns) & set(families["season_state"]))
        self.assertEqual(v4_columns, v4b_columns)
        self.assertEqual(v5_columns, v5b_columns)
        self.assertEqual(v6_columns, v6b_columns)
        self.assertEqual(v7_columns, v7b_columns)
        manifest_lookup = {item["variant_name"]: item for item in manifest}
        self.assertEqual(manifest_lookup["v4"]["feature_columns"], v4_columns)
        self.assertIn("confidence_backoff", manifest_lookup["v4b"]["feature_families"])

    def test_recent_opponent_features_are_leakage_free(self) -> None:
        matches = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "Date": pd.Timestamp("2025-01-01"),
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "season": "2425",
                    "HomeTeam": "Chelsea",
                    "AwayTeam": "Tottenham",
                    "FTHG": 2,
                    "FTAG": 1,
                    "outcome": "home",
                },
                {
                    "match_id": 2,
                    "Date": pd.Timestamp("2025-01-08"),
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "season": "2425",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea",
                    "FTHG": 1,
                    "FTAG": 0,
                    "outcome": "home",
                },
                {
                    "match_id": 3,
                    "Date": pd.Timestamp("2025-01-15"),
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "season": "2425",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Liverpool",
                    "FTHG": 0,
                    "FTAG": 1,
                    "outcome": "away",
                },
            ]
        )
        features = build_feature_rows(matches)
        third = features.loc[features["match_id"].eq(3)].iloc[0]
        self.assertAlmostEqual(float(third["home_opponent_attack_overall_5"]), 14.0 / 9.0, places=6)
        self.assertEqual(float(third["home_sample_adequacy_overall_5"]), 0.0)
        self.assertEqual(float(third["home_low_confidence_overall"]), 1.0)

    def test_confidence_scores_and_blend_are_bounded_and_exact(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "home_shrinkage_ratio_overall": 0.50,
                    "away_shrinkage_ratio_overall": 0.50,
                    "home_shrinkage_ratio_side": 0.60,
                    "away_shrinkage_ratio_side": 0.60,
                    "home_long_sample_ratio_overall": 1.0,
                    "away_long_sample_ratio_overall": 1.0,
                    "home_long_sample_ratio_side": 1.0,
                    "away_long_sample_ratio_side": 1.0,
                    "season_progress_pct": 0.50,
                    "home_opponent_strength_volatility_20": 0.15,
                    "away_opponent_strength_volatility_20": 0.15,
                    "home_goal_volatility_20": 0.30,
                    "away_goal_volatility_20": 0.30,
                    "home_goals_against_volatility_20": 0.30,
                    "away_goals_against_volatility_20": 0.30,
                },
                {
                    "home_shrinkage_ratio_overall": 0.10,
                    "away_shrinkage_ratio_overall": 0.10,
                    "home_shrinkage_ratio_side": 0.10,
                    "away_shrinkage_ratio_side": 0.10,
                    "home_long_sample_ratio_overall": 0.0,
                    "away_long_sample_ratio_overall": 0.0,
                    "home_long_sample_ratio_side": 0.0,
                    "away_long_sample_ratio_side": 0.0,
                    "season_progress_pct": 0.10,
                    "home_opponent_strength_volatility_20": 3.0,
                    "away_opponent_strength_volatility_20": 3.0,
                    "home_goal_volatility_20": 3.0,
                    "away_goal_volatility_20": 3.0,
                    "home_goals_against_volatility_20": 3.0,
                    "away_goals_against_volatility_20": 3.0,
                },
            ]
        )
        primary = pd.DataFrame([[0.7, 0.2, 0.1], [0.2, 0.2, 0.6]], dtype=float).to_numpy()
        fallback = pd.DataFrame([[0.4, 0.3, 0.3], [0.5, 0.2, 0.3]], dtype=float).to_numpy()
        scores = _confidence_scores(rows, primary, fallback)
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 1.0)
        expected_first = 0.30 * 0.55 + 0.30 * 1.0 + 0.25 * (1.0 - (0.25 / 1.5)) + 0.15 * 0.50
        self.assertAlmostEqual(float(scores[0]), expected_first, places=6)
        divergence = np.array([0.30, 0.30], dtype=float)
        blended = _blend_probabilities(primary, fallback, scores, divergence)
        first_weight = np.clip(0.45 * scores[0] - 0.25 * divergence[0], 0.0, 1.0)
        second_weight = np.clip(0.45 * scores[1] - 0.25 * divergence[1], 0.0, 1.0)
        self.assertAlmostEqual(float(blended[0, 0]), float(first_weight * 0.7 + (1.0 - first_weight) * 0.4), places=6)
        self.assertAlmostEqual(float(blended[1, 2]), float(second_weight * 0.6 + (1.0 - second_weight) * 0.3), places=6)

    def test_regional_adjustment_learns_region_penalty_and_emits_adjusted_columns(self) -> None:
        candidate_rows = pd.DataFrame(
            [
                {"selection": "draw", "quoted_odds": 3.5, "model_prob_raw": 0.40, "won": 0},
                {"selection": "draw", "quoted_odds": 3.4, "model_prob_raw": 0.38, "won": 0},
                {"selection": "draw", "quoted_odds": 3.6, "model_prob_raw": 0.37, "won": 0},
                {"selection": "draw", "quoted_odds": 3.2, "model_prob_raw": 0.36, "won": 0},
                {"selection": "draw", "quoted_odds": 3.8, "model_prob_raw": 0.35, "won": 0},
                {"selection": "draw", "quoted_odds": 3.3, "model_prob_raw": 0.34, "won": 0},
                {"selection": "draw", "quoted_odds": 3.7, "model_prob_raw": 0.33, "won": 0},
                {"selection": "draw", "quoted_odds": 3.1, "model_prob_raw": 0.32, "won": 0},
                {"selection": "draw", "quoted_odds": 3.9, "model_prob_raw": 0.31, "won": 0},
                {"selection": "draw", "quoted_odds": 3.0, "model_prob_raw": 0.30, "won": 0},
                {"selection": "draw", "quoted_odds": 3.25, "model_prob_raw": 0.29, "won": 0},
                {"selection": "draw", "quoted_odds": 3.45, "model_prob_raw": 0.28, "won": 0},
                {"selection": "home", "quoted_odds": 1.8, "model_prob_raw": 0.62, "won": 1},
                {"selection": "home", "quoted_odds": 1.9, "model_prob_raw": 0.58, "won": 1},
                {"selection": "home", "quoted_odds": 1.7, "model_prob_raw": 0.56, "won": 1},
            ]
        )
        model = fit_regional_adjustment_model(
            candidate_rows,
            probability_source="raw",
            prior_strength=4.0,
            min_rows=4,
        )
        self.assertIsNotNone(model)
        assert model is not None
        draw_region = model.factors["draw|3-6"]
        self.assertLess(float(draw_region["factor"]), float(model.global_factor))
        adjusted = apply_regional_adjustment(candidate_rows, model)
        self.assertIn("regional_adjusted_prob", adjusted.columns)
        self.assertIn("regional_adjusted_edge", adjusted.columns)
        self.assertIn("regional_adjusted_ev", adjusted.columns)
        self.assertTrue(np.isfinite(adjusted["regional_adjusted_prob"]).all())
        self.assertEqual(
            set(adjusted["regional_adjustment_mode"].astype(str).unique()),
            {REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK},
        )
        penalized = adjusted[adjusted["selection"].eq("draw")]["regional_adjusted_prob"]
        baseline = adjusted[adjusted["selection"].eq("draw")]["model_prob_raw"]
        self.assertTrue((penalized < baseline).all())

    def test_soft_regional_adjustment_preserves_more_probability_than_hard_shrink(self) -> None:
        candidate_rows = pd.DataFrame(
            [
                {"selection": "draw", "quoted_odds": 3.5, "model_prob_raw": 0.40, "won": 0},
                {"selection": "draw", "quoted_odds": 3.4, "model_prob_raw": 0.38, "won": 0},
                {"selection": "draw", "quoted_odds": 3.6, "model_prob_raw": 0.37, "won": 0},
                {"selection": "draw", "quoted_odds": 3.2, "model_prob_raw": 0.36, "won": 0},
                {"selection": "home", "quoted_odds": 1.8, "model_prob_raw": 0.62, "won": 1},
                {"selection": "home", "quoted_odds": 1.9, "model_prob_raw": 0.58, "won": 1},
                {"selection": "home", "quoted_odds": 1.7, "model_prob_raw": 0.56, "won": 1},
                {"selection": "home", "quoted_odds": 1.6, "model_prob_raw": 0.60, "won": 1},
            ]
        )
        hard_model = fit_regional_adjustment_model(
            candidate_rows,
            probability_source="raw",
            prior_strength=2.0,
            min_rows=2,
        )
        soft_model = fit_regional_adjustment_model(
            candidate_rows,
            probability_source="raw",
            adjustment_mode=REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK,
            prior_strength=2.0,
            min_rows=2,
        )
        assert hard_model is not None and soft_model is not None
        hard = apply_regional_adjustment(candidate_rows, hard_model)
        soft = apply_regional_adjustment(candidate_rows, soft_model)
        draw_mask = candidate_rows["selection"].eq("draw")
        self.assertEqual(set(soft["regional_adjustment_mode"].astype(str)), {REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK})
        self.assertTrue((soft.loc[draw_mask, "regional_adjusted_prob"] > hard.loc[draw_mask, "regional_adjusted_prob"]).all())
        self.assertTrue((soft.loc[draw_mask, "regional_adjusted_prob"] < candidate_rows.loc[draw_mask, "model_prob_raw"]).all())

    def test_regional_adjustment_trains_on_base_policy_region_when_available(self) -> None:
        policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0)
        noisy_non_policy_rows = [
            {
                "selection": "draw",
                "quoted_odds": 3.4,
                "model_prob_raw": 0.34,
                "won": 1,
                "quote_status": "eligible",
                "policy_edge": -0.01,
                "policy_ev": -0.02,
            }
            for _ in range(40)
        ]
        bad_policy_draw_rows = [
            {
                "selection": "draw",
                "quoted_odds": 3.5,
                "model_prob_raw": 0.40,
                "won": 0,
                "quote_status": "eligible",
                "policy_edge": 0.11,
                "policy_ev": 0.40,
            }
            for _ in range(12)
        ]
        good_policy_home_rows = [
            {
                "selection": "home",
                "quoted_odds": 1.8,
                "model_prob_raw": 0.62,
                "won": 1,
                "quote_status": "eligible",
                "policy_edge": 0.06,
                "policy_ev": 0.12,
            }
            for _ in range(12)
        ]
        candidate_rows = pd.DataFrame([*noisy_non_policy_rows, *bad_policy_draw_rows, *good_policy_home_rows])
        model = fit_regional_adjustment_model(
            candidate_rows,
            probability_source="raw",
            policy=policy,
            prior_strength=4.0,
            min_rows=4,
        )
        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model.training_scope, "base_policy_region")
        self.assertEqual(model.training_rows, 24)
        self.assertEqual(model.source_rows, len(candidate_rows))
        self.assertLess(float(model.factors["draw|3-6"]["factor"]), float(model.global_factor))

    def test_regional_adjustment_infers_wins_from_actual_outcome(self) -> None:
        candidate_rows = pd.DataFrame(
            [
                {
                    "selection": "home",
                    "actual_outcome": "home",
                    "quoted_odds": 1.8,
                    "model_prob_raw": 0.60,
                    "quote_status": "eligible",
                    "policy_edge": 0.05,
                    "policy_ev": 0.08,
                },
                {
                    "selection": "home",
                    "actual_outcome": "away",
                    "quoted_odds": 1.9,
                    "model_prob_raw": 0.58,
                    "quote_status": "eligible",
                    "policy_edge": 0.05,
                    "policy_ev": 0.10,
                },
                {
                    "selection": "away",
                    "actual_outcome": "away",
                    "quoted_odds": 2.4,
                    "model_prob_raw": 0.45,
                    "quote_status": "eligible",
                    "policy_edge": 0.04,
                    "policy_ev": 0.08,
                },
            ]
        )
        model = fit_regional_adjustment_model(
            candidate_rows,
            probability_source="raw",
            policy=BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0),
            prior_strength=2.0,
            min_rows=1,
        )
        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model.training_scope, "all_eligible_fallback")
        self.assertGreater(model.training_rows, 0)
        self.assertGreater(float(model.global_factor), 0.0)

    def test_stability_gate_blocks_bad_oof_region_and_emits_finite_gated_columns(self) -> None:
        policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0)
        rows = pd.DataFrame(
            [
                {
                    "match_id": f"m{i}",
                    "league_code": "E0",
                    "selection": "draw",
                    "actual_outcome": "home",
                    "quote_status": "eligible",
                    "quoted_odds": 3.5,
                    "top_ask": 0.25,
                    "policy_prob": 0.36,
                    "policy_edge": 0.11,
                    "policy_ev": 0.26,
                    "confidence_score_v2": 0.72,
                    "retro_fold_id": i % 3,
                    "won": 0.0,
                }
                for i in range(30)
            ]
        )
        model = fit_stability_gate_model(rows, policy, probability_source="raw")
        gated = apply_stability_gate(rows, model)
        self.assertEqual(set(gated["stability_gate_mode"].astype(str).unique()), {STABILITY_GATE_HIERARCHICAL})
        self.assertFalse(bool(gated["candidate_gate_pass"].all()))
        self.assertTrue(np.isfinite(gated["gated_prob"]).all())
        self.assertTrue(np.isfinite(gated["gated_edge"]).all())
        self.assertTrue(np.isfinite(gated["gated_ev"]).all())
        self.assertTrue(gated["candidate_gate_reason"].astype(str).isin({"posterior_lower_ev_below_floor", "positive_fold_ratio_below_floor", "flat_roi_below_floor"}).all())

    def test_soft_parent_stability_gate_softens_parent_regions_but_blocks_supported_specific_regions(self) -> None:
        policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0)
        parent_rows = pd.DataFrame(
            [
                {
                    "match_id": f"parent_{i}",
                    "league_code": f"L{i % 6}",
                    "selection": "draw",
                    "actual_outcome": "home",
                    "quote_status": "eligible",
                    "quoted_odds": [1.8, 2.4, 3.5][i % 3],
                    "top_ask": 0.25,
                    "policy_prob": 0.36,
                    "policy_edge": 0.11,
                    "policy_ev": 0.26,
                    "confidence_score_v2": 0.72,
                    "retro_fold_id": i % 3,
                    "won": 0.0,
                }
                for i in range(30)
            ]
        )
        parent_model = fit_stability_gate_model(
            parent_rows,
            policy,
            probability_source="raw",
            mode=STABILITY_GATE_SOFT_PARENT,
        )
        parent_gated = apply_stability_gate(parent_rows, parent_model)
        self.assertEqual(set(parent_gated["stability_gate_mode"].astype(str).unique()), {STABILITY_GATE_SOFT_PARENT})
        self.assertTrue(bool(parent_gated["candidate_gate_pass"].all()))
        self.assertTrue(parent_gated["candidate_gate_reason"].astype(str).eq("parent_soft_adjusted").any())
        self.assertTrue((parent_gated["gated_prob"] < parent_gated["policy_prob"]).all())

        specific_rows = parent_rows.copy()
        specific_rows["match_id"] = [f"specific_{i}" for i in range(len(specific_rows))]
        specific_rows["league_code"] = "E0"
        specific_rows["quoted_odds"] = 3.5
        specific_model = fit_stability_gate_model(
            specific_rows,
            policy,
            probability_source="raw",
            mode=STABILITY_GATE_SOFT_PARENT,
        )
        specific_gated = apply_stability_gate(specific_rows, specific_model)
        self.assertFalse(bool(specific_gated["candidate_gate_pass"].all()))
        self.assertTrue(specific_gated["effective_region_level"].astype(str).isin({"league_selection_odds", "selection_odds"}).all())

    def test_outcome_parent_gate_can_block_supported_selection_parent(self) -> None:
        policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0)
        rows = pd.DataFrame(
            [
                {
                    "match_id": f"m{i}",
                    "league_code": f"L{i % 8}",
                    "selection": "draw",
                    "actual_outcome": "home",
                    "quote_status": "eligible",
                    "quoted_odds": [1.8, 2.4, 3.5][i % 3],
                    "top_ask": 0.25,
                    "policy_prob": 0.36,
                    "policy_edge": 0.11,
                    "policy_ev": 0.26,
                    "confidence_score_v2": 0.72,
                    "retro_fold_id": i % 3,
                    "won": 0.0,
                }
                for i in range(30)
            ]
        )
        model = fit_stability_gate_model(rows, policy, probability_source="raw", mode=STABILITY_GATE_OUTCOME_PARENT)
        gated = apply_stability_gate(rows, model)
        self.assertEqual(set(gated["stability_gate_mode"].astype(str).unique()), {STABILITY_GATE_OUTCOME_PARENT})
        self.assertFalse(bool(gated["candidate_gate_pass"].all()))
        self.assertTrue(gated["effective_region_level"].astype(str).eq("selection").any())

    def test_expanding_stability_gate_trains_holdout_on_oof_plus_pre_holdout(self) -> None:
        policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0)
        rows = pd.DataFrame(
            [
                {
                    "match_id": f"m{i}",
                    "league_code": "E0",
                    "selection": "home",
                    "actual_outcome": "home" if i % 2 == 0 else "away",
                    "quote_status": "eligible",
                    "quoted_odds": 2.0,
                    "top_ask": 0.45,
                    "policy_prob": 0.55,
                    "policy_edge": 0.10,
                    "policy_ev": 0.10,
                    "confidence_score_v2": 0.75,
                    "retro_fold_id": (i % 3) + 1,
                    "won": 1.0 if i % 2 == 0 else 0.0,
                }
                for i in range(30)
            ]
        )
        pre_holdout = rows.iloc[:9].copy()
        holdout = rows.iloc[9:18].copy()
        _, _, _, diagnostics, _ = crossfit_stability_gate(
            rows,
            pre_holdout,
            holdout,
            policy=policy,
            probability_source="raw",
            fold_column="retro_fold_id",
            mode=STABILITY_GATE_EXPANDING_OUTCOME,
        )
        self.assertEqual(diagnostics["holdout_training_segments"], ["oof", "pre_holdout"])
        self.assertEqual(int(diagnostics["holdout_training_rows"]), len(rows) + len(pre_holdout))

    def test_validated_outcome_gate_learns_stable_outcome_without_hardcoding_home(self) -> None:
        policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0)
        rows = pd.DataFrame(
            [
                {
                    "match_id": f"stable_{i}",
                    "league_code": f"L{i % 4}",
                    "selection": "home",
                    "actual_outcome": "home" if i % 3 != 0 else "away",
                    "quote_status": "eligible",
                    "quoted_odds": 2.1,
                    "top_ask": 0.43,
                    "policy_prob": 0.55,
                    "policy_edge": 0.12,
                    "policy_ev": 0.155,
                    "confidence_score_v2": 0.75,
                    "retro_fold_id": (i % 3) + 1,
                    "won": 1.0 if i % 3 != 0 else 0.0,
                }
                for i in range(18)
            ]
            + [
                {
                    "match_id": f"unstable_{i}",
                    "league_code": f"L{i % 4}",
                    "selection": "draw",
                    "actual_outcome": "away",
                    "quote_status": "eligible",
                    "quoted_odds": 3.4,
                    "top_ask": 0.26,
                    "policy_prob": 0.36,
                    "policy_edge": 0.10,
                    "policy_ev": 0.224,
                    "confidence_score_v2": 0.75,
                    "retro_fold_id": (i % 3) + 1,
                    "won": 0.0,
                }
                for i in range(18)
            ]
        )
        model = fit_stability_gate_model(rows, policy, probability_source="raw", mode=STABILITY_GATE_VALIDATED_OUTCOME)
        self.assertEqual(model.regions["__validated_outcomes__"]["stable_outcomes"], ["home"])
        gated = apply_stability_gate(rows, model)
        self.assertEqual(set(gated["stability_gate_mode"].astype(str).unique()), {STABILITY_GATE_VALIDATED_OUTCOME})
        self.assertTrue(bool(gated[gated["selection"].eq("home")]["candidate_gate_pass"].all()))
        self.assertFalse(bool(gated[gated["selection"].eq("draw")]["candidate_gate_pass"].any()))
        self.assertTrue(gated[gated["selection"].eq("draw")]["candidate_gate_reason"].astype(str).eq("validated_outcome_unstable").all())

    def test_validated_outcome_soft_gate_penalizes_unstable_outcome_without_hard_blocking(self) -> None:
        policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0)
        rows = pd.DataFrame(
            [
                {
                    "match_id": f"m{i}",
                    "league_code": "E0",
                    "selection": "draw",
                    "actual_outcome": "home",
                    "quote_status": "eligible",
                    "quoted_odds": 3.4,
                    "top_ask": 0.26,
                    "policy_prob": 0.36,
                    "policy_edge": 0.10,
                    "policy_ev": 0.224,
                    "confidence_score_v2": 0.75,
                    "retro_fold_id": (i % 3) + 1,
                    "won": 0.0,
                }
                for i in range(18)
            ]
        )
        model = fit_stability_gate_model(rows, policy, probability_source="raw", mode=STABILITY_GATE_VALIDATED_OUTCOME_SOFT)
        gated = apply_stability_gate(rows, model)
        self.assertEqual(set(gated["stability_gate_mode"].astype(str).unique()), {STABILITY_GATE_VALIDATED_OUTCOME_SOFT})
        self.assertTrue(bool(gated["candidate_gate_pass"].all()))
        self.assertTrue((gated["gated_prob"] < gated["policy_prob"]).all())
        self.assertTrue(gated["candidate_gate_reason"].astype(str).eq("validated_outcome_soft_penalty").all())

    def test_stability_gate_crossfit_uses_only_past_folds(self) -> None:
        policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0)
        rows = pd.DataFrame(
            [
                {
                    "match_id": f"m{i}",
                    "league_code": "E0",
                    "selection": "home",
                    "actual_outcome": "home" if i % 2 == 0 else "away",
                    "quote_status": "eligible",
                    "quoted_odds": 2.0,
                    "top_ask": 0.45,
                    "policy_prob": 0.55,
                    "policy_edge": 0.10,
                    "policy_ev": 0.10,
                    "confidence_score_v2": 0.75,
                    "retro_fold_id": (i % 3) + 1,
                    "won": 1.0 if i % 2 == 0 else 0.0,
                }
                for i in range(30)
            ]
        )
        gated_oof, _, _, diagnostics, audit = crossfit_stability_gate(
            rows,
            rows.iloc[:0].copy(),
            rows.iloc[:0].copy(),
            policy=policy,
            probability_source="raw",
            fold_column="retro_fold_id",
        )
        self.assertEqual(len(diagnostics["crossfit_folds"]), 3)
        self.assertEqual([int(item["train_rows"]) for item in diagnostics["crossfit_folds"]], [0, 10, 20])
        self.assertEqual(diagnostics["crossfit_training_direction"], "past_folds_only")
        self.assertEqual(len(gated_oof), len(rows))
        self.assertEqual(len(audit[audit["stability_gate_split"].eq("oof")]), len(rows))

    def test_select_candidate_rows_respects_gated_columns_without_creating_new_eligibility(self) -> None:
        policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0)
        candidates = pd.DataFrame(
            [
                {
                    "match_id": "m1",
                    "selection": "home",
                    "actual_outcome": "home",
                    "quote_status": "eligible",
                    "quoted_odds": 2.0,
                    "top_ask": 0.45,
                    "model_prob_raw": 0.60,
                    "edge_raw": 0.15,
                    "ev_raw": 0.20,
                    "model_prob_calibrated": 0.60,
                    "edge_calibrated": 0.15,
                    "ev_calibrated": 0.20,
                    "gated_prob": 0.001,
                    "gated_edge": -1.0,
                    "gated_ev": -1.0,
                    "decision_time": pd.Timestamp("2026-01-01", tz="UTC"),
                },
                {
                    "match_id": "m2",
                    "selection": "away",
                    "actual_outcome": "away",
                    "quote_status": "eligible",
                    "quoted_odds": 2.0,
                    "top_ask": 0.45,
                    "model_prob_raw": 0.40,
                    "edge_raw": -0.05,
                    "ev_raw": -0.20,
                    "model_prob_calibrated": 0.40,
                    "edge_calibrated": -0.05,
                    "ev_calibrated": -0.20,
                    "gated_prob": 0.70,
                    "gated_edge": 0.25,
                    "gated_ev": 0.40,
                    "decision_time": pd.Timestamp("2026-01-01", tz="UTC"),
                },
            ]
        )
        selected = select_candidate_rows(candidates, policy, probability_source="raw")
        self.assertTrue(selected[selected["match_id"].eq("m1")].empty)
        # The gate may harden eligible candidates, but it must not rescue candidates
        # whose original model edge/EV never cleared the policy threshold.
        self.assertTrue(selected[selected["match_id"].eq("m2")].empty)

    def test_decision_adjusted_columns_soften_policy_without_creating_new_eligibility(self) -> None:
        policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.05, min_odds=1.2, max_odds=6.0)
        candidates = pd.DataFrame(
            [
                {
                    "match_id": "m1",
                    "selection": "home",
                    "actual_outcome": "home",
                    "quote_status": "eligible",
                    "quoted_odds": 2.0,
                    "top_ask": 0.45,
                    "model_prob_raw": 0.60,
                    "edge_raw": 0.15,
                    "ev_raw": 0.20,
                    "model_prob_calibrated": 0.60,
                    "edge_calibrated": 0.15,
                    "ev_calibrated": 0.20,
                    "decision_adjusted_prob": 0.47,
                    "decision_adjusted_edge": 0.02,
                    "decision_adjusted_ev": -0.06,
                    "decision_time": pd.Timestamp("2026-01-01", tz="UTC"),
                },
                {
                    "match_id": "m2",
                    "selection": "away",
                    "actual_outcome": "away",
                    "quote_status": "eligible",
                    "quoted_odds": 2.0,
                    "top_ask": 0.45,
                    "model_prob_raw": 0.40,
                    "edge_raw": -0.05,
                    "ev_raw": -0.20,
                    "model_prob_calibrated": 0.40,
                    "edge_calibrated": -0.05,
                    "ev_calibrated": -0.20,
                    "decision_adjusted_prob": 0.70,
                    "decision_adjusted_edge": 0.25,
                    "decision_adjusted_ev": 0.40,
                    "decision_time": pd.Timestamp("2026-01-01", tz="UTC"),
                },
            ]
        )
        scored = candidate_score_columns(candidates, probability_source="raw", slippage_cushion=0.0)
        self.assertLess(float(scored.loc[0, "policy_edge"]), float(candidates.loc[0, "edge_raw"]))
        self.assertLessEqual(float(scored.loc[1, "policy_prob"]), float(candidates.loc[1, "model_prob_raw"]))
        selected = select_candidate_rows(candidates, policy, probability_source="raw")
        self.assertTrue(selected.empty)

    def test_logit_probability_scorer_emits_conservative_policy_adjustment(self) -> None:
        rows = []
        for idx in range(18):
            fold_id = (idx % 3) + 1
            wins = idx % 2 == 0
            probability = 0.62 if wins else 0.48
            rows.append(
                {
                    "match_id": f"m{idx}",
                    "selection": "home",
                    "actual_outcome": "home" if wins else "away",
                    "quote_status": "eligible",
                    "quoted_odds": 2.0,
                    "top_ask": 0.50,
                    "model_prob_raw": probability,
                    "edge_raw": probability - 0.50,
                    "ev_raw": probability - 0.50,
                    "model_prob_calibrated": probability,
                    "edge_calibrated": probability - 0.50,
                    "ev_calibrated": probability - 0.50,
                    "league_code": "E0" if idx % 2 == 0 else "SP1",
                    "retro_fold_id": fold_id,
                    "decision_time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=idx),
                }
            )
        scored_rows = candidate_score_columns(pd.DataFrame(rows), probability_source="raw", slippage_cushion=0.0)
        oof, dev, holdout, diagnostics = crossfit_decision_region_scores(
            scored_rows,
            scored_rows.iloc[:3].copy(),
            scored_rows.iloc[3:6].copy(),
            scorer_name=DECISION_SCORER_LOGIT_PROB,
            training_scope=TRAINING_SCOPE_ELIGIBLE,
        )
        self.assertEqual(diagnostics["probability_adjustment_mode"], "conservative_model_prob")
        self.assertEqual(diagnostics["crossfit_training_direction"], "past_folds_only")
        self.assertEqual([int(item["train_rows"]) for item in diagnostics["crossfit_folds"]], [0, 6, 12])
        self.assertIn("decision_adjusted_prob", oof.columns)
        self.assertIn("decision_adjusted_prob", dev.columns)
        self.assertIn("decision_adjusted_prob", holdout.columns)
        self.assertTrue(np.isfinite(pd.to_numeric(oof["decision_adjusted_prob"], errors="coerce")).all())
        self.assertTrue(
            (
                pd.to_numeric(oof["decision_adjusted_prob"], errors="coerce")
                <= pd.to_numeric(oof["policy_prob"], errors="coerce") + 1e-12
            ).all()
        )

    def test_reliability_probability_scorer_shrinks_by_empirical_regions(self) -> None:
        rows = []
        for idx in range(48):
            fold_id = (idx % 4) + 1
            wins = idx % 4 == 0
            rows.append(
                {
                    "match_id": f"m{idx}",
                    "selection": "home",
                    "actual_outcome": "home" if wins else "away",
                    "quote_status": "eligible",
                    "quoted_odds": 2.0,
                    "top_ask": 0.50,
                    "model_prob_raw": 0.62,
                    "edge_raw": 0.12,
                    "ev_raw": 0.12,
                    "model_prob_calibrated": 0.62,
                    "edge_calibrated": 0.12,
                    "ev_calibrated": 0.12,
                    "league_code": "E0",
                    "retro_fold_id": fold_id,
                    "decision_time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=idx),
                }
            )
        scored_rows = candidate_score_columns(pd.DataFrame(rows), probability_source="raw", slippage_cushion=0.0)
        oof, _, _, diagnostics = crossfit_decision_region_scores(
            scored_rows,
            scored_rows.iloc[:6].copy(),
            scored_rows.iloc[6:12].copy(),
            scorer_name=DECISION_SCORER_RELIABILITY_PROB,
            training_scope=TRAINING_SCOPE_ELIGIBLE,
        )
        self.assertEqual(diagnostics["probability_adjustment_mode"], "conservative_model_prob")
        self.assertIn("decision_adjusted_prob", oof.columns)
        self.assertLess(
            float(pd.to_numeric(oof["decision_adjusted_prob"], errors="coerce").mean()),
            float(pd.to_numeric(oof["policy_prob"], errors="coerce").mean()),
        )

    def test_confidence_backoff_report_marks_not_engaged_when_scores_stay_at_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            predictions = pd.DataFrame(
                {
                    "match_id": ["m1", "m2"],
                    "confidence_score": [1.0, 1.0],
                    "confidence_divergence_to_v1": [0.0, 0.0],
                    "confidence_backoff_variant": [1.0, 1.0],
                    "confidence_backoff_applied": [0.0, 0.0],
                    "confidence_backoff_inactive_reason": ["probabilities_identical_to_v1", "probabilities_identical_to_v1"],
                }
            )
            report = _confidence_backoff_report(predictions, pd.DataFrame(), pd.DataFrame(), settings)
            self.assertFalse(report["summary"]["applied"])
            self.assertTrue(report["summary"]["variant_supports_backoff"])
            self.assertTrue(report["summary"]["backoff_not_engaged"])
            self.assertEqual(report["summary"]["backoff_inactive_reason"], "probabilities_identical_to_v1")
            self.assertEqual(float(report["summary"]["mean_divergence_to_v1"]), 0.0)

    def test_decision_region_report_collapses_duplicate_fills_and_keeps_selected_confidence(self) -> None:
        selected = pd.DataFrame(
            [
                {
                    "decision_id": "d1",
                    "match_id": "m1",
                    "selection": "home",
                    "confidence_score": 0.81,
                    "confidence_divergence_to_v1": 0.19,
                    "policy_confidence_score": 0.81,
                    "policy_confidence_divergence_to_v1": 0.19,
                    "selection_odds": 2.0,
                    "retro_fold_id": 1,
                    "pre_holdout_window": "pre_holdout_a",
                    "created_at": "2026-04-01T10:00:00Z",
                }
            ]
        )
        fills = pd.DataFrame(
            [
                {
                    "decision_id": "d1",
                    "fill_id": "f1",
                    "match_id": "m1",
                    "net_profit": 1.0,
                    "cost_basis": 4.0,
                    "won": 1,
                    "created_at": "2026-04-01T10:05:00Z",
                },
                {
                    "decision_id": "d1",
                    "fill_id": "f2",
                    "match_id": "m1",
                    "net_profit": 2.0,
                    "cost_basis": 6.0,
                    "won": 1,
                    "created_at": "2026-04-01T10:06:00Z",
                },
            ]
        )

        report = _decision_region_report(selected, fills, label="selection_dev")

        self.assertEqual(report["overview"]["bets"], 1)
        self.assertAlmostEqual(report["overview"]["stake"], 10.0, places=6)
        self.assertAlmostEqual(report["overview"]["profit"], 3.0, places=6)
        self.assertAlmostEqual(report["overview"]["roi"], 0.3, places=6)
        self.assertAlmostEqual(report["overview"]["avg_confidence"], 0.81, places=6)
        self.assertEqual(report["by_fold"]["1"]["bets"], 1)
        self.assertEqual(report["by_window"]["pre_holdout_a"]["bets"], 1)

    def test_confidence_backoff_report_uses_selected_scores_when_combined_is_flat(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "match_id": "m1",
                    "Date": pd.Timestamp("2026-04-01"),
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "season": "2526",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea",
                    "selection": "home",
                    "quote_status": "eligible",
                    "quoted_odds": 2.0,
                    "model_prob_raw": 0.60,
                    "model_prob_calibrated": 0.52,
                    "prob_home_raw": 0.60,
                    "prob_draw_raw": 0.25,
                    "prob_away_raw": 0.15,
                    "prob_home_calibrated": 0.52,
                    "prob_draw_calibrated": 0.28,
                    "prob_away_calibrated": 0.20,
                    "edge_raw": 0.08,
                    "ev_raw": 0.20,
                    "market_prob_home": 0.50,
                    "quote_age_minutes": 60.0,
                    "actual_outcome": "home",
                },
                {
                    "match_id": "m2",
                    "Date": pd.Timestamp("2026-04-01"),
                    "league_code": "E0",
                    "league_name": "Premier League",
                    "season": "2526",
                    "HomeTeam": "Liverpool",
                    "AwayTeam": "Arsenal",
                    "selection": "away",
                    "quote_status": "eligible",
                    "quoted_odds": 2.2,
                    "model_prob_raw": 0.53,
                    "model_prob_calibrated": 0.49,
                    "prob_home_raw": 0.24,
                    "prob_draw_raw": 0.23,
                    "prob_away_raw": 0.53,
                    "prob_home_calibrated": 0.26,
                    "prob_draw_calibrated": 0.25,
                    "prob_away_calibrated": 0.49,
                    "edge_raw": 0.06,
                    "ev_raw": 0.16,
                    "market_prob_home": 0.45,
                    "quote_age_minutes": 45.0,
                    "actual_outcome": "away",
                }
            ]
        )
        policy = BetPolicy(edge_threshold=0.0, ev_threshold=0.0, min_odds=1.2, max_odds=6.0)
        selected = select_candidate_rows(candidates, policy, probability_source="raw")
        self.assertEqual(len(selected), 2)
        self.assertGreater(float(selected.iloc[0]["selection_confidence_score"]), 0.0)
        self.assertLess(float(selected.iloc[0]["selection_confidence_score"]), 1.0)
        self.assertGreater(float(selected.iloc[0]["selection_confidence_divergence_to_v1"]), 0.0)
        self.assertGreater(float(selected.iloc[1]["selection_confidence_score"]), 0.0)
        self.assertLess(float(selected.iloc[1]["selection_confidence_score"]), 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            combined_predictions = pd.DataFrame(
                [
                    {
                        "match_id": "m1",
                        "confidence_score": 1.0,
                        "confidence_divergence_to_v1": 0.0,
                        "confidence_backoff_variant": 1.0,
                        "confidence_backoff_applied": 1.0,
                    },
                    {
                        "match_id": "m2",
                        "confidence_score": 1.0,
                        "confidence_divergence_to_v1": 0.0,
                        "confidence_backoff_variant": 1.0,
                        "confidence_backoff_applied": 1.0,
                    }
                ]
            )
            fills = pd.DataFrame(
                [
                    {
                        "match_id": "m1",
                        "won": 1,
                        "net_profit": 2.0,
                        "cost_basis": 10.0,
                    },
                    {
                        "match_id": "m2",
                        "won": 0,
                        "net_profit": -1.0,
                        "cost_basis": 5.0,
                    }
                ]
            )

            report = _confidence_backoff_report(combined_predictions, selected, fills, settings)
            self.assertTrue(report["summary"]["applied"])
            self.assertTrue(report["summary"]["variant_supports_backoff"])
            self.assertFalse(report["summary"]["backoff_not_engaged"])
            self.assertLess(float(report["summary"]["mean"]), 1.0)
            self.assertAlmostEqual(float(report["summary"]["mean"]), float(selected["selection_confidence_score"].mean()), places=6)
            self.assertGreater(float(report["summary"]["spread"]), 0.0)

    def test_prune_feature_columns_for_fold_drops_low_support_and_zero_variance(self) -> None:
        rows = pd.DataFrame(
            {
                "good_col": list(range(25)),
                "low_support": [1.0] * 19 + [None] * 6,
                "zero_var": [5.0] * 25,
                "league_code": ["E0"] * 25,
            }
        )
        used, pruned = _prune_feature_columns_for_fold(rows, ["good_col", "low_support", "zero_var", "league_code"])
        self.assertEqual(used, ["good_col"])
        self.assertIn("low_support", pruned["low_support"])
        self.assertIn("zero_var", pruned["zero_variance"])
        self.assertIn("league_code", pruned["zero_variance"])

    def test_oof_frozen_policy_report_keeps_folds_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            policy = BetPolicy(edge_threshold=0.0, ev_threshold=0.0, min_odds=1.2, max_odds=6.0)
            candidates = pd.DataFrame(
                [
                    {
                        "decision_id": "d1",
                        "match_id": "m1",
                        "retro_fold_id": 1,
                        "selection": "home",
                        "quote_status": "eligible",
                        "quoted_odds": 1.8,
                        "model_prob_raw": 0.60,
                        "edge_raw": 0.05,
                        "ev_raw": 0.04,
                        "decision_time": pd.Timestamp("2026-04-01T10:00:00Z"),
                        "kickoff_time": pd.Timestamp("2026-04-01T10:45:00Z"),
                        "group_key": "g1",
                        "asks_json": '[{"price": 0.55, "size": 100}]',
                        "fee_rate": 0.03,
                        "actual_outcome": "home",
                        "actual_target": 2,
                        "league_code": "E0",
                        "quality_tier": "history_proxy",
                        "source_type": "bookmaker_proxy",
                    },
                    {
                        "decision_id": "d2",
                        "match_id": "m2",
                        "retro_fold_id": 2,
                        "selection": "away",
                        "quote_status": "eligible",
                        "quoted_odds": 2.1,
                        "model_prob_raw": 0.58,
                        "edge_raw": 0.05,
                        "ev_raw": 0.04,
                        "decision_time": pd.Timestamp("2026-04-02T10:00:00Z"),
                        "kickoff_time": pd.Timestamp("2026-04-02T10:45:00Z"),
                        "group_key": "g2",
                        "asks_json": '[{"price": 0.48, "size": 100}]',
                        "fee_rate": 0.03,
                        "actual_outcome": "away",
                        "actual_target": 0,
                        "league_code": "E0",
                        "quality_tier": "history_proxy",
                        "source_type": "bookmaker_proxy",
                    },
                ]
            )
            report, selected, _ = _oof_frozen_policy_report(candidates, "raw", policy, settings)
            self.assertEqual(report["oof_fold_bets"], {"fold_1": 1, "fold_2": 1})
            self.assertEqual(report["oof_validation_unit_bets"], {"unit_1": 2})
            self.assertEqual(report["oof_raw_min_fold_bets"], 1)
            self.assertEqual(report["oof_min_validation_unit_bets"], 2)
            self.assertEqual(set(selected["retro_fold_id"].astype(int)), {1, 2})

    def test_select_candidate_rows_respects_no_draw_and_mid_odds_scope(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "match_id": "m1",
                    "selection": "draw",
                    "quote_status": "eligible",
                    "quoted_odds": 1.70,
                    "model_prob_raw": 0.62,
                    "edge_raw": 0.05,
                    "ev_raw": 0.04,
                    "decision_time": pd.Timestamp("2026-04-01T10:00:00Z"),
                    "kickoff_time": pd.Timestamp("2026-04-01T10:45:00Z"),
                    "group_key": "g1",
                    "asks_json": '[{"price": 0.58, "size": 100}]',
                    "fee_rate": 0.03,
                    "actual_outcome": "draw",
                    "league_code": "E0",
                    "quality_tier": "history_proxy",
                    "source_type": "bookmaker_proxy",
                },
                {
                    "match_id": "m2",
                    "selection": "home",
                    "quote_status": "eligible",
                    "quoted_odds": 2.00,
                    "model_prob_raw": 0.58,
                    "edge_raw": 0.05,
                    "ev_raw": 0.04,
                    "decision_time": pd.Timestamp("2026-04-02T10:00:00Z"),
                    "kickoff_time": pd.Timestamp("2026-04-02T10:45:00Z"),
                    "group_key": "g2",
                    "asks_json": '[{"price": 0.50, "size": 100}]',
                    "fee_rate": 0.03,
                    "actual_outcome": "home",
                    "league_code": "E0",
                    "quality_tier": "history_proxy",
                    "source_type": "bookmaker_proxy",
                },
                {
                    "match_id": "m3",
                    "selection": "away",
                    "quote_status": "eligible",
                    "quoted_odds": 1.99,
                    "model_prob_raw": 0.59,
                    "edge_raw": 0.05,
                    "ev_raw": 0.04,
                    "decision_time": pd.Timestamp("2026-04-03T10:00:00Z"),
                    "kickoff_time": pd.Timestamp("2026-04-03T10:45:00Z"),
                    "group_key": "g3",
                    "asks_json": '[{"price": 0.50, "size": 100}]',
                    "fee_rate": 0.03,
                    "actual_outcome": "away",
                    "league_code": "E0",
                    "quality_tier": "history_proxy",
                    "source_type": "bookmaker_proxy",
                },
            ]
        )
        policy = BetPolicy(
            edge_threshold=0.0,
            ev_threshold=0.0,
            min_odds=1.5,
            max_odds=2.0,
            family="edge_ev_threshold",
            allowed_outcomes=("home", "away"),
            scope_name="mid_odds_no_draw",
        )
        selected = select_candidate_rows(candidates, policy, probability_source="raw", slippage_cushion=0.0)
        self.assertEqual(set(selected["match_id"].astype(str)), {"m3"})
        self.assertTrue(selected["selection"].astype(str).isin(["home", "away"]).all())

    def test_pre_holdout_windows_are_contiguous_and_non_overlapping(self) -> None:
        rows = pd.DataFrame(
            {
                "match_id": [f"m{i}" for i in range(1, 10)],
                "kickoff_time": pd.date_range("2026-01-01", periods=9, freq="D", tz="UTC"),
            }
        )
        windows = _pre_holdout_windows(rows)
        self.assertEqual(set(windows["pre_holdout_window"].astype(str)), {"pre_holdout_a", "pre_holdout_b", "pre_holdout_c"})
        self.assertEqual(int(windows["match_id"].nunique()), 9)
        ordered = rows.merge(windows, on="match_id", how="left").sort_values("kickoff_time")
        sequences = ordered["pre_holdout_window"].astype(str).tolist()
        self.assertEqual(sequences[:3], ["pre_holdout_a"] * 3)
        self.assertEqual(sequences[3:6], ["pre_holdout_b"] * 3)
        self.assertEqual(sequences[6:], ["pre_holdout_c"] * 3)

    def test_model_status_distinguishes_overfit_predictive_only_and_promotable(self) -> None:
        baseline_summary = {
            "selection_dev_summary": {
                "current_policy_frozen": {"net_roi": 0.10},
                "pre_holdout_aggregate_roi": 0.10,
                "pre_holdout_median_roi": 0.10,
                "pre_holdout_positive_window_ratio": 1.0,
            },
            "locked_holdout_summary": {"current_policy_frozen": {"net_roi": 0.12}},
        }
        oof_report = {
            "oof_aggregate_roi": -0.01,
            "oof_positive_fold_ratio": 0.25,
            "oof_total_bets": 80,
            "oof_min_fold_bets": 12,
        }
        selection_dev_summary = {
            "current_policy_frozen": {"bets": 24, "net_roi": -0.02},
            "pre_holdout_aggregate_roi": -0.02,
            "pre_holdout_median_roi": -0.03,
            "pre_holdout_positive_window_ratio": 1 / 3,
            "pre_holdout_total_bets": 24,
            "pre_holdout_min_window_bets": 8,
        }
        locked_holdout_summary = {"current_policy_frozen": {"bets": 45, "net_roi": 0.30}}
        status, eligibility, _ = _model_status(
            "coverage_ready",
            oof_log_loss=0.80,
            v1_raw_log_loss=0.90,
            oof_report=oof_report,
            selection_dev_summary=selection_dev_summary,
            locked_holdout_summary=locked_holdout_summary,
            baseline_summary=baseline_summary,
        )
        self.assertEqual(status, "overfit_rejected")
        self.assertEqual(eligibility, "overfit_rejected")

        oof_report = {
            "oof_aggregate_roi": 0.08,
            "oof_positive_fold_ratio": 0.75,
            "oof_total_bets": 80,
            "oof_min_fold_bets": 12,
        }
        selection_dev_summary = {
            "current_policy_frozen": {"bets": 24, "net_roi": 0.11},
            "pre_holdout_aggregate_roi": 0.11,
            "pre_holdout_median_roi": 0.10,
            "pre_holdout_positive_window_ratio": 2 / 3,
            "pre_holdout_total_bets": 24,
            "pre_holdout_min_window_bets": 8,
        }
        locked_holdout_summary = {"current_policy_frozen": {"bets": 45, "net_roi": 0.10}}
        status, eligibility, _ = _model_status(
            "coverage_ready",
            oof_log_loss=0.80,
            v1_raw_log_loss=0.90,
            oof_report=oof_report,
            selection_dev_summary=selection_dev_summary,
            locked_holdout_summary=locked_holdout_summary,
            baseline_summary=baseline_summary,
        )
        self.assertEqual(status, "predictive_only_improvement")
        self.assertEqual(eligibility, "predictive_only_improvement")

        locked_holdout_summary = {"current_policy_frozen": {"bets": 45, "net_roi": 0.20}}
        status, eligibility, _ = _model_status(
            "coverage_ready",
            oof_log_loss=0.80,
            v1_raw_log_loss=0.90,
            oof_report=oof_report,
            selection_dev_summary=selection_dev_summary,
            locked_holdout_summary=locked_holdout_summary,
            baseline_summary=baseline_summary,
        )
        self.assertEqual(status, "promotable_for_forward")
        self.assertEqual(eligibility, "promotable_for_forward")

        selection_dev_summary = {
            "current_policy_frozen": {"bets": 12, "net_roi": 0.15},
            "pre_holdout_aggregate_roi": 0.15,
            "pre_holdout_median_roi": 0.14,
            "pre_holdout_positive_window_ratio": 1.0,
            "pre_holdout_total_bets": 12,
            "pre_holdout_min_window_bets": 4,
        }
        status, eligibility, _ = _model_status(
            "coverage_ready",
            oof_log_loss=0.80,
            v1_raw_log_loss=0.90,
            oof_report={
                "oof_aggregate_roi": 0.08,
                "oof_positive_fold_ratio": 0.75,
                "oof_total_bets": 40,
                "oof_min_fold_bets": 5,
            },
            selection_dev_summary=selection_dev_summary,
            locked_holdout_summary=locked_holdout_summary,
            baseline_summary=baseline_summary,
        )
        self.assertEqual(status, "pre_holdout_insufficient_sample")
        self.assertEqual(eligibility, "pre_holdout_insufficient_sample")

    def test_probability_source_keeps_raw_when_calibrated_hurts_frozen_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            baseline_policy = BetPolicy(edge_threshold=0.05, ev_threshold=0.0, min_odds=1.2, max_odds=6.0)
            candidates = pd.DataFrame(
                [
                    {
                        "match_id": "m1",
                        "selection": "home",
                        "actual_outcome": "home",
                        "actual_target": 2,
                        "quote_status": "eligible",
                        "quoted_odds": 1.90,
                        "model_prob_raw": 0.55,
                        "model_prob_calibrated": 0.70,
                        "edge_raw": 0.08,
                        "edge_calibrated": 0.01,
                        "ev_raw": 0.06,
                        "ev_calibrated": -0.01,
                        "decision_time": pd.Timestamp("2026-04-01T10:00:00Z"),
                        "kickoff_time": pd.Timestamp("2026-04-01T10:45:00Z"),
                        "group_key": "g1",
                        "asks_json": '[{"price": 0.52, "size": 100}]',
                        "fee_rate": 0.03,
                        "league_code": "E0",
                        "quality_tier": "history_proxy",
                        "source_type": "bookmaker_proxy",
                    },
                    {
                        "match_id": "m1",
                        "selection": "draw",
                        "actual_outcome": "home",
                        "actual_target": 2,
                        "quote_status": "eligible",
                        "quoted_odds": 3.20,
                        "model_prob_raw": 0.20,
                        "model_prob_calibrated": 0.15,
                        "edge_raw": -0.02,
                        "edge_calibrated": -0.03,
                        "ev_raw": -0.04,
                        "ev_calibrated": -0.05,
                        "decision_time": pd.Timestamp("2026-04-01T10:00:00Z"),
                        "kickoff_time": pd.Timestamp("2026-04-01T10:45:00Z"),
                        "group_key": "g1",
                        "asks_json": '[{"price": 0.31, "size": 100}]',
                        "fee_rate": 0.03,
                        "league_code": "E0",
                        "quality_tier": "history_proxy",
                        "source_type": "bookmaker_proxy",
                    },
                    {
                        "match_id": "m1",
                        "selection": "away",
                        "actual_outcome": "home",
                        "actual_target": 2,
                        "quote_status": "eligible",
                        "quoted_odds": 4.20,
                        "model_prob_raw": 0.25,
                        "model_prob_calibrated": 0.15,
                        "edge_raw": -0.01,
                        "edge_calibrated": -0.05,
                        "ev_raw": -0.03,
                        "ev_calibrated": -0.06,
                        "decision_time": pd.Timestamp("2026-04-01T10:00:00Z"),
                        "kickoff_time": pd.Timestamp("2026-04-01T10:45:00Z"),
                        "group_key": "g1",
                        "asks_json": '[{"price": 0.24, "size": 100}]',
                        "fee_rate": 0.03,
                        "league_code": "E0",
                        "quality_tier": "history_proxy",
                        "source_type": "bookmaker_proxy",
                    },
                ]
            )
            chosen, report = _select_probability_source(candidates, candidates, baseline_policy, settings)
            self.assertEqual(chosen, "raw")
            self.assertGreater(report["relative_log_loss_improvement"], 0.01)

    def test_rank_policies_returns_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            candidates = pd.DataFrame(
                [
                    {"match_id": "1", "selection": "home", "quote_status": "eligible", "quoted_odds": 2.0, "edge_raw": 0.10, "ev_raw": 0.08, "model_prob_raw": 0.58, "actual_outcome": "home", "decision_time": pd.Timestamp("2026-04-01T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-01T10:45:00Z"), "asks_json": '[{"price": 0.5, "size": 1000}]', "fee_rate": 0.03, "group_key": "g1", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "2", "selection": "away", "quote_status": "eligible", "quoted_odds": 2.5, "edge_raw": 0.12, "ev_raw": 0.09, "model_prob_raw": 0.50, "actual_outcome": "away", "decision_time": pd.Timestamp("2026-04-02T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-02T10:45:00Z"), "asks_json": '[{"price": 0.4, "size": 1000}]', "fee_rate": 0.03, "group_key": "g2", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "3", "selection": "draw", "quote_status": "eligible", "quoted_odds": 3.3, "edge_raw": 0.15, "ev_raw": 0.11, "model_prob_raw": 0.38, "actual_outcome": "draw", "decision_time": pd.Timestamp("2026-04-03T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-03T10:45:00Z"), "asks_json": '[{"price": 0.3, "size": 1000}]', "fee_rate": 0.03, "group_key": "g3", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "4", "selection": "home", "quote_status": "eligible", "quoted_odds": 1.8, "edge_raw": 0.09, "ev_raw": 0.07, "model_prob_raw": 0.64, "actual_outcome": "home", "decision_time": pd.Timestamp("2026-04-04T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-04T10:45:00Z"), "asks_json": '[{"price": 0.55, "size": 1000}]', "fee_rate": 0.03, "group_key": "g4", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "5", "selection": "away", "quote_status": "eligible", "quoted_odds": 2.2, "edge_raw": 0.11, "ev_raw": 0.08, "model_prob_raw": 0.53, "actual_outcome": "away", "decision_time": pd.Timestamp("2026-04-05T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-05T10:45:00Z"), "asks_json": '[{"price": 0.45, "size": 1000}]', "fee_rate": 0.03, "group_key": "g5", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "6", "selection": "draw", "quote_status": "eligible", "quoted_odds": 3.0, "edge_raw": 0.13, "ev_raw": 0.10, "model_prob_raw": 0.40, "actual_outcome": "draw", "decision_time": pd.Timestamp("2026-04-06T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-06T10:45:00Z"), "asks_json": '[{"price": 0.33, "size": 1000}]', "fee_rate": 0.03, "group_key": "g6", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "7", "selection": "home", "quote_status": "eligible", "quoted_odds": 2.1, "edge_raw": 0.10, "ev_raw": 0.08, "model_prob_raw": 0.55, "actual_outcome": "home", "decision_time": pd.Timestamp("2026-04-07T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-07T10:45:00Z"), "asks_json": '[{"price": 0.48, "size": 1000}]', "fee_rate": 0.03, "group_key": "g7", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "8", "selection": "away", "quote_status": "eligible", "quoted_odds": 2.4, "edge_raw": 0.12, "ev_raw": 0.09, "model_prob_raw": 0.51, "actual_outcome": "away", "decision_time": pd.Timestamp("2026-04-08T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-08T10:45:00Z"), "asks_json": '[{"price": 0.42, "size": 1000}]', "fee_rate": 0.03, "group_key": "g8", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "9", "selection": "draw", "quote_status": "eligible", "quoted_odds": 2.9, "edge_raw": 0.14, "ev_raw": 0.10, "model_prob_raw": 0.39, "actual_outcome": "draw", "decision_time": pd.Timestamp("2026-04-09T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-09T10:45:00Z"), "asks_json": '[{"price": 0.34, "size": 1000}]', "fee_rate": 0.03, "group_key": "g9", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "10", "selection": "home", "quote_status": "eligible", "quoted_odds": 1.9, "edge_raw": 0.09, "ev_raw": 0.07, "model_prob_raw": 0.60, "actual_outcome": "home", "decision_time": pd.Timestamp("2026-04-10T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-10T10:45:00Z"), "asks_json": '[{"price": 0.52, "size": 1000}]', "fee_rate": 0.03, "group_key": "g10", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "11", "selection": "away", "quote_status": "eligible", "quoted_odds": 2.3, "edge_raw": 0.11, "ev_raw": 0.08, "model_prob_raw": 0.52, "actual_outcome": "away", "decision_time": pd.Timestamp("2026-04-11T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-11T10:45:00Z"), "asks_json": '[{"price": 0.43, "size": 1000}]', "fee_rate": 0.03, "group_key": "g11", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "12", "selection": "draw", "quote_status": "eligible", "quoted_odds": 3.1, "edge_raw": 0.13, "ev_raw": 0.10, "model_prob_raw": 0.41, "actual_outcome": "draw", "decision_time": pd.Timestamp("2026-04-12T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-12T10:45:00Z"), "asks_json": '[{"price": 0.32, "size": 1000}]', "fee_rate": 0.03, "group_key": "g12", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "13", "selection": "home", "quote_status": "eligible", "quoted_odds": 2.0, "edge_raw": 0.10, "ev_raw": 0.08, "model_prob_raw": 0.57, "actual_outcome": "home", "decision_time": pd.Timestamp("2026-04-13T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-13T10:45:00Z"), "asks_json": '[{"price": 0.5, "size": 1000}]', "fee_rate": 0.03, "group_key": "g13", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "14", "selection": "away", "quote_status": "eligible", "quoted_odds": 2.6, "edge_raw": 0.12, "ev_raw": 0.09, "model_prob_raw": 0.49, "actual_outcome": "away", "decision_time": pd.Timestamp("2026-04-14T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-14T10:45:00Z"), "asks_json": '[{"price": 0.39, "size": 1000}]', "fee_rate": 0.03, "group_key": "g14", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "15", "selection": "draw", "quote_status": "eligible", "quoted_odds": 3.2, "edge_raw": 0.14, "ev_raw": 0.11, "model_prob_raw": 0.42, "actual_outcome": "draw", "decision_time": pd.Timestamp("2026-04-15T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-15T10:45:00Z"), "asks_json": '[{"price": 0.31, "size": 1000}]', "fee_rate": 0.03, "group_key": "g15", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "16", "selection": "home", "quote_status": "eligible", "quoted_odds": 2.0, "edge_raw": 0.10, "ev_raw": 0.08, "model_prob_raw": 0.56, "actual_outcome": "home", "decision_time": pd.Timestamp("2026-04-16T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-16T10:45:00Z"), "asks_json": '[{"price": 0.5, "size": 1000}]', "fee_rate": 0.03, "group_key": "g16", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "17", "selection": "away", "quote_status": "eligible", "quoted_odds": 2.5, "edge_raw": 0.12, "ev_raw": 0.09, "model_prob_raw": 0.50, "actual_outcome": "away", "decision_time": pd.Timestamp("2026-04-17T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-17T10:45:00Z"), "asks_json": '[{"price": 0.4, "size": 1000}]', "fee_rate": 0.03, "group_key": "g17", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "18", "selection": "draw", "quote_status": "eligible", "quoted_odds": 3.0, "edge_raw": 0.13, "ev_raw": 0.10, "model_prob_raw": 0.40, "actual_outcome": "draw", "decision_time": pd.Timestamp("2026-04-18T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-18T10:45:00Z"), "asks_json": '[{"price": 0.33, "size": 1000}]', "fee_rate": 0.03, "group_key": "g18", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "19", "selection": "home", "quote_status": "eligible", "quoted_odds": 1.9, "edge_raw": 0.09, "ev_raw": 0.07, "model_prob_raw": 0.61, "actual_outcome": "home", "decision_time": pd.Timestamp("2026-04-19T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-19T10:45:00Z"), "asks_json": '[{"price": 0.52, "size": 1000}]', "fee_rate": 0.03, "group_key": "g19", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                    {"match_id": "20", "selection": "away", "quote_status": "eligible", "quoted_odds": 2.2, "edge_raw": 0.11, "ev_raw": 0.08, "model_prob_raw": 0.53, "actual_outcome": "away", "decision_time": pd.Timestamp("2026-04-20T10:00:00Z"), "kickoff_time": pd.Timestamp("2026-04-20T10:45:00Z"), "asks_json": '[{"price": 0.45, "size": 1000}]', "fee_rate": 0.03, "group_key": "g20", "league_code": "E0", "quality_tier": "history_proxy", "source_type": "bookmaker_proxy"},
                ]
            )
            ranking = _rank_policies(candidates, probability_source="raw", settings=settings)
            self.assertFalse(ranking.empty)
            self.assertIn("score", ranking.columns)


if __name__ == "__main__":
    unittest.main()
