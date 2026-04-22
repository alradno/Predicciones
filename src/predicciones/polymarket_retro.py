from __future__ import annotations

import json
import math
import re
import uuid
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from .backtest import _build_prediction_frame
from .config import Settings
from .contracts import (
    OUTCOME_AWAY,
    OUTCOME_DRAW,
    OUTCOME_HOME,
    OUTCOME_ORDER,
    OUTCOME_TO_TARGET,
    PolymarketCoverageAuditResult,
    PolymarketHistoryBackfillResult,
    PolymarketRetroResult,
)
from .data_sources import LEAGUE_NAMES, PolymarketClobClient, PolymarketGammaClient
from .dataset import (
    build_feature_rows,
    build_fixture_feature_rows,
    model_feature_columns_for_variant,
    variant_feature_families,
    variant_feature_manifest,
)
from .ingestion import build_market_odds, canonicalize_matches, normalize_team_name
from .models import (
    OutcomeCalibrator,
    build_goal_model,
    fit_dixon_coles_rho,
    fit_goal_model,
    multiclass_brier_score,
    outcome_probabilities_from_lambdas,
)
from .reporting import _save_json, create_run_context
from .research import select_candidate_bets
from .strategy import BetPolicy, candidate_score_columns, select_candidate_rows
from .decision_region_diagnostics import (
    _collapse_selected_bet_rows,
    build_decision_region_diagnostics,
    format_decision_region_summary,
)
from .decision_region_model import (
    DECISION_SCORER_HEURISTIC,
    DECISION_SCORER_HGB,
    DECISION_SCORER_HGB_PROB,
    DECISION_SCORER_LOGIT,
    DECISION_SCORER_LOGIT_PROB,
    DECISION_SCORER_RELIABILITY_PROB,
    TRAINING_SCOPE_ARGMAX,
    TRAINING_SCOPE_ELIGIBLE,
    crossfit_decision_region_scores,
)
from .decision_region_adjustment import (
    REGIONAL_ADJUSTMENT_NONE,
    REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK,
    REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK,
    crossfit_regional_adjustment,
)
from .candidate_stability_gate import (
    STABILITY_GATE_EXPANDING_OUTCOME,
    STABILITY_GATE_HIERARCHICAL,
    STABILITY_GATE_NONE,
    STABILITY_GATE_OUTCOME_PARENT,
    STABILITY_GATE_SOFT_PARENT,
    STABILITY_GATE_VALIDATED_OUTCOME,
    STABILITY_GATE_VALIDATED_OUTCOME_SOFT,
    crossfit_stability_gate,
    stability_gate_audit_columns,
)
from .polymarket_retro_backoff import (
    backoff_inactive_reason as helper_backoff_inactive_reason,
    blend_probabilities as helper_blend_probabilities,
    build_confidence_backoff_report as helper_build_confidence_backoff_report,
    confidence_score_summary as helper_confidence_score_summary,
    confidence_scores as helper_confidence_scores,
)
from . import strategy as pm_strategy
from . import polymarket_shadow as pm_shadow


def _select_candidate_rows_with_quote_age_fallback(
    candidate_rows: pd.DataFrame,
    policy: BetPolicy,
    probability_source: str,
    slippage_cushion: float = 0.0,
) -> pd.DataFrame:
    working = candidate_rows.copy() if isinstance(candidate_rows, pd.DataFrame) else pd.DataFrame(candidate_rows)
    if "quote_age_minutes" not in working.columns:
        working["quote_age_minutes"] = 0.0
    return _ORIGINAL_SELECT_CANDIDATE_ROWS(working, policy, probability_source, slippage_cushion)


_ORIGINAL_SELECT_CANDIDATE_ROWS = pm_strategy.select_candidate_rows
pm_strategy.select_candidate_rows = _select_candidate_rows_with_quote_age_fallback
select_candidate_rows = pm_strategy.select_candidate_rows


from .polymarket_retro_parsing import *  # noqa: F401,F403
from .polymarket_retro_reporting import *  # noqa: F401,F403
def backfill_polymarket_history(
    settings: Settings,
    history_matches: pd.DataFrame,
    db_path: Path | str | None = None,
) -> PolymarketHistoryBackfillResult:
    db_path = Path(db_path) if db_path else pm_shadow.default_polymarket_db_path(settings)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = pm_shadow.init_polymarket_db(db_path)
    gamma = PolymarketGammaClient()
    clob = PolymarketClobClient()
    run = create_run_context(settings.paths.runs_dir, "backfill_polymarket_history")
    matches = _history_matches(settings, {"history_matches": history_matches}, history_matches)
    start, end = _history_window(settings, matches)

    events_by_slug, page_rows = _fetch_closed_soccer_events(settings, gamma, matches)
    updated_at = pm_shadow._iso_timestamp()
    raw_events = _flatten_event_rows(settings, events_by_slug, updated_at)
    raw_markets = _flatten_market_rows(settings, events_by_slug, updated_at)
    candidates = _classify_market_candidates(settings, matches, raw_markets, events_by_slug)
    groups_v2, classification_audit = _build_fixture_groups_v2(candidates)
    catalog, groups = _legacy_tables_from_v2(candidates, groups_v2)
    if not raw_events.empty and "game_start_time" in raw_events.columns:
        raw_events["game_start_time"] = pd.to_datetime(raw_events["game_start_time"], utc=True, errors="coerce")
    if not raw_markets.empty and "game_start_time" in raw_markets.columns:
        raw_markets["game_start_time"] = pd.to_datetime(raw_markets["game_start_time"], utc=True, errors="coerce")
    if not candidates.empty:
        candidates["game_start_time"] = pd.to_datetime(candidates["game_start_time"], utc=True, errors="coerce")
        candidates["match_date"] = pd.to_datetime(candidates["match_date"], utc=True, errors="coerce")
    if not groups_v2.empty:
        groups_v2["game_start_time"] = pd.to_datetime(groups_v2["game_start_time"], utc=True, errors="coerce")
        groups_v2["match_date"] = pd.to_datetime(groups_v2["match_date"], utc=True, errors="coerce")
    if not catalog.empty and "game_start_time" in catalog.columns:
        catalog["game_start_time"] = pd.to_datetime(catalog["game_start_time"], utc=True, errors="coerce")
    if not groups.empty and "game_start_time" in groups.columns:
        groups["game_start_time"] = pd.to_datetime(groups["game_start_time"], utc=True, errors="coerce")

    connection.executescript(
        """
        DELETE FROM pm_event_raw;
        DELETE FROM pm_market_raw;
        DELETE FROM pm_fixture_market_candidates;
        DELETE FROM pm_fixture_groups_v2;
        DELETE FROM pm_market_classification_audit;
        DELETE FROM pm_market_catalog;
        DELETE FROM pm_market_groups;
        DELETE FROM pm_team_aliases;
        DELETE FROM pm_price_history;
        """
    )
    connection.commit()
    pm_shadow._upsert_rows(connection, "pm_event_raw", raw_events.to_dict(orient="records") if not raw_events.empty else [])
    pm_shadow._upsert_rows(connection, "pm_market_raw", raw_markets.to_dict(orient="records") if not raw_markets.empty else [])
    candidate_store = candidates[
        [
            "market_id",
            "event_slug",
            "league_code",
            "league_name",
            "sport_code",
            "match_id",
            "match_date",
            "game_start_time",
            "fixture_key",
            "undirected_fixture_key",
            "home_team_raw",
            "away_team_raw",
            "home_team_canonical",
            "away_team_canonical",
            "parse_source",
            "parse_method",
            "market_shape",
            "market_role",
            "market_quality_rank",
            "classification_status",
            "classification_reason",
            "raw_json",
            "updated_at",
        ]
    ].copy() if not candidates.empty else pd.DataFrame()
    groups_v2_store = groups_v2[
        [
            "group_key",
            "fixture_key",
            "undirected_fixture_key",
            "event_slug",
            "event_title",
            "league_code",
            "league_name",
            "sport_code",
            "match_id",
            "match_date",
            "game_start_time",
            "home_team",
            "away_team",
            "home_market_id",
            "draw_market_id",
            "away_market_id",
            "group_status",
            "group_reason",
            "source_event_slugs_json",
            "source_market_ids_json",
            "duplicate_market_ids_json",
            "updated_at",
        ]
    ].copy() if not groups_v2.empty else pd.DataFrame()
    audit_store = classification_audit[
        [
            "market_id",
            "event_slug",
            "league_code",
            "league_name",
            "sport_code",
            "match_id",
            "question",
            "group_item_title",
            "market_slug",
            "market_shape",
            "classification_status",
            "classification_reason",
            "fixture_key",
            "group_key",
            "updated_at",
        ]
    ].copy() if not classification_audit.empty else pd.DataFrame()
    pm_shadow._upsert_rows(
        connection,
        "pm_fixture_market_candidates",
        candidate_store.to_dict(orient="records") if not candidate_store.empty else [],
    )
    pm_shadow._upsert_rows(
        connection,
        "pm_fixture_groups_v2",
        groups_v2_store.to_dict(orient="records") if not groups_v2_store.empty else [],
    )
    pm_shadow._upsert_rows(
        connection,
        "pm_market_classification_audit",
        audit_store.to_dict(orient="records") if not audit_store.empty else [],
    )
    pm_shadow.sync_discovery_to_db(connection, catalog=catalog, groups=groups)

    alias_frame = _team_alias_rows(matches, groups if not groups.empty else pd.DataFrame(columns=["league_code", "home_team", "away_team"]))
    pm_shadow._upsert_rows(connection, "pm_team_aliases", alias_frame.to_dict(orient="records") if not alias_frame.empty else [])
    resolutions_updated = pm_shadow.refresh_resolutions(connection, gamma=gamma)

    history_rows: list[dict[str, Any]] = []
    if not groups_v2.empty:
        complete_groups = groups_v2[groups_v2["group_status"].astype(str).eq("complete_group")].copy()
        for row in complete_groups.itertuples(index=False):
            decision_time = pd.Timestamp(row.game_start_time) - pd.to_timedelta(settings.polymarket.decision_offset_minutes, unit="m")
            for market_id in (str(row.home_market_id), str(row.draw_market_id), str(row.away_market_id)):
                if not market_id:
                    continue
                try:
                    payload = clob.get_prices_history(
                        market_id=market_id,
                        interval="1h",
                        start_ts=int((decision_time - pd.to_timedelta(12, unit="h")).timestamp()),
                        end_ts=int(decision_time.timestamp()),
                    )
                except requests.HTTPError:
                    continue
                history = payload.get("history", []) if isinstance(payload, dict) else []
                for item in history:
                    timestamp = pm_shadow._parse_timestamp(item.get("t") or item.get("timestamp"))
                    price = item.get("p") or item.get("price")
                    if pd.isna(timestamp) or price is None:
                        continue
                    price = float(price)
                    if not (0.0 < price < 1.0) or timestamp > decision_time:
                        continue
                    history_rows.append(
                        {
                            "price_key": f"{market_id}:{int(timestamp.timestamp())}",
                            "market_id": market_id,
                            "group_key": str(row.group_key),
                            "timestamp": pm_shadow._iso_timestamp(timestamp),
                            "price": price,
                            "source": "prices_history_backfill",
                            "decision_time": pm_shadow._iso_timestamp(decision_time),
                            "lag_seconds": float((decision_time - timestamp).total_seconds()),
                            "raw_json": pm_shadow._clean_json(item),
                        }
                    )
        pm_shadow._upsert_rows(connection, "pm_price_history", history_rows)

    chunk_rows: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _chunk_bounds(start, end, settings.polymarket.historical_chunk_days):
        chunk_groups = (
            groups_v2[groups_v2["match_date"].between(chunk_start, chunk_end, inclusive="both")].copy()
            if not groups_v2.empty
            else pd.DataFrame()
        )
        chunk_catalog = (
            raw_markets[raw_markets["game_start_time"].between(chunk_start, chunk_end + pd.to_timedelta(1, unit="D"), inclusive="both")].copy()
            if not raw_markets.empty
            else pd.DataFrame()
        )
        chunk_history = [
            row
            for row in history_rows
            if chunk_start <= pd.Timestamp(row["timestamp"]) <= chunk_end + pd.to_timedelta(1, unit="D")
        ]
        chunk_rows.append(
            {
                "chunk_start": pm_shadow._iso_timestamp(chunk_start),
                "chunk_end": pm_shadow._iso_timestamp(chunk_end),
                "group_rows": int(len(chunk_groups)),
                "complete_groups": int(chunk_groups["group_status"].astype(str).eq("complete_group").sum()) if not chunk_groups.empty else 0,
                "catalog_rows": int(len(chunk_catalog)),
                "price_history_rows": int(len(chunk_history)),
            }
        )

    audit_summary = _coverage_audit_summary(raw_events, raw_markets, candidates, groups_v2, classification_audit)
    summary = {
        "database_path": str(db_path),
        "history_start": pm_shadow._iso_timestamp(start),
        "history_end": pm_shadow._iso_timestamp(end),
        "page_scans": page_rows,
        "events_discovered": int(len(raw_events)),
        "catalog_rows": int(len(catalog)),
        "group_rows": int(len(groups_v2)),
        "complete_groups": int(groups_v2["group_status"].astype(str).eq("complete_group").sum()) if not groups_v2.empty else 0,
        "unmapped_groups": int(groups_v2["group_status"].astype(str).ne("complete_group").sum()) if not groups_v2.empty else 0,
        "alias_rows": int(len(alias_frame)),
        "price_history_rows": int(len(history_rows)),
        "price_history_markets": int(len({row["market_id"] for row in history_rows})),
        "resolutions_updated": int(resolutions_updated),
        "chunk_coverage": chunk_rows,
        "backfill_complete": True,
        **audit_summary,
    }
    summary_path = run.run_dir / "backfill_summary.json"
    chunk_path = run.run_dir / "chunk_coverage.csv"
    event_path = run.run_dir / "raw_event_rows.csv"
    market_path = run.run_dir / "raw_market_rows.csv"
    candidate_path = run.run_dir / "fixture_market_candidates.csv"
    group_v2_path = run.run_dir / "fixture_groups_v2.csv"
    audit_path = run.run_dir / "market_classification_audit.csv"
    catalog_path = run.run_dir / "catalog_rows.csv"
    group_path = run.run_dir / "group_rows.csv"
    alias_path = run.run_dir / "team_aliases.csv"
    coverage_audit_path = run.run_dir / "coverage_audit_summary.json"
    _save_json(summary_path, summary)
    _save_json(coverage_audit_path, audit_summary)
    pd.DataFrame(chunk_rows).to_csv(chunk_path, index=False)
    raw_events.to_csv(event_path, index=False)
    raw_markets.to_csv(market_path, index=False)
    candidates.to_csv(candidate_path, index=False)
    groups_v2.to_csv(group_v2_path, index=False)
    classification_audit.to_csv(audit_path, index=False)
    catalog.to_csv(catalog_path, index=False)
    groups.to_csv(group_path, index=False)
    alias_frame.to_csv(alias_path, index=False)
    connection.close()
    return PolymarketHistoryBackfillResult(
        run=run,
        database_path=db_path,
        summary_path=summary_path,
        summary=summary,
        artifacts={
            "backfill_summary": summary_path,
            "coverage_audit_summary": coverage_audit_path,
            "chunk_coverage": chunk_path,
            "raw_event_rows": event_path,
            "raw_market_rows": market_path,
            "fixture_market_candidates": candidate_path,
            "fixture_groups_v2": group_v2_path,
            "market_classification_audit": audit_path,
            "catalog_rows": catalog_path,
            "group_rows": group_path,
            "team_aliases": alias_path,
        },
    )


def audit_polymarket_coverage(
    settings: Settings,
    history_matches: pd.DataFrame | None = None,
    db_path: Path | str | None = None,
) -> PolymarketCoverageAuditResult:
    database_path = Path(db_path) if db_path else pm_shadow.default_polymarket_db_path(settings)
    if history_matches is not None:
        connection = pm_shadow.init_polymarket_db(database_path)
        raw_markets = _table_or_empty(connection, "SELECT * FROM pm_market_raw LIMIT 1")
        connection.close()
        if raw_markets.empty:
            backfill_polymarket_history(settings=settings, history_matches=history_matches, db_path=database_path)

    connection = pm_shadow.init_polymarket_db(database_path)
    raw_events = _table_or_empty(connection, "SELECT * FROM pm_event_raw")
    raw_markets = _table_or_empty(connection, "SELECT * FROM pm_market_raw")
    candidates = _table_or_empty(connection, "SELECT * FROM pm_fixture_market_candidates")
    groups_v2 = _table_or_empty(connection, "SELECT * FROM pm_fixture_groups_v2")
    audit = _table_or_empty(connection, "SELECT * FROM pm_market_classification_audit")
    connection.close()

    summary = _coverage_audit_summary(raw_events, raw_markets, candidates, groups_v2, audit)
    summary["database_path"] = str(database_path)
    summary["complete_groups"] = int(groups_v2["group_status"].astype(str).eq("complete_group").sum()) if not groups_v2.empty else 0
    run = create_run_context(settings.paths.runs_dir, "audit_polymarket_coverage")
    summary_path = run.run_dir / "coverage_audit_summary.json"
    fixture_audit_path = run.run_dir / "fixture_market_audit.csv"
    unmatched_path = run.run_dir / "unmatched_markets.csv"
    duplicate_path = run.run_dir / "duplicate_markets.csv"
    non_1x2_path = run.run_dir / "non_1x2_markets.csv"
    partial_1x2_path = run.run_dir / "partial_1x2_markets.csv"
    _save_json(summary_path, summary)
    audit.to_csv(fixture_audit_path, index=False)
    audit[audit["classification_status"].astype(str).eq("unmatched")].to_csv(unmatched_path, index=False)
    audit[audit["classification_status"].astype(str).eq("duplicate")].to_csv(duplicate_path, index=False)
    audit[audit["classification_status"].astype(str).eq("non_1x2")].to_csv(non_1x2_path, index=False)
    audit[audit["classification_status"].astype(str).eq("partial_1x2")].to_csv(partial_1x2_path, index=False)
    return PolymarketCoverageAuditResult(
        run=run,
        database_path=database_path,
        summary_path=summary_path,
        summary=summary,
        artifacts={
            "coverage_audit_summary": summary_path,
            "fixture_market_audit": fixture_audit_path,
            "unmatched_markets": unmatched_path,
            "duplicate_markets": duplicate_path,
            "non_1x2_markets": non_1x2_path,
            "partial_1x2_markets": partial_1x2_path,
        },
    )


def _needs_backfill(connection: Any, matches: pd.DataFrame, settings: Settings) -> bool:
    groups = pm_shadow._group_lookup(connection)
    if groups.empty:
        return True
    start, end = _history_window(settings, matches)
    game_times = pd.to_datetime(groups.get("game_start_time"), utc=True, errors="coerce")
    complete_mask = groups["mapping_status"].astype(str).eq("complete")
    relevant = groups[
        complete_mask
        & game_times.between(start - pd.to_timedelta(2, unit="D"), end + pd.to_timedelta(2, unit="D"))
    ]
    if relevant.empty:
        return True
    latest_complete = game_times.loc[complete_mask].max()
    if pd.isna(latest_complete):
        return True
    return bool(latest_complete < end - pd.to_timedelta(7, unit="D"))


def _prepare_context(
    settings: Settings,
    history_matches: pd.DataFrame | None,
    db_path: Path | str | None,
    model_path: Path | str | None,
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _, payload = _load_model_payload(settings, model_path)
    matches = _history_matches(settings, payload, history_matches)
    database_path = Path(db_path) if db_path else pm_shadow.default_polymarket_db_path(settings)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = pm_shadow.init_polymarket_db(database_path)
    backfill_summary: dict[str, Any] = {}
    if _needs_backfill(connection, matches, settings):
        result = backfill_polymarket_history(settings=settings, history_matches=matches, db_path=database_path)
        backfill_summary = result.summary
        connection.close()
        connection = pm_shadow.init_polymarket_db(database_path)
    catalog = pm_shadow._catalog_lookup(connection)
    groups = pm_shadow._group_lookup(connection)
    checkpoints = pm_shadow._checkpoint_lookup(connection)
    price_history = pm_shadow._price_history_lookup(connection)
    alias_frame = pm_shadow._team_alias_lookup(connection)
    if not backfill_summary:
        raw_events = _table_or_empty(connection, "SELECT * FROM pm_event_raw")
        raw_markets = _table_or_empty(connection, "SELECT * FROM pm_market_raw")
        candidate_audit = _table_or_empty(connection, "SELECT * FROM pm_fixture_market_candidates")
        groups_v2 = _table_or_empty(connection, "SELECT * FROM pm_fixture_groups_v2")
        market_audit = _table_or_empty(connection, "SELECT * FROM pm_market_classification_audit")
        if not raw_markets.empty:
            backfill_summary = _coverage_audit_summary(raw_events, raw_markets, candidate_audit, groups_v2, market_audit)
            backfill_summary["events_discovered"] = int(len(raw_events))
            backfill_summary["complete_groups"] = int(groups_v2["group_status"].astype(str).eq("complete_group").sum()) if not groups_v2.empty else 0
    if alias_frame.empty:
        alias_frame = _team_alias_rows(matches, groups)
        pm_shadow._upsert_rows(connection, "pm_team_aliases", alias_frame.to_dict(orient="records"))
    predictions = _historical_predictions(settings, payload, matches)
    mapping_audit = _mapping_audit_rows(settings, predictions, groups, alias_frame)
    candidates, mappings = _retro_candidates(
        settings=settings,
        predictions=predictions,
        groups=groups,
        catalog=catalog,
        checkpoints=checkpoints,
        price_history=price_history,
        mapping_audit=mapping_audit,
    )
    connection.close()
    return database_path, payload, predictions, candidates, mappings, mapping_audit, backfill_summary


def _historical_predictions(settings: Settings, payload: dict[str, Any], matches: pd.DataFrame) -> pd.DataFrame:
    fixtures = matches.copy()
    fixtures["kickoff_time"] = pd.to_datetime(fixtures["Date"], utc=True).dt.normalize() + pd.to_timedelta(
        settings.snapshot.default_kickoff_hour,
        unit="h",
    )
    feature_rows = build_fixture_feature_rows(
        history_matches=payload["history_matches"],
        fixtures=fixtures,
        rolling_window=int(payload["rolling_window"]),
    )
    market = build_market_odds(fixtures)
    dataset = feature_rows.merge(
        market,
        on=["match_id", "Date", "league_code", "season", "HomeTeam", "AwayTeam"],
        how="left",
    )
    actual = matches[["match_id", "outcome", "FTHG", "FTAG"]].copy()
    actual = actual.rename(columns={"FTHG": "home_goals", "FTAG": "away_goals"})
    actual["target"] = actual["outcome"].map(OUTCOME_TO_TARGET)
    dataset = dataset.merge(actual, on="match_id", how="left")
    dataset["league_name"] = dataset.get("league_name", dataset["league_code"])
    model = payload["model"]
    rho = float(payload["rho"])
    feature_columns = payload["feature_columns"]
    lambda_home, lambda_away = model.predict_lambdas(dataset[feature_columns])
    raw = outcome_probabilities_from_lambdas(
        lambda_home,
        lambda_away,
        rho=rho,
        max_goals=int(payload["max_poisson_goals"]),
    )
    calibrator = payload.get("calibrator")
    calibrated = raw.copy() if calibrator is None else calibrator.transform(raw)
    predictions = _build_prediction_frame(dataset, lambda_home, lambda_away, raw, calibrated)
    predictions["actual_outcome"] = dataset["outcome"].astype(str)
    predictions["actual_target"] = dataset["target"].astype(int)
    predictions["kickoff_time"] = fixtures["kickoff_time"].values
    predictions["league_name"] = dataset["league_name"]
    return predictions


RETRO_SEGMENT_DISCOVERY = "discovery_train"
RETRO_SEGMENT_DEV = "selection_dev"
RETRO_SEGMENT_HOLDOUT = "locked_holdout"
PRE_HOLDOUT_WINDOWS = ("pre_holdout_a", "pre_holdout_b", "pre_holdout_c")
RETRO_BASELINE_RUN_ID = "backtest_polymarket_retro_20260418_192807"
RETRO_PROMOTION_READY = "promoted_for_t45m"
RETRO_PROMOTION_REJECTED = "overfit_rejected"
RETRO_POLICY_SCOPES: tuple[dict[str, Any], ...] = (
    {
        "scope_name": "global_all",
        "allowed_outcomes": (),
        "min_odds": 1.2,
        "max_odds": 6.0,
    },
    {
        "scope_name": "global_no_draw",
        "allowed_outcomes": (OUTCOME_HOME, OUTCOME_AWAY),
        "min_odds": 1.2,
        "max_odds": 6.0,
    },
    {
        "scope_name": "mid_odds_all",
        "allowed_outcomes": (),
        "min_odds": 1.5,
        "max_odds": 2.0,
    },
    {
        "scope_name": "mid_odds_no_draw",
        "allowed_outcomes": (OUTCOME_HOME, OUTCOME_AWAY),
        "min_odds": 1.5,
        "max_odds": 2.0,
    },
)
RESEARCH_STATUS_COVERAGE_LIMITED = "coverage_limited"
RESEARCH_STATUS_OVERFIT_REJECTED = "overfit_rejected"
RESEARCH_STATUS_SMALL_SAMPLE = "research_positive_but_sample_small"
RESEARCH_STATUS_PROMOTABLE = "promotable_for_forward"
MODEL_STATUS_NOT_PROMOTED = "not_promoted"
MODEL_STATUS_PREDICTIVE_ONLY = "predictive_only_improvement"
MODEL_STATUS_PRE_HOLDOUT_INSUFFICIENT = "pre_holdout_insufficient_sample"
WEIGHTED_VARIANTS = {"v4", "v5", "v4b", "v5b", "v6", "v6b", "v7", "v7b"}


def _experiment_name(
    variant: str,
    scorer: str,
    training_scope: str,
    regional_adjustment: str = REGIONAL_ADJUSTMENT_NONE,
    stability_gate_mode: str = STABILITY_GATE_NONE,
) -> str:
    scorer_key = str(scorer or DECISION_SCORER_HEURISTIC)
    scope_key = str(training_scope or TRAINING_SCOPE_ELIGIBLE)
    adjustment_key = str(regional_adjustment or REGIONAL_ADJUSTMENT_NONE)
    gate_key = str(stability_gate_mode or STABILITY_GATE_NONE)
    if scorer_key == DECISION_SCORER_HEURISTIC and adjustment_key == REGIONAL_ADJUSTMENT_NONE and gate_key == STABILITY_GATE_NONE:
        return str(variant).lower()
    parts = [str(variant).lower(), scorer_key, scope_key]
    if adjustment_key != REGIONAL_ADJUSTMENT_NONE:
        parts.append(adjustment_key)
    if gate_key != STABILITY_GATE_NONE:
        parts.append(gate_key)
    return "__".join(parts)


def _primary_experiment_configs() -> list[dict[str, str]]:
    configs = [
        {
            "variant": variant,
            "decision_scorer": DECISION_SCORER_HEURISTIC,
            "decision_training_scope": TRAINING_SCOPE_ELIGIBLE,
            "regional_adjustment": REGIONAL_ADJUSTMENT_NONE,
            "stability_gate_mode": STABILITY_GATE_NONE,
        }
        for variant in ("v1", "v2", "v3", "v2r", "v3r", "v4", "v5", "v4b", "v5b")
    ]
    configs.extend(
        [
            {"variant": "v6", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v3r", "decision_scorer": DECISION_SCORER_LOGIT_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v5", "decision_scorer": DECISION_SCORER_LOGIT_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6", "decision_scorer": DECISION_SCORER_LOGIT_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_LOGIT_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7", "decision_scorer": DECISION_SCORER_LOGIT_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_LOGIT_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_HGB_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_HGB_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v3r", "decision_scorer": DECISION_SCORER_RELIABILITY_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v4", "decision_scorer": DECISION_SCORER_RELIABILITY_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v5", "decision_scorer": DECISION_SCORER_RELIABILITY_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6", "decision_scorer": DECISION_SCORER_RELIABILITY_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_RELIABILITY_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7", "decision_scorer": DECISION_SCORER_RELIABILITY_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_RELIABILITY_PROB, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
            {"variant": "v6", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK},
            {"variant": "v7", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK},
            {"variant": "v6", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK},
            {"variant": "v7", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK},
            {"variant": "v6", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_SOFT_PARENT},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_SOFT_PARENT},
            {"variant": "v7", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_SOFT_PARENT},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_SOFT_PARENT},
            {"variant": "v6", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_VALIDATED_OUTCOME},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_VALIDATED_OUTCOME},
            {"variant": "v7", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_VALIDATED_OUTCOME},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_VALIDATED_OUTCOME},
            {"variant": "v6", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_VALIDATED_OUTCOME_SOFT},
            {"variant": "v6b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_VALIDATED_OUTCOME_SOFT},
            {"variant": "v7", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_VALIDATED_OUTCOME_SOFT},
            {"variant": "v7b", "decision_scorer": DECISION_SCORER_HEURISTIC, "decision_training_scope": TRAINING_SCOPE_ELIGIBLE, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE, "stability_gate_mode": STABILITY_GATE_VALIDATED_OUTCOME_SOFT},
        ]
    )
    for config in configs:
        config.setdefault("stability_gate_mode", STABILITY_GATE_NONE)
    return configs


def _secondary_experiment_configs() -> list[dict[str, str]]:
    return [
        {"variant": "v5", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v5", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v6b", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v6b", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v7b", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v7b", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v5", "decision_scorer": DECISION_SCORER_LOGIT_PROB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v6b", "decision_scorer": DECISION_SCORER_LOGIT_PROB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v7b", "decision_scorer": DECISION_SCORER_LOGIT_PROB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v5", "decision_scorer": DECISION_SCORER_RELIABILITY_PROB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v6b", "decision_scorer": DECISION_SCORER_RELIABILITY_PROB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v7b", "decision_scorer": DECISION_SCORER_RELIABILITY_PROB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_NONE},
        {"variant": "v6b", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK},
        {"variant": "v6b", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK},
        {"variant": "v7b", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK},
        {"variant": "v7b", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK},
        {"variant": "v6b", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK},
        {"variant": "v6b", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK},
        {"variant": "v7b", "decision_scorer": DECISION_SCORER_LOGIT, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK},
        {"variant": "v7b", "decision_scorer": DECISION_SCORER_HGB, "decision_training_scope": TRAINING_SCOPE_ARGMAX, "regional_adjustment": REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK},
    ]


def _candidate_context_columns(mapped_dataset: pd.DataFrame) -> list[str]:
    desired = [
        "league_code",
        "season_phase_early",
        "season_phase_mid",
        "season_phase_late",
        "season_progress_pct",
        "home_long_sample_ratio_overall",
        "away_long_sample_ratio_overall",
        "home_long_sample_ratio_side",
        "away_long_sample_ratio_side",
        "home_goal_volatility_20",
        "away_goal_volatility_20",
        "home_goals_against_volatility_20",
        "away_goals_against_volatility_20",
        "home_opponent_strength_volatility_20",
        "away_opponent_strength_volatility_20",
        "home_schedule_irregularity_20",
        "away_schedule_irregularity_20",
        "confidence_floor_continuous",
        "confidence_score_v2",
        "points_form_diff_5",
        "points_form_diff_10",
        "rest_advantage",
        "home_attack_minus_away_defense_shrunk",
        "away_attack_minus_home_defense_shrunk",
        "low_confidence_match_flag",
    ]
    return [column for column in desired if column in mapped_dataset.columns]


def _attach_candidate_context(candidate_rows: pd.DataFrame, mapped_dataset: pd.DataFrame) -> pd.DataFrame:
    if candidate_rows.empty:
        return candidate_rows.copy()
    context_columns = _candidate_context_columns(mapped_dataset)
    if not context_columns:
        return candidate_rows.copy()
    context_frame = mapped_dataset[["match_id", *context_columns]].drop_duplicates("match_id")
    merged = candidate_rows.merge(context_frame, on="match_id", how="left", suffixes=("", "_context"))
    for column in context_columns:
        context_column = f"{column}_context"
        if context_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[column].combine_first(merged[context_column])
            merged = merged.drop(columns=[context_column])
        else:
            merged = merged.rename(columns={context_column: column})
    return merged


def _gate_policy_track(
    candidates: pd.DataFrame,
    *,
    policy: BetPolicy,
    probability_source: str,
    settings: Settings,
) -> dict[str, Any]:
    selected = select_candidate_rows(candidates, policy, probability_source, settings.polymarket.slippage_cushion)
    fills = _simulate_selected(selected, settings)
    return _summarize_track(selected, fills, settings)


def _gate_before_after_report(
    before_after_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    policy: BetPolicy,
    probability_source: str,
    settings: Settings,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split_name, (before_frame, after_frame) in before_after_frames.items():
        before_summary = _gate_policy_track(
            before_frame,
            policy=policy,
            probability_source=probability_source,
            settings=settings,
        )
        after_summary = _gate_policy_track(
            after_frame,
            policy=policy,
            probability_source=probability_source,
            settings=settings,
        )
        report[split_name] = {
            "before_gate": before_summary,
            "after_gate": after_summary,
            "delta_bets": int(after_summary.get("bets", 0) - before_summary.get("bets", 0)),
            "delta_net_roi": float(after_summary.get("net_roi", 0.0) - before_summary.get("net_roi", 0.0)),
            "delta_net_pnl": float(after_summary.get("net_pnl", 0.0) - before_summary.get("net_pnl", 0.0)),
        }
    return report


def _apply_decision_region_scorer(
    oof_candidates: pd.DataFrame,
    dev_candidates: pd.DataFrame,
    holdout_candidates: pd.DataFrame,
    *,
    probability_source: str,
    decision_scorer: str,
    decision_training_scope: str,
    regional_adjustment: str,
    stability_gate_mode: str,
    baseline_policy: BetPolicy,
    slippage_cushion: float,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    diagnostics: dict[str, Any] = {}
    oof_working = oof_candidates.copy()
    dev_working = dev_candidates.copy()
    holdout_working = holdout_candidates.copy()
    oof_base = candidate_score_columns(oof_working, probability_source=probability_source, slippage_cushion=slippage_cushion)
    dev_base = candidate_score_columns(dev_working, probability_source=probability_source, slippage_cushion=slippage_cushion)
    holdout_base = candidate_score_columns(holdout_working, probability_source=probability_source, slippage_cushion=slippage_cushion)
    if str(regional_adjustment) in {REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK, REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK}:
        oof_adjusted, dev_adjusted, holdout_adjusted, regional_diagnostics = crossfit_regional_adjustment(
            oof_base,
            dev_base,
            holdout_base,
            probability_source=probability_source,
            fold_column="retro_fold_id",
            adjustment_mode=str(regional_adjustment),
            policy=baseline_policy,
        )
        oof_base = candidate_score_columns(oof_adjusted, probability_source=probability_source, slippage_cushion=slippage_cushion)
        dev_base = candidate_score_columns(dev_adjusted, probability_source=probability_source, slippage_cushion=slippage_cushion)
        holdout_base = candidate_score_columns(holdout_adjusted, probability_source=probability_source, slippage_cushion=slippage_cushion)
        diagnostics["regional_adjustment"] = regional_diagnostics
    else:
        diagnostics["regional_adjustment"] = {"mode": REGIONAL_ADJUSTMENT_NONE, "regions": {}, "crossfit_folds": []}
    scored = crossfit_decision_region_scores(
        oof_base,
        dev_base,
        holdout_base,
        scorer_name=decision_scorer,
        training_scope=decision_training_scope,
    )
    diagnostics["decision_region"] = scored[3]
    scored_oof, scored_dev, scored_holdout = scored[0], scored[1], scored[2]
    if any("decision_adjusted_prob" in frame.columns for frame in (scored_oof, scored_dev, scored_holdout)):
        scored_oof = candidate_score_columns(scored_oof, probability_source=probability_source, slippage_cushion=slippage_cushion)
        scored_dev = candidate_score_columns(scored_dev, probability_source=probability_source, slippage_cushion=slippage_cushion)
        scored_holdout = candidate_score_columns(scored_holdout, probability_source=probability_source, slippage_cushion=slippage_cushion)
        diagnostics["decision_region"]["policy_probability_adjustment_applied"] = True
    else:
        diagnostics["decision_region"]["policy_probability_adjustment_applied"] = False
    stability_gate_audit = pd.DataFrame()
    if str(stability_gate_mode) in {
        STABILITY_GATE_HIERARCHICAL,
        STABILITY_GATE_SOFT_PARENT,
        STABILITY_GATE_OUTCOME_PARENT,
        STABILITY_GATE_EXPANDING_OUTCOME,
        STABILITY_GATE_VALIDATED_OUTCOME,
        STABILITY_GATE_VALIDATED_OUTCOME_SOFT,
    }:
        oof_gated, dev_gated, holdout_gated, stability_diagnostics, stability_gate_audit = crossfit_stability_gate(
            scored_oof,
            scored_dev,
            scored_holdout,
            policy=baseline_policy,
            probability_source=probability_source,
            fold_column="retro_fold_id",
            mode=str(stability_gate_mode),
        )
        stability_diagnostics["before_after"] = _gate_before_after_report(
            {
                "oof": (scored_oof, oof_gated),
                "pre_holdout": (scored_dev, dev_gated),
                "locked_holdout": (scored_holdout, holdout_gated),
            },
            policy=baseline_policy,
            probability_source=probability_source,
            settings=settings,
        )
        diagnostics["stability_gate"] = stability_diagnostics
        return oof_gated, dev_gated, holdout_gated, diagnostics, stability_gate_audit
    diagnostics["stability_gate"] = {"mode": STABILITY_GATE_NONE, "regions": {}, "crossfit_folds": [], "blocked_by_split": {}}
    return scored_oof, scored_dev, scored_holdout, diagnostics, stability_gate_audit


def _history_dataset_rows(settings: Settings, matches: pd.DataFrame) -> pd.DataFrame:
    feature_rows = build_feature_rows(matches, rolling_window=settings.backtest.rolling_window)
    market_odds = build_market_odds(matches)
    dataset = feature_rows.merge(
        market_odds,
        on=["match_id", "Date", "league_code", "season", "HomeTeam", "AwayTeam"],
        how="left",
    )
    dataset["kickoff_time"] = pd.to_datetime(dataset["Date"], utc=True).dt.normalize() + pd.to_timedelta(
        settings.snapshot.default_kickoff_hour,
        unit="h",
    )
    dataset["match_id"] = dataset["match_id"].astype(str)
    dataset["league_name"] = dataset.get("league_name", dataset["league_code"])
    return dataset.sort_values(["Date", "match_id"]).reset_index(drop=True)


def _prepare_retro_base_context(
    settings: Settings,
    history_matches: pd.DataFrame | None,
    db_path: Path | str | None,
    model_path: Path | str | None,
) -> tuple[
    Path,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    _, payload = _load_model_payload(settings, model_path)
    matches = _history_matches(settings, payload, history_matches)
    dataset = _history_dataset_rows(settings, matches)
    database_path = Path(db_path) if db_path else pm_shadow.default_polymarket_db_path(settings)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = pm_shadow.init_polymarket_db(database_path)
    backfill_summary: dict[str, Any] = {}
    if _needs_backfill(connection, matches, settings):
        result = backfill_polymarket_history(settings=settings, history_matches=matches, db_path=database_path)
        backfill_summary = result.summary
        connection.close()
        connection = pm_shadow.init_polymarket_db(database_path)
    catalog = pm_shadow._catalog_lookup(connection)
    groups = pm_shadow._group_lookup(connection)
    checkpoints = pm_shadow._checkpoint_lookup(connection)
    price_history = pm_shadow._price_history_lookup(connection)
    alias_frame = pm_shadow._team_alias_lookup(connection)
    if not backfill_summary:
        raw_events = _table_or_empty(connection, "SELECT * FROM pm_event_raw")
        raw_markets = _table_or_empty(connection, "SELECT * FROM pm_market_raw")
        candidate_audit = _table_or_empty(connection, "SELECT * FROM pm_fixture_market_candidates")
        groups_v2 = _table_or_empty(connection, "SELECT * FROM pm_fixture_groups_v2")
        market_audit = _table_or_empty(connection, "SELECT * FROM pm_market_classification_audit")
        if not raw_markets.empty:
            backfill_summary = _coverage_audit_summary(raw_events, raw_markets, candidate_audit, groups_v2, market_audit)
            backfill_summary["events_discovered"] = int(len(raw_events))
            backfill_summary["complete_groups"] = int(groups_v2["group_status"].astype(str).eq("complete_group").sum()) if not groups_v2.empty else 0
    if alias_frame.empty:
        alias_frame = _team_alias_rows(matches, groups)
        pm_shadow._upsert_rows(connection, "pm_team_aliases", alias_frame.to_dict(orient="records"))
    connection.close()
    mapping_input = dataset[["match_id", "league_code", "league_name", "HomeTeam", "AwayTeam", "kickoff_time"]].drop_duplicates("match_id")
    mapping_audit = _mapping_audit_rows(settings, mapping_input, groups, alias_frame)
    return database_path, payload, matches, dataset, catalog, groups, checkpoints, price_history, mapping_audit, backfill_summary


def _segment_boundaries(sorted_match_rows: pd.DataFrame) -> pd.DataFrame:
    unique_matches = sorted_match_rows.drop_duplicates("match_id").sort_values(["kickoff_time", "match_id"]).reset_index(drop=True)
    total = len(unique_matches)
    if total < 10:
        raise ValueError("No hay suficientes partidos mapeados para una validacion retro bloqueada.")
    discovery_end = max(1, int(np.floor(total * 0.6)))
    dev_end = max(discovery_end + 1, int(np.floor(total * 0.8)))
    if dev_end >= total:
        dev_end = total - 1
    if discovery_end >= dev_end:
        discovery_end = max(1, dev_end - 1)
    unique_matches["retro_segment"] = RETRO_SEGMENT_HOLDOUT
    unique_matches.loc[: discovery_end - 1, "retro_segment"] = RETRO_SEGMENT_DISCOVERY
    unique_matches.loc[discovery_end: dev_end - 1, "retro_segment"] = RETRO_SEGMENT_DEV
    return unique_matches[["match_id", "kickoff_time", "retro_segment"]]


def _pre_holdout_windows(dev_rows: pd.DataFrame) -> pd.DataFrame:
    unique_matches = dev_rows.drop_duplicates("match_id").sort_values(["kickoff_time", "match_id"]).reset_index(drop=True)
    if unique_matches.empty:
        return unique_matches.assign(pre_holdout_window=pd.Series(dtype=str))[["match_id", "pre_holdout_window"]]
    index_chunks = [chunk for chunk in np.array_split(unique_matches.index.to_numpy(), len(PRE_HOLDOUT_WINDOWS)) if len(chunk)]
    window_rows: list[pd.DataFrame] = []
    for index, chunk_index in enumerate(index_chunks):
        window_name = PRE_HOLDOUT_WINDOWS[index]
        chunk = unique_matches.loc[list(chunk_index)].copy()
        chunk["pre_holdout_window"] = window_name
        window_rows.append(chunk[["match_id", "pre_holdout_window"]])
    return pd.concat(window_rows, ignore_index=True) if window_rows else unique_matches[["match_id"]].assign(pre_holdout_window=pd.Series(dtype=str))


def _build_discovery_folds(mapped_discovery: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = mapped_discovery.drop_duplicates("match_id").sort_values(["kickoff_time", "match_id"]).reset_index(drop=True)
    unique_dates = pd.to_datetime(ordered["kickoff_time"], utc=True).dt.normalize().drop_duplicates().sort_values().tolist()
    if not unique_dates:
        return []
    folds: list[dict[str, Any]] = []
    current_start = unique_dates[0] + pd.Timedelta(days=28)
    max_date = unique_dates[-1] + pd.Timedelta(days=1)
    fold_id = 1
    while current_start < max_date:
        test_end = current_start + pd.Timedelta(days=28)
        test_rows = ordered[(ordered["kickoff_time"] >= current_start) & (ordered["kickoff_time"] < test_end)].copy()
        if not test_rows.empty:
            folds.append(
                {
                    "fold_id": fold_id,
                    "test_start": current_start,
                    "test_end": test_end,
                    "match_ids": tuple(test_rows["match_id"].astype(str).tolist()),
                }
            )
            fold_id += 1
        current_start = test_end
    if len(folds) >= 4:
        return folds

    fallback: list[dict[str, Any]] = []
    for index, part in enumerate([chunk.tolist() for chunk in np.array_split(unique_dates, 5) if len(chunk)][1:], start=1):
        start = pd.Timestamp(min(part))
        end = pd.Timestamp(max(part)) + pd.Timedelta(days=1)
        test_rows = ordered[(ordered["kickoff_time"] >= start) & (ordered["kickoff_time"] < end)].copy()
        if test_rows.empty:
            continue
        fallback.append(
            {
                "fold_id": index,
                "test_start": start,
                "test_end": end,
                "match_ids": tuple(test_rows["match_id"].astype(str).tolist()),
            }
        )
    return fallback[:4]


def _split_for_calibration(train_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train_rows.sort_values(["Date", "match_id"]).reset_index(drop=True)
    if len(ordered) < 160:
        return ordered, pd.DataFrame()
    calibration_rows = min(150, max(50, len(ordered) // 5))
    calibration_anchor = ordered.iloc[-calibration_rows]["Date"]
    subtrain = ordered[ordered["Date"] < calibration_anchor].copy()
    calibration = ordered[ordered["Date"] >= calibration_anchor].copy()
    if subtrain.empty or calibration.empty:
        return ordered, pd.DataFrame()
    return subtrain, calibration


def _prune_feature_columns_for_fold(
    train_rows: pd.DataFrame,
    feature_columns: list[str],
    minimum_support: int = 20,
) -> tuple[list[str], dict[str, list[str]]]:
    used_columns: list[str] = []
    pruned = {"low_support": [], "zero_variance": []}
    for column in feature_columns:
        if column not in train_rows.columns:
            pruned["low_support"].append(column)
            continue
        series = train_rows[column]
        support = int(series.notna().sum())
        if support < int(minimum_support):
            pruned["low_support"].append(column)
            continue
        observed = series.dropna()
        if observed.empty:
            pruned["low_support"].append(column)
            continue
        if pd.api.types.is_numeric_dtype(observed):
            if float(pd.to_numeric(observed, errors="coerce").var(ddof=0)) == 0.0:
                pruned["zero_variance"].append(column)
                continue
        elif int(observed.astype(str).nunique(dropna=True)) <= 1:
            pruned["zero_variance"].append(column)
            continue
        used_columns.append(column)
    return used_columns, pruned


def _train_league_mix_summary(train_rows: pd.DataFrame) -> dict[str, Any]:
    if train_rows.empty or "league_code" not in train_rows.columns:
        return {"count": 0, "by_league": []}
    grouped = (
        train_rows.assign(league_code=train_rows["league_code"].astype(str))
        .groupby("league_code", observed=True)
        .agg(count=("match_id", "nunique"))
        .reset_index()
        .rename(columns={"league_code": "league"})
        .sort_values(["count", "league"], ascending=[False, True])
    )
    total = int(grouped["count"].sum()) if not grouped.empty else 0
    records = []
    for row in grouped.to_dict(orient="records"):
        count = int(row["count"])
        records.append(
            {
                "league": str(row["league"]),
                "count": count,
                "share": float(count / total) if total > 0 else 0.0,
            }
        )
    return {"count": total, "by_league": records}


def _challenger_train_weights(train_rows: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    if train_rows.empty:
        empty_summary = {
            "count": 0,
            "mean_weight": 0.0,
            "by_league": [],
            "by_time_tercile": [],
        }
        return np.array([], dtype=float), empty_summary

    working = train_rows[["league_code", "kickoff_time", "Date"]].copy()
    working["league_code"] = working["league_code"].astype(str)
    league_counts = working["league_code"].value_counts()
    mean_league_count = float(league_counts.mean()) if not league_counts.empty else 1.0
    working["league_weight"] = working["league_code"].map(
        lambda code: math.sqrt(mean_league_count / float(league_counts.get(code, mean_league_count)))
    )
    kickoff_times = pd.to_datetime(working["kickoff_time"], utc=True, errors="coerce")
    fallback_dates = pd.to_datetime(working["Date"], utc=True, errors="coerce")
    order_times = kickoff_times.where(kickoff_times.notna(), fallback_dates)
    percentile_rank = order_times.rank(method="average", pct=True).fillna(0.5)
    working["time_weight"] = 0.85 + (0.30 * percentile_rank.astype(float))
    working["final_weight"] = (working["league_weight"] * working["time_weight"]).clip(lower=0.75, upper=1.50)
    try:
        terciles = pd.qcut(percentile_rank.rank(method="first"), q=3, labels=["early", "mid", "late"], duplicates="drop")
    except ValueError:
        terciles = pd.Series(["mid"] * len(working), index=working.index, dtype=object)
    working["time_tercile"] = pd.Series(terciles, index=working.index).astype(str)
    by_league = (
        working.groupby("league_code", observed=True)["final_weight"]
        .agg(count="size", mean_weight="mean", min_weight="min", max_weight="max")
        .reset_index()
        .rename(columns={"league_code": "league"})
        .to_dict(orient="records")
    )
    by_time_tercile = (
        working.groupby("time_tercile", observed=True)["final_weight"]
        .agg(count="size", mean_weight="mean", min_weight="min", max_weight="max")
        .reset_index()
        .rename(columns={"time_tercile": "bucket"})
        .to_dict(orient="records")
    )
    summary = {
        "count": int(len(working)),
        "mean_weight": float(working["final_weight"].mean()),
        "by_league": by_league,
        "by_time_tercile": by_time_tercile,
    }
    return working["final_weight"].to_numpy(dtype=float), summary


def _aggregate_weight_summaries(weight_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not weight_summaries:
        return {"weighted_calls": 0, "mean_weight": 0.0, "by_league": [], "by_time_tercile": []}

    total_count = sum(int(summary.get("count", 0)) for summary in weight_summaries)
    mean_weight = (
        sum(float(summary.get("mean_weight", 0.0)) * int(summary.get("count", 0)) for summary in weight_summaries) / total_count
        if total_count > 0
        else 0.0
    )

    league_accumulator: dict[str, dict[str, float]] = {}
    tercile_accumulator: dict[str, dict[str, float]] = {}
    for summary in weight_summaries:
        for row in summary.get("by_league", []):
            league = str(row.get("league", "unknown"))
            bucket = league_accumulator.setdefault(league, {"count": 0.0, "weighted_sum": 0.0, "min_weight": float("inf"), "max_weight": float("-inf")})
            count = float(row.get("count", 0.0))
            mean = float(row.get("mean_weight", 0.0))
            bucket["count"] += count
            bucket["weighted_sum"] += mean * count
            bucket["min_weight"] = min(bucket["min_weight"], float(row.get("min_weight", mean)))
            bucket["max_weight"] = max(bucket["max_weight"], float(row.get("max_weight", mean)))
        for row in summary.get("by_time_tercile", []):
            name = str(row.get("bucket", "unknown"))
            bucket = tercile_accumulator.setdefault(name, {"count": 0.0, "weighted_sum": 0.0, "min_weight": float("inf"), "max_weight": float("-inf")})
            count = float(row.get("count", 0.0))
            mean = float(row.get("mean_weight", 0.0))
            bucket["count"] += count
            bucket["weighted_sum"] += mean * count
            bucket["min_weight"] = min(bucket["min_weight"], float(row.get("min_weight", mean)))
            bucket["max_weight"] = max(bucket["max_weight"], float(row.get("max_weight", mean)))

    by_league = [
        {
            "league": league,
            "count": int(values["count"]),
            "mean_weight": float(values["weighted_sum"] / values["count"]) if values["count"] > 0 else 0.0,
            "min_weight": float(values["min_weight"]) if values["count"] > 0 else 0.0,
            "max_weight": float(values["max_weight"]) if values["count"] > 0 else 0.0,
        }
        for league, values in sorted(league_accumulator.items())
    ]
    by_time_tercile = [
        {
            "bucket": bucket_name,
            "count": int(values["count"]),
            "mean_weight": float(values["weighted_sum"] / values["count"]) if values["count"] > 0 else 0.0,
            "min_weight": float(values["min_weight"]) if values["count"] > 0 else 0.0,
            "max_weight": float(values["max_weight"]) if values["count"] > 0 else 0.0,
        }
        for bucket_name, values in sorted(tercile_accumulator.items())
    ]
    return {
        "weighted_calls": int(len(weight_summaries)),
        "mean_weight": float(mean_weight),
        "by_league": by_league,
        "by_time_tercile": by_time_tercile,
    }


def _aggregate_train_mix_summaries(train_mix_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not train_mix_summaries:
        return {"observations": 0, "by_league": []}
    accumulator: dict[str, int] = {}
    total = 0
    for summary in train_mix_summaries:
        for row in summary.get("by_league", []):
            league = str(row.get("league", "unknown"))
            count = int(row.get("count", 0))
            accumulator[league] = accumulator.get(league, 0) + count
            total += count
    by_league = [
        {
            "league": league,
            "count": count,
            "share": float(count / total) if total > 0 else 0.0,
        }
        for league, count in sorted(accumulator.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {"observations": int(total), "by_league": by_league}


def _confidence_scores(
    rows: pd.DataFrame,
    variant_raw: np.ndarray | None = None,
    v1_raw: np.ndarray | None = None,
) -> np.ndarray:
    return helper_confidence_scores(rows, variant_raw=variant_raw, v1_raw=v1_raw)


def _confidence_score_summary(scores: np.ndarray | list[float], applied: bool) -> dict[str, Any]:
    return helper_confidence_score_summary(scores, applied=applied)


def _blend_probabilities(
    primary: np.ndarray,
    fallback: np.ndarray,
    confidence_scores: np.ndarray,
    divergence_scores: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    return helper_blend_probabilities(primary, fallback, confidence_scores, divergence_scores=divergence_scores)


def _backoff_inactive_reason(
    backoff_variant: bool,
    confidence_scores: np.ndarray | list[float],
    divergence_scores: np.ndarray | list[float],
) -> str | None:
    return helper_backoff_inactive_reason(backoff_variant, confidence_scores, divergence_scores)


def _merge_selected_and_fills(
    selected: pd.DataFrame,
    fills: pd.DataFrame,
    combined_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if selected.empty and fills.empty:
        return pd.DataFrame()
    if selected.empty:
        merged = fills.copy()
    elif fills.empty:
        merged = selected.copy()
    elif "decision_id" in selected.columns and "decision_id" in fills.columns:
        merged = fills.merge(selected, on="decision_id", how="left", suffixes=("", "_selected"))
    else:
        selected_frame = selected.reset_index(drop=True)
        fills_frame = fills.reset_index(drop=True)
        merged = pd.concat([fills_frame, selected_frame], axis=1)

    if combined_predictions is not None and not combined_predictions.empty and "match_id" in merged.columns and "match_id" in combined_predictions.columns:
        prediction_columns = [
            column
            for column in ("confidence_score", "confidence_divergence_to_v1")
            if column in combined_predictions.columns
        ]
        if prediction_columns:
            prediction_frame = combined_predictions[["match_id", *prediction_columns]].drop_duplicates("match_id")
            merged = merged.merge(prediction_frame, on="match_id", how="left", suffixes=("", "_prediction"))
            for column in prediction_columns:
                prediction_column = f"{column}_prediction"
                if prediction_column in merged.columns:
                    if column in merged.columns:
                        merged[column] = pd.to_numeric(merged[column], errors="coerce").combine_first(
                            pd.to_numeric(merged[prediction_column], errors="coerce")
                        )
                        merged = merged.drop(columns=[prediction_column])
                    else:
                        merged = merged.rename(columns={prediction_column: column})
    return merged


def _decision_region_report(
    selected: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    label: str,
    combined_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    merged = _merge_selected_and_fills(selected, fills, combined_predictions=combined_predictions)
    merged = _collapse_selected_bet_rows(merged)
    report = build_decision_region_diagnostics(
        merged,
        fold_columns=("retro_fold_id", "fold_id", "policy_fold", "fold_segment"),
        window_columns=("pre_holdout_window", "policy_window", "window_id", "window_name", "policy_week", "decision_window", "fold_window"),
    )
    report["label"] = str(label)
    report["summary_text"] = format_decision_region_summary(report)
    return report


def _fit_variant_model(
    settings: Settings,
    train_rows: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_columns = model_feature_columns_for_variant(train_rows, variant=variant)
    model = build_goal_model(train_rows[feature_columns])
    fit_goal_model(model, train_rows[feature_columns], train_rows["home_goals"], train_rows["away_goals"])
    lambda_home, lambda_away = model.predict_lambdas(prediction_rows[feature_columns])
    return lambda_home, lambda_away, feature_columns


def _raw_probabilities_for_variant(
    settings: Settings,
    train_rows: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    lambda_home, lambda_away, feature_columns = _fit_variant_model(settings, train_rows, prediction_rows, variant)
    raw = outcome_probabilities_from_lambdas(
        lambda_home,
        lambda_away,
        rho=0.0,
        max_goals=settings.backtest.max_poisson_goals,
    )
    return lambda_home, lambda_away, raw, feature_columns


def _train_variant_predictions(
    settings: Settings,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subtrain, calibration = _split_for_calibration(train_rows)
    train_source = subtrain if not subtrain.empty else train_rows
    variant_key = str(variant).lower()
    calibrator: OutcomeCalibrator | None = None
    confidence_summary = _confidence_score_summary([], applied=False)
    feature_columns: list[str]
    lambda_home: np.ndarray
    lambda_away: np.ndarray
    raw: np.ndarray
    rho = 0.0
    feature_usage: dict[str, Any]
    weight_summary: dict[str, Any] | None = None
    train_league_mix_summary = _train_league_mix_summary(train_source)
    confidence_divergence = np.zeros(len(test_rows), dtype=float)
    confidence_scores = np.ones(len(test_rows), dtype=float)
    backoff_variant = variant_key in {"v4b", "v5b", "v6b", "v7b"}
    backoff_inactive_reason: str | None = "not_backoff_variant"
    if backoff_variant:
        primary_variant = {
            "v4b": "v4",
            "v5b": "v5",
            "v6b": "v6",
            "v7b": "v7",
        }.get(variant_key, "v5")
        declared_primary_columns = model_feature_columns_for_variant(train_source, variant=primary_variant)
        primary_feature_columns, primary_pruned = _prune_feature_columns_for_fold(train_source, declared_primary_columns)
        if not primary_feature_columns:
            raise ValueError(f"La variante {variant_key} se ha quedado sin columnas usables tras la poda por fold.")
        declared_fallback_columns = model_feature_columns_for_variant(train_source, variant="v1")
        fallback_feature_columns, fallback_pruned = _prune_feature_columns_for_fold(train_source, declared_fallback_columns)
        if not fallback_feature_columns:
            raise ValueError(f"La variante fallback v1 se ha quedado sin columnas usables para {variant_key}.")

        primary_weights, weight_summary = _challenger_train_weights(train_source)
        primary_model = build_goal_model(train_source[primary_feature_columns])
        fallback_model = build_goal_model(train_source[fallback_feature_columns])
        fit_goal_model(
            primary_model,
            train_source[primary_feature_columns],
            train_source["home_goals"],
            train_source["away_goals"],
            sample_weight=primary_weights,
        )
        fit_goal_model(
            fallback_model,
            train_source[fallback_feature_columns],
            train_source["home_goals"],
            train_source["away_goals"],
            sample_weight=None,
        )

        primary_rho = 0.0
        fallback_rho = 0.0
        if not calibration.empty:
            primary_calibration_home, primary_calibration_away = primary_model.predict_lambdas(calibration[primary_feature_columns])
            fallback_calibration_home, fallback_calibration_away = fallback_model.predict_lambdas(calibration[fallback_feature_columns])
            primary_rho = fit_dixon_coles_rho(
                primary_calibration_home,
                primary_calibration_away,
                calibration["home_goals"].to_numpy(),
                calibration["away_goals"].to_numpy(),
                settings.backtest.dixon_coles_bounds,
            )
            fallback_rho = fit_dixon_coles_rho(
                fallback_calibration_home,
                fallback_calibration_away,
                calibration["home_goals"].to_numpy(),
                calibration["away_goals"].to_numpy(),
                settings.backtest.dixon_coles_bounds,
            )
            primary_calibration_raw = outcome_probabilities_from_lambdas(
                primary_calibration_home,
                primary_calibration_away,
                rho=primary_rho,
                max_goals=settings.backtest.max_poisson_goals,
            )
            fallback_calibration_raw = outcome_probabilities_from_lambdas(
                fallback_calibration_home,
                fallback_calibration_away,
                rho=fallback_rho,
                max_goals=settings.backtest.max_poisson_goals,
            )
            calibration_confidence = _confidence_scores(calibration, primary_calibration_raw, fallback_calibration_raw)
            calibration_divergence = np.clip(np.sum(np.abs(primary_calibration_raw - fallback_calibration_raw), axis=1) / 2.0, 0.0, 1.0)
            calibration_raw = _blend_probabilities(
                primary_calibration_raw,
                fallback_calibration_raw,
                calibration_confidence,
                calibration_divergence,
            )
            if calibration["target"].nunique() >= 2:
                calibrator = OutcomeCalibrator(epsilon=settings.backtest.calibrator_epsilon).fit(
                    calibration_raw,
                    calibration["target"].to_numpy(),
                )

        primary_test_home, primary_test_away = primary_model.predict_lambdas(test_rows[primary_feature_columns])
        fallback_test_home, fallback_test_away = fallback_model.predict_lambdas(test_rows[fallback_feature_columns])
        primary_raw = outcome_probabilities_from_lambdas(
            primary_test_home,
            primary_test_away,
            rho=primary_rho,
            max_goals=settings.backtest.max_poisson_goals,
        )
        fallback_raw = outcome_probabilities_from_lambdas(
            fallback_test_home,
            fallback_test_away,
            rho=fallback_rho,
            max_goals=settings.backtest.max_poisson_goals,
        )
        confidence_scores = _confidence_scores(test_rows, primary_raw, fallback_raw)
        confidence_divergence = np.clip(np.sum(np.abs(primary_raw - fallback_raw), axis=1) / 2.0, 0.0, 1.0)
        raw = _blend_probabilities(primary_raw, fallback_raw, confidence_scores, confidence_divergence)
        feature_columns = primary_feature_columns
        lambda_home, lambda_away = primary_test_home, primary_test_away
        rho = float(primary_rho)
        backoff_inactive_reason = _backoff_inactive_reason(True, confidence_scores, confidence_divergence)
        confidence_summary = _confidence_score_summary(
            confidence_scores,
            applied=backoff_inactive_reason is None,
        )
        feature_usage = {
            "declared_feature_columns": declared_primary_columns,
            "used_feature_columns": primary_feature_columns,
            "pruned_feature_columns": primary_pruned,
            "fallback_feature_columns": fallback_feature_columns,
            "fallback_pruned_feature_columns": fallback_pruned,
        }
    else:
        declared_feature_columns = model_feature_columns_for_variant(train_source, variant=variant_key)
        feature_columns, pruned_feature_columns = _prune_feature_columns_for_fold(train_source, declared_feature_columns)
        if not feature_columns:
            raise ValueError(f"La variante {variant_key} se ha quedado sin columnas usables tras la poda por fold.")
        model = build_goal_model(train_source[feature_columns])
        sample_weight = None
        if variant_key in WEIGHTED_VARIANTS:
            sample_weight, weight_summary = _challenger_train_weights(train_source)
        fit_goal_model(
            model,
            train_source[feature_columns],
            train_source["home_goals"],
            train_source["away_goals"],
            sample_weight=sample_weight,
        )
        if not calibration.empty:
            calibration_home, calibration_away = model.predict_lambdas(calibration[feature_columns])
            rho = fit_dixon_coles_rho(
                calibration_home,
                calibration_away,
                calibration["home_goals"].to_numpy(),
                calibration["away_goals"].to_numpy(),
                settings.backtest.dixon_coles_bounds,
            )
            calibration_raw = outcome_probabilities_from_lambdas(
                calibration_home,
                calibration_away,
                rho=rho,
                max_goals=settings.backtest.max_poisson_goals,
            )
            if calibration["target"].nunique() >= 2:
                calibrator = OutcomeCalibrator(epsilon=settings.backtest.calibrator_epsilon).fit(
                    calibration_raw,
                    calibration["target"].to_numpy(),
                )

        lambda_home, lambda_away = model.predict_lambdas(test_rows[feature_columns])
        raw = outcome_probabilities_from_lambdas(
            lambda_home,
            lambda_away,
            rho=rho,
            max_goals=settings.backtest.max_poisson_goals,
        )
        confidence_scores = _confidence_scores(test_rows)
        confidence_summary = _confidence_score_summary(confidence_scores, applied=False)
        feature_usage = {
            "declared_feature_columns": declared_feature_columns,
            "used_feature_columns": feature_columns,
            "pruned_feature_columns": pruned_feature_columns,
        }
    calibrated = raw.copy() if calibrator is None else calibrator.transform(raw)
    predictions = _build_prediction_frame(test_rows, lambda_home, lambda_away, raw, calibrated)
    predictions["kickoff_time"] = test_rows["kickoff_time"].values
    predictions["league_name"] = test_rows["league_name"].values
    predictions["model_variant"] = str(variant_key)
    predictions["confidence_score"] = confidence_scores
    predictions["confidence_score_v2"] = confidence_scores
    predictions["confidence_divergence_to_v1"] = confidence_divergence
    predictions["confidence_backoff_variant"] = float(backoff_variant)
    predictions["confidence_backoff_applied"] = float(confidence_summary["applied"])
    predictions["confidence_backoff_not_engaged"] = float(backoff_variant and not confidence_summary["applied"])
    predictions["confidence_backoff_inactive_reason"] = "" if backoff_inactive_reason is None else str(backoff_inactive_reason)
    return predictions, {
        "variant": str(variant_key),
        "rho": float(rho),
        "feature_columns": feature_columns,
        "calibrator_fitted": bool(calibrator is not None),
        "confidence_score_summary": confidence_summary,
        "backoff_inactive_reason": backoff_inactive_reason,
        "backoff_variant": bool(backoff_variant),
        "variant_feature_families": variant_feature_families(variant_key),
        "feature_usage": feature_usage,
        "train_weight_summary": weight_summary,
        "train_league_mix_summary": train_league_mix_summary,
    }


def _retro_predictions_for_rows(
    settings: Settings,
    rows: pd.DataFrame,
    full_history: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_rows = full_history[full_history["Date"] < pd.Timestamp(rows["Date"].min())].copy()
    if train_rows.empty:
        raise ValueError("No hay historial previo suficiente para generar predicciones retro sin leakage.")
    return _train_variant_predictions(settings=settings, train_rows=train_rows, test_rows=rows, variant=variant)


def _base_notional_fills(fills: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if fills.empty:
        return fills
    return fills[fills["notional"].eq(float(settings.polymarket.notionals_ladder[0]))].copy()


def _month_concentration(fills: pd.DataFrame) -> float:
    if fills.empty:
        return 0.0
    month_series = pd.to_datetime(fills["kickoff_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    grouped = fills.assign(month=month_series).groupby("month", observed=True)["net_profit"].sum()
    total = float(grouped.sum())
    if total <= 0:
        return 1.0
    return float(grouped.max() / total)


def _policy_fold_metrics(fills: pd.DataFrame) -> dict[str, float]:
    if fills.empty:
        return {"bets": 0, "roi": 0.0, "drawdown_norm": 0.0, "pnl": 0.0}
    total_cost = float(fills["cost_basis"].sum())
    drawdown = pm_shadow._drawdown_from_profit(fills.sort_values("created_at")["net_profit"])
    return {
        "bets": int(len(fills)),
        "roi": float(fills["net_profit"].sum() / total_cost) if total_cost > 0 else 0.0,
        "drawdown_norm": float(abs(drawdown) / total_cost) if total_cost > 0 else 0.0,
        "pnl": float(fills["net_profit"].sum()),
    }


def _odds_band(value: float) -> str:
    odds = float(value)
    if not np.isfinite(odds):
        return "unknown"
    if 1.2 <= odds < 1.5:
        return "1.2-1.5"
    if 1.5 <= odds < 2.0:
        return "1.5-2.0"
    if 2.0 <= odds < 3.0:
        return "2.0-3.0"
    if 3.0 <= odds <= 6.0:
        return "3.0-6.0"
    return "other"


def _aggregate_cohort(rows: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    grouped = (
        rows.groupby(group_column, observed=True)
        .agg(
            bets=("decision_id", "nunique"),
            wins=("won", "sum"),
            pnl=("net_profit", "sum"),
            total_cost=("cost_basis", "sum"),
        )
        .reset_index()
    )
    grouped["hit_rate"] = np.where(grouped["bets"] > 0, grouped["wins"] / grouped["bets"], 0.0)
    grouped["roi"] = np.where(grouped["total_cost"] > 0, grouped["pnl"] / grouped["total_cost"], 0.0)
    return grouped.rename(columns={group_column: "group"}).to_dict(orient="records")


def _cohort_snapshot(selected: pd.DataFrame, fills: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    base_fills = _base_notional_fills(fills, settings)
    if selected.empty or base_fills.empty:
        return {
            "summary": {"bets": 0, "hit_rate": 0.0, "net_pnl": 0.0, "net_roi": 0.0},
            "by_league": [],
            "by_outcome": [],
            "by_odds_band": [],
            "by_month": [],
        }
    merged = base_fills.merge(
        selected[
            [
                "decision_id",
                "league_code",
                "selection",
                "quoted_odds",
                "kickoff_time",
            ]
        ],
        on="decision_id",
        how="left",
        suffixes=("", "_selected"),
    )
    merged["league_bucket"] = merged["league_code"].astype(str)
    merged["outcome_bucket"] = merged["selection"].astype(str)
    merged["odds_band"] = merged["quoted_odds"].map(_odds_band)
    merged["month_bucket"] = pd.to_datetime(merged["kickoff_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    total_cost = float(merged["cost_basis"].sum())
    return {
        "summary": {
            "bets": int(merged["decision_id"].nunique()),
            "hit_rate": float(merged["won"].mean()) if not merged.empty else 0.0,
            "net_pnl": float(merged["net_profit"].sum()),
            "net_roi": float(merged["net_profit"].sum() / total_cost) if total_cost > 0 else 0.0,
        },
        "by_league": _aggregate_cohort(merged, "league_bucket"),
        "by_outcome": _aggregate_cohort(merged, "outcome_bucket"),
        "by_odds_band": _aggregate_cohort(merged, "odds_band"),
        "by_month": _aggregate_cohort(merged, "month_bucket"),
    }


def _cohort_report(
    selection_dev_selected: pd.DataFrame,
    selection_dev_fills: pd.DataFrame,
    holdout_selected: pd.DataFrame,
    holdout_fills: pd.DataFrame,
    settings: Settings,
) -> dict[str, Any]:
    return {
        RETRO_SEGMENT_DEV: _cohort_snapshot(selection_dev_selected, selection_dev_fills, settings),
        RETRO_SEGMENT_HOLDOUT: _cohort_snapshot(holdout_selected, holdout_fills, settings),
    }


def _pre_holdout_window_report(
    dev_candidates: pd.DataFrame,
    probability_source: str,
    baseline_policy: BetPolicy,
    settings: Settings,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reports: dict[str, Any] = {}
    selected_parts: list[pd.DataFrame] = []
    fill_parts: list[pd.DataFrame] = []
    argmax_selected_parts: list[pd.DataFrame] = []
    argmax_fill_parts: list[pd.DataFrame] = []
    for window_name in PRE_HOLDOUT_WINDOWS:
        window_candidates = dev_candidates[dev_candidates["pre_holdout_window"].astype(str).eq(window_name)].copy()
        selected = select_candidate_rows(window_candidates, baseline_policy, probability_source, settings.polymarket.slippage_cushion)
        fills = _simulate_selected(selected, settings)
        argmax_selected = _argmax_from_candidates(window_candidates, probability_source)
        argmax_fills = _simulate_selected(argmax_selected, settings)
        reports[window_name] = {
            "model_pure": _model_metrics(window_candidates, probability_source),
            "argmax_all_bets": _summarize_track(argmax_selected, argmax_fills, settings),
            "current_policy_frozen": _summarize_track(selected, fills, settings),
            "candidate_policy": _summarize_track(selected, fills, settings),
        }
        if not selected.empty:
            selected_parts.append(selected.assign(pre_holdout_window=window_name))
        if not fills.empty:
            fill_parts.append(fills)
        if not argmax_selected.empty:
            argmax_selected_parts.append(argmax_selected.assign(pre_holdout_window=window_name))
        if not argmax_fills.empty:
            argmax_fill_parts.append(argmax_fills)
    selected_frame = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    fill_frame = pd.concat(fill_parts, ignore_index=True) if fill_parts else pd.DataFrame()
    argmax_selected_frame = pd.concat(argmax_selected_parts, ignore_index=True) if argmax_selected_parts else pd.DataFrame()
    argmax_fill_frame = pd.concat(argmax_fill_parts, ignore_index=True) if argmax_fill_parts else pd.DataFrame()
    return reports, selected_frame, fill_frame, argmax_selected_frame, argmax_fill_frame


def _pre_holdout_metrics(window_report: dict[str, Any]) -> dict[str, Any]:
    window_rois = {
        window_name: float(window_report.get(window_name, {}).get("current_policy_frozen", {}).get("net_roi", 0.0))
        for window_name in PRE_HOLDOUT_WINDOWS
    }
    window_bets = {
        window_name: int(window_report.get(window_name, {}).get("current_policy_frozen", {}).get("bets", 0))
        for window_name in PRE_HOLDOUT_WINDOWS
    }
    roi_values = np.array(list(window_rois.values()), dtype=float)
    total_bets = int(sum(window_bets.values()))
    positive_ratio = float(np.mean(roi_values > 0.0)) if roi_values.size else 0.0
    weighted_numerator = float(sum(window_rois[name] * window_bets[name] for name in PRE_HOLDOUT_WINDOWS))
    aggregate_roi = float(weighted_numerator / total_bets) if total_bets > 0 else 0.0
    minimum_window_bets = int(min(window_bets.values())) if window_bets else 0
    return {
        "pre_holdout_window_rois": window_rois,
        "pre_holdout_window_bets": window_bets,
        "pre_holdout_positive_window_ratio": positive_ratio,
        "pre_holdout_median_roi": float(np.median(roi_values)) if roi_values.size else 0.0,
        "pre_holdout_aggregate_roi": aggregate_roi,
        "pre_holdout_total_bets": total_bets,
        "pre_holdout_min_window_bets": minimum_window_bets,
    }


def _oof_frozen_policy_report(
    oof_candidates: pd.DataFrame,
    probability_source: str,
    baseline_policy: BetPolicy,
    settings: Settings,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    fold_reports: dict[str, Any] = {}
    selected_parts: list[pd.DataFrame] = []
    fill_parts: list[pd.DataFrame] = []
    fold_rois: dict[str, float] = {}
    fold_bets: dict[str, int] = {}
    fold_ids = sorted(
        int(value)
        for value in pd.to_numeric(oof_candidates.get("retro_fold_id"), errors="coerce").dropna().unique().tolist()
        if int(value) > 0
    )
    for fold_id in fold_ids:
        fold_name = f"fold_{fold_id}"
        fold_candidates = oof_candidates[pd.to_numeric(oof_candidates["retro_fold_id"], errors="coerce").eq(float(fold_id))].copy()
        selected = select_candidate_rows(fold_candidates, baseline_policy, probability_source, settings.polymarket.slippage_cushion)
        fills = _simulate_selected(selected, settings)
        fold_report = {
            "model_pure": _model_metrics(fold_candidates, probability_source),
            "current_policy_frozen": _summarize_track(selected, fills, settings),
            "policy_reoptimized": False,
        }
        fold_reports[fold_name] = fold_report
        fold_rois[fold_name] = float(fold_report["current_policy_frozen"]["net_roi"])
        fold_bets[fold_name] = int(fold_report["current_policy_frozen"]["bets"])
        if not selected.empty:
            selected_parts.append(selected.assign(retro_fold_id=fold_id))
        if not fills.empty:
            fill_parts.append(fills.assign(retro_fold_id=fold_id))
    selected_frame = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    fill_frame = pd.concat(fill_parts, ignore_index=True) if fill_parts else pd.DataFrame()
    aggregate = _summarize_track(selected_frame, fill_frame, settings)
    raw_roi_values = np.array(list(fold_rois.values()), dtype=float)
    validation_units = _contiguous_oof_policy_units(fold_ids, fold_bets, min_bets=6)
    unit_reports: dict[str, Any] = {}
    unit_rois: dict[str, float] = {}
    unit_bets: dict[str, int] = {}
    for unit_name, source_folds in validation_units.items():
        selected_unit = pd.DataFrame()
        fill_unit = pd.DataFrame()
        if not selected_frame.empty and "retro_fold_id" in selected_frame.columns:
            selected_fold_ids = pd.to_numeric(selected_frame["retro_fold_id"], errors="coerce")
            selected_unit = selected_frame[selected_fold_ids.isin(source_folds)].copy()
        if not fill_frame.empty and "retro_fold_id" in fill_frame.columns:
            fill_fold_ids = pd.to_numeric(fill_frame["retro_fold_id"], errors="coerce")
            fill_unit = fill_frame[fill_fold_ids.isin(source_folds)].copy()
        unit_summary = _summarize_track(selected_unit, fill_unit, settings)
        unit_reports[unit_name] = {
            "source_folds": [f"fold_{fold_id}" for fold_id in source_folds],
            "current_policy_frozen": unit_summary,
        }
        unit_rois[unit_name] = float(unit_summary["net_roi"])
        unit_bets[unit_name] = int(unit_summary["bets"])
    validation_roi_values = np.array(list(unit_rois.values()), dtype=float)
    total_bets = int(sum(fold_bets.values()))
    return {
        "folds": fold_reports,
        "oof_fold_rois": fold_rois,
        "oof_fold_bets": fold_bets,
        "oof_raw_positive_fold_ratio": float(np.mean(raw_roi_values > 0.0)) if raw_roi_values.size else 0.0,
        "oof_raw_median_roi": float(np.median(raw_roi_values)) if raw_roi_values.size else 0.0,
        "oof_raw_min_fold_bets": int(min(fold_bets.values())) if fold_bets else 0,
        "oof_validation_units": unit_reports,
        "oof_validation_unit_rois": unit_rois,
        "oof_validation_unit_bets": unit_bets,
        "oof_positive_fold_ratio": float(np.mean(validation_roi_values > 0.0)) if validation_roi_values.size else 0.0,
        "oof_median_roi": float(np.median(validation_roi_values)) if validation_roi_values.size else 0.0,
        "oof_aggregate_roi": float(aggregate["net_roi"]),
        "oof_total_bets": total_bets,
        "oof_min_fold_bets": int(min(unit_bets.values())) if unit_bets else 0,
        "oof_min_validation_unit_bets": int(min(unit_bets.values())) if unit_bets else 0,
        "current_policy_frozen": aggregate,
        "policy_reoptimized": False,
    }, selected_frame, fill_frame


def _oof_validation_status(oof_report: dict[str, Any]) -> str:
    if int(oof_report.get("oof_total_bets", 0)) < 60 or int(oof_report.get("oof_min_fold_bets", 0)) < 6:
        return MODEL_STATUS_PRE_HOLDOUT_INSUFFICIENT
    if float(oof_report.get("oof_aggregate_roi", 0.0)) <= 0.0 or float(oof_report.get("oof_positive_fold_ratio", 0.0)) < 0.5:
        return RESEARCH_STATUS_OVERFIT_REJECTED
    return MODEL_STATUS_NOT_PROMOTED


def _pre_holdout_validation_status(selection_dev_summary: dict[str, Any]) -> str:
    if int(selection_dev_summary.get("pre_holdout_total_bets", 0)) < 24 or int(selection_dev_summary.get("pre_holdout_min_window_bets", 0)) < 8:
        return MODEL_STATUS_PRE_HOLDOUT_INSUFFICIENT
    if (
        float(selection_dev_summary.get("pre_holdout_aggregate_roi", 0.0)) <= 0.0
        or float(selection_dev_summary.get("pre_holdout_positive_window_ratio", 0.0)) < (2.0 / 3.0)
    ):
        return RESEARCH_STATUS_OVERFIT_REJECTED
    return MODEL_STATUS_NOT_PROMOTED


def _contiguous_oof_policy_units(
    fold_ids: list[int],
    fold_bets: dict[str, int],
    *,
    min_bets: int = 6,
) -> dict[str, list[int]]:
    units: list[list[int]] = []
    current: list[int] = []
    current_bets = 0
    for fold_id in fold_ids:
        current.append(int(fold_id))
        current_bets += int(fold_bets.get(f"fold_{int(fold_id)}", 0))
        if current_bets >= min_bets:
            units.append(current)
            current = []
            current_bets = 0
    if current:
        if units:
            units[-1].extend(current)
        else:
            units.append(current)
    return {f"unit_{index + 1}": folds for index, folds in enumerate(units)}


def _confidence_backoff_report(
    combined_predictions: pd.DataFrame,
    selected: pd.DataFrame,
    fills: pd.DataFrame,
    settings: Settings,
) -> dict[str, Any]:
    backoff_variant = bool(pd.to_numeric(combined_predictions.get("confidence_backoff_variant"), errors="coerce").fillna(0.0).gt(0.0).any())
    applied = bool(pd.to_numeric(combined_predictions.get("confidence_backoff_applied"), errors="coerce").fillna(0.0).gt(0.0).any())
    report = helper_build_confidence_backoff_report(
        combined_predictions,
        selected,
        fills,
        backoff_variant=backoff_variant,
        applied=applied,
    )
    report["summary"]["variant_supports_backoff"] = backoff_variant
    report["summary"]["backoff_not_engaged"] = bool(backoff_variant and not applied)
    return report


def _stability_gate_report(diagnostics: dict[str, Any], audit: pd.DataFrame) -> dict[str, Any]:
    gate = dict(diagnostics.get("stability_gate", {}))
    mode = str(gate.get("mode", STABILITY_GATE_NONE))
    if audit.empty:
        return {
            "mode": mode,
            "regions": gate.get("regions", {}),
            "crossfit_folds": gate.get("crossfit_folds", []),
            "blocked_by_split": gate.get("blocked_by_split", {}),
            "before_after": gate.get("before_after", {}),
            "summary": {
                "candidates": 0,
                "blocked_candidates": 0,
                "blocked_candidate_share": 0.0,
                "blocked_base_policy_pick_share": 0.0,
            },
        }
    working = audit.copy()
    pass_mask = working.get("candidate_gate_pass", pd.Series(True, index=working.index)).fillna(True).astype(bool)
    base_policy_like = (
        pd.to_numeric(working.get("pre_gate_policy_edge", working.get("policy_edge")), errors="coerce").ge(0.03)
        & pd.to_numeric(working.get("pre_gate_policy_ev", working.get("policy_ev")), errors="coerce").ge(0.05)
        & working.get("quote_status", pd.Series("eligible", index=working.index)).fillna("eligible").astype(str).eq("eligible")
    )
    blocked_base = int((base_policy_like & ~pass_mask).sum())
    base_total = int(base_policy_like.sum())
    return {
        "mode": mode,
        "regions": gate.get("regions", {}),
        "crossfit_folds": gate.get("crossfit_folds", []),
        "blocked_by_split": gate.get("blocked_by_split", {}),
        "before_after": gate.get("before_after", {}),
        "summary": {
            "candidates": int(len(working)),
            "blocked_candidates": int((~pass_mask).sum()),
            "blocked_candidate_share": float((~pass_mask).mean()) if len(working) else 0.0,
            "base_policy_candidates": base_total,
            "blocked_base_policy_candidates": blocked_base,
            "blocked_base_policy_pick_share": float(blocked_base / base_total) if base_total > 0 else 0.0,
            "reasons": {
                str(key): int(value)
                for key, value in working.loc[~pass_mask, "candidate_gate_reason"].astype(str).value_counts().to_dict().items()
            },
        },
    }


def _load_frozen_baseline_summary(settings: Settings) -> dict[str, Any]:
    baseline_path = settings.paths.runs_dir / RETRO_BASELINE_RUN_ID / "retro_shadow_summary.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"No encuentro el baseline congelado en {baseline_path}. Ejecuta o conserva ese run antes de esta iteracion."
        )
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def _feature_family_set(variant: str) -> list[str]:
    return variant_feature_families(variant)


def _series_stats(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "count": float(len(clean)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
    }


def _model_diagnostics(
    mapped_dataset: pd.DataFrame,
    weight_diagnostics_by_variant: dict[str, Any] | None = None,
    effective_train_mix_by_variant: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mapped_dataset.empty:
        return {
            "mapped_matches": 0,
            "sample_sizes": {},
            "shrinkage_ratios": {},
            "volatility_by_league": [],
            "low_confidence_match_share": 0.0,
            "weight_diagnostics_by_variant": weight_diagnostics_by_variant or {},
            "effective_train_mix_by_variant": effective_train_mix_by_variant or {},
        }
    low_confidence = mapped_dataset["low_confidence_match_flag"] if "low_confidence_match_flag" in mapped_dataset.columns else pd.Series(dtype=float)
    diagnostics = {
        "mapped_matches": int(len(mapped_dataset)),
        "sample_sizes": {
            "home_overall": _series_stats(mapped_dataset.get("home_long_overall_sample_size", pd.Series(dtype=float))),
            "away_overall": _series_stats(mapped_dataset.get("away_long_overall_sample_size", pd.Series(dtype=float))),
            "home_side": _series_stats(mapped_dataset.get("home_long_side_sample_size", pd.Series(dtype=float))),
            "away_side": _series_stats(mapped_dataset.get("away_long_side_sample_size", pd.Series(dtype=float))),
        },
        "shrinkage_ratios": {
            "home_overall": _series_stats(mapped_dataset.get("home_shrinkage_ratio_overall", pd.Series(dtype=float))),
            "away_overall": _series_stats(mapped_dataset.get("away_shrinkage_ratio_overall", pd.Series(dtype=float))),
            "home_side": _series_stats(mapped_dataset.get("home_shrinkage_ratio_side", pd.Series(dtype=float))),
            "away_side": _series_stats(mapped_dataset.get("away_shrinkage_ratio_side", pd.Series(dtype=float))),
        },
        "volatility_by_league": [],
        "low_confidence_match_share": float(pd.to_numeric(low_confidence, errors="coerce").fillna(0.0).mean()) if not low_confidence.empty else 0.0,
        "weight_diagnostics_by_variant": weight_diagnostics_by_variant or {},
        "effective_train_mix_by_variant": effective_train_mix_by_variant or {},
    }
    by_league = (
        mapped_dataset.groupby("league_code", observed=True)
        .agg(
            matches=("match_id", "nunique"),
            avg_goals_volatility_5=("home_goals_for_volatility_5", "mean"),
            avg_goals_volatility_10=("home_goals_for_volatility_10", "mean"),
            avg_opponent_strength_volatility_5=("home_opponent_strength_volatility_5", "mean"),
            avg_opponent_strength_volatility_10=("home_opponent_strength_volatility_10", "mean"),
            low_confidence_share=("low_confidence_match_flag", "mean"),
        )
        .reset_index()
    )
    diagnostics["volatility_by_league"] = by_league.rename(columns={"league_code": "league"}).to_dict(orient="records")
    return diagnostics


def _baseline_pre_holdout_summary(baseline_summary: dict[str, Any]) -> dict[str, float]:
    selection_dev = baseline_summary.get("selection_dev_summary", {})
    frozen = selection_dev.get("current_policy_frozen", {})
    aggregate_roi = float(selection_dev.get("pre_holdout_aggregate_roi", frozen.get("net_roi", 0.0)))
    median_roi = float(selection_dev.get("pre_holdout_median_roi", aggregate_roi))
    positive_ratio = float(selection_dev.get("pre_holdout_positive_window_ratio", 1.0 if aggregate_roi > 0 else 0.0))
    return {
        "aggregate_roi": aggregate_roi,
        "median_roi": median_roi,
        "positive_window_ratio": positive_ratio,
    }


def _model_status(
    coverage_status: str,
    oof_log_loss: float,
    v1_raw_log_loss: float,
    oof_report: dict[str, Any],
    selection_dev_summary: dict[str, Any],
    locked_holdout_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> tuple[str, str, dict[str, float]]:
    selection_dev_frozen = selection_dev_summary["current_policy_frozen"]
    holdout_frozen = locked_holdout_summary["current_policy_frozen"]
    baseline_pre_holdout = _baseline_pre_holdout_summary(baseline_summary)
    baseline_holdout_roi = float(baseline_summary["locked_holdout_summary"]["current_policy_frozen"]["net_roi"])
    oof_improvement_vs_v1 = ((float(v1_raw_log_loss) - float(oof_log_loss)) / float(v1_raw_log_loss)) if float(v1_raw_log_loss) > 0 else 0.0
    oof_aggregate_roi = float(oof_report.get("oof_aggregate_roi", 0.0))
    oof_positive_fold_ratio = float(oof_report.get("oof_positive_fold_ratio", 0.0))
    oof_total_bets = int(oof_report.get("oof_total_bets", 0))
    oof_min_fold_bets = int(oof_report.get("oof_min_fold_bets", 0))
    pre_holdout_aggregate_roi = float(selection_dev_summary.get("pre_holdout_aggregate_roi", selection_dev_frozen.get("net_roi", 0.0)))
    pre_holdout_positive_window_ratio = float(selection_dev_summary.get("pre_holdout_positive_window_ratio", 0.0))
    pre_holdout_total_bets = int(selection_dev_summary.get("pre_holdout_total_bets", selection_dev_frozen.get("bets", 0)))
    pre_holdout_min_window_bets = int(selection_dev_summary.get("pre_holdout_min_window_bets", 0))
    holdout_bets = int(holdout_frozen.get("bets", 0))
    validation_total_bets = int(oof_total_bets + pre_holdout_total_bets + holdout_bets)
    holdout_sample_status = "holdout_sample_ready" if holdout_bets >= 40 else RESEARCH_STATUS_SMALL_SAMPLE
    validation_sample_status = "validation_sample_ready" if validation_total_bets >= 100 else MODEL_STATUS_PRE_HOLDOUT_INSUFFICIENT
    pre_holdout_delta = pre_holdout_aggregate_roi - float(baseline_pre_holdout["aggregate_roi"])
    holdout_delta = float(holdout_frozen["net_roi"]) - baseline_holdout_roi
    improved_oof = oof_improvement_vs_v1 > 0.0
    improved_selection = pre_holdout_delta > 0.0
    improved_holdout = holdout_delta > 0.0
    oof_validation_status = _oof_validation_status(oof_report)
    pre_holdout_validation_status = _pre_holdout_validation_status(selection_dev_summary)
    if oof_validation_status != MODEL_STATUS_NOT_PROMOTED:
        status = oof_validation_status
    elif pre_holdout_validation_status != MODEL_STATUS_NOT_PROMOTED:
        status = pre_holdout_validation_status
    elif improved_oof and not improved_holdout:
        status = MODEL_STATUS_PREDICTIVE_ONLY
    elif (
        improved_selection
        and improved_holdout
        and holdout_sample_status == "holdout_sample_ready"
        and validation_sample_status == "validation_sample_ready"
    ):
        status = RESEARCH_STATUS_PROMOTABLE
    elif improved_selection and improved_holdout and validation_sample_status == "validation_sample_ready":
        status = RESEARCH_STATUS_SMALL_SAMPLE
    else:
        status = MODEL_STATUS_NOT_PROMOTED
    if status == RESEARCH_STATUS_PROMOTABLE:
        promotion_eligibility = RESEARCH_STATUS_PROMOTABLE
    elif status in {
        RESEARCH_STATUS_OVERFIT_REJECTED,
        RESEARCH_STATUS_SMALL_SAMPLE,
        MODEL_STATUS_PREDICTIVE_ONLY,
        MODEL_STATUS_PRE_HOLDOUT_INSUFFICIENT,
    }:
        promotion_eligibility = status
    else:
        promotion_eligibility = MODEL_STATUS_NOT_PROMOTED
    return status, promotion_eligibility, {
        "oof_log_loss_vs_v1": float(oof_improvement_vs_v1),
        "oof_aggregate_roi": float(oof_aggregate_roi),
        "oof_positive_fold_ratio": float(oof_positive_fold_ratio),
        "pre_holdout_delta_vs_baseline": float(pre_holdout_delta),
        "locked_holdout_delta_vs_baseline": float(holdout_delta),
        "validation_total_bets": float(validation_total_bets),
        "holdout_sample_status": str(holdout_sample_status),
        "validation_sample_status": str(validation_sample_status),
        "oof_validation_status": str(oof_validation_status),
        "pre_holdout_validation_status": str(pre_holdout_validation_status),
    }


def _feature_ablation_report(experiments: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    chain = ["v1", "v2", "v2r", "v4", "v4b", "v5", "v5b", "v6", "v6b", "v7", "v7b"]
    report: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for variant in chain:
        preferred_key = next(
            (
                key
                for key, experiment in experiments.items()
                if str(experiment.get("variant")) == variant
                and str(experiment.get("decision_scorer", DECISION_SCORER_HEURISTIC)) == DECISION_SCORER_HEURISTIC
                and str(experiment.get("decision_training_scope", TRAINING_SCOPE_ELIGIBLE)) == TRAINING_SCOPE_ELIGIBLE
            ),
            None,
        )
        if preferred_key is None:
            preferred_key = next((key for key, experiment in experiments.items() if str(experiment.get("variant")) == variant), None)
        if preferred_key is None:
            continue
        experiment = experiments[preferred_key]
        probability_source = str(experiment["probability_source"])
        chosen_metrics = experiment["source_report"][probability_source]["oof_model_metrics"]
        current = {
            "variant": variant,
            "feature_family_set": _feature_family_set(variant),
            "probability_source": probability_source,
            "oof_log_loss": float(chosen_metrics.get("log_loss", 0.0)),
            "oof_brier": float(chosen_metrics.get("brier", 0.0)),
            "oof_aggregate_roi": float(experiment["oof_frozen_policy_report"]["oof_aggregate_roi"]),
            "oof_median_roi": float(experiment["oof_frozen_policy_report"]["oof_median_roi"]),
            "oof_positive_fold_ratio": float(experiment["oof_frozen_policy_report"]["oof_positive_fold_ratio"]),
            "pre_holdout_aggregate_roi": float(experiment["selection_dev_summary"]["pre_holdout_aggregate_roi"]),
            "pre_holdout_median_roi": float(experiment["selection_dev_summary"]["pre_holdout_median_roi"]),
            "locked_holdout_frozen_policy_roi": float(experiment["locked_holdout_summary"]["current_policy_frozen"]["net_roi"]),
            "status": str(experiment["status"]),
            "oof_validation_status": str(experiment.get("oof_validation_status", MODEL_STATUS_NOT_PROMOTED)),
            "pre_holdout_validation_status": str(experiment.get("pre_holdout_validation_status", MODEL_STATUS_NOT_PROMOTED)),
        }
        if previous is None:
            current["delta_vs_previous_oof_log_loss"] = 0.0
            current["delta_vs_previous_oof_roi"] = 0.0
            current["delta_vs_previous_pre_holdout"] = 0.0
            current["delta_vs_previous_holdout"] = 0.0
        else:
            current["delta_vs_previous_oof_log_loss"] = float(previous["oof_log_loss"] - current["oof_log_loss"])
            current["delta_vs_previous_oof_roi"] = float(current["oof_aggregate_roi"] - previous["oof_aggregate_roi"])
            current["delta_vs_previous_pre_holdout"] = float(current["pre_holdout_aggregate_roi"] - previous["pre_holdout_aggregate_roi"])
            current["delta_vs_previous_holdout"] = float(current["locked_holdout_frozen_policy_roi"] - previous["locked_holdout_frozen_policy_roi"])
        report.append(current)
        previous = current
    return report


def _decision_scorer_ablation(experiments: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_name, experiment in experiments.items():
        probability_source = str(experiment["probability_source"])
        chosen_metrics = experiment["source_report"][probability_source]["oof_model_metrics"]
        regional_diagnostics = dict(
            (experiment.get("decision_scorer_diagnostics", {}) or {}).get("regional_adjustment", {}) or {}
        )
        rows.append(
            {
                "experiment_name": str(experiment_name),
                "variant": str(experiment.get("variant")),
                "decision_scorer": str(experiment.get("decision_scorer", DECISION_SCORER_HEURISTIC)),
                "decision_training_scope": str(experiment.get("decision_training_scope", TRAINING_SCOPE_ELIGIBLE)),
                "regional_adjustment": str(experiment.get("regional_adjustment", REGIONAL_ADJUSTMENT_NONE)),
                "regional_adjustment_training_scope": str(regional_diagnostics.get("training_scope", "")),
                "regional_adjustment_training_rows": int(regional_diagnostics.get("training_rows", 0) or 0),
                "regional_adjustment_source_rows": int(regional_diagnostics.get("source_rows", 0) or 0),
                "stability_gate_mode": str(experiment.get("stability_gate_mode", STABILITY_GATE_NONE)),
                "probability_source": probability_source,
                "oof_log_loss": float(chosen_metrics.get("log_loss", 0.0)),
                "oof_aggregate_roi": float(experiment["oof_frozen_policy_report"]["oof_aggregate_roi"]),
                "pre_holdout_aggregate_roi": float(experiment["selection_dev_summary"]["pre_holdout_aggregate_roi"]),
                "locked_holdout_frozen_policy_roi": float(experiment["locked_holdout_summary"]["current_policy_frozen"]["net_roi"]),
                "status": str(experiment.get("status", MODEL_STATUS_NOT_PROMOTED)),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            float(item["oof_aggregate_roi"]),
            float(item["pre_holdout_aggregate_roi"]),
            float(item["locked_holdout_frozen_policy_roi"]),
            -float(item["oof_log_loss"]),
        ),
        reverse=True,
    )


def _roi_ladder_report(champion: dict[str, Any]) -> dict[str, Any]:
    oof_roi = float(champion["oof_frozen_policy_report"]["oof_aggregate_roi"])
    pre_holdout_roi = float(champion["selection_dev_summary"]["pre_holdout_aggregate_roi"])
    holdout_roi = float(champion["locked_holdout_summary"]["current_policy_frozen"]["net_roi"])
    return {
        "model_variant": str(champion.get("variant")),
        "decision_scorer": str(champion.get("decision_scorer", DECISION_SCORER_HEURISTIC)),
        "decision_training_scope": str(champion.get("decision_training_scope", TRAINING_SCOPE_ELIGIBLE)),
        "regional_adjustment": str(champion.get("regional_adjustment", REGIONAL_ADJUSTMENT_NONE)),
        "stability_gate_mode": str(champion.get("stability_gate_mode", STABILITY_GATE_NONE)),
        "milestone_1": bool(oof_roi > 0.0),
        "milestone_2": bool(pre_holdout_roi > 0.0),
        "milestone_3": bool(holdout_roi >= 0.15),
        "milestone_4": bool(holdout_roi >= 0.30),
        "milestone_5": bool(holdout_roi >= 0.45),
        "oof_aggregate_roi": oof_roi,
        "pre_holdout_aggregate_roi": pre_holdout_roi,
        "locked_holdout_frozen_policy_roi": holdout_roi,
    }


def _select_probability_source(
    oof_candidates: pd.DataFrame,
    dev_candidates: pd.DataFrame,
    baseline_policy: BetPolicy,
    settings: Settings,
) -> tuple[str, dict[str, Any]]:
    raw_metrics = _model_metrics(oof_candidates, "raw")
    calibrated_metrics = _model_metrics(oof_candidates, "calibrated")
    raw_selected = select_candidate_rows(dev_candidates, baseline_policy, "raw", settings.polymarket.slippage_cushion)
    calibrated_selected = select_candidate_rows(dev_candidates, baseline_policy, "calibrated", settings.polymarket.slippage_cushion)
    raw_fills = _base_notional_fills(_simulate_selected(raw_selected, settings), settings)
    calibrated_fills = _base_notional_fills(_simulate_selected(calibrated_selected, settings), settings)
    raw_dev = _policy_fold_metrics(raw_fills)
    calibrated_dev = _policy_fold_metrics(calibrated_fills)
    raw_log_loss = float(raw_metrics.get("log_loss", 0.0))
    calibrated_log_loss = float(calibrated_metrics.get("log_loss", raw_log_loss))
    raw_brier = float(raw_metrics.get("brier", float("inf")))
    calibrated_brier = float(calibrated_metrics.get("brier", raw_brier))
    improvement = ((raw_log_loss - calibrated_log_loss) / raw_log_loss) if raw_log_loss > 0 else 0.0
    calibrated_quality_better = calibrated_log_loss < raw_log_loss or (
        np.isclose(calibrated_log_loss, raw_log_loss) and calibrated_brier <= raw_brier
    )
    chosen = "calibrated" if calibrated_quality_better and float(calibrated_dev.get("roi", 0.0)) >= float(raw_dev.get("roi", 0.0)) else "raw"
    return chosen, {
        "raw": {"oof_model_metrics": raw_metrics, "oof_baseline_policy": raw_dev},
        "calibrated": {"oof_model_metrics": calibrated_metrics, "oof_baseline_policy": calibrated_dev},
        "chosen_probability_source": chosen,
        "relative_log_loss_improvement": float(improvement),
        "criterion": "oof_probability_quality",
    }


def _scoped_policy_grid(settings: Settings) -> list[BetPolicy]:
    policies: list[BetPolicy] = []
    for scope in RETRO_POLICY_SCOPES:
        min_odds = float(scope["min_odds"])
        max_odds = float(scope["max_odds"])
        allowed_outcomes = tuple(scope.get("allowed_outcomes", ()))
        scope_name = str(scope["scope_name"])
        for edge, ev in product(
            settings.backtest.policy_search.edge_thresholds,
            settings.backtest.policy_search.ev_thresholds,
        ):
            policies.append(
                BetPolicy(
                    edge_threshold=float(edge),
                    ev_threshold=float(ev),
                    min_odds=min_odds,
                    max_odds=max_odds,
                    kelly_fraction=settings.backtest.policy_search.max_kelly_fraction,
                    family="edge_ev_threshold",
                    allowed_leagues=(),
                    allowed_outcomes=allowed_outcomes,
                    scope_name=scope_name,
                )
            )
        for quantile in settings.backtest.policy_search.top_quantiles:
            policies.append(
                BetPolicy(
                    edge_threshold=0.0,
                    ev_threshold=0.0,
                    min_odds=min_odds,
                    max_odds=max_odds,
                    kelly_fraction=settings.backtest.policy_search.max_kelly_fraction,
                    family="top_quantile",
                    top_quantile=float(quantile),
                    allowed_leagues=(),
                    allowed_outcomes=allowed_outcomes,
                    scope_name=scope_name,
                )
            )
        policies.append(
            BetPolicy(
                edge_threshold=0.0,
                ev_threshold=0.0,
                min_odds=min_odds,
                max_odds=max_odds,
                kelly_fraction=settings.backtest.policy_search.max_kelly_fraction,
                family="ranked_one_pick",
                allowed_leagues=(),
                allowed_outcomes=allowed_outcomes,
                scope_name=scope_name,
            )
        )
    return policies


def _search_candidate_policy(
    settings: Settings,
    oof_candidates: pd.DataFrame,
    dev_candidates: pd.DataFrame,
    current_policy: BetPolicy,
    probability_source: str,
) -> tuple[BetPolicy, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    min_total_bets = max(10, int(settings.backtest.policy_search.min_bets))
    for policy in _scoped_policy_grid(settings):
        oof_selected = select_candidate_rows(oof_candidates, policy, probability_source, settings.polymarket.slippage_cushion)
        dev_selected = select_candidate_rows(dev_candidates, policy, probability_source, settings.polymarket.slippage_cushion)
        oof_fills = _base_notional_fills(_simulate_selected(oof_selected, settings), settings)
        dev_fills = _base_notional_fills(_simulate_selected(dev_selected, settings), settings)
        total_bets = int(len(oof_fills) + len(dev_fills))
        if total_bets < min_total_bets:
            continue
        fold_stats = [_policy_fold_metrics(group) for _, group in oof_fills.groupby("retro_fold_id", observed=True)]
        if not fold_stats:
            continue
        positive_ratio = float(np.mean([item["roi"] > 0 for item in fold_stats]))
        combined_fills = pd.concat([oof_fills, dev_fills], ignore_index=True)
        month_concentration = _month_concentration(combined_fills)
        mean_fold_roi = float(np.mean([item["roi"] for item in fold_stats]))
        normalized_drawdown = float(np.mean([item["drawdown_norm"] for item in fold_stats]))
        dev_metrics = _policy_fold_metrics(dev_fills)
        rows.append(
            {
                "scope_name": str(policy.scope_name),
                "family": policy.family,
                "edge_threshold": float(policy.edge_threshold),
                "ev_threshold": float(policy.ev_threshold),
                "min_odds": float(policy.min_odds),
                "max_odds": float(policy.max_odds),
                "top_quantile": float(policy.top_quantile),
                "allowed_leagues": ",".join(policy.allowed_leagues),
                "allowed_outcomes": ",".join(policy.allowed_outcomes),
                "oof_bets": int(len(oof_fills)),
                "selection_dev_bets": int(len(dev_fills)),
                "combined_bets": total_bets,
                "mean_fold_roi": mean_fold_roi,
                "normalized_drawdown": normalized_drawdown,
                "positive_fold_ratio": positive_ratio,
                "selection_dev_roi": float(dev_metrics["roi"]),
                "selection_dev_pnl": float(dev_metrics["pnl"]),
                "month_concentration": month_concentration,
                "score": mean_fold_roi - (0.20 * normalized_drawdown),
                "generalization_gap": abs(mean_fold_roi - float(dev_metrics["roi"])),
                "policy_payload": policy.to_dict(),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return current_policy, ranking
    ranking = ranking.sort_values(["score", "generalization_gap", "combined_bets"], ascending=[False, True, False]).reset_index(drop=True)
    payload = ranking.iloc[0]["policy_payload"]
    champion = BetPolicy(
        edge_threshold=float(payload["edge_threshold"]),
        ev_threshold=float(payload["ev_threshold"]),
        min_odds=float(payload["min_odds"]),
        max_odds=float(payload["max_odds"]),
        kelly_fraction=float(payload["kelly_fraction"]),
        family=str(payload["family"]),
        top_quantile=float(payload["top_quantile"]),
        allowed_leagues=tuple(payload.get("allowed_leagues", [])),
        allowed_outcomes=tuple(payload.get("allowed_outcomes", [])),
        scope_name=str(payload.get("scope_name", "global_all")),
    )
    return champion, ranking


def _summarize_track(selected: pd.DataFrame, fills: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    base_fills = _base_notional_fills(fills, settings)
    total_cost = float(base_fills["cost_basis"].sum()) if not base_fills.empty else 0.0
    return {
        "bets": int(len(base_fills)),
        "wins": int(base_fills["won"].eq(1).sum()) if not base_fills.empty else 0,
        "hit_rate": float(selected["selection"].astype(str).eq(selected["actual_outcome"].astype(str)).mean()) if not selected.empty else 0.0,
        "net_pnl": float(base_fills["net_profit"].sum()) if not base_fills.empty else 0.0,
        "net_roi": float(base_fills["net_profit"].sum() / total_cost) if total_cost > 0 else 0.0,
        "drawdown": pm_shadow._drawdown_from_profit(base_fills.sort_values("created_at")["net_profit"]) if not base_fills.empty else 0.0,
    }


def _argmax_from_candidates(candidates: pd.DataFrame, probability_source: str) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    working = candidates[candidates["quote_status"].astype(str).eq("eligible")].copy()
    if working.empty:
        return working
    working = working.sort_values(
        ["match_id", f"model_prob_{probability_source}", f"ev_{probability_source}", "quoted_odds"],
        ascending=[True, False, False, False],
    )
    return working.groupby("match_id", observed=True).head(1).copy().reset_index(drop=True)


def _build_variant_base_context(
    settings: Settings,
    dataset: pd.DataFrame,
    mapping_audit: pd.DataFrame,
    catalog: pd.DataFrame,
    groups: pd.DataFrame,
    checkpoints: pd.DataFrame,
    price_history: pd.DataFrame,
    payload: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    mapping_complete = mapping_audit[mapping_audit["mapping_status"].astype(str).eq("complete")][
        ["match_id", "group_key", "mapping_status", "audit_reason", "mapping_stage", "mapping_score"]
    ].copy()
    mapped_dataset = dataset.merge(mapping_complete, on="match_id", how="inner").sort_values(["kickoff_time", "match_id"]).reset_index(drop=True)
    segment_rows = _segment_boundaries(mapped_dataset)
    mapped_dataset = mapped_dataset.merge(segment_rows[["match_id", "retro_segment"]], on="match_id", how="left")
    discovery_rows = mapped_dataset[mapped_dataset["retro_segment"].eq(RETRO_SEGMENT_DISCOVERY)].copy()
    dev_rows = mapped_dataset[mapped_dataset["retro_segment"].eq(RETRO_SEGMENT_DEV)].copy()
    holdout_rows = mapped_dataset[mapped_dataset["retro_segment"].eq(RETRO_SEGMENT_HOLDOUT)].copy()
    pre_holdout_window_rows = _pre_holdout_windows(dev_rows)
    dev_rows = dev_rows.merge(pre_holdout_window_rows, on="match_id", how="left")
    folds = _build_discovery_folds(discovery_rows)
    if len(folds) < 4:
        raise ValueError("No hay suficientes folds discovery para una validacion retro honesta.")

    oof_predictions_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        test_rows = discovery_rows[discovery_rows["match_id"].astype(str).isin(fold["match_ids"])].copy()
        fold_cutoff = pd.Timestamp(fold["test_start"]).tz_localize(None).normalize()
        train_rows = dataset[dataset["Date"] < fold_cutoff].copy()
        if test_rows.empty or train_rows.empty:
            continue
        predictions, metadata = _train_variant_predictions(settings, train_rows, test_rows, variant)
        predictions["retro_segment"] = RETRO_SEGMENT_DISCOVERY
        predictions["retro_fold_id"] = int(fold["fold_id"])
        oof_predictions_parts.append(predictions)
        fold_rows.append(
            {
                "fold_id": int(fold["fold_id"]),
                "test_start": pm_shadow._iso_timestamp(fold["test_start"]),
                "test_end": pm_shadow._iso_timestamp(fold["test_end"]),
                "test_matches": int(len(test_rows)),
                "train_rows": int(len(train_rows)),
                "variant": str(variant),
                "metadata": metadata,
            }
        )
    oof_predictions = pd.concat(oof_predictions_parts, ignore_index=True) if oof_predictions_parts else pd.DataFrame()
    if oof_predictions.empty:
        raise ValueError("No he podido generar predicciones OOF para discovery.")

    dev_predictions, dev_metadata = _retro_predictions_for_rows(settings, dev_rows, dataset, variant)
    dev_predictions["retro_segment"] = RETRO_SEGMENT_DEV
    dev_predictions["retro_fold_id"] = 0
    holdout_train = dataset[dataset["Date"] < pd.Timestamp(holdout_rows["Date"].min())].copy()
    holdout_predictions, holdout_metadata = _train_variant_predictions(settings, holdout_train, holdout_rows, variant)
    holdout_predictions["retro_segment"] = RETRO_SEGMENT_HOLDOUT
    holdout_predictions["retro_fold_id"] = 0

    combined_predictions = pd.concat([oof_predictions, dev_predictions, holdout_predictions], ignore_index=True)
    candidate_rows, mappings = _retro_candidates(
        settings=settings,
        predictions=combined_predictions,
        groups=groups,
        catalog=catalog,
        checkpoints=checkpoints,
        price_history=price_history,
        mapping_audit=mapping_audit,
    )
    candidate_rows = _attach_candidate_context(candidate_rows, mapped_dataset)
    prediction_meta = combined_predictions[["match_id", "retro_segment", "retro_fold_id", "model_variant"]].drop_duplicates("match_id")
    candidate_rows = candidate_rows.merge(prediction_meta, on="match_id", how="left")
    candidate_rows = candidate_rows.merge(pre_holdout_window_rows, on="match_id", how="left")
    oof_candidates = candidate_rows[candidate_rows["retro_segment"].eq(RETRO_SEGMENT_DISCOVERY)].copy()
    dev_candidates = candidate_rows[candidate_rows["retro_segment"].eq(RETRO_SEGMENT_DEV)].copy()
    holdout_candidates = candidate_rows[candidate_rows["retro_segment"].eq(RETRO_SEGMENT_HOLDOUT)].copy()

    baseline_policy = pm_shadow._policy_from_raw(payload.get("policy", {}), settings)
    probability_source, source_report = _select_probability_source(oof_candidates, dev_candidates, baseline_policy, settings)
    return {
        "variant": str(variant),
        "mapped_dataset": mapped_dataset,
        "pre_holdout_window_rows": pre_holdout_window_rows,
        "fold_rows": fold_rows,
        "dev_metadata": dev_metadata,
        "holdout_metadata": holdout_metadata,
        "combined_predictions": combined_predictions,
        "oof_predictions": oof_predictions,
        "holdout_predictions": holdout_predictions,
        "candidate_rows": candidate_rows,
        "mappings": mappings,
        "oof_candidates": oof_candidates,
        "dev_candidates": dev_candidates,
        "holdout_candidates": holdout_candidates,
        "baseline_policy": baseline_policy,
        "probability_source": probability_source,
        "source_report": source_report,
    }


def _evaluate_variant_experiment(
    settings: Settings,
    dataset: pd.DataFrame,
    mapping_audit: pd.DataFrame,
    catalog: pd.DataFrame,
    groups: pd.DataFrame,
    checkpoints: pd.DataFrame,
    price_history: pd.DataFrame,
    backfill_summary: dict[str, Any],
    payload: dict[str, Any],
    variant: str,
    decision_scorer: str = DECISION_SCORER_HEURISTIC,
    decision_training_scope: str = TRAINING_SCOPE_ELIGIBLE,
    regional_adjustment: str = REGIONAL_ADJUSTMENT_NONE,
    stability_gate_mode: str = STABILITY_GATE_NONE,
    base_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if base_context is None:
        base_context = _build_variant_base_context(
            settings=settings,
            dataset=dataset,
            mapping_audit=mapping_audit,
            catalog=catalog,
            groups=groups,
            checkpoints=checkpoints,
            price_history=price_history,
            payload=payload,
            variant=variant,
        )
    mapped_dataset = base_context["mapped_dataset"]
    pre_holdout_window_rows = base_context["pre_holdout_window_rows"]
    fold_rows = base_context["fold_rows"]
    dev_metadata = base_context["dev_metadata"]
    holdout_metadata = base_context["holdout_metadata"]
    combined_predictions = base_context["combined_predictions"].copy()
    oof_predictions = base_context["oof_predictions"].copy()
    holdout_predictions = base_context["holdout_predictions"].copy()
    candidate_rows = base_context["candidate_rows"].copy()
    mappings = base_context["mappings"].copy()
    oof_candidates = base_context["oof_candidates"].copy()
    dev_candidates = base_context["dev_candidates"].copy()
    holdout_candidates = base_context["holdout_candidates"].copy()
    baseline_policy = base_context["baseline_policy"]
    probability_source = str(base_context["probability_source"])
    source_report = base_context["source_report"]
    (
        oof_candidates,
        dev_candidates,
        holdout_candidates,
        decision_scorer_diagnostics,
        stability_gate_audit,
    ) = _apply_decision_region_scorer(
        oof_candidates,
        dev_candidates,
        holdout_candidates,
        probability_source=probability_source,
        decision_scorer=decision_scorer,
        decision_training_scope=decision_training_scope,
        regional_adjustment=regional_adjustment,
        stability_gate_mode=stability_gate_mode,
        baseline_policy=baseline_policy,
        slippage_cushion=settings.polymarket.slippage_cushion,
        settings=settings,
    )
    candidate_rows = pd.concat([oof_candidates, dev_candidates, holdout_candidates], ignore_index=True, sort=False)
    champion_policy = baseline_policy
    ranking = pd.DataFrame()
    oof_frozen_policy_report, frozen_oof_selected, frozen_oof_fills = _oof_frozen_policy_report(
        oof_candidates,
        probability_source,
        baseline_policy,
        settings,
    )

    pre_holdout_window_report, frozen_dev_selected, frozen_dev_fills, argmax_dev_selected, argmax_dev_fills = _pre_holdout_window_report(
        dev_candidates,
        probability_source,
        baseline_policy,
        settings,
    )
    frozen_holdout_selected = select_candidate_rows(holdout_candidates, baseline_policy, probability_source, settings.polymarket.slippage_cushion)
    argmax_holdout_selected = _argmax_from_candidates(holdout_candidates, probability_source)
    frozen_holdout_fills = _simulate_selected(frozen_holdout_selected, settings)
    argmax_holdout_fills = _simulate_selected(argmax_holdout_selected, settings)

    coverage_summary = _coverage_summary(settings, combined_predictions, mappings, mapping_audit, candidate_rows, frozen_holdout_selected)
    if backfill_summary:
        coverage_summary["backfill_summary"] = backfill_summary
        for key, value in backfill_summary.items():
            if key not in coverage_summary:
                coverage_summary[key] = value

    selection_dev_summary = {
        "model_variant": variant,
        "probability_source": probability_source,
        "scope_name": champion_policy.scope_name,
        "model_pure": _model_metrics(dev_candidates, probability_source),
        "argmax_all_bets": _summarize_track(argmax_dev_selected, argmax_dev_fills, settings),
        "current_policy_frozen": _summarize_track(frozen_dev_selected, frozen_dev_fills, settings),
        "candidate_policy": _summarize_track(frozen_dev_selected, frozen_dev_fills, settings),
        "policy_reoptimized": False,
    }
    selection_dev_summary.update(_pre_holdout_metrics(pre_holdout_window_report))
    selection_dev_summary["pre_holdout_window_report"] = pre_holdout_window_report
    locked_holdout_summary = {
        "model_variant": variant,
        "probability_source": probability_source,
        "scope_name": champion_policy.scope_name,
        "model_pure": _model_metrics(holdout_candidates, probability_source),
        "argmax_all_bets": _summarize_track(argmax_holdout_selected, argmax_holdout_fills, settings),
        "current_policy_frozen": _summarize_track(frozen_holdout_selected, frozen_holdout_fills, settings),
        "candidate_policy": _summarize_track(frozen_holdout_selected, frozen_holdout_fills, settings),
        "policy_reoptimized": False,
    }
    holdout_frozen = locked_holdout_summary["current_policy_frozen"]
    cohort_report = _cohort_report(
        frozen_dev_selected,
        frozen_dev_fills,
        frozen_holdout_selected,
        frozen_holdout_fills,
        settings,
    )
    combined_selected = pd.concat(
        [frame for frame in (frozen_oof_selected, frozen_dev_selected, frozen_holdout_selected) if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in (frozen_oof_selected, frozen_dev_selected, frozen_holdout_selected)) else pd.DataFrame()
    combined_fills = pd.concat(
        [frame for frame in (frozen_oof_fills, frozen_dev_fills, frozen_holdout_fills) if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in (frozen_oof_fills, frozen_dev_fills, frozen_holdout_fills)) else pd.DataFrame()
    confidence_scores = pd.to_numeric(combined_predictions.get("confidence_score"), errors="coerce").dropna().to_numpy(dtype=float)
    confidence_score_summary = _confidence_score_summary(confidence_scores, applied=bool(np.any(confidence_scores < 0.999999)))
    confidence_backoff_report = _confidence_backoff_report(combined_predictions, combined_selected, combined_fills, settings)
    decision_region_report = {
        RETRO_SEGMENT_DEV: _decision_region_report(
            frozen_dev_selected,
            frozen_dev_fills,
            label=RETRO_SEGMENT_DEV,
            combined_predictions=combined_predictions,
        ),
        RETRO_SEGMENT_HOLDOUT: _decision_region_report(
            frozen_holdout_selected,
            frozen_holdout_fills,
            label=RETRO_SEGMENT_HOLDOUT,
            combined_predictions=combined_predictions,
        ),
        "combined": _decision_region_report(
            combined_selected,
            combined_fills,
            label="combined",
            combined_predictions=combined_predictions,
        ),
    }
    gate_before_after = (
        decision_scorer_diagnostics.get("stability_gate", {})
        if isinstance(decision_scorer_diagnostics.get("stability_gate", {}), dict)
        else {}
    ).get("before_after", {})
    if gate_before_after:
        decision_region_report["stability_gate_comparison"] = gate_before_after
    feature_usage_rows: list[dict[str, Any]] = []
    weight_summaries: list[dict[str, Any]] = []
    train_mix_summaries: list[dict[str, Any]] = []
    for fold_row in fold_rows:
        metadata = dict(fold_row.get("metadata", {}))
        usage = metadata.get("feature_usage") or {}
        feature_usage_rows.append(
            {
                "variant": variant,
                "fold": f"discovery_fold_{int(fold_row['fold_id'])}",
                "declared_feature_columns": usage.get("declared_feature_columns", []),
                "used_feature_columns": usage.get("used_feature_columns", []),
                "pruned_feature_columns": usage.get("pruned_feature_columns", {}),
                "fallback_feature_columns": usage.get("fallback_feature_columns", []),
                "fallback_pruned_feature_columns": usage.get("fallback_pruned_feature_columns", {}),
                "training_input_columns": usage.get("training_input_columns", []),
                "external_model_adapter": usage.get("external_model_adapter"),
            }
        )
        if metadata.get("train_weight_summary"):
            weight_summaries.append(dict(metadata["train_weight_summary"]))
        if metadata.get("train_league_mix_summary"):
            train_mix_summaries.append(dict(metadata["train_league_mix_summary"]))
    for fold_name, metadata in ((RETRO_SEGMENT_DEV, dev_metadata), (RETRO_SEGMENT_HOLDOUT, holdout_metadata)):
        usage = dict(metadata.get("feature_usage", {}))
        feature_usage_rows.append(
            {
                "variant": variant,
                "fold": fold_name,
                "declared_feature_columns": usage.get("declared_feature_columns", []),
                "used_feature_columns": usage.get("used_feature_columns", []),
                "pruned_feature_columns": usage.get("pruned_feature_columns", {}),
                "fallback_feature_columns": usage.get("fallback_feature_columns", []),
                "fallback_pruned_feature_columns": usage.get("fallback_pruned_feature_columns", {}),
                "training_input_columns": usage.get("training_input_columns", []),
                "external_model_adapter": usage.get("external_model_adapter"),
            }
        )
        if metadata.get("train_weight_summary"):
            weight_summaries.append(dict(metadata["train_weight_summary"]))
        if metadata.get("train_league_mix_summary"):
            train_mix_summaries.append(dict(metadata["train_league_mix_summary"]))
    weight_diagnostics = _aggregate_weight_summaries(weight_summaries)
    effective_train_mix = _aggregate_train_mix_summaries(train_mix_summaries)
    promotion_decision = {
        "status": MODEL_STATUS_NOT_PROMOTED,
        "promotion_status": RETRO_PROMOTION_REJECTED,
        "coverage_status": coverage_summary["coverage_status"],
        "holdout_candidate_roi": holdout_frozen["net_roi"],
        "holdout_frozen_roi": holdout_frozen["net_roi"],
        "holdout_candidate_bets": holdout_frozen["bets"],
        "selection_dev_candidate_roi": selection_dev_summary["pre_holdout_aggregate_roi"],
        "scope_name": champion_policy.scope_name,
        "research_status": RESEARCH_STATUS_COVERAGE_LIMITED,
        "promotion_eligibility": RESEARCH_STATUS_COVERAGE_LIMITED,
        "reason": "pending_model_comparison",
    }

    return {
        "variant": variant,
        "decision_scorer": str(decision_scorer),
        "decision_training_scope": str(decision_training_scope),
        "stability_gate_mode": str(stability_gate_mode),
        "combined_predictions": combined_predictions,
        "oof_predictions": oof_predictions,
        "candidate_rows": candidate_rows,
        "mappings": mappings,
        "coverage_summary": coverage_summary,
        "selection_dev_summary": selection_dev_summary,
        "locked_holdout_summary": locked_holdout_summary,
        "policy_ranking": ranking,
        "baseline_policy": baseline_policy,
        "candidate_policy": champion_policy,
        "probability_source": probability_source,
        "regional_adjustment": str(regional_adjustment),
        "stability_gate_report": _stability_gate_report(decision_scorer_diagnostics, stability_gate_audit),
        "stability_gate_audit": stability_gate_audit_columns(stability_gate_audit),
        "source_report": source_report,
        "promotion_decision": promotion_decision,
        "research_status": RESEARCH_STATUS_COVERAGE_LIMITED,
        "promotion_eligibility": RESEARCH_STATUS_COVERAGE_LIMITED,
        "cohort_report": cohort_report,
        "decision_region_report": decision_region_report,
        "model_diagnostics": _model_diagnostics(
            mapped_dataset,
            {variant: weight_diagnostics} if weight_summaries else {},
            {variant: effective_train_mix} if train_mix_summaries else {},
        ),
        "weight_diagnostics": weight_diagnostics,
        "effective_train_mix": effective_train_mix,
        "oof_frozen_policy_report": oof_frozen_policy_report,
        "pre_holdout_window_report": pre_holdout_window_report,
        "confidence_score_summary": confidence_score_summary,
        "confidence_backoff_report": confidence_backoff_report,
        "decision_scorer_diagnostics": decision_scorer_diagnostics,
        "fold_feature_usage": feature_usage_rows,
        "fold_rows": fold_rows,
        "metadata": {"selection_dev": dev_metadata, "locked_holdout": holdout_metadata},
        "holdout_selected": frozen_holdout_selected,
        "holdout_fills": frozen_holdout_fills,
        "holdout_decisions": _build_decisions(holdout_predictions, mappings, holdout_candidates, frozen_holdout_selected, probability_source),
    }


def _save_retro_experiment_artifacts(
    run_dir: Path,
    experiment: dict[str, Any],
    mapping_audit: pd.DataFrame,
    policy_bundle_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Path], Path | None]:
    summary_path = run_dir / "retro_shadow_summary.json"
    coverage_path = run_dir / "retro_coverage_summary.json"
    selection_dev_path = run_dir / "selection_dev_summary.json"
    holdout_path = run_dir / "locked_holdout_summary.json"
    cohort_path = run_dir / "cohort_report.json"
    pre_holdout_path = run_dir / "pre_holdout_window_report.json"
    oof_policy_path = run_dir / "oof_frozen_policy_report.json"
    ablation_path = run_dir / "ablation_report.json"
    feature_ablation_path = run_dir / "feature_ablation_report.json"
    diagnostics_path = run_dir / "model_diagnostics.json"
    manifest_path = run_dir / "variant_feature_manifest.json"
    feature_usage_path = run_dir / "fold_feature_usage.json"
    confidence_backoff_path = run_dir / "confidence_backoff_report.json"
    decision_region_path = run_dir / "decision_region_report.json"
    decision_scorer_ablation_path = run_dir / "decision_scorer_ablation.json"
    roi_ladder_path = run_dir / "roi_ladder_report.json"
    stability_gate_report_path = run_dir / "stability_gate_report.json"
    stability_gate_audit_path = run_dir / "stability_gate_audit.csv"
    promotion_path = run_dir / "promotion_decision.json"
    oof_path = run_dir / "oof_predictions.csv"
    candidate_path = run_dir / "retro_candidate_rows.csv"
    decision_path = run_dir / "retro_decision_rows.csv"
    fill_path = run_dir / "retro_fill_rows.csv"
    mapping_path = run_dir / "market_mapping.csv"
    audit_path = run_dir / "mapping_audit.csv"
    bank_curve_path = run_dir / "bank_curve.png"
    bundle_path = run_dir / "policy_bundle.json" if policy_bundle_payload is not None else None

    _save_json(summary_path, experiment["summary"])
    _save_json(coverage_path, experiment["coverage_summary"])
    _save_json(selection_dev_path, experiment["selection_dev_summary"])
    _save_json(holdout_path, experiment["locked_holdout_summary"])
    _save_json(cohort_path, experiment["cohort_report"])
    _save_json(pre_holdout_path, experiment["pre_holdout_window_report"])
    _save_json(oof_policy_path, experiment["oof_frozen_policy_report"])
    _save_json(ablation_path, experiment["ablation_report"])
    _save_json(feature_ablation_path, experiment["feature_ablation_report"])
    _save_json(diagnostics_path, experiment["model_diagnostics"])
    _save_json(manifest_path, experiment["variant_feature_manifest"])
    _save_json(feature_usage_path, experiment["fold_feature_usage"])
    _save_json(confidence_backoff_path, experiment["confidence_backoff_report"])
    _save_json(decision_region_path, experiment["decision_region_report"])
    _save_json(decision_scorer_ablation_path, experiment["decision_scorer_ablation"])
    _save_json(roi_ladder_path, experiment["roi_ladder_report"])
    _save_json(stability_gate_report_path, experiment.get("stability_gate_report", {}))
    _save_json(promotion_path, experiment["promotion_decision"])
    experiment["oof_predictions"].to_csv(oof_path, index=False)
    experiment["candidate_rows"].to_csv(candidate_path, index=False)
    experiment["holdout_decisions"].to_csv(decision_path, index=False)
    experiment["holdout_fills"].to_csv(fill_path, index=False)
    experiment["mappings"].to_csv(mapping_path, index=False)
    mapping_audit.to_csv(audit_path, index=False)
    experiment.get("stability_gate_audit", pd.DataFrame()).to_csv(stability_gate_audit_path, index=False)
    pm_shadow._plot_shadow_bank_curve(experiment["holdout_fills"], bank_curve_path)
    if bundle_path is not None:
        _save_json(bundle_path, policy_bundle_payload)
    if not experiment["policy_ranking"].empty:
        experiment["policy_ranking"].to_csv(run_dir / "policy_candidates.csv", index=False)
    artifacts = {
        "retro_shadow_summary": summary_path,
        "retro_coverage_summary": coverage_path,
        "selection_dev_summary": selection_dev_path,
        "locked_holdout_summary": holdout_path,
        "cohort_report": cohort_path,
        "pre_holdout_window_report": pre_holdout_path,
        "oof_frozen_policy_report": oof_policy_path,
        "ablation_report": ablation_path,
        "feature_ablation_report": feature_ablation_path,
        "model_diagnostics": diagnostics_path,
        "variant_feature_manifest": manifest_path,
        "fold_feature_usage": feature_usage_path,
        "confidence_backoff_report": confidence_backoff_path,
        "decision_region_report": decision_region_path,
        "decision_scorer_ablation": decision_scorer_ablation_path,
        "roi_ladder_report": roi_ladder_path,
        "stability_gate_report": stability_gate_report_path,
        "stability_gate_audit": stability_gate_audit_path,
        "promotion_decision": promotion_path,
        "oof_predictions": oof_path,
        "retro_candidate_rows": candidate_path,
        "retro_decision_rows": decision_path,
        "retro_fill_rows": fill_path,
        "market_mapping": mapping_path,
        "mapping_audit": audit_path,
        "bank_curve": bank_curve_path,
    }
    if bundle_path is not None:
        artifacts["policy_bundle"] = bundle_path
    if not experiment["policy_ranking"].empty:
        artifacts["policy_candidates"] = run_dir / "policy_candidates.csv"
    return artifacts, bundle_path


def backtest_polymarket_retro(
    settings: Settings,
    history_matches: pd.DataFrame | None = None,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
    policy_bundle_path: Path | str | None = None,
) -> PolymarketRetroResult:
    (
        database_path,
        payload,
        _matches,
        dataset,
        catalog,
        groups,
        checkpoints,
        price_history,
        mapping_audit,
        backfill_summary,
    ) = _prepare_retro_base_context(settings, history_matches, db_path, model_path)
    baseline_summary = _load_frozen_baseline_summary(settings)
    all_variant_names = (
        "v1",
        "v2",
        "v3",
        "v2r",
        "v3r",
        "v4",
        "v5",
        "v4b",
        "v5b",
        "v6",
        "v6b",
        "v7",
        "v7b",
    )
    full_variant_manifest = variant_feature_manifest(dataset, variants=all_variant_names)
    experiments: dict[str, dict[str, Any]] = {}
    variant_base_contexts: dict[str, dict[str, Any]] = {}
    for config in _primary_experiment_configs():
        variant_key = str(config["variant"])
        if variant_key not in variant_base_contexts:
            variant_base_contexts[variant_key] = _build_variant_base_context(
                settings=settings,
                dataset=dataset,
                mapping_audit=mapping_audit,
                catalog=catalog,
                groups=groups,
                checkpoints=checkpoints,
                price_history=price_history,
                payload=payload,
                variant=variant_key,
            )
        experiment_name = _experiment_name(
            config["variant"],
            config["decision_scorer"],
            config["decision_training_scope"],
            config.get("regional_adjustment", REGIONAL_ADJUSTMENT_NONE),
            config.get("stability_gate_mode", STABILITY_GATE_NONE),
        )
        experiments[experiment_name] = _evaluate_variant_experiment(
            settings=settings,
            dataset=dataset,
            mapping_audit=mapping_audit,
            catalog=catalog,
            groups=groups,
            checkpoints=checkpoints,
            price_history=price_history,
            backfill_summary=backfill_summary,
            payload=payload,
            variant=config["variant"],
            decision_scorer=config["decision_scorer"],
            decision_training_scope=config["decision_training_scope"],
            regional_adjustment=config.get("regional_adjustment", REGIONAL_ADJUSTMENT_NONE),
            stability_gate_mode=config.get("stability_gate_mode", STABILITY_GATE_NONE),
            base_context=variant_base_contexts[variant_key],
        )
    primary_tree_success = any(
        str(experiment.get("variant")) in {"v6", "v6b", "v7", "v7b"}
        and float(experiment["oof_frozen_policy_report"]["oof_aggregate_roi"]) > 0.0
        and float(experiment["selection_dev_summary"]["pre_holdout_aggregate_roi"]) > 0.0
        for experiment in experiments.values()
    )
    if not primary_tree_success:
        for config in _secondary_experiment_configs():
            variant_key = str(config["variant"])
            if variant_key not in variant_base_contexts:
                variant_base_contexts[variant_key] = _build_variant_base_context(
                    settings=settings,
                    dataset=dataset,
                    mapping_audit=mapping_audit,
                    catalog=catalog,
                    groups=groups,
                    checkpoints=checkpoints,
                    price_history=price_history,
                    payload=payload,
                    variant=variant_key,
                )
            experiment_name = _experiment_name(
                config["variant"],
                config["decision_scorer"],
                config["decision_training_scope"],
                config.get("regional_adjustment", REGIONAL_ADJUSTMENT_NONE),
                config.get("stability_gate_mode", STABILITY_GATE_NONE),
            )
            experiments[experiment_name] = _evaluate_variant_experiment(
                settings=settings,
                dataset=dataset,
                mapping_audit=mapping_audit,
                catalog=catalog,
                groups=groups,
                checkpoints=checkpoints,
                price_history=price_history,
                backfill_summary=backfill_summary,
                payload=payload,
                variant=config["variant"],
                decision_scorer=config["decision_scorer"],
                decision_training_scope=config["decision_training_scope"],
                regional_adjustment=config.get("regional_adjustment", REGIONAL_ADJUSTMENT_NONE),
                stability_gate_mode=config.get("stability_gate_mode", STABILITY_GATE_NONE),
                base_context=variant_base_contexts[variant_key],
            )
    v1_oof_log_loss = float(experiments["v1"]["source_report"]["raw"]["oof_model_metrics"]["log_loss"])
    weight_diagnostics_by_variant = {
        variant: experiment.get("weight_diagnostics", {})
        for variant, experiment in experiments.items()
        if int(experiment.get("weight_diagnostics", {}).get("weighted_calls", 0)) > 0
    }
    for experiment in experiments.values():
        experiment["model_diagnostics"]["weight_diagnostics_by_variant"] = weight_diagnostics_by_variant
    ablations: list[dict[str, Any]] = []
    champion_variant: str | None = None
    champion_tuple: tuple[float, ...] | None = None
    for experiment_name, experiment in experiments.items():
        variant = str(experiment["variant"])
        probability_source = str(experiment["probability_source"])
        chosen_oof_metrics = experiment["source_report"][probability_source]["oof_model_metrics"]
        oof_validation_status = _oof_validation_status(experiment["oof_frozen_policy_report"])
        pre_holdout_validation_status = _pre_holdout_validation_status(experiment["selection_dev_summary"])
        status, promotion_eligibility, deltas = _model_status(
            experiment["coverage_summary"]["coverage_status"],
            float(chosen_oof_metrics.get("log_loss", 0.0)),
            v1_oof_log_loss,
            experiment["oof_frozen_policy_report"],
            experiment["selection_dev_summary"],
            experiment["locked_holdout_summary"],
            baseline_summary,
        )
        experiment["status"] = status
        experiment["research_status"] = status
        experiment["promotion_eligibility"] = promotion_eligibility
        experiment["oof_validation_status"] = oof_validation_status
        experiment["pre_holdout_validation_status"] = pre_holdout_validation_status
        experiment["promotion_decision"] = {
            "status": status,
            "promotion_status": RETRO_PROMOTION_READY if promotion_eligibility == RESEARCH_STATUS_PROMOTABLE else RETRO_PROMOTION_REJECTED,
            "coverage_status": experiment["coverage_summary"]["coverage_status"],
            "holdout_candidate_roi": float(experiment["locked_holdout_summary"]["current_policy_frozen"]["net_roi"]),
            "holdout_frozen_roi": float(experiment["locked_holdout_summary"]["current_policy_frozen"]["net_roi"]),
            "holdout_candidate_bets": int(experiment["locked_holdout_summary"]["current_policy_frozen"]["bets"]),
            "oof_candidate_roi": float(experiment["oof_frozen_policy_report"]["oof_aggregate_roi"]),
            "oof_median_roi": float(experiment["oof_frozen_policy_report"]["oof_median_roi"]),
            "selection_dev_candidate_roi": float(experiment["selection_dev_summary"]["pre_holdout_aggregate_roi"]),
            "pre_holdout_median_roi": float(experiment["selection_dev_summary"]["pre_holdout_median_roi"]),
            "scope_name": experiment["candidate_policy"].scope_name,
            "oof_validation_status": oof_validation_status,
            "pre_holdout_validation_status": pre_holdout_validation_status,
            "holdout_sample_status": str(deltas.get("holdout_sample_status", MODEL_STATUS_PRE_HOLDOUT_INSUFFICIENT)),
            "validation_sample_status": str(deltas.get("validation_sample_status", MODEL_STATUS_PRE_HOLDOUT_INSUFFICIENT)),
            "validation_total_bets": int(deltas.get("validation_total_bets", 0)),
            "research_status": status,
            "promotion_eligibility": promotion_eligibility,
            "reason": status,
        }
        experiment["deltas_vs_baseline"] = deltas
        oof_total_bets = int(experiment["oof_frozen_policy_report"].get("oof_total_bets", 0))
        oof_min_fold_bets = int(experiment["oof_frozen_policy_report"].get("oof_min_fold_bets", 0))
        oof_min_validation_unit_bets = int(experiment["oof_frozen_policy_report"].get("oof_min_validation_unit_bets", oof_min_fold_bets))
        pre_holdout_total_bets = int(experiment["selection_dev_summary"].get("pre_holdout_total_bets", 0))
        pre_holdout_min_window_bets = int(experiment["selection_dev_summary"].get("pre_holdout_min_window_bets", 0))
        locked_holdout_bets = int(experiment["locked_holdout_summary"]["current_policy_frozen"].get("bets", 0))
        ranking_sample_ready = oof_total_bets >= 60 and pre_holdout_total_bets >= 24
        ranking_validation_ready = (
            oof_validation_status == MODEL_STATUS_NOT_PROMOTED
            and pre_holdout_validation_status == MODEL_STATUS_NOT_PROMOTED
        )
        experiment["ranking_sample_status"] = "ranking_sample_ready" if ranking_sample_ready else MODEL_STATUS_PRE_HOLDOUT_INSUFFICIENT
        experiment["ranking_validation_ready"] = bool(ranking_validation_ready)
        ranking_tuple = (
            1.0 if ranking_sample_ready else 0.0,
            1.0 if ranking_validation_ready else 0.0,
            float(deltas["oof_log_loss_vs_v1"]),
            float(experiment["oof_frozen_policy_report"]["oof_median_roi"]),
            float(experiment["selection_dev_summary"]["pre_holdout_median_roi"]),
            float(experiment["selection_dev_summary"]["pre_holdout_aggregate_roi"]),
            float(experiment["locked_holdout_summary"]["current_policy_frozen"]["net_roi"]),
            -abs(
                float(experiment["selection_dev_summary"]["pre_holdout_aggregate_roi"])
                - float(experiment["locked_holdout_summary"]["current_policy_frozen"]["net_roi"])
            ),
        )
        if champion_tuple is None or ranking_tuple > champion_tuple:
            champion_tuple = ranking_tuple
            champion_variant = experiment_name
        regional_diagnostics = dict(
            (experiment.get("decision_scorer_diagnostics", {}) or {}).get("regional_adjustment", {}) or {}
        )
        ablations.append(
            {
                "experiment_name": experiment_name,
                "variant": variant,
                "variant_name": variant,
                "decision_scorer": str(experiment.get("decision_scorer", DECISION_SCORER_HEURISTIC)),
                "decision_training_scope": str(experiment.get("decision_training_scope", TRAINING_SCOPE_ELIGIBLE)),
                "regional_adjustment": str(experiment.get("regional_adjustment", REGIONAL_ADJUSTMENT_NONE)),
                "regional_adjustment_training_scope": str(regional_diagnostics.get("training_scope", "")),
                "regional_adjustment_training_rows": int(regional_diagnostics.get("training_rows", 0) or 0),
                "regional_adjustment_source_rows": int(regional_diagnostics.get("source_rows", 0) or 0),
                "stability_gate_mode": str(experiment.get("stability_gate_mode", STABILITY_GATE_NONE)),
                "blocked_candidate_share": float(
                    experiment.get("stability_gate_report", {}).get("summary", {}).get("blocked_candidate_share", 0.0)
                ),
                "blocked_base_policy_pick_share": float(
                    experiment.get("stability_gate_report", {}).get("summary", {}).get("blocked_base_policy_pick_share", 0.0)
                ),
                "feature_family_set": _feature_family_set(variant),
                "scope_name": experiment["candidate_policy"].scope_name,
                "probability_source": probability_source,
                "oof_log_loss": float(chosen_oof_metrics.get("log_loss", 0.0)),
                "oof_brier": float(chosen_oof_metrics.get("brier", 0.0)),
                "oof_aggregate_roi": float(experiment["oof_frozen_policy_report"]["oof_aggregate_roi"]),
                "oof_median_roi": float(experiment["oof_frozen_policy_report"]["oof_median_roi"]),
                "oof_positive_fold_ratio": float(experiment["oof_frozen_policy_report"]["oof_positive_fold_ratio"]),
                "oof_total_bets": oof_total_bets,
                "oof_min_fold_bets": oof_min_fold_bets,
                "oof_min_validation_unit_bets": oof_min_validation_unit_bets,
                "oof_raw_min_fold_bets": int(experiment["oof_frozen_policy_report"].get("oof_raw_min_fold_bets", oof_min_fold_bets)),
                "pre_holdout_aggregate_roi": float(experiment["selection_dev_summary"]["pre_holdout_aggregate_roi"]),
                "pre_holdout_median_roi": float(experiment["selection_dev_summary"]["pre_holdout_median_roi"]),
                "pre_holdout_positive_window_ratio": float(experiment["selection_dev_summary"]["pre_holdout_positive_window_ratio"]),
                "pre_holdout_total_bets": pre_holdout_total_bets,
                "pre_holdout_min_window_bets": pre_holdout_min_window_bets,
                "locked_holdout_frozen_policy_roi": float(experiment["locked_holdout_summary"]["current_policy_frozen"]["net_roi"]),
                "locked_holdout_bets": locked_holdout_bets,
                "holdout_sample_status": str(deltas.get("holdout_sample_status", MODEL_STATUS_PRE_HOLDOUT_INSUFFICIENT)),
                "validation_sample_status": str(deltas.get("validation_sample_status", MODEL_STATUS_PRE_HOLDOUT_INSUFFICIENT)),
                "ranking_sample_status": str(experiment["ranking_sample_status"]),
                "ranking_validation_ready": bool(experiment["ranking_validation_ready"]),
                "validation_total_bets": int(deltas.get("validation_total_bets", 0)),
                "oof_validation_status": oof_validation_status,
                "pre_holdout_validation_status": pre_holdout_validation_status,
                "delta_vs_baseline_pre_holdout": float(deltas["pre_holdout_delta_vs_baseline"]),
                "delta_vs_baseline_holdout": float(deltas["locked_holdout_delta_vs_baseline"]),
                "selection_dev_decision_region": experiment["decision_region_report"].get(RETRO_SEGMENT_DEV, {}),
                "locked_holdout_decision_region": experiment["decision_region_report"].get(RETRO_SEGMENT_HOLDOUT, {}),
                "status": status,
                "promotion_eligibility": promotion_eligibility,
                "relative_log_loss_improvement": float(deltas["oof_log_loss_vs_v1"]),
            }
        )
    assert champion_variant is not None
    champion = experiments[champion_variant]
    if policy_bundle_path:
        bundle = pm_shadow._load_policy_bundle(policy_bundle_path)
        champion["candidate_policy"] = pm_shadow._policy_from_raw(bundle.get("policy", {}), settings)

    coverage_summary = champion["coverage_summary"]
    coverage_summary["backfill_summary"] = backfill_summary
    if backfill_summary:
        coverage_summary["discovered_events"] = int(backfill_summary.get("events_discovered", coverage_summary.get("discovered_events", 0)))
        coverage_summary["complete_groups"] = int(backfill_summary.get("complete_groups", coverage_summary.get("complete_groups", 0)))
    locked_holdout_summary = champion["locked_holdout_summary"]
    selection_dev_summary = champion["selection_dev_summary"]
    price_provenance_counts = (
        champion["candidate_rows"]["quality_tier"].astype(str).map(pm_shadow._price_provenance_from_quality_tier).value_counts().to_dict()
        if not champion["candidate_rows"].empty and "quality_tier" in champion["candidate_rows"].columns
        else {}
    )
    history_proxy_used = (
        str(settings.polymarket.historical_tuning_quality_min).strip() == "history_proxy"
        or int(price_provenance_counts.get(pm_shadow.PRICE_PROVENANCE_PROXY, 0) or 0) > 0
        or (
            not champion["candidate_rows"].empty
            and "quality_tier" in champion["candidate_rows"].columns
            and champion["candidate_rows"]["quality_tier"].astype(str).eq("history_proxy").any()
        )
    )
    bundle_status = (
        pm_shadow.BUNDLE_STATUS_PROMOTABLE
        if champion["promotion_decision"]["promotion_status"] == RETRO_PROMOTION_READY and not history_proxy_used
        else pm_shadow.BUNDLE_STATUS_PROVISIONAL
    )
    summary = {
        "source_mode": pm_shadow.SOURCE_MODE_RETRO,
        "model_variant": champion["variant"],
        "decision_scorer": str(champion.get("decision_scorer", DECISION_SCORER_HEURISTIC)),
        "decision_training_scope": str(champion.get("decision_training_scope", TRAINING_SCOPE_ELIGIBLE)),
        "regional_adjustment": str(champion.get("regional_adjustment", REGIONAL_ADJUSTMENT_NONE)),
        "stability_gate_mode": str(champion.get("stability_gate_mode", STABILITY_GATE_NONE)),
        "probability_source": champion["probability_source"],
        "net_roi": float(locked_holdout_summary["current_policy_frozen"]["net_roi"]),
        "net_pnl": float(locked_holdout_summary["current_policy_frozen"]["net_pnl"]),
        "settled_bets": int(locked_holdout_summary["current_policy_frozen"]["bets"]),
        "approximate_only": True,
        "coverage_status": coverage_summary["coverage_status"],
        "bundle_status": bundle_status,
        "insufficient_sample": bool(coverage_summary.get("insufficient_sample", False)),
        "scope_name": champion["candidate_policy"].scope_name,
        "research_status": champion["status"],
        "promotion_eligibility": champion["promotion_eligibility"],
        "diagnostic_only": True,
        "policy_written": False,
        "history_proxy_used": bool(history_proxy_used),
        "minimum_quality_tier": settings.polymarket.historical_tuning_quality_min,
        "coverage_summary": coverage_summary,
        "selection_dev_summary": selection_dev_summary,
        "locked_holdout_summary": locked_holdout_summary,
        "oof_frozen_policy_report": champion["oof_frozen_policy_report"],
        "pre_holdout_window_report": champion["pre_holdout_window_report"],
        "oof_validation_status": champion["oof_validation_status"],
        "pre_holdout_validation_status": champion["pre_holdout_validation_status"],
        "cohort_report": champion["cohort_report"],
        "decision_region_report": champion["decision_region_report"],
        "ablation_report": ablations,
        "feature_ablation_report": _feature_ablation_report(experiments),
        "decision_scorer_ablation": _decision_scorer_ablation(experiments),
        "promotion_decision": champion["promotion_decision"],
        "model_metrics": locked_holdout_summary["model_pure"],
        "policy_metrics": locked_holdout_summary["current_policy_frozen"],
        "current_policy_frozen": locked_holdout_summary["current_policy_frozen"],
        "model_diagnostics": champion["model_diagnostics"],
        "variant_feature_manifest": full_variant_manifest,
        "confidence_score_summary": champion["confidence_score_summary"],
        "confidence_backoff_report": champion["confidence_backoff_report"],
        "decision_scorer_diagnostics": champion["decision_scorer_diagnostics"],
        "stability_gate_report": champion["stability_gate_report"],
        "roi_ladder_report": _roi_ladder_report(champion),
        "policy_reoptimized": False,
    }
    lifecycle = pm_shadow.build_polymarket_lifecycle_summary(
        source_mode=pm_shadow.SOURCE_MODE_RETRO,
        bundle_status=summary["bundle_status"],
        price_provenance_counts=price_provenance_counts,
        validation_stage=pm_shadow.VALIDATION_STAGE_RETRO,
    )
    summary.update(
        {
            "validation_stage": lifecycle["validation_stage"],
            "price_provenance_counts": price_provenance_counts,
            "price_provenance": lifecycle["price_provenance"],
            "bundle_readiness": lifecycle["bundle_readiness"],
            "lifecycle": lifecycle,
        }
    )
    run = create_run_context(settings.paths.runs_dir, "backtest_polymarket_retro")
    if not mapping_audit.empty:
        mapping_audit = mapping_audit.copy()
        mapping_audit["run_id"] = run.run_id
    connection = pm_shadow.init_polymarket_db(database_path)
    _persist_mapping_audit(connection, run.run_id, mapping_audit)
    connection.close()
    policy_bundle = {
        "source_mode": pm_shadow.SOURCE_MODE_RETRO,
        "probability_source": champion["probability_source"],
        "model_variant": champion["variant"],
        "variant_name": champion["variant"],
        "decision_scorer": str(champion.get("decision_scorer", DECISION_SCORER_HEURISTIC)),
        "decision_training_scope": str(champion.get("decision_training_scope", TRAINING_SCOPE_ELIGIBLE)),
        "regional_adjustment": str(champion.get("regional_adjustment", REGIONAL_ADJUSTMENT_NONE)),
        "stability_gate_mode": str(champion.get("stability_gate_mode", STABILITY_GATE_NONE)),
        "scope_name": champion["candidate_policy"].scope_name,
        "research_status": champion["status"],
        "promotion_eligibility": champion["promotion_eligibility"],
        "variant_feature_families": _feature_family_set(champion["variant"]),
        "policy": champion["baseline_policy"].to_dict(),
        "coverage_status": coverage_summary["coverage_status"],
        "bundle_status": summary["bundle_status"],
        "minimum_quality_tier": settings.polymarket.historical_tuning_quality_min,
        "diagnostic_only": True,
        "policy_written": False,
        "history_proxy_used": bool(history_proxy_used),
        "promotion_decision": champion["promotion_decision"],
        "oof_validation_status": champion["oof_validation_status"],
        "pre_holdout_validation_status": champion["pre_holdout_validation_status"],
        "holdout_sample_status": champion["promotion_decision"].get("holdout_sample_status"),
        "validation_sample_status": champion["promotion_decision"].get("validation_sample_status"),
        "confidence_score_summary": champion["confidence_score_summary"],
        "decision_region_report": champion["decision_region_report"],
        "decision_scorer_diagnostics": champion["decision_scorer_diagnostics"],
        "stability_gate_report": champion["stability_gate_report"],
        "stability_gate_validation_status": champion.get("status", MODEL_STATUS_NOT_PROMOTED),
        "policy_reoptimized": False,
    }
    champion["summary"] = summary
    champion["ablation_report"] = ablations
    champion["feature_ablation_report"] = summary["feature_ablation_report"]
    champion["decision_scorer_ablation"] = summary["decision_scorer_ablation"]
    champion["roi_ladder_report"] = summary["roi_ladder_report"]
    champion["variant_feature_manifest"] = full_variant_manifest
    artifacts, bundle_path = _save_retro_experiment_artifacts(run.run_dir, champion, mapping_audit, policy_bundle_payload=policy_bundle)
    return PolymarketRetroResult(
        run=run,
        database_path=database_path,
        candidates=champion["candidate_rows"],
        decisions=champion["holdout_decisions"],
        fills=champion["holdout_fills"],
        mappings=champion["mappings"],
        mapping_audit=mapping_audit,
        summary=summary,
        coverage_summary=coverage_summary,
        artifacts=artifacts,
        policy_bundle_path=bundle_path,
    )


def tune_polymarket_policy(
    settings: Settings,
    history_matches: pd.DataFrame | None = None,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
) -> PolymarketRetroResult:
    return backtest_polymarket_retro(
        settings=settings,
        history_matches=history_matches,
        db_path=db_path,
        model_path=model_path,
        policy_bundle_path=None,
    )


def report_polymarket_retro(run_dir: Path | str) -> tuple[dict[str, Any], str]:
    root = Path(run_dir)
    summary = json.loads((root / "retro_shadow_summary.json").read_text(encoding="utf-8"))
    coverage_path = root / "retro_coverage_summary.json"
    coverage = (
        json.loads(coverage_path.read_text(encoding="utf-8"))
        if coverage_path.exists()
        else summary.get("coverage_summary", {})
    )
    lifecycle = summary.get("lifecycle", {}) or {
        "validation_stage": summary.get("validation_stage", pm_shadow.VALIDATION_STAGE_RETRO),
        "price_provenance": summary.get("price_provenance", "unknown"),
        "bundle_readiness": summary.get("bundle_readiness", summary.get("bundle_status", pm_shadow.BUNDLE_STATUS_PROVISIONAL)),
    }
    clv_summary = summary.get("clv_summary", {}) or {}
    fill_adjusted_ev_summary = summary.get("fill_adjusted_ev_summary", {}) or {}
    lines = [
        f"- Source mode: {summary.get('source_mode', pm_shadow.SOURCE_MODE_RETRO)}",
        f"- Lifecycle: {pm_shadow.format_polymarket_lifecycle_label(lifecycle)}",
        f"- Validation stage: {lifecycle.get('validation_stage', summary.get('validation_stage', pm_shadow.VALIDATION_STAGE_RETRO))}",
        f"- Price provenance: {lifecycle.get('price_provenance', summary.get('price_provenance', 'unknown'))}",
        f"- Bundle readiness: {lifecycle.get('bundle_readiness', summary.get('bundle_readiness', summary.get('bundle_status', pm_shadow.BUNDLE_STATUS_PROVISIONAL)))}",
        f"- Model variant: {summary.get('model_variant', 'v1')}",
        f"- Scope name: {summary.get('scope_name', 'global_all')}",
        f"- Probability source: {summary.get('probability_source', 'raw')}",
        f"- Coverage status: {coverage.get('coverage_status', pm_shadow.COVERAGE_STATUS_LIMITED)}",
        f"- Bundle status: {coverage.get('bundle_status', pm_shadow.BUNDLE_STATUS_PROVISIONAL)}",
        f"- Research status: {summary.get('research_status', RESEARCH_STATUS_COVERAGE_LIMITED)}",
        f"- Promotion eligibility: {summary.get('promotion_eligibility', RESEARCH_STATUS_COVERAGE_LIMITED)}",
        f"- Mapped matches: {coverage.get('mapped_matches', 0)}",
        f"- Holdout candidate bets: {summary.get('policy_metrics', {}).get('bets', 0)}",
        f"- Holdout model argmax hit rate: {summary.get('model_metrics', {}).get('argmax_hit_rate', 0.0):.4f}",
        f"- Holdout policy hit rate: {summary.get('policy_metrics', {}).get('hit_rate', 0.0):.4f}",
        f"- Holdout net ROI: {summary.get('net_roi', 0.0):.4f}",
        f"- Holdout net PnL: {summary.get('net_pnl', 0.0):.4f}",
        f"- Fill-adjusted EV mean: {fill_adjusted_ev_summary.get('mean', 0.0):.4f}",
        f"- Mean CLV: {clv_summary.get('mean_clv', 0.0):.4f}",
        f"- CLV coverage: {clv_summary.get('coverage', 0.0):.4f}",
        f"- Promotion status: {summary.get('promotion_decision', {}).get('status', RETRO_PROMOTION_REJECTED)}",
        f"- Approximate only: {summary.get('approximate_only', True)}",
    ]
    return summary, "\n".join(lines)
