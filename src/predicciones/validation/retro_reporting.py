from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Iterable, Mapping
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from ..backtest import _build_prediction_frame
from ..config import Settings
from ..contracts import (
    OUTCOME_AWAY,
    OUTCOME_DRAW,
    OUTCOME_HOME,
    OUTCOME_ORDER,
    OUTCOME_TO_TARGET,
    PolymarketCoverageAuditResult,
    PolymarketHistoryBackfillResult,
    PolymarketRetroResult,
)
from ..data_sources import LEAGUE_NAMES, PolymarketClobClient, PolymarketGammaClient
from ..football.dataset import (
    build_feature_rows,
    build_fixture_feature_rows,
    model_feature_columns_for_variant,
    variant_feature_families,
    variant_feature_manifest,
)
from ..execution_quality import (
    attach_fill_adjusted_ev,
    build_clv_rows,
    fit_fill_probability_priors,
    summarize_clv,
    summarize_fill_adjusted_ev,
    summarize_sizing,
)
from ..ingestion import build_market_odds, canonicalize_matches, normalize_team_name
from ..models import (
    OutcomeCalibrator,
    build_goal_model,
    fit_dixon_coles_rho,
    fit_goal_model,
    multiclass_brier_score,
    outcome_probabilities_from_lambdas,
)
from ..reporting import _save_json, create_run_context
from ..research import select_candidate_bets
from ..strategy import BetPolicy, select_candidate_rows
from ..markets import shadow as pm_shadow
from ..decision_region_diagnostics import (
    _collapse_selected_bet_rows,
    build_decision_region_diagnostics as _build_decision_region_diagnostics,
    format_decision_region_summary as _format_decision_region_summary,
)
from ..selection_scoring import conservative_score_breakdown


from .retro_parsing import *  # noqa: F401,F403

SELECTED_BET_DIAGNOSTICS_FILENAME = "selected_bet_diagnostics.json"
SELECTED_BET_DIAGNOSTICS_TEXT_FILENAME = "selected_bet_diagnostics.txt"


def _is_decision_region_summary(payload: Any) -> bool:
    return isinstance(payload, Mapping) and any(
        key in payload
        for key in (
            "overview",
            "by_outcome",
            "by_odds_band",
            "by_fold",
            "by_window",
            "by_confidence_decile",
        )
    )


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonify_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_selected_bet_diagnostics(
    selected_bets: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    include_summary_text: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    if _is_decision_region_summary(selected_bets):
        payload = dict(selected_bets)
    elif isinstance(selected_bets, Mapping):
        payload = _build_decision_region_diagnostics([selected_bets], **kwargs)
    else:
        payload = _build_decision_region_diagnostics(selected_bets, **kwargs)
    payload = _jsonify_value(payload)
    payload.setdefault("artifact_type", "selected_bet_diagnostics")
    if include_summary_text:
        payload.setdefault("summary_text", _format_decision_region_summary(payload))
    return payload


def format_selected_bet_diagnostics(
    selected_bets_diagnostics: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any],
    **kwargs: Any,
) -> str:
    if isinstance(selected_bets_diagnostics, Mapping) and str(selected_bets_diagnostics.get("summary_text", "")).strip():
        return str(selected_bets_diagnostics["summary_text"])
    if _is_decision_region_summary(selected_bets_diagnostics):
        return _format_decision_region_summary(_jsonify_value(dict(selected_bets_diagnostics)))
    if isinstance(selected_bets_diagnostics, Mapping):
        return _format_decision_region_summary(
            build_selected_bet_diagnostics([selected_bets_diagnostics], include_summary_text=False, **kwargs)
        )
    return _format_decision_region_summary(build_selected_bet_diagnostics(selected_bets_diagnostics, include_summary_text=False, **kwargs))


def save_selected_bet_diagnostics(
    run_dir: Path,
    selected_bets: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    json_filename: str = SELECTED_BET_DIAGNOSTICS_FILENAME,
    text_filename: str = SELECTED_BET_DIAGNOSTICS_TEXT_FILENAME,
    include_text_summary: bool = True,
    **kwargs: Any,
) -> tuple[Path, Path | None]:
    payload = build_selected_bet_diagnostics(selected_bets, include_summary_text=include_text_summary, **kwargs)
    json_path = run_dir / json_filename
    _save_json(json_path, payload)
    text_path: Path | None = None
    if include_text_summary and str(payload.get("summary_text", "")).strip():
        text_path = run_dir / text_filename
        text_path.write_text(f"{payload['summary_text']}\n", encoding="utf-8")
    return json_path, text_path


def _coverage_audit_summary(
    raw_events: pd.DataFrame,
    raw_markets: pd.DataFrame,
    candidates: pd.DataFrame,
    groups_v2: pd.DataFrame,
    audit: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "raw_event_rows": int(len(raw_events)),
        "raw_market_rows": int(len(raw_markets)),
        "football_market_rows": int(len(raw_markets)),
        "match_market_rows": int(candidates["parse_method"].astype(str).ne("unparsed").sum()) if not candidates.empty else 0,
        "true_1x2_legs": int(candidates["market_shape"].astype(str).isin(TRUE_1X2_SHAPES).sum()) if not candidates.empty else 0,
        "true_1x2_complete_groups": int(groups_v2["group_status"].astype(str).eq("complete_group").sum()) if not groups_v2.empty else 0,
        "binary_match_markets": int(audit["classification_status"].astype(str).eq("non_1x2").sum()) if not audit.empty else 0,
        "partial_true_1x2_markets": int(audit["classification_status"].astype(str).eq("partial_1x2").sum()) if not audit.empty else 0,
        "duplicate_markets": int(audit["classification_status"].astype(str).eq("duplicate").sum()) if not audit.empty else 0,
        "ambiguous_markets": int(audit["classification_status"].astype(str).eq("ambiguous").sum()) if not audit.empty else 0,
        "unmatched_markets": int(audit["classification_status"].astype(str).eq("unmatched").sum()) if not audit.empty else 0,
        "out_of_scope_markets": int(audit["classification_status"].astype(str).eq("out_of_scope").sum()) if not audit.empty else 0,
        "counted_true_1x2_legs": int(audit["classification_status"].astype(str).eq("1x2_counted").sum()) if not audit.empty else 0,
    }


def _table_or_empty(connection: Any, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, connection, params=params)
    except Exception:
        return pd.DataFrame()


def _load_checkpoints(settings: Settings, db_path: Path | str | None) -> tuple[Path | None, pd.DataFrame]:
    path = Path(db_path) if db_path else pm_shadow.default_polymarket_db_path(settings)
    if not path.exists():
        return None, pd.DataFrame(columns=["market_id", "timestamp", "asks_json", "asset_id", "event_type"])
    connection = pm_shadow.init_polymarket_db(path)
    checkpoints = pm_shadow._checkpoint_lookup(connection)
    connection.close()
    return path, checkpoints


def _load_price_history(settings: Settings, db_path: Path | str | None) -> pd.DataFrame:
    path = Path(db_path) if db_path else pm_shadow.default_polymarket_db_path(settings)
    if not path.exists():
        return pd.DataFrame(columns=["market_id", "timestamp", "price", "decision_time", "lag_seconds"])
    connection = pm_shadow.init_polymarket_db(path)
    price_history = pm_shadow._price_history_lookup(connection)
    connection.close()
    return price_history


def _history_point_local(price_history: pd.DataFrame, market_id: str, decision_time: pd.Timestamp) -> tuple[float, pd.Timestamp] | None:
    if price_history.empty:
        return None
    rows = price_history[
        price_history["market_id"].astype(str).eq(str(market_id))
        & price_history["timestamp"].notna()
        & price_history["timestamp"].le(decision_time)
    ].copy()
    if rows.empty:
        return None
    rows["lag_seconds"] = (decision_time - rows["timestamp"]).dt.total_seconds()
    row = rows.sort_values(["lag_seconds", "timestamp"], ascending=[True, False]).iloc[0]
    return float(row["price"]), pd.Timestamp(row["timestamp"])


def _history_point_remote(clob: PolymarketClobClient, market_id: str, decision_time: pd.Timestamp) -> tuple[float, pd.Timestamp] | None:
    try:
        payload = clob.get_prices_history(
            market_id=market_id,
            interval="1h",
            start_ts=int((decision_time - pd.to_timedelta(12, unit="h")).timestamp()),
            end_ts=int(decision_time.timestamp()),
        )
    except requests.HTTPError:
        return None
    history = payload.get("history", []) if isinstance(payload, dict) else []
    best: tuple[pd.Timestamp, float] | None = None
    for item in history:
        timestamp = pm_shadow._parse_timestamp(item.get("t") or item.get("timestamp"))
        price = item.get("p") or item.get("price")
        if pd.isna(timestamp) or price is None:
            continue
        price = float(price)
        if not (0.0 < price < 1.0) or timestamp > decision_time:
            continue
        if best is None or timestamp > best[0]:
            best = (timestamp, price)
    return (best[1], best[0]) if best is not None else None


def _retro_price_payload(
    settings: Settings,
    checkpoints: pd.DataFrame,
    price_history: pd.DataFrame,
    clob: PolymarketClobClient,
    market_row: pd.Series,
    decision_time: pd.Timestamp,
    market_probability: float,
) -> dict[str, Any]:
    market_id = str(market_row["market_id"])
    fee_rate = float(market_row.get("fee_rate", 0.0)) if int(market_row.get("fees_enabled", 0)) else 0.0
    checkpoint = pm_shadow._latest_checkpoint_before(checkpoints, market_id=market_id, decision_time=decision_time)
    freshness_limit = max(settings.polymarket.book_freshness_seconds, settings.polymarket.checkpoint_interval_seconds + 5)
    if checkpoint is not None:
        asks = pm_shadow._json_list(checkpoint["asks_json"])
        age_seconds = float((decision_time - checkpoint["timestamp"]).total_seconds())
        if asks and age_seconds <= freshness_limit:
            top_ask = float(sorted(asks, key=lambda item: float(item["price"]))[0]["price"])
            return {
                "quality_tier": "history_exact",
                "quote_status": "eligible"
                if pm_shadow._quality_at_or_above("history_exact", settings.polymarket.historical_quality_min)
                else "rejected_quality",
                "top_ask": top_ask,
                "asks_json": pm_shadow._clean_json(asks),
                "snapshot_time": pm_shadow._iso_timestamp(checkpoint["timestamp"]),
                "snapshot_lag_seconds": age_seconds,
                "source_type": "polymarket_checkpoint",
                "book_ref_json": pm_shadow._clean_json(
                    {
                        "market_id": market_id,
                        "asset_id": str(checkpoint["asset_id"]),
                        "event_type": str(checkpoint["event_type"]),
                    }
                ),
                "fee_rate": fee_rate,
                "proxy_haircut": 0.0,
            }

    local_history = _history_point_local(price_history, market_id=market_id, decision_time=decision_time)
    if local_history is not None:
        price, snapshot_time = local_history
        lag_seconds = float((decision_time - snapshot_time).total_seconds())
        return {
            "quality_tier": "history_exact",
            "quote_status": "eligible"
            if pm_shadow._quality_at_or_above("history_exact", settings.polymarket.historical_quality_min)
            else "rejected_quality",
            "top_ask": float(price),
            "asks_json": pm_shadow._clean_json([{"price": float(price), "size": 1_000_000.0}]),
            "snapshot_time": pm_shadow._iso_timestamp(snapshot_time),
            "snapshot_lag_seconds": lag_seconds,
            "source_type": "polymarket_history_local",
            "book_ref_json": pm_shadow._clean_json({"market_id": market_id, "event_type": "prices_history_local"}),
            "fee_rate": fee_rate,
            "proxy_haircut": 0.0,
        }

    if settings.polymarket.historical_price_mode == "exact_first":
        history = _history_point_remote(clob, market_id=market_id, decision_time=decision_time)
        if history is not None:
            price, snapshot_time = history
            lag_seconds = float((decision_time - snapshot_time).total_seconds())
            return {
                "quality_tier": "history_exact",
                "quote_status": "eligible"
                if pm_shadow._quality_at_or_above("history_exact", settings.polymarket.historical_quality_min)
                else "rejected_quality",
                "top_ask": float(price),
                "asks_json": pm_shadow._clean_json([{"price": float(price), "size": 1_000_000.0}]),
                "snapshot_time": pm_shadow._iso_timestamp(snapshot_time),
                "snapshot_lag_seconds": lag_seconds,
                "source_type": "polymarket_history_remote",
                "book_ref_json": pm_shadow._clean_json({"market_id": market_id, "event_type": "prices_history_remote"}),
                "fee_rate": fee_rate,
                "proxy_haircut": 0.0,
            }

    if pd.notna(market_probability) and 0 < float(market_probability) < 1:
        proxy_price = min(max(float(market_probability) + settings.polymarket.historical_proxy_haircut, 0.01), 0.99)
        return {
            "quality_tier": "history_proxy",
            "quote_status": "eligible"
            if pm_shadow._quality_at_or_above("history_proxy", settings.polymarket.historical_quality_min)
            else "rejected_quality",
            "top_ask": proxy_price,
            "asks_json": pm_shadow._clean_json([{"price": proxy_price, "size": 1_000_000.0}]),
            "snapshot_time": pm_shadow._iso_timestamp(decision_time),
            "snapshot_lag_seconds": 0.0,
            "source_type": "bookmaker_proxy",
            "book_ref_json": pm_shadow._clean_json({"market_id": market_id, "event_type": "bookmaker_proxy"}),
            "fee_rate": fee_rate,
            "proxy_haircut": settings.polymarket.historical_proxy_haircut,
        }

    return {
        "quality_tier": "resolution_only",
        "quote_status": "resolution_only",
        "top_ask": np.nan,
        "asks_json": pm_shadow._clean_json([]),
        "snapshot_time": "",
        "snapshot_lag_seconds": np.nan,
        "source_type": "resolution_only",
        "book_ref_json": pm_shadow._clean_json({"market_id": market_id}),
        "fee_rate": fee_rate,
        "proxy_haircut": 0.0,
    }


def _mapping_audit_rows(
    settings: Settings,
    predictions: pd.DataFrame,
    groups: pd.DataFrame,
    alias_frame: pd.DataFrame,
) -> pd.DataFrame:
    tolerance = pd.to_timedelta(settings.polymarket.league_match_tolerance_minutes, unit="m")
    extended_tolerance = max(tolerance, pd.to_timedelta(12, unit="h"))
    direct_reprogrammed_tolerance = pd.to_timedelta(120, unit="D")
    audit_rows: list[dict[str, Any]] = []
    groups = groups[groups["mapping_status"].astype(str).eq("complete")].copy()

    for row in predictions.itertuples(index=False):
        kickoff = pd.Timestamp(row.kickoff_time)
        kickoff = kickoff.tz_localize("UTC") if kickoff.tzinfo is None else kickoff.tz_convert("UTC")
        league_groups = groups[groups["league_code"].astype(str).eq(str(row.league_code))].copy()
        payload = {
            "audit_id": str(uuid.uuid4()),
            "run_id": "",
            "source_mode": pm_shadow.SOURCE_MODE_RETRO,
            "match_id": str(row.match_id),
            "league_code": str(row.league_code),
            "league_name": str(row.league_name),
            "HomeTeam": str(row.HomeTeam),
            "AwayTeam": str(row.AwayTeam),
            "kickoff_time": pm_shadow._iso_timestamp(kickoff),
            "group_key": "",
            "mapping_status": "missing",
            "mapping_stage": "missing",
            "mapping_score": 0.0,
            "home_score": 0.0,
            "away_score": 0.0,
            "kickoff_delta_minutes": np.nan,
            "candidate_count": 0,
            "selected_event_slug": "",
            "audit_reason": "no_complete_groups_for_league",
            "candidates_json": pm_shadow._clean_json([]),
            "created_at": pm_shadow._iso_timestamp(),
        }
        if league_groups.empty:
            audit_rows.append(payload)
            continue

        direct_rejected_reason = ""
        direct_match_rows = (
            league_groups[league_groups["match_id"].astype(str).eq(str(row.match_id))].copy()
            if "match_id" in league_groups.columns
            else pd.DataFrame()
        )
        if not direct_match_rows.empty:
            direct_match_rows["kickoff_delta_minutes"] = (
                (direct_match_rows["game_start_time"] - kickoff).abs().dt.total_seconds() / 60.0
            )
            direct_match_rows["same_day"] = (
                direct_match_rows["game_start_time"].dt.normalize().eq(kickoff.normalize())
            )
            direct_candidates: list[dict[str, Any]] = []
            for candidate in direct_match_rows.itertuples(index=False):
                home_score = pm_shadow._team_match_score(str(row.HomeTeam), str(candidate.home_team), str(candidate.league_code))
                away_score = pm_shadow._team_match_score(str(row.AwayTeam), str(candidate.away_team), str(candidate.league_code))
                direct_candidates.append(
                    {
                        "group_key": str(candidate.group_key),
                        "event_slug": str(candidate.event_slug),
                        "home_team": str(candidate.home_team),
                        "away_team": str(candidate.away_team),
                        "home_score": home_score,
                        "away_score": away_score,
                        "mapping_score": (home_score + away_score) / 2.0,
                        "kickoff_delta_minutes": float(candidate.kickoff_delta_minutes),
                        "same_day": bool(candidate.same_day),
                        "exact_match": home_score >= 0.999 and away_score >= 0.999,
                    }
                )
            direct_candidates = sorted(
                direct_candidates,
                key=lambda item: (item["exact_match"], item["same_day"], item["mapping_score"], -item["kickoff_delta_minutes"]),
                reverse=True,
            )
            payload["candidate_count"] = len(direct_candidates)
            payload["candidates_json"] = pm_shadow._clean_json(direct_candidates)
            extended_minutes = float(extended_tolerance / pd.Timedelta(minutes=1))
            direct_reprogrammed_minutes = float(direct_reprogrammed_tolerance / pd.Timedelta(minutes=1))
            viable_direct_candidates = [
                item
                for item in direct_candidates
                if item["mapping_score"] >= float(settings.polymarket.mapping_score_threshold)
                and (
                    item["same_day"]
                    or item["kickoff_delta_minutes"] <= extended_minutes
                    or (item["exact_match"] and item["kickoff_delta_minutes"] <= direct_reprogrammed_minutes)
                )
            ]
            if viable_direct_candidates:
                chosen = viable_direct_candidates[0]
                if (
                    len(viable_direct_candidates) > 1
                    and abs(chosen["kickoff_delta_minutes"] - viable_direct_candidates[1]["kickoff_delta_minutes"]) < 1e-6
                ):
                    payload["mapping_status"] = "ambiguous_mapping"
                    payload["mapping_stage"] = "match_id"
                    payload["audit_reason"] = "multiple_match_id_candidates"
                    audit_rows.append(payload)
                    continue
                payload.update(
                    {
                        "group_key": chosen["group_key"],
                        "mapping_status": "complete",
                        "mapping_stage": "match_id",
                        "mapping_score": chosen["mapping_score"],
                        "home_score": chosen["home_score"],
                        "away_score": chosen["away_score"],
                        "kickoff_delta_minutes": chosen["kickoff_delta_minutes"],
                        "selected_event_slug": chosen["event_slug"],
                        "audit_reason": (
                            "direct_match_id"
                            if chosen["same_day"] or chosen["kickoff_delta_minutes"] <= extended_minutes
                            else "direct_match_id_reprogrammed"
                        ),
                    }
                )
                audit_rows.append(payload)
                continue
            direct_rejected_reason = "direct_match_id_rejected_team_or_time_mismatch"

        within_window = league_groups.copy()
        within_window["kickoff_delta_minutes"] = (
            (within_window["game_start_time"] - kickoff).abs().dt.total_seconds() / 60.0
        )
        within_window["same_day"] = (
            within_window["game_start_time"].dt.normalize().eq(kickoff.normalize())
        )
        within_window = within_window[
            within_window["same_day"]
            | within_window["game_start_time"].between(kickoff - extended_tolerance, kickoff + extended_tolerance)
        ]
        if within_window.empty:
            payload["mapping_status"] = "out_of_window"
            payload["mapping_stage"] = "out_of_window"
            payload["audit_reason"] = direct_rejected_reason or "no_groups_within_time_tolerance"
            payload["candidate_count"] = int(len(league_groups))
            audit_rows.append(payload)
            continue

        candidates: list[dict[str, Any]] = []
        for candidate in within_window.itertuples(index=False):
            home_score = pm_shadow._team_match_score(str(row.HomeTeam), str(candidate.home_team), str(candidate.league_code))
            away_score = pm_shadow._team_match_score(str(row.AwayTeam), str(candidate.away_team), str(candidate.league_code))
            exact = home_score >= 0.999 and away_score >= 0.999
            candidates.append(
                {
                    "group_key": str(candidate.group_key),
                    "event_slug": str(candidate.event_slug),
                    "home_team": str(candidate.home_team),
                    "away_team": str(candidate.away_team),
                    "home_score": home_score,
                    "away_score": away_score,
                    "mapping_score": (home_score + away_score) / 2.0,
                    "kickoff_delta_minutes": float(candidate.kickoff_delta_minutes),
                    "same_day": bool(candidate.same_day),
                    "exact_match": exact,
                }
            )

        candidates = sorted(
            candidates,
            key=lambda item: (item["exact_match"], item["same_day"], item["mapping_score"], -item["kickoff_delta_minutes"]),
            reverse=True,
        )
        payload["candidate_count"] = len(candidates)
        payload["candidates_json"] = pm_shadow._clean_json(candidates)

        exact_candidates = sorted(
            [item for item in candidates if item["exact_match"]],
            key=lambda item: (item["same_day"], item["mapping_score"], -item["kickoff_delta_minutes"]),
            reverse=True,
        )
        if exact_candidates:
            chosen = exact_candidates[0]
            if len(exact_candidates) > 1 and abs(chosen["kickoff_delta_minutes"] - exact_candidates[1]["kickoff_delta_minutes"]) < 1e-6:
                payload["mapping_status"] = "ambiguous_mapping"
                payload["mapping_stage"] = "exact"
                payload["audit_reason"] = "multiple_exact_candidates"
                audit_rows.append(payload)
                continue
            payload.update(
                {
                    "group_key": chosen["group_key"],
                    "mapping_status": "complete",
                    "mapping_stage": "exact",
                    "mapping_score": chosen["mapping_score"],
                    "home_score": chosen["home_score"],
                    "away_score": chosen["away_score"],
                    "kickoff_delta_minutes": chosen["kickoff_delta_minutes"],
                    "selected_event_slug": chosen["event_slug"],
                    "audit_reason": "exact_alias_and_time_match",
                }
            )
            audit_rows.append(payload)
            continue

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        if best["mapping_score"] < settings.polymarket.mapping_score_threshold:
            payload["audit_reason"] = "best_fuzzy_score_below_threshold"
            audit_rows.append(payload)
            continue
        if second is not None and (
            second["mapping_score"] >= settings.polymarket.mapping_score_threshold
            and (best["mapping_score"] - second["mapping_score"]) < settings.polymarket.mapping_score_gap
        ):
            payload["mapping_status"] = "ambiguous_mapping"
            payload["mapping_stage"] = "fuzzy"
            payload["audit_reason"] = "fuzzy_score_gap_too_small"
            audit_rows.append(payload)
            continue

        payload.update(
            {
                "group_key": best["group_key"],
                "mapping_status": "complete",
                "mapping_stage": "fuzzy",
                "mapping_score": best["mapping_score"],
                "home_score": best["home_score"],
                "away_score": best["away_score"],
                "kickoff_delta_minutes": best["kickoff_delta_minutes"],
                "selected_event_slug": best["event_slug"],
                "audit_reason": "best_unique_fuzzy_match",
            }
        )
        audit_rows.append(payload)

    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit["kickoff_time"] = pd.to_datetime(audit["kickoff_time"], utc=True, errors="coerce")
        audit["created_at"] = pd.to_datetime(audit["created_at"], utc=True, errors="coerce")
    return audit


def _retro_candidates(
    settings: Settings,
    predictions: pd.DataFrame,
    groups: pd.DataFrame,
    catalog: pd.DataFrame,
    checkpoints: pd.DataFrame,
    price_history: pd.DataFrame,
    mapping_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clob = PolymarketClobClient()
    catalog_lookup = catalog.set_index("market_id").to_dict(orient="index") if not catalog.empty else {}
    audit_lookup = mapping_audit.set_index("match_id").to_dict(orient="index") if not mapping_audit.empty else {}
    rows: list[dict[str, Any]] = []
    mappings = (
        mapping_audit[
            [
                "match_id",
                "league_code",
                "league_name",
                "HomeTeam",
                "AwayTeam",
                "group_key",
                "mapping_status",
                "audit_reason",
            ]
        ].copy()
        if not mapping_audit.empty
        else pd.DataFrame(
            columns=["match_id", "league_code", "league_name", "HomeTeam", "AwayTeam", "group_key", "mapping_status", "audit_reason"]
        )
    )
    mappings["source_mode"] = pm_shadow.SOURCE_MODE_RETRO

    for row in predictions.itertuples(index=False):
        audit = audit_lookup.get(str(row.match_id))
        if not audit or str(audit.get("mapping_status", "")) != "complete":
            continue
        group_key = str(audit["group_key"])
        group = groups[groups["group_key"].astype(str).eq(group_key)]
        if group.empty:
            continue
        group_row = group.iloc[0]
        kickoff_time = pd.Timestamp(group_row["game_start_time"])
        decision_time = kickoff_time - pd.to_timedelta(settings.polymarket.decision_offset_minutes, unit="m")
        for outcome, market_id in {
            OUTCOME_HOME: group_row["home_market_id"],
            OUTCOME_DRAW: group_row["draw_market_id"],
            OUTCOME_AWAY: group_row["away_market_id"],
        }.items():
            market_data = dict(catalog_lookup[str(market_id)])
            market_data["market_id"] = str(market_id)
            market_row = pd.Series(market_data)
            price = _retro_price_payload(
                settings=settings,
                checkpoints=checkpoints,
                price_history=price_history,
                clob=clob,
                market_row=market_row,
                decision_time=decision_time,
                market_probability=float(getattr(row, f"market_prob_{outcome}", np.nan)),
            )
            top_ask = float(price["top_ask"]) if pd.notna(price["top_ask"]) else np.nan
            fee_rate = float(price["fee_rate"])
            fee_cost = fee_rate * top_ask * (1.0 - top_ask) if pd.notna(top_ask) else np.nan
            record = {
                "match_id": str(row.match_id),
                "Date": row.Date,
                "kickoff_time": kickoff_time,
                "decision_time": decision_time,
                "snapshot_time": pd.to_datetime(price["snapshot_time"], utc=True, errors="coerce"),
                "snapshot_lag_seconds": float(price["snapshot_lag_seconds"])
                if pd.notna(price["snapshot_lag_seconds"])
                else np.nan,
                "league_code": str(row.league_code),
                "league_name": str(row.league_name),
                "season": str(row.season),
                "HomeTeam": str(row.HomeTeam),
                "AwayTeam": str(row.AwayTeam),
                "group_key": group_key,
                "market_id": str(market_id),
                "selection": outcome,
                "quality_tier": price["quality_tier"],
                "quote_status": price["quote_status"],
                "source_type": price["source_type"],
                "source_mode": pm_shadow.SOURCE_MODE_RETRO,
                "top_ask": top_ask,
                "quoted_odds": (1.0 / top_ask) if pd.notna(top_ask) and top_ask > 0 else np.nan,
                "fee_rate": fee_rate,
                "proxy_haircut": float(price["proxy_haircut"]),
                "asks_json": price["asks_json"],
                "book_ref_json": price["book_ref_json"],
                "actual_outcome": str(row.actual_outcome),
                "actual_target": int(row.actual_target),
                "mapping_stage": str(audit["mapping_stage"]),
                "mapping_score": float(audit["mapping_score"]),
                "mapping_reason": str(audit["audit_reason"]),
                "model_prob_raw": float(getattr(row, f"prob_{outcome}_raw")),
                "model_prob_calibrated": float(getattr(row, f"prob_{outcome}_calibrated")),
            }
            for source in ("raw", "calibrated"):
                prob = float(record[f"model_prob_{source}"])
                record[f"edge_{source}"] = prob - top_ask if pd.notna(top_ask) else np.nan
                record[f"ev_{source}"] = (
                    prob - (top_ask + fee_cost + settings.polymarket.slippage_cushion)
                    if pd.notna(top_ask)
                    else np.nan
                )
            rows.append(record)
    return pd.DataFrame(rows), mappings


def _selected_rows(
    candidates: pd.DataFrame,
    policy: BetPolicy,
    probability_source: str,
    minimum_quality: str | None = None,
) -> pd.DataFrame:
    working = candidates.copy()
    if minimum_quality:
        working = working[
            working["quality_tier"].astype(str).map(
                lambda value: pm_shadow._quality_at_or_above(str(value), minimum_quality)
            )
        ].copy()
    selected = select_candidate_rows(working, policy, probability_source, 0.0) if not working.empty else pd.DataFrame()
    if not selected.empty:
        selected = selected.copy()
        selected["decision_id"] = [str(uuid.uuid4()) for _ in range(len(selected))]
    return selected


def _simulate_selected(selected: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    working = selected.copy()
    if "decision_id" not in working.columns:
        working["decision_id"] = [str(uuid.uuid4()) for _ in range(len(working))]
    rows: list[dict[str, Any]] = []
    for row in working.itertuples(index=False):
        asks = pm_shadow._json_list(row.asks_json)
        for notional in settings.polymarket.notionals_ladder:
            fill = pm_shadow.simulate_taker_yes_fill(
                asks,
                float(notional),
                float(row.fee_rate),
                settings.polymarket.slippage_cushion,
            )
            won = int(str(row.selection) == str(row.actual_outcome))
            payout = float(fill["shares_filled"]) if won == 1 else 0.0
            rows.append(
                {
                    "fill_id": str(uuid.uuid4()),
                    "decision_id": str(row.decision_id),
                    "run_id": "",
                    "group_key": str(row.group_key),
                    "selection": str(row.selection),
                    "notional": float(notional),
                    "raw_spend": fill["raw_spend"],
                    "shares_filled": fill["shares_filled"],
                    "fill_rate": fill["fill_rate"],
                    "partial_fill": int(bool(fill["partial_fill"])),
                    "top_ask": fill["top_ask"],
                    "raw_vwap": fill["raw_vwap"],
                    "effective_vwap": fill["effective_vwap"],
                    "fee_paid": fill["fee_paid"],
                    "slippage_cost": fill["slippage_cost"],
                    "cost_basis": fill["cost_basis"],
                    "available_ask_size": float(
                        sum(
                            float(item["price"]) * float(item["size"])
                            for item in pm_shadow._json_list(row.asks_json)
                            if item.get("price") is not None and item.get("size") is not None
                        )
                    ),
                    "minutes_to_kickoff": float((pd.Timestamp(row.kickoff_time) - pd.Timestamp(row.decision_time)).total_seconds() / 60.0),
                    "price_provenance": pm_shadow._price_provenance_from_quality_tier(str(row.quality_tier)),
                    "model_prob": float(getattr(row, "selection_prob", np.nan)),
                    "expected_edge": float(getattr(row, "selection_edge", np.nan)),
                    "expected_ev": float(getattr(row, "selection_ev", np.nan)),
                    "closing_reference_odds": np.nan,
                    "closing_reference_prob": np.nan,
                    "clv_source": "",
                    "payout": payout,
                    "net_profit": payout - float(fill["cost_basis"]),
                    "won": won,
                    "resolution_outcome": str(row.actual_outcome),
                    "status": "settled",
                    "levels_used_json": pm_shadow._clean_json(fill["levels_used"]),
                    "created_at": pm_shadow._iso_timestamp(row.decision_time),
                    "league_code": str(row.league_code),
                    "kickoff_time": pm_shadow._iso_timestamp(row.kickoff_time),
                    "quality_tier": str(row.quality_tier),
                    "source_type": str(row.source_type),
                    "retro_fold_id": int(getattr(row, "retro_fold_id", 0)),
                    "retro_segment": str(getattr(row, "retro_segment", "")),
                }
            )
    fills = pd.DataFrame(rows)
    if not fills.empty:
        fills["created_at"] = pd.to_datetime(fills["created_at"], utc=True)
        fills["kickoff_time"] = pd.to_datetime(fills["kickoff_time"], utc=True)
    return fills


def _multiclass_log_loss(probabilities: np.ndarray, targets: np.ndarray, epsilon: float = 1e-12) -> float:
    clipped = np.clip(probabilities, epsilon, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    row_ids = np.arange(len(targets))
    return float(-np.mean(np.log(clipped[row_ids, targets])))


def _multiclass_brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    encoded = np.zeros_like(probabilities)
    encoded[np.arange(len(targets)), targets] = 1.0
    return float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1)))


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    phat = successes / trials
    denominator = 1.0 + (z * z / trials)
    center = (phat + (z * z / (2 * trials))) / denominator
    margin = (z / denominator) * math.sqrt((phat * (1.0 - phat) / trials) + ((z * z) / (4 * trials * trials)))
    return (float(center - margin), float(center + margin))


def _bootstrap_roi_interval(fills: pd.DataFrame, iterations: int = 1000, seed: int = 42) -> tuple[float, float]:
    if fills.empty:
        return (0.0, 0.0)
    profits = fills["net_profit"].to_numpy(dtype=float)
    costs = fills["cost_basis"].to_numpy(dtype=float)
    if profits.size == 0 or float(costs.sum()) <= 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    rois = np.zeros(iterations, dtype=float)
    for index in range(iterations):
        sample_ids = rng.integers(0, profits.size, size=profits.size)
        sample_profit = float(profits[sample_ids].sum())
        sample_cost = float(costs[sample_ids].sum())
        rois[index] = sample_profit / sample_cost if sample_cost > 0 else 0.0
    return (float(np.quantile(rois, 0.025)), float(np.quantile(rois, 0.975)))


def _model_metrics(candidates: pd.DataFrame, probability_source: str) -> dict[str, Any]:
    if candidates.empty:
        return {
            "mapped_matches": 0,
            "argmax_hit_rate": 0.0,
            "log_loss": 0.0,
            "brier": 0.0,
        }

    mapped_rows: list[dict[str, Any]] = []
    probability_lookup = {
        outcome: f"model_prob_{probability_source}"
        for outcome in (OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME)
    }
    for _, group in candidates.groupby("match_id", observed=True):
        payload = {
            "actual_target": int(group["actual_target"].iloc[0]),
            "actual_outcome": str(group["actual_outcome"].iloc[0]),
        }
        for outcome in (OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME):
            value = group[group["selection"].astype(str).eq(outcome)][probability_lookup[outcome]]
            payload[outcome] = float(value.iloc[0]) if not value.empty else 0.0
        mapped_rows.append(payload)
    mapped = pd.DataFrame(mapped_rows)
    probabilities = mapped[[OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME]].to_numpy(dtype=float)
    targets = mapped["actual_target"].to_numpy(dtype=int)
    predictions = probabilities.argmax(axis=1)
    return {
        "mapped_matches": int(len(mapped)),
        "argmax_hit_rate": float((predictions == targets).mean()) if len(mapped) else 0.0,
        "log_loss": _multiclass_log_loss(probabilities, targets) if len(mapped) else 0.0,
        "brier": _multiclass_brier(probabilities, targets) if len(mapped) else 0.0,
    }


def _coverage_summary(
    settings: Settings,
    predictions: pd.DataFrame,
    mappings: pd.DataFrame,
    mapping_audit: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    mapped_matches = int(mapping_audit["mapping_status"].astype(str).eq("complete").sum()) if not mapping_audit.empty else 0
    selected_unique = int(len(selected))
    selected_resolution_only = int(selected["quality_tier"].astype(str).eq("resolution_only").sum()) if not selected.empty else 0
    coverage_status = (
        pm_shadow.COVERAGE_STATUS_READY
        if mapped_matches >= settings.polymarket.retro_min_mapped_matches
        and selected_unique >= settings.polymarket.retro_min_selected_predictions
        and selected_resolution_only == 0
        else pm_shadow.COVERAGE_STATUS_LIMITED
    )
    bundle_status = (
        pm_shadow.BUNDLE_STATUS_PROMOTABLE
        if coverage_status == pm_shadow.COVERAGE_STATUS_READY
        else pm_shadow.BUNDLE_STATUS_PROVISIONAL
    )
    return {
        "source_mode": pm_shadow.SOURCE_MODE_RETRO,
        "total_history_matches": int(len(predictions)),
        "discovered_events": int(mappings["group_key"].astype(str).replace("", np.nan).notna().sum()) if not mappings.empty else 0,
        "complete_groups": int(mappings["group_key"].astype(str).replace("", np.nan).notna().sum()) if not mappings.empty else 0,
        "mapped_matches": mapped_matches,
        "missing_matches": int(mapping_audit["mapping_status"].astype(str).eq("missing").sum()) if not mapping_audit.empty else 0,
        "out_of_window_matches": int(mapping_audit["mapping_status"].astype(str).eq("out_of_window").sum()) if not mapping_audit.empty else 0,
        "ambiguous_matches": int(mapping_audit["mapping_status"].astype(str).eq("ambiguous_mapping").sum()) if not mapping_audit.empty else 0,
        "candidate_rows": int(len(candidates)),
        "exact_candidates": int(candidates["quality_tier"].astype(str).eq("history_exact").sum()) if not candidates.empty else 0,
        "proxy_candidates": int(candidates["quality_tier"].astype(str).eq("history_proxy").sum()) if not candidates.empty else 0,
        "resolution_only_candidates": int(candidates["quality_tier"].astype(str).eq("resolution_only").sum()) if not candidates.empty else 0,
        "eligible_candidates": int(candidates["quote_status"].astype(str).eq("eligible").sum()) if not candidates.empty else 0,
        "selected_unique_predictions": selected_unique,
        "selected_exact_predictions": int(selected["quality_tier"].astype(str).eq("history_exact").sum()) if not selected.empty else 0,
        "selected_proxy_predictions": int(selected["quality_tier"].astype(str).eq("history_proxy").sum()) if not selected.empty else 0,
        "selected_resolution_only_predictions": selected_resolution_only,
        "coverage_status": coverage_status,
        "bundle_status": bundle_status,
        "insufficient_sample": bool(coverage_status != pm_shadow.COVERAGE_STATUS_READY),
        "minimum_mapped_matches": int(settings.polymarket.retro_min_mapped_matches),
        "minimum_selected_predictions": int(settings.polymarket.retro_min_selected_predictions),
        "true_1x2_complete_groups": 0,
        "binary_match_markets": 0,
        "duplicate_markets": 0,
        "ambiguous_markets": 0,
    }


def _policy_metrics(selected: pd.DataFrame, fills: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    if selected.empty:
        return {
            "selected_unique_predictions": 0,
            "hit_rate": 0.0,
            "hit_rate_wilson": [0.0, 0.0],
            "roi_ci": [0.0, 0.0],
        }
    hits = int((selected["selection"].astype(str) == selected["actual_outcome"].astype(str)).sum())
    trials = int(len(selected))
    base_notional = float(settings.polymarket.notionals_ladder[0])
    base_fills = fills[fills["notional"].eq(base_notional)].copy() if not fills.empty else pd.DataFrame()
    return {
        "selected_unique_predictions": trials,
        "hit_rate": float(hits / trials) if trials else 0.0,
        "hit_rate_wilson": list(_wilson_interval(hits, trials)),
        "roi_ci": list(_bootstrap_roi_interval(base_fills)),
    }


def _build_decisions(
    predictions: pd.DataFrame,
    mappings: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    probability_source: str,
) -> pd.DataFrame:
    selected_lookup = {str(row["match_id"]): row for _, row in selected.iterrows()} if not selected.empty else {}
    candidate_lookup = (
        {
            str(match_id): group.sort_values(
                [f"ev_{probability_source}", f"edge_{probability_source}", "quoted_odds"],
                ascending=[False, False, False],
            ).iloc[0]
            for match_id, group in candidates.groupby("match_id", observed=True)
        }
        if not candidates.empty
        else {}
    )
    mapping_lookup = mappings.set_index("match_id").to_dict(orient="index") if not mappings.empty else {}
    rows: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        match_id = str(row.match_id)
        mapping = mapping_lookup.get(match_id, {})
        selected_row = selected_lookup.get(match_id)
        fallback = candidate_lookup.get(match_id)
        decision_time = (
            pd.Timestamp(selected_row["decision_time"])
            if selected_row is not None
            else (pd.Timestamp(fallback["decision_time"]) if fallback is not None else pd.Timestamp(row.kickoff_time))
        )
        payload = {
            "decision_id": str(selected_row["decision_id"]) if selected_row is not None else str(uuid.uuid4()),
            "run_id": "",
            "group_key": str(mapping.get("group_key", "")),
            "match_id": match_id,
            "Date": pm_shadow._iso_timestamp(row.Date),
            "kickoff_time": pm_shadow._iso_timestamp(row.kickoff_time),
            "decision_time": pm_shadow._iso_timestamp(decision_time),
            "league_code": str(row.league_code),
            "league_name": str(row.league_name),
            "HomeTeam": str(row.HomeTeam),
            "AwayTeam": str(row.AwayTeam),
            "mapping_status": str(mapping.get("mapping_status", "missing")),
            "mapping_reason": str(mapping.get("audit_reason", "no_market_group")),
            "selection": "",
            "probability_source": probability_source,
            "model_prob": np.nan,
            "top_ask": np.nan,
            "fee_rate": np.nan,
            "expected_edge": np.nan,
            "expected_ev": np.nan,
            "book_age_seconds": np.nan,
            "snapshot_time": "",
            "skip_reason": "no_market_group",
            "book_ref_json": pm_shadow._clean_json({}),
            "created_at": pm_shadow._iso_timestamp(decision_time),
            "source_mode": pm_shadow.SOURCE_MODE_RETRO,
            "quality_tier": "",
            "quoted_odds": np.nan,
            "source_type": "",
        }
        chosen = selected_row if selected_row is not None else fallback
        if chosen is not None:
            payload.update(
                {
                    "model_prob": float(chosen[f"model_prob_{probability_source}"]),
                    "top_ask": float(chosen["top_ask"]) if pd.notna(chosen["top_ask"]) else np.nan,
                    "fee_rate": float(chosen["fee_rate"]),
                    "expected_edge": float(chosen[f"edge_{probability_source}"])
                    if pd.notna(chosen[f"edge_{probability_source}"])
                    else np.nan,
                    "expected_ev": float(chosen[f"ev_{probability_source}"])
                    if pd.notna(chosen[f"ev_{probability_source}"])
                    else np.nan,
                    "snapshot_time": pm_shadow._iso_timestamp(chosen["snapshot_time"])
                    if pd.notna(chosen["snapshot_time"])
                    else "",
                    "book_ref_json": str(chosen["book_ref_json"]),
                    "quality_tier": str(chosen["quality_tier"]),
                    "quoted_odds": float(chosen["quoted_odds"]) if pd.notna(chosen["quoted_odds"]) else np.nan,
                    "source_type": str(chosen["source_type"]),
                }
            )
        if selected_row is not None:
            payload["selection"] = str(selected_row["selection"])
            payload["skip_reason"] = ""
        elif fallback is not None:
            payload["skip_reason"] = (
                "policy_rejected" if str(fallback["quote_status"]) == "eligible" else str(fallback["quote_status"])
            )
        rows.append(payload)
    return pd.DataFrame(rows)


def _summarize(
    decisions: pd.DataFrame,
    fills: pd.DataFrame,
    mappings: pd.DataFrame,
    candidates: pd.DataFrame,
    policy: BetPolicy,
    probability_source: str,
    coverage_summary: dict[str, Any],
    model_metrics: dict[str, Any],
    policy_metrics: dict[str, Any],
    selected_bet_rows: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = pm_shadow._summarize_shadow(
        decisions,
        fills,
        mappings,
        bundle_status=coverage_summary["bundle_status"],
    )
    settled = fills[fills["status"].eq("settled")].copy() if not fills.empty else pd.DataFrame()
    quality_counts = (
        candidates.groupby("quality_tier", observed=True).size().reset_index(name="count")
        if not candidates.empty
        else pd.DataFrame(columns=["quality_tier", "count"])
    )
    price_provenance_counts = (
        quality_counts.assign(
            price_provenance=quality_counts["quality_tier"].map(pm_shadow._price_provenance_from_quality_tier)
        )
        .groupby("price_provenance", observed=True)["count"]
        .sum()
        .to_dict()
        if not quality_counts.empty
        else {}
    )
    roi_by_quality = (
        settled.groupby("quality_tier", observed=True)
        .agg(bets=("fill_id", "count"), pnl=("net_profit", "sum"), total_cost=("cost_basis", "sum"))
        .reset_index()
        if not settled.empty
        else pd.DataFrame(columns=["quality_tier", "bets", "pnl", "total_cost"])
    )
    if not roi_by_quality.empty:
        roi_by_quality["roi"] = np.where(roi_by_quality["total_cost"] > 0, roi_by_quality["pnl"] / roi_by_quality["total_cost"], 0.0)
    fill_model_summary = fit_fill_probability_priors(fills)
    fill_quality_rows = attach_fill_adjusted_ev(
        fills.copy(),
        probability_column="model_prob",
        edge_column="expected_edge",
        priors=fill_model_summary,
        default_notional=float(fills["notional"].median()) if not fills.empty and "notional" in fills.columns else 10.0,
    )
    if not fill_quality_rows.empty:
        closing_reference_prob = pd.to_numeric(
            pd.Series(fill_quality_rows.get("closing_reference_prob", np.nan), index=fill_quality_rows.index),
            errors="coerce",
        )
        fill_quality_rows["executed_odds"] = np.where(
            pd.to_numeric(fill_quality_rows["effective_vwap"], errors="coerce").gt(0.0),
            1.0 / pd.to_numeric(fill_quality_rows["effective_vwap"], errors="coerce"),
            np.nan,
        )
        fill_quality_rows["closing_reference_odds"] = np.where(
            closing_reference_prob.gt(0.0),
            1.0 / closing_reference_prob,
            np.nan,
        )
    clv_rows = build_clv_rows(
        fill_quality_rows,
        executed_odds_column="executed_odds",
        closing_reference_odds_column="closing_reference_odds",
        executed_prob_column="effective_vwap",
        closing_reference_prob_column="closing_reference_prob",
        source_column="clv_source",
    )
    clv_summary = summarize_clv(clv_rows, total_rows=int(len(fill_quality_rows)))
    fill_adjusted_ev_summary = summarize_fill_adjusted_ev(fill_quality_rows)
    sizing_summary = summarize_sizing(fills, requested_column="notional")

    selected_bet_source = selected_bet_rows if selected_bet_rows is not None else fills
    selected_bet_payload = build_selected_bet_diagnostics(
        _collapse_selected_bet_rows(selected_bet_source) if isinstance(selected_bet_source, pd.DataFrame) else selected_bet_source,
        include_summary_text=True,
    )

    summary.update(
        {
            "source_mode": pm_shadow.SOURCE_MODE_RETRO,
            "validation_stage": pm_shadow.VALIDATION_STAGE_RETRO,
            "price_provenance_counts": price_provenance_counts,
            "price_provenance": pm_shadow._dominant_label(
                price_provenance_counts,
                (
                    pm_shadow.PRICE_PROVENANCE_EXACT,
                    pm_shadow.PRICE_PROVENANCE_PROXY,
                    pm_shadow.PRICE_PROVENANCE_RESOLUTION_ONLY,
                ),
                pm_shadow.PRICE_PROVENANCE_RESOLUTION_ONLY if price_provenance_counts else pm_shadow.PRICE_PROVENANCE_EXACT,
            ),
            "probability_source": probability_source,
            "policy": policy.to_dict(),
            "candidate_rows": int(len(candidates)),
            "eligible_candidates": int(candidates["quote_status"].eq("eligible").sum()) if not candidates.empty else 0,
            "matched_groups": int(mappings["mapping_status"].eq("complete").sum()) if not mappings.empty else 0,
            "quality_counts": quality_counts.to_dict(orient="records"),
            "roi_by_quality": roi_by_quality.to_dict(orient="records"),
            "approximate_only": True,
            "coverage_status": coverage_summary["coverage_status"],
            "bundle_status": coverage_summary["bundle_status"],
            "bundle_readiness": coverage_summary["bundle_status"],
            "insufficient_sample": coverage_summary["insufficient_sample"],
            "coverage_summary": coverage_summary,
            "model_metrics": model_metrics,
            "policy_metrics": policy_metrics,
            "observed_fill_rate": float(pd.to_numeric(fills["fill_rate"], errors="coerce").mean()) if not fills.empty else 0.0,
            "clv_summary": clv_summary,
            "fill_model_summary": fill_model_summary,
            "fill_adjusted_ev_summary": fill_adjusted_ev_summary,
            "sizing_summary": sizing_summary,
            "clv_rows": clv_rows.to_dict(orient="records"),
            "selected_bet_diagnostics": selected_bet_payload,
            "selected_bet_diagnostics_text": str(selected_bet_payload.get("summary_text", "")),
            "lifecycle": {
                **summary.get("lifecycle", {}),
                "source_mode": pm_shadow.SOURCE_MODE_RETRO,
                "validation_stage": pm_shadow.VALIDATION_STAGE_RETRO,
                "price_provenance_counts": price_provenance_counts,
                "price_provenance": pm_shadow._dominant_label(
                    price_provenance_counts,
                    (
                        pm_shadow.PRICE_PROVENANCE_EXACT,
                        pm_shadow.PRICE_PROVENANCE_PROXY,
                        pm_shadow.PRICE_PROVENANCE_RESOLUTION_ONLY,
                    ),
                    pm_shadow.PRICE_PROVENANCE_RESOLUTION_ONLY if price_provenance_counts else pm_shadow.PRICE_PROVENANCE_EXACT,
                ),
                "bundle_status": coverage_summary["bundle_status"],
                "bundle_readiness": coverage_summary["bundle_status"],
                "lifecycle_label": f"{pm_shadow.VALIDATION_STAGE_RETRO} / {pm_shadow._dominant_label(price_provenance_counts, (pm_shadow.PRICE_PROVENANCE_EXACT, pm_shadow.PRICE_PROVENANCE_PROXY, pm_shadow.PRICE_PROVENANCE_RESOLUTION_ONLY), pm_shadow.PRICE_PROVENANCE_RESOLUTION_ONLY if price_provenance_counts else pm_shadow.PRICE_PROVENANCE_EXACT)} / {coverage_summary['bundle_status']}",
            },
        }
    )
    return summary


def _rank_policies(candidates: pd.DataFrame, probability_source: str, settings: Settings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    min_bets = max(settings.backtest.policy_search.min_bets, settings.polymarket.historical_min_markets)
    filtered_candidates = (
        candidates[
            candidates["quality_tier"].astype(str).map(
                lambda value: pm_shadow._quality_at_or_above(value, settings.polymarket.historical_tuning_quality_min)
            )
        ].copy()
        if not candidates.empty
        else candidates
    )
    for edge, ev, min_odds, max_odds in product(
        settings.backtest.policy_search.edge_thresholds,
        settings.backtest.policy_search.ev_thresholds,
        settings.backtest.policy_search.min_odds_options,
        settings.backtest.policy_search.max_odds_options,
    ):
        if min_odds >= max_odds:
            continue
        policy = BetPolicy(
            float(edge),
            float(ev),
            float(min_odds),
            float(max_odds),
            settings.backtest.policy_search.max_kelly_fraction,
        )
        selected = _selected_rows(
            filtered_candidates,
            policy,
            probability_source,
            minimum_quality=settings.polymarket.historical_tuning_quality_min,
        )
        fills = _simulate_selected(selected, settings)
        fills = fills[fills["notional"].eq(float(settings.polymarket.notionals_ladder[0]))].copy() if not fills.empty else fills
        if len(fills) < min_bets:
            continue
        total_cost = float(fills["cost_basis"].sum())
        roi = float(fills["net_profit"].sum() / total_cost) if total_cost > 0 else 0.0
        fold_roi = fills.groupby("retro_fold_id", observed=True).apply(
            lambda group: float(group["net_profit"].sum() / group["cost_basis"].sum()) if float(group["cost_basis"].sum()) > 0 else 0.0
        )
        metrics = {
            "roi": roi,
            "profit": float(fills["net_profit"].sum()),
            "stake": total_cost,
            "max_drawdown": abs(pm_shadow._drawdown_from_profit(fills.sort_values("created_at")["net_profit"])),
            "executed": int(len(fills)),
        }
        breakdown = conservative_score_breakdown(
            metrics,
            fold_roi=fold_roi,
            positive_fold_target=settings.research.positive_fold_ratio,
            prior_bets=min_bets,
            drawdown_weight=settings.research.drawdown_weight,
            generalization_gap_weight=settings.research.generalization_gap_weight,
            positive_penalty_weight=settings.research.positive_penalty_weight,
        )
        rows.append(
            {
                "edge_threshold": float(edge),
                "ev_threshold": float(ev),
                "min_odds": float(min_odds),
                "max_odds": float(max_odds),
                "bets": int(len(fills)),
                "pnl": float(fills["net_profit"].sum()),
                "total_cost": total_cost,
                "roi": roi,
                "drawdown": metrics["max_drawdown"],
                **breakdown,
            }
        )
    ranking = pd.DataFrame(rows)
    return (
        ranking.sort_values(["score", "generalization_gap", "bets"], ascending=[False, True, False]).reset_index(drop=True)
        if not ranking.empty
        else ranking
    )


def _save_artifacts(
    run_dir: Path,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    fills: pd.DataFrame,
    mappings: pd.DataFrame,
    mapping_audit: pd.DataFrame,
    summary: dict[str, Any],
    coverage_summary: dict[str, Any],
    policy_bundle_payload: dict[str, Any] | None = None,
    policy_candidates: pd.DataFrame | None = None,
    selected_bet_diagnostics: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Path], Path | None]:
    summary_path = run_dir / "retro_shadow_summary.json"
    coverage_path = run_dir / "retro_coverage_summary.json"
    candidate_path = run_dir / "retro_candidate_rows.csv"
    decision_path = run_dir / "retro_decision_rows.csv"
    fill_path = run_dir / "retro_fill_rows.csv"
    mapping_path = run_dir / "market_mapping.csv"
    audit_path = run_dir / "mapping_audit.csv"
    skip_path = run_dir / "retro_skip_reasons.csv"
    clv_path = run_dir / "clv_rows.csv"
    execution_quality_path = run_dir / "execution_quality.json"
    bank_curve_path = run_dir / "bank_curve.png"
    bundle_path = run_dir / "policy_bundle.json" if policy_bundle_payload is not None else None
    _save_json(summary_path, summary)
    _save_json(coverage_path, coverage_summary)
    diagnostics_source = selected_bet_diagnostics
    if diagnostics_source is None and isinstance(summary, Mapping):
        diagnostics_source = summary.get("selected_bet_diagnostics")
    if diagnostics_source is not None:
        diagnostics_json_path, diagnostics_text_path = save_selected_bet_diagnostics(run_dir, diagnostics_source)
    else:
        diagnostics_json_path = None
        diagnostics_text_path = None
    candidates.to_csv(candidate_path, index=False)
    decisions.to_csv(decision_path, index=False)
    fills.to_csv(fill_path, index=False)
    mappings.to_csv(mapping_path, index=False)
    mapping_audit.to_csv(audit_path, index=False)
    pd.DataFrame(summary.get("skip_reasons", [])).to_csv(skip_path, index=False)
    pd.DataFrame(summary.get("clv_rows", [])).to_csv(clv_path, index=False)
    _save_json(
        execution_quality_path,
        {
            "clv_summary": summary.get("clv_summary", {}),
            "fill_model_summary": summary.get("fill_model_summary", {}),
            "fill_adjusted_ev_summary": summary.get("fill_adjusted_ev_summary", {}),
            "sizing_summary": summary.get("sizing_summary", {}),
        },
    )
    pm_shadow._plot_shadow_bank_curve(fills, bank_curve_path)
    if bundle_path is not None:
        _save_json(bundle_path, policy_bundle_payload or {})
    if policy_candidates is not None:
        policy_candidates.to_csv(run_dir / "policy_candidates.csv", index=False)
    artifacts = {
        "retro_shadow_summary": summary_path,
        "retro_coverage_summary": coverage_path,
        "retro_candidate_rows": candidate_path,
        "retro_decision_rows": decision_path,
        "retro_fill_rows": fill_path,
        "market_mapping": mapping_path,
        "mapping_audit": audit_path,
        "retro_skip_reasons": skip_path,
        "clv_rows": clv_path,
        "execution_quality": execution_quality_path,
        "bank_curve": bank_curve_path,
    }
    if bundle_path is not None:
        artifacts["policy_bundle"] = bundle_path
    if policy_candidates is not None:
        artifacts["policy_candidates"] = run_dir / "policy_candidates.csv"
    if diagnostics_json_path is not None:
        artifacts["selected_bet_diagnostics"] = diagnostics_json_path
    if diagnostics_text_path is not None:
        artifacts["selected_bet_diagnostics_text"] = diagnostics_text_path
    return artifacts, bundle_path


def _persist_mapping_audit(connection: Any, run_id: str, mapping_audit: pd.DataFrame) -> None:
    if mapping_audit.empty:
        return
    audit = mapping_audit.copy()
    audit["run_id"] = run_id
    pm_shadow._upsert_rows(connection, "pm_mapping_audit", audit.to_dict(orient="records"))


__all__ = [name for name in globals() if not name.startswith("__")]
