from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import Settings
from .contracts import OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME, PolymarketCollectResult
from .data_sources import PolymarketClobClient, PolymarketGammaClient
from .polymarket_shadow_common import (
    GROUP_ROLE_ORDER,
    _clean_json,
    _event_game_start,
    _event_within_window,
    _infer_league_from_event,
    _infer_market_role,
    _iso_timestamp,
    _json_list,
    _order_levels,
    _parse_match_title,
    _parse_timestamp,
    _resolve_team_alias,
    _status_label,
    _utcnow,
)
from .polymarket_shadow_db import (
    _catalog_lookup,
    _checkpoint_lookup,
    _group_lookup,
    _resolution_lookup,
    _upsert_rows,
    default_polymarket_db_path,
    init_polymarket_db,
)
from .reporting import _save_json, create_run_context


def _book_fetch_failure_label(error: Exception) -> str:
    if isinstance(error, requests.HTTPError):
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {404, 410}:
            return "missing"
    return "failed"


def _build_catalog_from_events(
    settings: Settings,
    events_by_slug: dict[str, dict[str, Any]],
    updated_at: str,
    team_lookup: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for event_slug, event in sorted(events_by_slug.items()):
        title = str(event.get("title", "")).strip()
        teams = _parse_match_title(title)
        if teams is None:
            continue
        event_home_team, event_away_team = teams
        home_team, away_team = event_home_team, event_away_team
        sport_code, league_code, league_name = _infer_league_from_event(event_slug, event)
        if not sport_code or sport_code not in settings.polymarket.supported_sports or not league_code:
            continue
        if team_lookup:
            candidates = team_lookup.get(str(league_code), [])
            home_team = _resolve_team_alias(home_team, candidates, league_code=league_code)
            away_team = _resolve_team_alias(away_team, candidates, league_code=league_code)
        market_role_map: dict[str, list[str]] = {role: [] for role in GROUP_ROLE_ORDER}

        for market in event.get("markets", []):
            outcomes = [str(item).lower() for item in _json_list(market.get("outcomes"))]
            market_type = str(market.get("sportsMarketType", "")).strip().lower()
            if outcomes != ["yes", "no"] or (market_type not in {"", "moneyline"}):
                continue
            market_role = _infer_market_role(
                str(market.get("question", "")),
                home_team=event_home_team,
                away_team=event_away_team,
            )
            token_ids = _json_list(market.get("clobTokenIds"))
            if not market_role or len(token_ids) < 1:
                continue
            market_role_map[market_role].append(str(market.get("id")))
            catalog_rows.append(
                {
                    "market_id": str(market.get("id")),
                    "event_id": str(event.get("id", "")),
                    "event_slug": event_slug,
                    "event_title": title,
                    "market_slug": str(market.get("slug", "")),
                    "question": str(market.get("question", "")),
                    "league_code": league_code,
                    "league_name": league_name,
                    "sport_code": sport_code,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_role": market_role,
                    "game_start_time": _iso_timestamp(_parse_timestamp(market.get("gameStartTime"))),
                    "yes_token_id": str(token_ids[0]),
                    "no_token_id": str(token_ids[1]) if len(token_ids) > 1 else "",
                    "fees_enabled": int(bool(market.get("feesEnabled", False))),
                    "fee_rate": float(PolymarketGammaClient.extract_fee_rate(market)),
                    "status": _status_label(
                        active=bool(market.get("active", False)),
                        closed=bool(market.get("closed", False)),
                        accepting_orders=bool(market.get("acceptingOrders", market.get("active", False))),
                    ),
                    "active": int(bool(market.get("active", False))),
                    "closed": int(bool(market.get("closed", False))),
                    "accepting_orders": int(bool(market.get("acceptingOrders", market.get("active", False)))),
                    "raw_json": _clean_json(market),
                    "updated_at": updated_at,
                }
            )

        group_rows.append(
            {
                "group_key": event_slug,
                "event_slug": event_slug,
                "event_title": title,
                "league_code": league_code,
                "league_name": league_name,
                "sport_code": sport_code,
                "home_team": home_team,
                "away_team": away_team,
                "game_start_time": _iso_timestamp(_event_game_start(event)),
                "home_market_id": market_role_map[OUTCOME_HOME][0] if len(market_role_map[OUTCOME_HOME]) == 1 else "",
                "draw_market_id": market_role_map[OUTCOME_DRAW][0] if len(market_role_map[OUTCOME_DRAW]) == 1 else "",
                "away_market_id": market_role_map[OUTCOME_AWAY][0] if len(market_role_map[OUTCOME_AWAY]) == 1 else "",
                "mapping_status": "complete" if all(len(market_role_map[item]) == 1 for item in GROUP_ROLE_ORDER) else "unmapped",
                "mapping_reason": "" if all(len(market_role_map[item]) == 1 for item in GROUP_ROLE_ORDER) else _clean_json(market_role_map),
                "raw_market_ids_json": _clean_json(market_role_map),
                "updated_at": updated_at,
            }
        )

    return pd.DataFrame(catalog_rows), pd.DataFrame(group_rows)


def discover_market_catalog(settings: Settings, gamma: PolymarketGammaClient, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    now = now or _utcnow()
    events_by_slug: dict[str, dict[str, Any]] = {}
    search_terms = set(settings.polymarket.discovery_queries)
    horizon_days = max(1, math.ceil(settings.polymarket.discovery_window_hours / 24))
    for offset in range(horizon_days + 1):
        date_text = (now + pd.to_timedelta(offset, unit="D")).strftime("%Y-%m-%d")
        search_terms.add(date_text)
        search_terms.add(f"win on {date_text}")

    for query in sorted(search_terms):
        for event in gamma.search_events(query=query, limit=50):
            slug = str(event.get("slug", "")).strip()
            if not slug or slug in events_by_slug:
                continue
            if not _event_within_window(event, now=now, window_hours=float(settings.polymarket.discovery_window_hours)):
                continue
            full_event = gamma.event_by_slug(slug)
            if full_event:
                events_by_slug[slug] = full_event[0]

    return _build_catalog_from_events(
        settings=settings,
        events_by_slug=events_by_slug,
        updated_at=_iso_timestamp(now),
    )


def sync_discovery_to_db(connection, catalog: pd.DataFrame, groups: pd.DataFrame) -> None:
    _upsert_rows(connection, "pm_market_catalog", catalog.to_dict(orient="records") if not catalog.empty else [])
    _upsert_rows(connection, "pm_market_groups", groups.to_dict(orient="records") if not groups.empty else [])


def _group_key_for_market(groups: pd.DataFrame, market_id: str) -> str:
    if groups.empty:
        return ""
    for row in groups.itertuples(index=False):
        if market_id in {str(row.home_market_id), str(row.draw_market_id), str(row.away_market_id)}:
            return str(row.group_key)
    return ""


def _capture_book_rows(
    order_book: dict[str, Any],
    market_id: str,
    group_key: str,
    event_type: str,
    source: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamp_ms = order_book.get("timestamp")
    timestamp = pd.to_datetime(int(timestamp_ms), unit="ms", utc=True) if timestamp_ms is not None else _utcnow()
    bids = _order_levels(order_book.get("bids", []), ascending=False)
    asks = _order_levels(order_book.get("asks", []), ascending=True)
    best_bid = float(bids[0]["price"]) if bids else np.nan
    best_ask = float(asks[0]["price"]) if asks else np.nan
    spread = float(best_ask - best_bid) if not (np.isnan(best_bid) or np.isnan(best_ask)) else np.nan
    asset_id = str(order_book.get("asset_id", ""))
    best_row = {
        "asset_id": asset_id,
        "market_id": market_id,
        "group_key": group_key,
        "timestamp": _iso_timestamp(timestamp),
        "event_type": event_type,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "source": source,
        "raw_json": _clean_json(order_book),
    }
    checkpoint_row = {
        "asset_id": asset_id,
        "market_id": market_id,
        "group_key": group_key,
        "timestamp": _iso_timestamp(timestamp),
        "event_type": event_type,
        "top_ask": best_ask,
        "asks_json": _clean_json(asks),
        "bids_json": _clean_json(bids),
        "source": source,
        "raw_json": _clean_json(order_book),
    }
    return best_row, checkpoint_row


def capture_rest_books(
    connection,
    settings: Settings,
    clob: PolymarketClobClient,
    event_type: str = "rest_checkpoint",
    market_ids: list[str] | None = None,
) -> dict[str, int]:
    catalog = _catalog_lookup(connection)
    groups = _group_lookup(connection)
    stats = {
        "book_fetch_attempts": 0,
        "book_fetch_successes": 0,
        "book_fetch_missing": 0,
        "book_fetch_failed": 0,
        "book_fetch_retry_attempts": 0,
        "book_fetch_retry_successes": 0,
        "last_trade_attempts": 0,
        "last_trade_successes": 0,
        "last_trade_missing": 0,
        "last_trade_failed": 0,
    }
    if catalog.empty:
        return {"best_rows": 0, "checkpoint_rows": 0, "trade_rows": 0, "capture": stats}
    if market_ids:
        market_ids = [str(item) for item in market_ids]
        catalog = catalog[catalog["market_id"].astype(str).isin(market_ids)].copy()
    best_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    retry_budget = max(1, int(settings.polymarket.decision_capture_retry_attempts)) if str(event_type).startswith("decision_precheck") else 1

    for row in catalog.itertuples(index=False):
        if not str(row.yes_token_id):
            continue
        order_book = None
        fetch_failed = False
        fetch_missing = False
        retried = False
        for attempt in range(retry_budget):
            stats["book_fetch_attempts"] += 1
            try:
                order_book = clob.get_order_book(str(row.yes_token_id))
                break
            except requests.RequestException as exc:
                fetch_failed = True
                if _book_fetch_failure_label(exc) == "missing":
                    fetch_missing = True
                if attempt + 1 < retry_budget:
                    retried = True
                    stats["book_fetch_retry_attempts"] += 1
                continue
            except Exception:
                fetch_failed = True
                if attempt + 1 < retry_budget:
                    retried = True
                    stats["book_fetch_retry_attempts"] += 1
                continue
        if order_book is None:
            if fetch_failed:
                stats["book_fetch_failed"] += 1
            if fetch_missing:
                stats["book_fetch_missing"] += 1
            continue
        stats["book_fetch_successes"] += 1
        if not isinstance(order_book, dict):
            stats["book_fetch_failed"] += 1
            continue
        if retried:
            stats["book_fetch_retry_successes"] += 1
        group_key = _group_key_for_market(groups, str(row.market_id))
        best_row, checkpoint_row = _capture_book_rows(
            order_book=order_book,
            market_id=str(row.market_id),
            group_key=group_key,
            event_type=event_type,
            source="clob_rest",
        )
        best_rows.append(best_row)
        checkpoint_rows.append(checkpoint_row)
        stats["last_trade_attempts"] += 1
        try:
            last_trade = clob.get_last_trade_price(str(row.yes_token_id))
            trade_rows.append(
                {
                    "trade_key": f"{row.yes_token_id}:{best_row['timestamp']}:last_trade_price_rest",
                    "asset_id": str(row.yes_token_id),
                    "market_id": str(row.market_id),
                    "group_key": group_key,
                    "timestamp": best_row["timestamp"],
                    "event_type": "last_trade_price_rest",
                    "price": float(last_trade.get("price", np.nan)),
                    "side": str(last_trade.get("side", "")),
                    "size": np.nan,
                    "trade_hash": "",
                    "raw_json": _clean_json(last_trade),
                }
            )
        except requests.RequestException as exc:
            stats["last_trade_failed"] += 1
            if _book_fetch_failure_label(exc) == "missing":
                stats["last_trade_missing"] += 1
            continue
        except Exception:
            stats["last_trade_failed"] += 1
            continue
        stats["last_trade_successes"] += 1

    _upsert_rows(connection, "pm_book_best", best_rows)
    _upsert_rows(connection, "pm_book_checkpoints", checkpoint_rows)
    _upsert_rows(connection, "pm_trades", trade_rows)
    return {
        "best_rows": len(best_rows),
        "checkpoint_rows": len(checkpoint_rows),
        "trade_rows": len(trade_rows),
        "capture": stats,
    }


def refresh_resolutions(connection, gamma: PolymarketGammaClient) -> int:
    groups = _group_lookup(connection)
    catalog = _catalog_lookup(connection)
    if groups.empty or catalog.empty:
        return 0

    resolution_rows: list[dict[str, Any]] = []
    for group in groups.itertuples(index=False):
        if str(group.mapping_status) != "complete":
            continue
        event_payload = gamma.event_by_slug(str(group.event_slug))
        if not event_payload:
            continue
        event = event_payload[0]
        role_prices: dict[str, float] = {}
        for market in event.get("markets", []):
            market_id = str(market.get("id"))
            match = catalog[catalog["market_id"].astype(str) == market_id]
            if match.empty:
                continue
            role = str(match.iloc[0]["market_role"])
            prices = _json_list(market.get("outcomePrices"))
            if prices:
                role_prices[role] = float(prices[0])
        winning_role = ""
        if all(role in role_prices for role in GROUP_ROLE_ORDER):
            for role in GROUP_ROLE_ORDER:
                if math.isclose(role_prices[role], 1.0, abs_tol=1e-9):
                    winning_role = role
                    break
        resolution_rows.append(
            {
                "group_key": str(group.group_key),
                "event_slug": str(group.event_slug),
                "resolved_at": _iso_timestamp(),
                "winning_role": winning_role,
                "source": "gamma_event",
                "raw_json": _clean_json({"event": event, "role_prices": role_prices}),
            }
        )
    _upsert_rows(connection, "pm_resolutions", resolution_rows)
    return len(resolution_rows)


def _decision_window_diagnostics(
    groups: pd.DataFrame,
    settings: Settings,
    now: pd.Timestamp | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    now = now or _utcnow()
    if groups.empty:
        return {
            "status": "waiting_for_fixtures",
            "upcoming_complete_groups": 0,
            "next_decision_time": "",
            "seconds_until_next_decision": None,
            "recommended_stream_seconds": 0,
            "upcoming_windows": [],
        }
    frame = groups.copy()
    frame["game_start_time"] = pd.to_datetime(frame["game_start_time"], utc=True, errors="coerce")
    frame = frame[frame["mapping_status"].astype(str).eq("complete") & frame["game_start_time"].notna()].copy()
    if frame.empty:
        return {
            "status": "waiting_for_fixtures",
            "upcoming_complete_groups": 0,
            "next_decision_time": "",
            "seconds_until_next_decision": None,
            "recommended_stream_seconds": 0,
            "upcoming_windows": [],
        }
    frame["decision_time"] = frame["game_start_time"] - pd.to_timedelta(settings.polymarket.decision_offset_minutes, unit="m")
    frame = frame[frame["decision_time"] >= now - pd.to_timedelta(5, unit="m")].sort_values("decision_time").head(int(limit))
    if frame.empty:
        return {
            "status": "waiting_for_fixtures",
            "upcoming_complete_groups": 0,
            "next_decision_time": "",
            "seconds_until_next_decision": None,
            "recommended_stream_seconds": 0,
            "upcoming_windows": [],
        }
    next_decision_time = pd.Timestamp(frame.iloc[0]["decision_time"])
    seconds_until_next = max(float((next_decision_time - now).total_seconds()), 0.0)
    recommended_stream_seconds = int(math.ceil(seconds_until_next + 600.0))
    return {
        "status": "ready_to_capture" if seconds_until_next <= float(settings.polymarket.discovery_window_hours) * 3600.0 else "waiting_for_fixtures",
        "upcoming_complete_groups": int(len(frame)),
        "next_decision_time": _iso_timestamp(next_decision_time),
        "seconds_until_next_decision": seconds_until_next,
        "recommended_stream_seconds": recommended_stream_seconds,
        "decision_offset_minutes": int(settings.polymarket.decision_offset_minutes),
        "decision_capture_lead_seconds": int(settings.polymarket.decision_capture_lead_seconds),
        "decision_book_freshness_seconds": int(settings.polymarket.decision_book_freshness_seconds),
        "upcoming_windows": [
            {
                "group_key": str(row.group_key),
                "league_code": str(row.league_code),
                "game_start_time": _iso_timestamp(pd.Timestamp(row.game_start_time)),
                "decision_time": _iso_timestamp(pd.Timestamp(row.decision_time)),
            }
            for row in frame.itertuples(index=False)
        ],
    }


async def _stream_polymarket_updates(
    connection,
    settings: Settings,
    clob: PolymarketClobClient,
    stream_seconds: int,
) -> dict[str, Any]:
    catalog = _catalog_lookup(connection)
    groups = _group_lookup(connection)
    asset_to_market = {
        str(row.yes_token_id): (str(row.market_id), _group_key_for_market(groups, str(row.market_id)))
        for row in catalog.itertuples(index=False)
        if str(row.yes_token_id)
    }
    asset_ids = sorted(asset_to_market.keys())
    if not asset_ids:
        return {"decision_precheck_attempts": 0, "decision_precheck_retry_attempts": 0, "decision_precheck_ready_groups": 0}

    diagnostics = {
        "decision_precheck_attempts": 0,
        "decision_precheck_retry_attempts": 0,
        "decision_precheck_retry_successes": 0,
        "decision_precheck_ready_groups": 0,
        "decision_precheck_fetch_failures": 0,
        "decision_precheck_missing_books": 0,
        "decision_precheck_stale_candidates": 0,
        "decision_precheck_retry_due_to_missing_books": 0,
        "decision_precheck_retry_due_to_failed_books": 0,
        "market_stream_completed": 0,
        "market_stream_errors": 0,
        "market_stream_last_error": "",
        "sports_stream_completed": 0,
        "sports_stream_errors": 0,
        "sports_stream_last_error": "",
    }

    def _group_has_fresh_checkpoint(group_row: pd.Series, checkpoints_frame: pd.DataFrame) -> bool:
        decision_time = pd.Timestamp(group_row["game_start_time"]) - pd.to_timedelta(
            settings.polymarket.decision_offset_minutes, unit="m"
        )
        for market_id in (str(group_row["home_market_id"]), str(group_row["draw_market_id"]), str(group_row["away_market_id"])):
            if not market_id:
                return False
            market_checkpoints = checkpoints_frame[checkpoints_frame["market_id"].astype(str) == market_id].copy()
            if market_checkpoints.empty:
                return False
            timestamps = pd.to_datetime(market_checkpoints["timestamp"], utc=True, errors="coerce").dropna()
            if timestamps.empty:
                return False
            latest = timestamps[timestamps <= decision_time].max()
            if pd.isna(latest):
                return False
            if float((decision_time - latest).total_seconds()) > float(settings.polymarket.decision_book_freshness_seconds):
                return False
        return True

    async def handle_market_message(message: dict[str, Any]) -> None:
        if not isinstance(message, dict):
            return
        event_type = str(message.get("event_type") or message.get("type") or "market_message")
        asset_id = str(message.get("asset_id") or message.get("assetId") or "")
        market_id, group_key = asset_to_market.get(asset_id, ("", ""))
        timestamp = _parse_timestamp(message.get("timestamp")) if message.get("timestamp") else _utcnow()
        if message.get("bids") is not None or message.get("asks") is not None:
            best_row, checkpoint_row = _capture_book_rows(
                {
                    "asset_id": asset_id,
                    "timestamp": int(pd.Timestamp(timestamp).timestamp() * 1000),
                    "bids": message.get("bids", []),
                    "asks": message.get("asks", []),
                },
                market_id=market_id,
                group_key=group_key,
                event_type=event_type,
                source="clob_ws",
            )
            _upsert_rows(connection, "pm_book_best", [best_row])
            _upsert_rows(connection, "pm_book_checkpoints", [checkpoint_row])
        if event_type in {"last_trade_price", "trade"}:
            _upsert_rows(
                connection,
                "pm_trades",
                [
                    {
                        "trade_key": f"{asset_id}:{_iso_timestamp(timestamp)}:{event_type}:{message.get('hash', '')}",
                        "asset_id": asset_id,
                        "market_id": market_id,
                        "group_key": group_key,
                        "timestamp": _iso_timestamp(timestamp),
                        "event_type": event_type,
                        "price": float(message.get("price", np.nan)),
                        "side": str(message.get("side", "")),
                        "size": float(message.get("size", np.nan)) if message.get("size") is not None else np.nan,
                        "trade_hash": str(message.get("hash", "")),
                        "raw_json": _clean_json(message),
                    }
                ],
            )
        if event_type == "market_resolved" and group_key:
            _upsert_rows(
                connection,
                "pm_resolutions",
                [
                    {
                        "group_key": group_key,
                        "event_slug": group_key,
                        "resolved_at": _iso_timestamp(timestamp),
                        "winning_role": str(message.get("winning_role", "")),
                        "source": "clob_market_ws",
                        "raw_json": _clean_json(message),
                    }
                ],
            )

    async def handle_sports_message(message: dict[str, Any]) -> None:
        if not isinstance(message, dict):
            return
        timestamp = _parse_timestamp(message.get("timestamp")) if message.get("timestamp") else _utcnow()
        _upsert_rows(
            connection,
            "pm_sports_results",
            [
                {
                    "event_key": f"{message.get('event_type', 'sports')}:{message.get('market_slug', '')}:{_iso_timestamp(timestamp)}",
                    "timestamp": _iso_timestamp(timestamp),
                    "event_type": str(message.get("event_type", "sports")),
                    "market_slug": str(message.get("market_slug", "")),
                    "status": str(message.get("status", "")),
                    "raw_json": _clean_json(message),
                }
            ],
        )

    async def periodic_checkpoints() -> None:
        end_at = _utcnow() + pd.to_timedelta(stream_seconds, unit="s")
        while _utcnow() < end_at:
            capture_rest_books(connection, settings=settings, clob=clob, event_type="periodic_checkpoint")
            remaining = max((end_at - _utcnow()).total_seconds(), 0.0)
            if remaining <= 0:
                break
            await asyncio.sleep(min(settings.polymarket.checkpoint_interval_seconds, remaining))

    async def decision_checkpoints() -> None:
        groups_frame = _group_lookup(connection)
        checkpoints_done: set[str] = set()
        end_at = _utcnow() + pd.to_timedelta(stream_seconds, unit="s")
        lead_seconds = float(settings.polymarket.decision_capture_lead_seconds)
        retry_attempts = max(1, int(settings.polymarket.decision_capture_retry_attempts))
        retry_delay_seconds = max(0.0, float(settings.polymarket.decision_capture_retry_delay_seconds))
        while True:
            now = _utcnow()
            if now >= end_at:
                break
            decision_time = groups_frame["game_start_time"] - pd.to_timedelta(settings.polymarket.decision_offset_minutes, unit="m")
            due = groups_frame[
                groups_frame["mapping_status"].eq("complete")
                & groups_frame["game_start_time"].notna()
                & (decision_time - pd.to_timedelta(lead_seconds, unit="s") <= now)
                & (now < decision_time)
            ]
            for row in due.itertuples(index=False):
                if str(row.group_key) in checkpoints_done:
                    continue
                diagnostics["decision_precheck_attempts"] += 1
                capture_summary = capture_rest_books(
                    connection,
                    settings=settings,
                    clob=clob,
                    event_type="decision_precheck",
                    market_ids=[str(row.home_market_id), str(row.draw_market_id), str(row.away_market_id)],
                )
                capture_stats = capture_summary.get("capture", {}) if isinstance(capture_summary, dict) else {}
                diagnostics["decision_precheck_fetch_failures"] += int(capture_stats.get("book_fetch_failed", 0) or 0)
                diagnostics["decision_precheck_missing_books"] += int(capture_stats.get("book_fetch_missing", 0) or 0)
                if int(capture_stats.get("book_fetch_missing", 0) or 0) > 0:
                    diagnostics["decision_precheck_retry_due_to_missing_books"] += 1
                if int(capture_stats.get("book_fetch_failed", 0) or 0) > 0:
                    diagnostics["decision_precheck_retry_due_to_failed_books"] += 1
                checkpoints_frame = _checkpoint_lookup(connection)
                if not _group_has_fresh_checkpoint(pd.Series(row._asdict()), checkpoints_frame):
                    diagnostics["decision_precheck_stale_candidates"] += 1
                    if (
                        int(capture_stats.get("book_fetch_failed", 0) or 0) > 0
                        and retry_attempts > 1
                        and retry_delay_seconds > 0.0
                    ):
                        diagnostics["decision_precheck_retry_attempts"] += 1
                        await asyncio.sleep(retry_delay_seconds)
                        capture_summary = capture_rest_books(
                            connection,
                            settings=settings,
                            clob=clob,
                            event_type="decision_precheck_retry",
                            market_ids=[str(row.home_market_id), str(row.draw_market_id), str(row.away_market_id)],
                        )
                        capture_stats = capture_summary.get("capture", {}) if isinstance(capture_summary, dict) else {}
                        diagnostics["decision_precheck_fetch_failures"] += int(capture_stats.get("book_fetch_failed", 0) or 0)
                        diagnostics["decision_precheck_missing_books"] += int(capture_stats.get("book_fetch_missing", 0) or 0)
                        checkpoints_frame = _checkpoint_lookup(connection)
                        if _group_has_fresh_checkpoint(pd.Series(row._asdict()), checkpoints_frame):
                            diagnostics["decision_precheck_retry_successes"] += 1
                            checkpoints_done.add(str(row.group_key))
                            diagnostics["decision_precheck_ready_groups"] += 1
                            continue
                    continue
                checkpoints_done.add(str(row.group_key))
                diagnostics["decision_precheck_ready_groups"] += 1
            await asyncio.sleep(1.0)
            if len(checkpoints_done) >= len(groups_frame.index):
                break

        return diagnostics

    async def guarded_stream_task(name: str, awaitable) -> None:
        try:
            await awaitable
            diagnostics[f"{name}_completed"] += 1
        except Exception as exc:  # pragma: no cover - depende de red real
            diagnostics[f"{name}_errors"] += 1
            diagnostics[f"{name}_last_error"] = f"{type(exc).__name__}: {exc}"

    await asyncio.gather(
        guarded_stream_task(
            "market_stream",
            clob.stream_market(
                websocket_url=settings.polymarket.market_ws_url,
                asset_ids=asset_ids,
                on_message=handle_market_message,
                duration_seconds=stream_seconds,
            ),
        ),
        guarded_stream_task(
            "sports_stream",
            clob.stream_sports(
                websocket_url=settings.polymarket.sports_ws_url,
                on_message=handle_sports_message,
                duration_seconds=stream_seconds,
            ),
        ),
        periodic_checkpoints(),
        decision_checkpoints(),
    )
    return diagnostics


def collect_polymarket(
    settings: Settings,
    db_path: Path | str | None = None,
    stream_seconds: int = 0,
    now: pd.Timestamp | None = None,
) -> PolymarketCollectResult:
    db_path = Path(db_path) if db_path else default_polymarket_db_path(settings)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = init_polymarket_db(db_path)
    gamma = PolymarketGammaClient()
    clob = PolymarketClobClient()
    run = create_run_context(settings.paths.runs_dir, "collect_polymarket")
    now = now or _utcnow()

    catalog, groups = discover_market_catalog(settings=settings, gamma=gamma, now=now)
    sync_discovery_to_db(connection, catalog=catalog, groups=groups)
    decision_window_diagnostics = _decision_window_diagnostics(groups, settings=settings, now=now)
    capture_summary = capture_rest_books(connection, settings=settings, clob=clob, event_type="initial_checkpoint")
    resolutions_updated = refresh_resolutions(connection, gamma=gamma)
    stream_diagnostics: dict[str, Any] = {}

    if stream_seconds > 0:
        try:  # pragma: no cover - depende de red real
            stream_diagnostics = asyncio.run(
                _stream_polymarket_updates(connection, settings=settings, clob=clob, stream_seconds=stream_seconds)
            )
        except RuntimeError:
            pass
        resolutions_updated = refresh_resolutions(connection, gamma=gamma)

    summary = {
        "database_path": str(db_path),
        "catalog_rows": int(len(catalog)),
        "group_rows": int(len(groups)),
        "complete_groups": int(groups["mapping_status"].eq("complete").sum()) if not groups.empty else 0,
        "unmapped_groups": int(groups["mapping_status"].ne("complete").sum()) if not groups.empty else 0,
        "tracked_assets": int(catalog["yes_token_id"].astype(str).ne("").sum()) if not catalog.empty else 0,
        "capture": capture_summary,
        "capture_diagnostics": capture_summary.get("capture", {}) if isinstance(capture_summary, dict) else {},
        "stream": stream_diagnostics,
        "decision_precheck_diagnostics": stream_diagnostics,
        "decision_capture_profile": {
            "decision_offset_minutes": int(settings.polymarket.decision_offset_minutes),
            "decision_book_freshness_seconds": int(settings.polymarket.decision_book_freshness_seconds),
            "decision_capture_lead_seconds": int(settings.polymarket.decision_capture_lead_seconds),
            "decision_capture_retry_attempts": int(settings.polymarket.decision_capture_retry_attempts),
            "decision_capture_retry_delay_seconds": float(settings.polymarket.decision_capture_retry_delay_seconds),
            "checkpoint_interval_seconds": int(settings.polymarket.checkpoint_interval_seconds),
        },
        "decision_window_diagnostics": decision_window_diagnostics,
        "stream_seconds": int(stream_seconds),
        "resolutions_updated": int(resolutions_updated),
    }
    summary_path = run.run_dir / "collect_summary.json"
    _save_json(summary_path, summary)
    return PolymarketCollectResult(database_path=db_path, summary_path=summary_path, summary=summary)
