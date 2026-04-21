from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .contracts import OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME
from .models import outcome_probabilities_from_lambdas
from .polymarket_shadow_common import (
    BUNDLE_STATUS_PROVISIONAL,
    GROUP_ROLE_ORDER,
    PRICE_PROVENANCE_EXACT,
    PRICE_PROVENANCE_RESOLUTION_ONLY,
    _clean_json,
    _iso_timestamp,
    _json_list,
    _resolve_team_alias,
    _team_match_score,
    _utcnow,
    VALIDATION_STAGE_SHADOW,
)
from .strategy import BetPolicy


def _derive_fixtures_from_groups(connection, history_matches: pd.DataFrame) -> pd.DataFrame:
    from .polymarket_shadow_db import _group_lookup
    from .polymarket_shadow_common import _history_team_lookup

    groups = _group_lookup(connection)
    groups = groups[groups["mapping_status"].eq("complete")].copy()
    if not groups.empty:
        groups["game_start_time"] = pd.to_datetime(groups["game_start_time"], utc=True, errors="coerce")
        groups = groups[
            groups["game_start_time"].notna()
            & groups["game_start_time"].ge(_utcnow() - pd.to_timedelta(5, unit="m"))
        ].copy()
    if groups.empty:
        return pd.DataFrame(columns=["Date", "league_code", "league_name", "HomeTeam", "AwayTeam", "group_key", "kickoff_time"])
    aliases = _history_team_lookup(history_matches)
    groups["league_candidates"] = groups["league_code"].map(lambda league: aliases.get(str(league), []))
    groups["HomeTeam"] = groups.apply(
        lambda row: _resolve_team_alias(str(row["home_team"]), row["league_candidates"], league_code=str(row["league_code"])),
        axis=1,
    )
    groups["AwayTeam"] = groups.apply(
        lambda row: _resolve_team_alias(str(row["away_team"]), row["league_candidates"], league_code=str(row["league_code"])),
        axis=1,
    )
    fixtures = groups[["league_code", "league_name", "HomeTeam", "AwayTeam", "group_key", "game_start_time"]].copy()
    fixtures["kickoff_time"] = pd.to_datetime(fixtures["game_start_time"], utc=True, errors="coerce").dt.tz_localize(None)
    fixtures["Date"] = fixtures["kickoff_time"].dt.normalize()
    return fixtures.drop(columns=["game_start_time"])


def _match_fixture_to_group(row: pd.Series, groups: pd.DataFrame, settings: Settings) -> str:
    if "group_key" in row and pd.notna(row["group_key"]) and str(row["group_key"]).strip():
        return str(row["group_key"])
    tolerance = pd.to_timedelta(settings.polymarket.league_match_tolerance_minutes, unit="m")
    kickoff = pd.Timestamp(row.get("kickoff_time", row["Date"]))
    kickoff = kickoff.tz_localize("UTC") if kickoff.tzinfo is None else kickoff.tz_convert("UTC")
    matches = groups[
        groups["mapping_status"].eq("complete")
        & groups["league_code"].astype(str).eq(str(row["league_code"]))
    ].copy()
    if matches.empty:
        return ""
    league_code = str(row["league_code"])
    matches = matches[
        matches["home_team"].map(lambda team: _team_match_score(str(row["HomeTeam"]), str(team), league_code) >= settings.polymarket.mapping_score_threshold)
        & matches["away_team"].map(lambda team: _team_match_score(str(row["AwayTeam"]), str(team), league_code) >= settings.polymarket.mapping_score_threshold)
    ].copy()
    if matches.empty:
        return ""
    matches["kickoff_delta"] = (matches["game_start_time"] - kickoff).abs()
    matches = matches[matches["kickoff_delta"] <= tolerance]
    if matches.empty:
        return ""
    return str(matches.sort_values("kickoff_delta").iloc[0]["group_key"])


def _latest_checkpoint_before(checkpoints: pd.DataFrame, market_id: str, decision_time: pd.Timestamp) -> pd.Series | None:
    rows = checkpoints[
        checkpoints["market_id"].astype(str).eq(str(market_id))
        & checkpoints["timestamp"].notna()
        & checkpoints["timestamp"].le(decision_time)
    ].copy()
    if rows.empty:
        return None
    rows["age_seconds"] = (decision_time - rows["timestamp"]).dt.total_seconds()
    return rows.sort_values(["age_seconds", "timestamp"], ascending=[True, False]).iloc[0]


def simulate_taker_yes_fill(
    asks: list[dict[str, Any]],
    notional: float,
    fee_rate: float,
    slippage_cushion: float,
) -> dict[str, Any]:
    ladder = []
    for item in asks or []:
        if item.get("price") is not None and item.get("size") is not None:
            ladder.append({"price": float(item["price"]), "size": float(item["size"])})
    ladder = sorted(ladder, key=lambda item: item["price"])
    if not ladder or notional <= 0:
        return {
            "raw_spend": 0.0,
            "shares_filled": 0.0,
            "fill_rate": 0.0,
            "partial_fill": False,
            "top_ask": np.nan,
            "raw_vwap": np.nan,
            "effective_vwap": np.nan,
            "fee_paid": 0.0,
            "slippage_cost": 0.0,
            "cost_basis": 0.0,
            "levels_used": [],
        }

    remaining = float(notional)
    raw_spend = 0.0
    shares_filled = 0.0
    fee_paid = 0.0
    levels_used: list[dict[str, float]] = []
    top_ask = float(ladder[0]["price"])

    for level in ladder:
        price = float(level["price"])
        size = float(level["size"])
        if price <= 0 or size <= 0:
            continue
        max_spend = price * size
        spend_here = min(remaining, max_spend)
        if spend_here <= 0:
            continue
        shares_here = spend_here / price
        raw_spend += spend_here
        shares_filled += shares_here
        fee_here = shares_here * fee_rate * price * (1.0 - price)
        fee_paid += fee_here
        levels_used.append({"price": price, "shares": shares_here, "spend": spend_here, "fee": fee_here})
        remaining -= spend_here
        if remaining <= 1e-9:
            break

    slippage_cost = shares_filled * max(float(slippage_cushion), 0.0)
    cost_basis = raw_spend + fee_paid + slippage_cost
    raw_vwap = raw_spend / shares_filled if shares_filled > 0 else np.nan
    effective_vwap = cost_basis / shares_filled if shares_filled > 0 else np.nan
    fill_rate = min(raw_spend / notional, 1.0) if notional > 0 else 0.0
    return {
        "raw_spend": raw_spend,
        "shares_filled": shares_filled,
        "fill_rate": fill_rate,
        "partial_fill": fill_rate < 0.999,
        "top_ask": top_ask,
        "raw_vwap": raw_vwap,
        "effective_vwap": effective_vwap,
        "fee_paid": fee_paid,
        "slippage_cost": slippage_cost,
        "cost_basis": cost_basis,
        "levels_used": levels_used,
    }


def _policy_from_raw(raw: dict[str, Any] | None, settings: Settings) -> BetPolicy:
    raw = raw or {}
    return BetPolicy(
        edge_threshold=float(raw.get("edge_threshold", settings.research.provisional_edge_threshold)),
        ev_threshold=float(raw.get("ev_threshold", settings.research.provisional_ev_threshold)),
        min_odds=float(raw.get("min_odds", min(settings.backtest.policy_search.min_odds_options))),
        max_odds=float(raw.get("max_odds", max(settings.backtest.policy_search.max_odds_options))),
        kelly_fraction=float(raw.get("kelly_fraction", settings.backtest.policy_search.max_kelly_fraction)),
        family=str(raw.get("family", "edge_ev_threshold")),
        top_quantile=float(raw.get("top_quantile", 0.10)),
        allowed_leagues=tuple(raw.get("allowed_leagues", [])),
        allowed_outcomes=tuple(raw.get("allowed_outcomes", [])),
        scope_name=str(raw.get("scope_name", "global_all")),
    )


def _load_policy_bundle(path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _empty_decision_row(
    decision_id: str,
    row: pd.Series,
    decision_time: pd.Timestamp,
    kickoff_time: pd.Timestamp,
    probability_source: str,
    mapping_status: str,
    skip_reason: str,
    group_key: str = "",
    price_provenance: str = PRICE_PROVENANCE_RESOLUTION_ONLY,
    validation_stage: str = VALIDATION_STAGE_SHADOW,
    book_ref_json: str | None = None,
    book_age_seconds: float | None = None,
    snapshot_time: str = "",
) -> dict[str, Any]:
    date_value = pd.Timestamp(row["Date"])
    date_value = date_value.tz_localize("UTC") if date_value.tzinfo is None else date_value.tz_convert("UTC")
    return {
        "decision_id": decision_id,
        "run_id": "",
        "group_key": group_key,
        "match_id": str(row["match_id"]),
        "Date": _iso_timestamp(date_value),
        "kickoff_time": _iso_timestamp(kickoff_time),
        "decision_time": _iso_timestamp(decision_time),
        "league_code": str(row["league_code"]),
        "league_name": str(row["league_name"]),
        "HomeTeam": str(row["HomeTeam"]),
        "AwayTeam": str(row["AwayTeam"]),
        "mapping_status": mapping_status,
        "selection": "",
        "probability_source": probability_source,
        "price_provenance": price_provenance,
        "validation_stage": validation_stage,
        "model_prob": np.nan,
        "top_ask": np.nan,
        "fee_rate": np.nan,
        "expected_edge": np.nan,
        "expected_ev": np.nan,
        "available_ask_size": np.nan,
        "minutes_to_kickoff": float((kickoff_time - decision_time).total_seconds() / 60.0),
        "expected_fill_probability": np.nan,
        "fill_adjusted_ev": np.nan,
        "book_age_seconds": book_age_seconds if book_age_seconds is not None else np.nan,
        "snapshot_time": snapshot_time,
        "skip_reason": skip_reason,
        "book_ref_json": book_ref_json or _clean_json({}),
        "created_at": _iso_timestamp(),
    }


def _decision_from_prediction(
    row: pd.Series,
    groups: pd.DataFrame,
    catalog: pd.DataFrame,
    checkpoints: pd.DataFrame,
    settings: Settings,
    probability_source: str,
    policy: BetPolicy | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decision_id = str(uuid.uuid4())
    kickoff_time = pd.Timestamp(row.get("kickoff_time", row["Date"]))
    kickoff_time = kickoff_time.tz_localize("UTC") if kickoff_time.tzinfo is None else kickoff_time.tz_convert("UTC")
    decision_time = kickoff_time - pd.to_timedelta(settings.polymarket.decision_offset_minutes, unit="m")
    group_key = str(row.get("group_key", "")).strip()
    if not group_key:
        return _empty_decision_row(
            decision_id,
            row,
            decision_time=decision_time,
            kickoff_time=kickoff_time,
            probability_source=probability_source,
            mapping_status="missing",
            skip_reason="no_market_group",
        ), []

    group_match = groups[groups["group_key"].astype(str) == group_key]
    if group_match.empty or str(group_match.iloc[0]["mapping_status"]) != "complete":
        return _empty_decision_row(
            decision_id,
            row,
            decision_time=decision_time,
            kickoff_time=kickoff_time,
            probability_source=probability_source,
            mapping_status="unmapped",
            skip_reason="incomplete_market_group",
            group_key=group_key,
        ), []

    group = group_match.iloc[0]
    market_ids = {
        OUTCOME_HOME: str(group["home_market_id"]),
        OUTCOME_DRAW: str(group["draw_market_id"]),
        OUTCOME_AWAY: str(group["away_market_id"]),
    }
    outcome_rows: dict[str, dict[str, Any]] = {}
    book_refs: dict[str, Any] = {}
    stale = False

    for outcome, market_id in market_ids.items():
        market = catalog[catalog["market_id"].astype(str) == market_id]
        if market.empty:
            return _empty_decision_row(
                decision_id,
                row,
                decision_time=decision_time,
                kickoff_time=kickoff_time,
                probability_source=probability_source,
                mapping_status="missing_catalog",
                skip_reason="market_not_in_catalog",
                group_key=group_key,
                book_ref_json=_clean_json(book_refs),
            ), []
        checkpoint = _latest_checkpoint_before(checkpoints, market_id=market_id, decision_time=decision_time)
        if checkpoint is None:
            return _empty_decision_row(
                decision_id,
                row,
                decision_time=decision_time,
                kickoff_time=kickoff_time,
                probability_source=probability_source,
                mapping_status="missing_book",
                skip_reason="no_book_checkpoint",
                group_key=group_key,
                book_ref_json=_clean_json(book_refs),
            ), []

        age_seconds = float((decision_time - checkpoint["timestamp"]).total_seconds())
        if age_seconds > settings.polymarket.decision_book_freshness_seconds:
            stale = True
        asks = _json_list(checkpoint["asks_json"])
        if not asks:
            return _empty_decision_row(
                decision_id,
                row,
                decision_time=decision_time,
                kickoff_time=kickoff_time,
                probability_source=probability_source,
                mapping_status="no_ask_ladder",
                skip_reason="no_ask_ladder",
                group_key=group_key,
                price_provenance=PRICE_PROVENANCE_EXACT,
                book_ref_json=_clean_json(book_refs),
                book_age_seconds=age_seconds,
                snapshot_time=_iso_timestamp(checkpoint["timestamp"]),
            ), []
        top_ask = float(sorted(asks, key=lambda item: float(item["price"]))[0]["price"])
        market_row = market.iloc[0]
        fee_rate = float(market_row.get("fee_rate", 0.0)) if int(market_row.get("fees_enabled", 0)) else 0.0
        probability = float(row[f"prob_{outcome}_{probability_source}"])
        fee_per_share = fee_rate * top_ask * (1.0 - top_ask)
        total_cost_per_share = top_ask + fee_per_share + settings.polymarket.slippage_cushion
        expected_ev = probability - total_cost_per_share
        available_ask_size = float(sum(float(item["price"]) * float(item["size"]) for item in asks if item.get("price") is not None and item.get("size") is not None))
        outcome_rows[outcome] = {
            "model_prob": probability,
            "top_ask": top_ask,
            "fee_rate": fee_rate,
            "age_seconds": age_seconds,
            "expected_edge": probability - top_ask,
            "expected_ev": expected_ev,
            "available_ask_size": available_ask_size,
            "market_id": market_id,
            "asks": asks,
            "snapshot_time": checkpoint["timestamp"],
        }
        book_refs[outcome] = {
            "market_id": market_id,
            "asset_id": str(checkpoint["asset_id"]),
            "snapshot_time": _iso_timestamp(checkpoint["timestamp"]),
            "event_type": str(checkpoint["event_type"]),
        }

    if stale:
            return _empty_decision_row(
                decision_id,
                row,
                decision_time=decision_time,
                kickoff_time=kickoff_time,
                probability_source=probability_source,
                mapping_status="stale_book",
                skip_reason="stale_book",
                group_key=group_key,
                price_provenance=PRICE_PROVENANCE_EXACT,
                book_ref_json=_clean_json(book_refs),
                book_age_seconds=max(item["age_seconds"] for item in outcome_rows.values()),
                snapshot_time=_iso_timestamp(max(item["snapshot_time"] for item in outcome_rows.values())),
            ), []

    policy = policy or _policy_from_raw(None, settings)
    eligible_outcomes = {
        outcome: details
        for outcome, details in outcome_rows.items()
        if details["expected_edge"] >= policy.edge_threshold
        and details["expected_ev"] >= policy.ev_threshold
        and (1.0 / details["top_ask"]) >= policy.min_odds
        and (1.0 / details["top_ask"]) <= policy.max_odds
    }
    if not eligible_outcomes:
        best_outcome, best_details = max(outcome_rows.items(), key=lambda item: item[1]["expected_ev"])
        decision_row = _empty_decision_row(
            decision_id,
            row,
            decision_time=decision_time,
            kickoff_time=kickoff_time,
            probability_source=probability_source,
            mapping_status="complete",
            skip_reason="policy_rejected",
            group_key=group_key,
            book_ref_json=_clean_json(book_refs),
            book_age_seconds=best_details["age_seconds"],
            snapshot_time=_iso_timestamp(best_details["snapshot_time"]),
        )
        decision_row.update(
            {
                "selection": "",
                "model_prob": best_details["model_prob"],
                "top_ask": best_details["top_ask"],
                "fee_rate": best_details["fee_rate"],
                "expected_edge": best_details["expected_edge"],
                "expected_ev": best_details["expected_ev"],
                "available_ask_size": best_details["available_ask_size"],
                "price_provenance": PRICE_PROVENANCE_EXACT,
            }
        )
        return decision_row, []

    best_outcome, best_details = max(eligible_outcomes.items(), key=lambda item: (item[1]["expected_ev"], item[1]["expected_edge"], -item[1]["top_ask"]))
    decision_row = _empty_decision_row(
        decision_id,
        row,
        decision_time=decision_time,
        kickoff_time=kickoff_time,
        probability_source=probability_source,
        mapping_status="complete",
        skip_reason="",
        group_key=group_key,
        book_ref_json=_clean_json(book_refs),
        book_age_seconds=best_details["age_seconds"],
        snapshot_time=_iso_timestamp(best_details["snapshot_time"]),
    )
    decision_row.update(
        {
            "selection": best_outcome,
            "model_prob": best_details["model_prob"],
            "top_ask": best_details["top_ask"],
            "fee_rate": best_details["fee_rate"],
            "expected_edge": best_details["expected_edge"],
            "expected_ev": best_details["expected_ev"],
            "available_ask_size": best_details["available_ask_size"],
            "skip_reason": "",
            "price_provenance": PRICE_PROVENANCE_EXACT,
        }
    )

    fills: list[dict[str, Any]] = []
    for notional in settings.polymarket.notionals_ladder:
        fill = simulate_taker_yes_fill(
            asks=best_details["asks"],
            notional=float(notional),
            fee_rate=best_details["fee_rate"],
            slippage_cushion=settings.polymarket.slippage_cushion,
        )
        fills.append(
            {
                "fill_id": str(uuid.uuid4()),
                "decision_id": decision_id,
                "run_id": "",
                "group_key": group_key,
                "selection": best_outcome,
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
                "available_ask_size": best_details["available_ask_size"],
                "minutes_to_kickoff": float((kickoff_time - decision_time).total_seconds() / 60.0),
                "price_provenance": PRICE_PROVENANCE_EXACT,
                "model_prob": best_details["model_prob"],
                "expected_edge": best_details["expected_edge"],
                "expected_ev": best_details["expected_ev"],
                "closing_reference_odds": np.nan,
                "closing_reference_prob": np.nan,
                "clv_source": "",
                "payout": np.nan,
                "net_profit": np.nan,
                "won": np.nan,
                "resolution_outcome": "",
                "status": "open",
                "levels_used_json": _clean_json(fill["levels_used"]),
                "created_at": _iso_timestamp(),
            }
        )
    return decision_row, fills


def _settle_fill_row(fill_row: pd.Series, winning_role: str) -> dict[str, Any]:
    shares = float(fill_row["shares_filled"])
    cost_basis = float(fill_row["cost_basis"])
    if not winning_role:
        payout = np.nan
        profit = np.nan
        won = np.nan
        status = "open"
    else:
        won = int(str(fill_row["selection"]) == winning_role)
        payout = shares if won == 1 else 0.0
        profit = payout - cost_basis
        status = "settled"
    payload = fill_row.to_dict()
    payload.update(
        {
            "payout": payout,
            "net_profit": profit,
            "won": won,
            "resolution_outcome": winning_role,
            "status": status,
        }
    )
    return payload
