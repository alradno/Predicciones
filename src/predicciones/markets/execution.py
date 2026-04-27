from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Settings
from ..core.edge_hypothesis import EdgeHypothesisStatus
from ..core.promotion_state import PromotionStatus
from ..lanes.evaluation import evaluate_forward_lane
from ..lanes.runtime import default_multi_market_db_path, get_market_lane_spec, init_multi_market_db
from .trading import (
    LimitOrderIntent,
    TradeMode,
    TradeRiskInputs,
    VenueConfigurationError,
    ensure_trade_audit_tables,
    evaluate_trade_risk,
    is_kill_switch_enabled,
    record_risk_event,
    set_kill_switch,
    venue_from_settings,
)


def _utcnow() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if pd.notna(number) else default


def _quote_age_seconds(row: pd.Series) -> float | None:
    decision = pd.to_datetime(row.get("decision_time"), utc=True, errors="coerce")
    book = pd.to_datetime(row.get("book_timestamp"), utc=True, errors="coerce")
    if pd.isna(decision) or pd.isna(book):
        return None
    return float((decision - book).total_seconds())


def _expires_at(row: pd.Series) -> str:
    kickoff = pd.to_datetime(row.get("game_start_time"), utc=True, errors="coerce")
    if pd.notna(kickoff):
        return (kickoff - pd.Timedelta(minutes=5)).isoformat()
    decision = pd.to_datetime(row.get("decision_time"), utc=True, errors="coerce")
    if pd.notna(decision):
        return (decision + pd.Timedelta(minutes=30)).isoformat()
    return (pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=30)).isoformat()


def _candidate_decisions(connection: sqlite3.Connection, lane_id: str, limit: int = 25) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT *
        FROM mm_lane_forward_ledger
        WHERE lane_id = ?
          AND decision_status IN ('valid_forward_sample', 'settled')
          AND COALESCE(asset_id, '') <> ''
          AND top_ask IS NOT NULL
        ORDER BY decision_time DESC, created_at DESC
        LIMIT ?
        """,
        connection,
        params=(lane_id, limit),
    )


def _exposure(connection: sqlite3.Connection, mode: TradeMode) -> tuple[float, float]:
    today = pd.Timestamp.now(tz="UTC").date().isoformat()
    daily_row = connection.execute(
        """
        SELECT COALESCE(SUM(notional), 0.0)
        FROM mm_trade_order_intents
        WHERE mode = ? AND created_at >= ?
        """,
        (mode.value, today),
    ).fetchone()
    open_row = connection.execute(
        """
        SELECT COALESCE(SUM(notional), 0.0)
        FROM mm_trade_order_intents
        WHERE mode = ? AND status IN ('paper_open', 'posted', 'partially_filled')
        """,
        (mode.value,),
    ).fetchone()
    return float(daily_row[0] or 0.0), float(open_row[0] or 0.0)


def _build_intent(settings: Settings, lane_id: str, mode: TradeMode, row: pd.Series) -> LimitOrderIntent:
    price = _safe_float(row.get("top_ask"))
    notional = min(settings.trade.default_order_size, settings.trade.paper_max_notional)
    if settings.trade.bankroll_usdc:
        notional = min(notional, settings.trade.bankroll_usdc * settings.trade.max_order_bankroll_fraction)
    size = notional / price if price > 0.0 else 0.0
    return LimitOrderIntent(
        intent_id=str(uuid.uuid4()),
        lane_id=lane_id,
        decision_id=str(row.get("decision_id") or ""),
        venue=settings.trade.venue,
        token_id=str(row.get("asset_id") or ""),
        side="BUY",
        price=price,
        size=float(size),
        notional=float(notional),
        time_in_force=settings.trade.order_time_in_force,
        expires_at=_expires_at(row),
        mode=mode,
    )


def _record_intent(
    connection: sqlite3.Connection,
    intent: LimitOrderIntent,
    *,
    risk_allowed: bool,
    risk_blockers: list[str],
    status: str,
    raw: dict[str, Any] | None = None,
) -> None:
    payload = intent.to_dict()
    connection.execute(
        """
        INSERT INTO mm_trade_order_intents (
            intent_id, lane_id, decision_id, venue, mode, token_id, side, price, size,
            notional, time_in_force, expires_at, risk_allowed, risk_blockers_json,
            status, raw_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            intent.intent_id,
            intent.lane_id,
            intent.decision_id,
            intent.venue,
            intent.mode.value,
            intent.token_id,
            intent.side,
            intent.price,
            intent.size,
            intent.notional,
            intent.time_in_force,
            intent.expires_at,
            1 if risk_allowed else 0,
            json.dumps(risk_blockers, sort_keys=True),
            status,
            json.dumps(raw or payload, sort_keys=True),
            _utcnow(),
        ),
    )
    connection.commit()


def run_trade_cycle(
    settings: Settings,
    lane_id: str,
    mode: TradeMode,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    spec = get_market_lane_spec(lane_id)
    database_path = Path(db_path) if db_path else default_multi_market_db_path(settings)
    forward_summary, forward_artifacts = evaluate_forward_lane(settings, spec.lane_id, database_path)
    connection = init_multi_market_db(database_path)
    try:
        ensure_trade_audit_tables(connection)
        kill_switch = is_kill_switch_enabled(connection)
        rows = _candidate_decisions(connection, spec.lane_id)
        daily_committed, open_exposure = _exposure(connection, mode)
        promotion_status = forward_summary.get("promotion_status", PromotionStatus.not_actionable.value)
        edge_status = (forward_summary.get("roi_45_hypothesis") or {}).get(
            "status",
            EdgeHypothesisStatus.collecting.value,
        )
        intents_written = 0
        posted_orders = 0
        blockers: list[str] = []
        venue = venue_from_settings(settings)

        if rows.empty:
            decision = evaluate_trade_risk(
                TradeRiskInputs(
                    lane_id=spec.lane_id,
                    mode=mode,
                    promotion_status=promotion_status,
                    edge_status=edge_status,
                    order_notional=0.0,
                    daily_committed=daily_committed,
                    open_exposure=open_exposure,
                    quote_age_seconds=None,
                    config=settings.trade,
                    kill_switch_enabled=kill_switch,
                    reconciliation_clean=True,
                )
            )
            record_risk_event(
                connection,
                lane_id=spec.lane_id,
                mode=mode,
                event_type="no_trade_candidates",
                decision=decision,
                raw={"forward_summary": forward_summary},
            )
            blockers.extend(decision.blockers or ("no_trade_candidates",))
        else:
            for _, row in rows.iterrows():
                intent = _build_intent(settings, spec.lane_id, mode, row)
                decision = evaluate_trade_risk(
                    TradeRiskInputs(
                        lane_id=spec.lane_id,
                        mode=mode,
                        promotion_status=promotion_status,
                        edge_status=edge_status,
                        order_notional=intent.notional,
                        daily_committed=daily_committed,
                        open_exposure=open_exposure,
                        quote_age_seconds=_quote_age_seconds(row),
                        config=settings.trade,
                        kill_switch_enabled=kill_switch,
                        reconciliation_clean=True,
                    )
                )
                record_risk_event(
                    connection,
                    lane_id=spec.lane_id,
                    mode=mode,
                    event_type="trade_intent_risk_check",
                    decision=decision,
                    raw={"decision_id": intent.decision_id, "intent_id": intent.intent_id},
                )
                preview = venue.preview_order(intent)
                intent_blockers = list(decision.blockers) + list(preview.blockers)
                status = "blocked"
                if decision.allowed and preview.accepted:
                    status = "paper_open" if mode == TradeMode.paper else "preview_accepted"
                    if mode == TradeMode.live:
                        try:
                            posted = venue.place_limit_order(intent)
                        except VenueConfigurationError as exc:
                            status = "blocked"
                            intent_blockers.append(str(exc))
                        else:
                            posted_orders += 1
                            connection.execute(
                                """
                                INSERT INTO mm_trade_posted_orders (
                                    order_id, intent_id, lane_id, venue, mode, status, raw_json, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    posted.order_id,
                                    intent.intent_id,
                                    intent.lane_id,
                                    intent.venue,
                                    intent.mode.value,
                                    posted.status,
                                    json.dumps(posted.raw or {}, sort_keys=True),
                                    _utcnow(),
                                ),
                            )
                _record_intent(
                    connection,
                    intent,
                    risk_allowed=decision.allowed and preview.accepted and not intent_blockers,
                    risk_blockers=intent_blockers,
                    status=status,
                    raw={"preview": preview.raw or {}},
                )
                daily_committed += intent.notional if status != "blocked" else 0.0
                open_exposure += intent.notional if status != "blocked" else 0.0
                intents_written += 1
                blockers.extend(intent_blockers)

        summary = {
            "lane_id": spec.lane_id,
            "mode": mode.value,
            "database_path": str(database_path),
            "promotion_status": promotion_status,
            "roi_45_hypothesis_status": edge_status,
            "kill_switch_enabled": kill_switch,
            "candidate_decisions": int(len(rows)),
            "intents_written": intents_written,
            "posted_orders": posted_orders,
            "blocked": bool(blockers),
            "blockers": sorted(set(blockers)),
            "live_trading_enabled": settings.trade.live_enabled,
            "venue": settings.trade.venue,
            "policy_reoptimized": False,
        }
    finally:
        connection.close()

    artifacts = dict(forward_artifacts)
    report_path = settings.paths.outputs_dir / "lanes" / spec.lane_id / f"trade_{mode.value}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    artifacts[f"trade_{mode.value}_report"] = report_path
    return summary, artifacts


def run_trade_reconciliation(
    settings: Settings,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    database_path = Path(db_path) if db_path else default_multi_market_db_path(settings)
    connection = init_multi_market_db(database_path)
    try:
        ensure_trade_audit_tables(connection)
        venue = venue_from_settings(settings)
        blockers: list[str] = []
        if settings.trade.venue not in {"global", "us"}:
            blockers.append("venue_not_configured")
        open_orders = venue.open_orders()
        positions = venue.positions()
        balances = venue.balances()
        clean = not blockers
        snapshot_id = str(uuid.uuid4())
        connection.execute(
            """
            INSERT INTO mm_trade_reconciliation_snapshots (
                snapshot_id, venue, open_orders_json, positions_json, balances_json,
                clean, blockers_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                settings.trade.venue,
                json.dumps(open_orders, sort_keys=True),
                json.dumps(positions, sort_keys=True),
                json.dumps(balances, sort_keys=True),
                1 if clean else 0,
                json.dumps(blockers, sort_keys=True),
                _utcnow(),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    summary = {
        "database_path": str(database_path),
        "venue": settings.trade.venue,
        "clean": clean,
        "blockers": blockers,
        "open_orders": len(open_orders),
        "positions": len(positions),
    }
    report_path = settings.paths.outputs_dir / "trade_reconciliation_report.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return summary, {"trade_reconciliation_report": report_path}


def update_trade_kill_switch(
    settings: Settings,
    *,
    enabled: bool,
    reason: str = "",
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    database_path = Path(db_path) if db_path else default_multi_market_db_path(settings)
    init_multi_market_db(database_path).close()
    summary = set_kill_switch(database_path, enabled=enabled, reason=reason)
    summary["database_path"] = str(database_path)
    report_path = settings.paths.outputs_dir / "trade_kill_switch_report.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return summary, {"trade_kill_switch_report": report_path}
