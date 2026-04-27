from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from ..config import Settings, TradingConfig
from ..core.edge_hypothesis import EdgeHypothesisStatus
from ..core.lane_governance import get_lane_governance
from ..core.promotion_state import PromotionStatus


class TradeMode(str, Enum):
    paper = "paper"
    live = "live"


class VenueName(str, Enum):
    disabled = "disabled"
    polymarket_global = "global"
    polymarket_us = "us"


class VenueConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LimitOrderIntent:
    intent_id: str
    lane_id: str
    decision_id: str
    venue: str
    token_id: str
    side: str
    price: float
    size: float
    notional: float
    time_in_force: str
    expires_at: str
    mode: TradeMode

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


@dataclass(frozen=True)
class OrderPreview:
    accepted: bool
    blockers: tuple[str, ...]
    estimated_cost: float
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class PostedOrder:
    order_id: str
    status: str
    raw: dict[str, Any] | None = None


class PolymarketVenue(Protocol):
    name: str

    def preview_order(self, intent: LimitOrderIntent) -> OrderPreview:
        ...

    def place_limit_order(self, intent: LimitOrderIntent) -> PostedOrder:
        ...

    def cancel_order(self, order_id: str) -> PostedOrder:
        ...

    def open_orders(self) -> list[dict[str, Any]]:
        ...

    def positions(self) -> list[dict[str, Any]]:
        ...

    def balances(self) -> dict[str, Any]:
        ...


class _ConfiguredVenue:
    def __init__(self, name: str, *, live_enabled: bool = False, client: Any | None = None) -> None:
        self.name = name
        self._live_enabled = live_enabled
        self._client = client

    def preview_order(self, intent: LimitOrderIntent) -> OrderPreview:
        blockers: list[str] = []
        if intent.price <= 0.0 or intent.price >= 1.0:
            blockers.append("limit_price_outside_probability_bounds")
        if intent.size <= 0.0:
            blockers.append("size_must_be_positive")
        if intent.time_in_force != "GTD":
            blockers.append("time_in_force_must_be_gtd")
        return OrderPreview(
            accepted=not blockers,
            blockers=tuple(blockers),
            estimated_cost=float(intent.notional),
            raw={"venue": self.name, "intent_id": intent.intent_id},
        )

    def place_limit_order(self, intent: LimitOrderIntent) -> PostedOrder:
        if not self._live_enabled:
            raise VenueConfigurationError("live trading is disabled for this venue adapter")
        if self._client is None:
            raise VenueConfigurationError("authenticated venue client is not configured")
        if hasattr(self._client, "place_limit_order"):
            return self._client.place_limit_order(intent)
        raise VenueConfigurationError("authenticated venue client does not expose place_limit_order")

    def cancel_order(self, order_id: str) -> PostedOrder:
        if self._client is None or not hasattr(self._client, "cancel_order"):
            raise VenueConfigurationError("authenticated venue client is not configured")
        return self._client.cancel_order(order_id)

    def open_orders(self) -> list[dict[str, Any]]:
        if self._client is not None and hasattr(self._client, "open_orders"):
            return list(self._client.open_orders())
        return []

    def positions(self) -> list[dict[str, Any]]:
        if self._client is not None and hasattr(self._client, "positions"):
            return list(self._client.positions())
        return []

    def balances(self) -> dict[str, Any]:
        if self._client is not None and hasattr(self._client, "balances"):
            return dict(self._client.balances())
        return {}


class PolymarketGlobalVenue(_ConfiguredVenue):
    def __init__(self, *, live_enabled: bool = False, client: Any | None = None) -> None:
        super().__init__(VenueName.polymarket_global.value, live_enabled=live_enabled, client=client)


class PolymarketUSVenue(_ConfiguredVenue):
    def __init__(self, *, live_enabled: bool = False, client: Any | None = None) -> None:
        super().__init__(VenueName.polymarket_us.value, live_enabled=live_enabled, client=client)


@dataclass(frozen=True)
class TradeRiskInputs:
    lane_id: str
    mode: TradeMode
    promotion_status: PromotionStatus | str
    edge_status: EdgeHypothesisStatus | str
    order_notional: float
    daily_committed: float
    open_exposure: float
    quote_age_seconds: float | None
    config: TradingConfig
    kill_switch_enabled: bool = False
    reconciliation_clean: bool = True


@dataclass(frozen=True)
class TradeRiskDecision:
    allowed: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "blockers": list(self.blockers)}


def evaluate_trade_risk(inputs: TradeRiskInputs) -> TradeRiskDecision:
    blockers: list[str] = []
    promotion_status = getattr(inputs.promotion_status, "value", str(inputs.promotion_status))
    edge_status = getattr(inputs.edge_status, "value", str(inputs.edge_status))
    governance = get_lane_governance(inputs.lane_id)
    if not governance.can_emit_shadow_picks:
        blockers.append("lane_cannot_emit_trade_intents")
    if inputs.kill_switch_enabled:
        blockers.append("kill_switch_enabled")
    if not inputs.reconciliation_clean:
        blockers.append("reconciliation_not_clean")
    if inputs.quote_age_seconds is None:
        blockers.append("quote_age_missing")
    elif inputs.quote_age_seconds > inputs.config.max_quote_age_seconds:
        blockers.append(
            f"quote_age_above_max:{inputs.quote_age_seconds:.2f}>"
            f"{inputs.config.max_quote_age_seconds}"
        )

    if inputs.mode == TradeMode.paper:
        if not inputs.config.paper_enabled:
            blockers.append("paper_trading_disabled")
        if promotion_status not in {
            PromotionStatus.paper_promotable.value,
            PromotionStatus.capital_promotable.value,
        }:
            blockers.append("paper_requires_paper_promotable")
    else:
        if not governance.can_emit_capital_picks:
            blockers.append("lane_capital_governance_disabled")
        if not inputs.config.live_enabled:
            blockers.append("live_trading_disabled")
        if inputs.config.venue not in {VenueName.polymarket_global.value, VenueName.polymarket_us.value}:
            blockers.append("venue_not_configured")
        if inputs.config.bankroll_usdc is None or inputs.config.bankroll_usdc <= 0.0:
            blockers.append("bankroll_usdc_missing")
        if not inputs.config.jurisdiction_confirmed:
            blockers.append("jurisdiction_not_confirmed")
        if inputs.config.allow_vpn_bypass:
            blockers.append("vpn_bypass_not_allowed")
        if promotion_status != PromotionStatus.capital_promotable.value:
            blockers.append("live_requires_capital_promotable")
        if edge_status != EdgeHypothesisStatus.supported.value:
            blockers.append("live_requires_roi45_supported")

    bankroll = inputs.config.bankroll_usdc
    if bankroll and bankroll > 0.0:
        max_order = bankroll * inputs.config.max_order_bankroll_fraction
        max_daily = bankroll * inputs.config.max_daily_bankroll_fraction
        max_open = bankroll * inputs.config.max_open_bankroll_fraction
        if inputs.order_notional > max_order:
            blockers.append(f"order_notional_above_cap:{inputs.order_notional:.2f}>{max_order:.2f}")
        if inputs.daily_committed + inputs.order_notional > max_daily:
            blockers.append("daily_exposure_cap_exceeded")
        if inputs.open_exposure + inputs.order_notional > max_open:
            blockers.append("open_exposure_cap_exceeded")
    elif inputs.mode == TradeMode.paper:
        max_order = inputs.config.paper_max_notional
        if inputs.order_notional > max_order:
            blockers.append(f"paper_order_notional_above_cap:{inputs.order_notional:.2f}>{max_order:.2f}")

    return TradeRiskDecision(allowed=not blockers, blockers=tuple(blockers))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_trade_audit_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS mm_trade_order_intents (
            intent_id TEXT PRIMARY KEY,
            lane_id TEXT NOT NULL,
            decision_id TEXT,
            venue TEXT NOT NULL,
            mode TEXT NOT NULL,
            token_id TEXT,
            side TEXT,
            price REAL,
            size REAL,
            notional REAL,
            time_in_force TEXT,
            expires_at TEXT,
            risk_allowed INTEGER,
            risk_blockers_json TEXT,
            status TEXT,
            raw_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mm_trade_posted_orders (
            order_id TEXT PRIMARY KEY,
            intent_id TEXT NOT NULL,
            lane_id TEXT NOT NULL,
            venue TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            raw_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mm_trade_fills (
            fill_id TEXT PRIMARY KEY,
            order_id TEXT,
            intent_id TEXT,
            lane_id TEXT,
            venue TEXT,
            price REAL,
            size REAL,
            notional REAL,
            fee REAL,
            status TEXT,
            raw_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mm_trade_positions (
            position_id TEXT PRIMARY KEY,
            lane_id TEXT,
            venue TEXT,
            token_id TEXT,
            size REAL,
            avg_price REAL,
            raw_json TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mm_trade_risk_events (
            event_id TEXT PRIMARY KEY,
            lane_id TEXT,
            mode TEXT,
            event_type TEXT NOT NULL,
            allowed INTEGER,
            blockers_json TEXT,
            raw_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mm_trade_reconciliation_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            venue TEXT,
            open_orders_json TEXT,
            positions_json TEXT,
            balances_json TEXT,
            clean INTEGER,
            blockers_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mm_trade_kill_switch (
            switch_id TEXT PRIMARY KEY,
            enabled INTEGER NOT NULL,
            reason TEXT,
            updated_at TEXT NOT NULL
        );
        """
    )
    connection.commit()


def set_kill_switch(db_path: Path | str, *, enabled: bool, reason: str = "") -> dict[str, Any]:
    connection = sqlite3.connect(str(db_path))
    try:
        ensure_trade_audit_tables(connection)
        payload = {
            "switch_id": "global",
            "enabled": 1 if enabled else 0,
            "reason": reason,
            "updated_at": _utcnow(),
        }
        connection.execute(
            """
            INSERT INTO mm_trade_kill_switch (switch_id, enabled, reason, updated_at)
            VALUES (:switch_id, :enabled, :reason, :updated_at)
            ON CONFLICT(switch_id) DO UPDATE SET
              enabled=excluded.enabled,
              reason=excluded.reason,
              updated_at=excluded.updated_at
            """,
            payload,
        )
        connection.commit()
        return {"kill_switch_enabled": enabled, "reason": reason, "updated_at": payload["updated_at"]}
    finally:
        connection.close()


def is_kill_switch_enabled(connection: sqlite3.Connection) -> bool:
    ensure_trade_audit_tables(connection)
    row = connection.execute(
        "SELECT enabled FROM mm_trade_kill_switch WHERE switch_id = 'global'"
    ).fetchone()
    return bool(row and int(row[0]) == 1)


def record_risk_event(
    connection: sqlite3.Connection,
    *,
    lane_id: str,
    mode: TradeMode,
    event_type: str,
    decision: TradeRiskDecision,
    raw: dict[str, Any] | None = None,
) -> str:
    ensure_trade_audit_tables(connection)
    event_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO mm_trade_risk_events (
            event_id, lane_id, mode, event_type, allowed, blockers_json, raw_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            lane_id,
            mode.value,
            event_type,
            1 if decision.allowed else 0,
            json.dumps(list(decision.blockers), sort_keys=True),
            json.dumps(raw or {}, sort_keys=True),
            _utcnow(),
        ),
    )
    connection.commit()
    return event_id


def venue_from_settings(settings: Settings) -> PolymarketVenue:
    if settings.trade.venue == VenueName.polymarket_global.value:
        return PolymarketGlobalVenue(live_enabled=settings.trade.live_enabled)
    if settings.trade.venue == VenueName.polymarket_us.value:
        return PolymarketUSVenue(live_enabled=settings.trade.live_enabled)
    return _ConfiguredVenue(VenueName.disabled.value, live_enabled=False)
