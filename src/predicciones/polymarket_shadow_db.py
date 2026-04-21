from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .polymarket_shadow_common import _iso_timestamp


def default_polymarket_db_path(settings: Settings) -> Path:
    return settings.paths.data_dir / settings.polymarket.database_filename


def _connect_db(path: Path | str) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    return connection


def init_polymarket_db(path: Path | str) -> sqlite3.Connection:
    connection = _connect_db(path)
    cursor = connection.cursor()
    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS pm_market_catalog (
            market_id TEXT PRIMARY KEY,
            event_id TEXT,
            event_slug TEXT,
            event_title TEXT,
            market_slug TEXT,
            question TEXT,
            league_code TEXT,
            league_name TEXT,
            sport_code TEXT,
            home_team TEXT,
            away_team TEXT,
            market_role TEXT,
            game_start_time TEXT,
            yes_token_id TEXT,
            no_token_id TEXT,
            fees_enabled INTEGER,
            fee_rate REAL,
            status TEXT,
            active INTEGER,
            closed INTEGER,
            accepting_orders INTEGER,
            raw_json TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_market_groups (
            group_key TEXT PRIMARY KEY,
            event_slug TEXT,
            event_title TEXT,
            league_code TEXT,
            league_name TEXT,
            sport_code TEXT,
            match_id TEXT,
            home_team TEXT,
            away_team TEXT,
            game_start_time TEXT,
            home_market_id TEXT,
            draw_market_id TEXT,
            away_market_id TEXT,
            mapping_status TEXT,
            mapping_reason TEXT,
            raw_market_ids_json TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_book_best (
            asset_id TEXT,
            market_id TEXT,
            group_key TEXT,
            timestamp TEXT,
            event_type TEXT,
            best_bid REAL,
            best_ask REAL,
            spread REAL,
            source TEXT,
            raw_json TEXT,
            PRIMARY KEY(asset_id, timestamp, event_type)
        );
        CREATE TABLE IF NOT EXISTS pm_book_checkpoints (
            asset_id TEXT,
            market_id TEXT,
            group_key TEXT,
            timestamp TEXT,
            event_type TEXT,
            top_ask REAL,
            asks_json TEXT,
            bids_json TEXT,
            source TEXT,
            raw_json TEXT,
            PRIMARY KEY(asset_id, timestamp, event_type)
        );
        CREATE TABLE IF NOT EXISTS pm_trades (
            trade_key TEXT PRIMARY KEY,
            asset_id TEXT,
            market_id TEXT,
            group_key TEXT,
            timestamp TEXT,
            event_type TEXT,
            price REAL,
            side TEXT,
            size REAL,
            trade_hash TEXT,
            raw_json TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_resolutions (
            group_key TEXT PRIMARY KEY,
            event_slug TEXT,
            resolved_at TEXT,
            winning_role TEXT,
            source TEXT,
            raw_json TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_price_history (
            price_key TEXT PRIMARY KEY,
            market_id TEXT,
            group_key TEXT,
            timestamp TEXT,
            price REAL,
            source TEXT,
            decision_time TEXT,
            lag_seconds REAL,
            raw_json TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_event_raw (
            event_slug TEXT PRIMARY KEY,
            event_id TEXT,
            series_slug TEXT,
            title TEXT,
            start_date TEXT,
            game_start_time TEXT,
            event_date TEXT,
            league_code TEXT,
            league_name TEXT,
            sport_code TEXT,
            closed INTEGER,
            archived INTEGER,
            raw_json TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_market_raw (
            market_id TEXT PRIMARY KEY,
            event_id TEXT,
            event_slug TEXT,
            event_title TEXT,
            series_slug TEXT,
            market_slug TEXT,
            question TEXT,
            group_item_title TEXT,
            league_code TEXT,
            league_name TEXT,
            sport_code TEXT,
            game_start_time TEXT,
            closed INTEGER,
            archived INTEGER,
            active INTEGER,
            accepting_orders INTEGER,
            sports_market_type TEXT,
            outcomes_json TEXT,
            outcome_prices_json TEXT,
            clob_token_ids_json TEXT,
            fees_enabled INTEGER,
            fee_rate REAL,
            raw_json TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_fixture_market_candidates (
            market_id TEXT PRIMARY KEY,
            event_slug TEXT,
            league_code TEXT,
            league_name TEXT,
            sport_code TEXT,
            match_id TEXT,
            match_date TEXT,
            game_start_time TEXT,
            fixture_key TEXT,
            undirected_fixture_key TEXT,
            home_team_raw TEXT,
            away_team_raw TEXT,
            home_team_canonical TEXT,
            away_team_canonical TEXT,
            parse_source TEXT,
            parse_method TEXT,
            market_shape TEXT,
            market_role TEXT,
            market_quality_rank REAL,
            classification_status TEXT,
            classification_reason TEXT,
            raw_json TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_fixture_groups_v2 (
            group_key TEXT PRIMARY KEY,
            fixture_key TEXT,
            undirected_fixture_key TEXT,
            event_slug TEXT,
            event_title TEXT,
            league_code TEXT,
            league_name TEXT,
            sport_code TEXT,
            match_id TEXT,
            match_date TEXT,
            game_start_time TEXT,
            home_team TEXT,
            away_team TEXT,
            home_market_id TEXT,
            draw_market_id TEXT,
            away_market_id TEXT,
            group_status TEXT,
            group_reason TEXT,
            source_event_slugs_json TEXT,
            source_market_ids_json TEXT,
            duplicate_market_ids_json TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_market_classification_audit (
            market_id TEXT PRIMARY KEY,
            event_slug TEXT,
            league_code TEXT,
            league_name TEXT,
            sport_code TEXT,
            match_id TEXT,
            question TEXT,
            group_item_title TEXT,
            market_slug TEXT,
            market_shape TEXT,
            classification_status TEXT,
            classification_reason TEXT,
            fixture_key TEXT,
            group_key TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_team_aliases (
            alias_id TEXT PRIMARY KEY,
            league_code TEXT,
            canonical_team TEXT,
            alias_text TEXT,
            alias_key TEXT,
            source TEXT,
            score REAL,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_mapping_audit (
            audit_id TEXT PRIMARY KEY,
            run_id TEXT,
            source_mode TEXT,
            match_id TEXT,
            league_code TEXT,
            league_name TEXT,
            HomeTeam TEXT,
            AwayTeam TEXT,
            kickoff_time TEXT,
            group_key TEXT,
            mapping_status TEXT,
            mapping_stage TEXT,
            mapping_score REAL,
            home_score REAL,
            away_score REAL,
            kickoff_delta_minutes REAL,
            candidate_count INTEGER,
            selected_event_slug TEXT,
            audit_reason TEXT,
            candidates_json TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_shadow_decisions (
            decision_id TEXT PRIMARY KEY,
            run_id TEXT,
            group_key TEXT,
            match_id TEXT,
            Date TEXT,
            kickoff_time TEXT,
            decision_time TEXT,
            league_code TEXT,
            league_name TEXT,
            HomeTeam TEXT,
            AwayTeam TEXT,
            mapping_status TEXT,
            selection TEXT,
            probability_source TEXT,
            price_provenance TEXT,
            validation_stage TEXT,
            model_prob REAL,
            top_ask REAL,
            fee_rate REAL,
            expected_edge REAL,
            expected_ev REAL,
            available_ask_size REAL,
            minutes_to_kickoff REAL,
            expected_fill_probability REAL,
            fill_adjusted_ev REAL,
            book_age_seconds REAL,
            snapshot_time TEXT,
            skip_reason TEXT,
            book_ref_json TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_shadow_fills (
            fill_id TEXT PRIMARY KEY,
            decision_id TEXT,
            run_id TEXT,
            group_key TEXT,
            selection TEXT,
            notional REAL,
            raw_spend REAL,
            shares_filled REAL,
            fill_rate REAL,
            partial_fill INTEGER,
            top_ask REAL,
            raw_vwap REAL,
            effective_vwap REAL,
            fee_paid REAL,
            slippage_cost REAL,
            cost_basis REAL,
            payout REAL,
            net_profit REAL,
            won INTEGER,
            resolution_outcome TEXT,
            status TEXT,
            price_provenance TEXT,
            model_prob REAL,
            expected_edge REAL,
            expected_ev REAL,
            available_ask_size REAL,
            minutes_to_kickoff REAL,
            closing_reference_odds REAL,
            closing_reference_prob REAL,
            clv_source TEXT,
            league_code TEXT,
            kickoff_time TEXT,
            levels_used_json TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pm_sports_results (
            event_key TEXT PRIMARY KEY,
            timestamp TEXT,
            event_type TEXT,
            market_slug TEXT,
            status TEXT,
            raw_json TEXT
        );
        """
    )
    group_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(pm_market_groups)").fetchall()
    }
    if "match_id" not in group_columns:
        connection.execute("ALTER TABLE pm_market_groups ADD COLUMN match_id TEXT")
    shadow_decision_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(pm_shadow_decisions)").fetchall()
    }
    for column_name, column_type in {
        "price_provenance": "TEXT",
        "validation_stage": "TEXT",
        "available_ask_size": "REAL",
        "minutes_to_kickoff": "REAL",
        "expected_fill_probability": "REAL",
        "fill_adjusted_ev": "REAL",
    }.items():
        if column_name not in shadow_decision_columns:
            connection.execute(f"ALTER TABLE pm_shadow_decisions ADD COLUMN {column_name} {column_type}")
    shadow_fill_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(pm_shadow_fills)").fetchall()
    }
    for column_name, column_type in {
        "price_provenance": "TEXT",
        "model_prob": "REAL",
        "expected_edge": "REAL",
        "expected_ev": "REAL",
        "available_ask_size": "REAL",
        "minutes_to_kickoff": "REAL",
        "closing_reference_odds": "REAL",
        "closing_reference_prob": "REAL",
        "clv_source": "TEXT",
        "league_code": "TEXT",
        "kickoff_time": "TEXT",
    }.items():
        if column_name not in shadow_fill_columns:
            connection.execute(f"ALTER TABLE pm_shadow_fills ADD COLUMN {column_name} {column_type}")
    connection.commit()
    return connection


def _upsert_rows(connection: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    def _sqlite_safe(value: Any) -> Any:
        if value is None or value is pd.NaT:
            return None
        if isinstance(value, pd.Timestamp):
            return _iso_timestamp(value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    prepared_rows = [{key: _sqlite_safe(value) for key, value in row.items()} for row in rows]
    columns = sorted(prepared_rows[0].keys())
    placeholders = ", ".join(f":{column}" for column in columns)
    protected = {
        "market_id",
        "group_key",
        "trade_key",
        "decision_id",
        "fill_id",
        "event_key",
        "price_key",
        "alias_id",
        "audit_id",
    }
    assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in protected)
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) ON CONFLICT DO UPDATE SET {assignments}"
    connection.executemany(sql, prepared_rows)
    connection.commit()


def _frame_from_query(connection: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, connection, params=params)


def _catalog_lookup(connection: sqlite3.Connection) -> pd.DataFrame:
    return _frame_from_query(connection, "SELECT * FROM pm_market_catalog")


def _group_lookup(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = _frame_from_query(connection, "SELECT * FROM pm_market_groups")
    if not frame.empty:
        frame["game_start_time"] = pd.to_datetime(frame["game_start_time"], utc=True, errors="coerce")
    return frame


def _checkpoint_lookup(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = _frame_from_query(connection, "SELECT * FROM pm_book_checkpoints")
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame


def _resolution_lookup(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = _frame_from_query(connection, "SELECT * FROM pm_resolutions")
    if not frame.empty:
        frame["resolved_at"] = pd.to_datetime(frame["resolved_at"], utc=True, errors="coerce")
    return frame


def _price_history_lookup(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = _frame_from_query(connection, "SELECT * FROM pm_price_history")
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    return frame


def _team_alias_lookup(connection: sqlite3.Connection) -> pd.DataFrame:
    return _frame_from_query(connection, "SELECT * FROM pm_team_aliases")


def _mapping_audit_lookup(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = _frame_from_query(connection, "SELECT * FROM pm_mapping_audit")
    if not frame.empty:
        frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    return frame
