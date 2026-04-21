from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from .config import Settings
from .data_sources import FootballDataClient, LEAGUE_NAMES
from .ingestion import canonicalize_matches, normalize_team_name


FOOTBALL_SIM_DATABASE_FILENAME = "football_sim_data.sqlite"
SIM_OUTPUT_DIRNAME = "sim_data"
STATSBOMB_RAW_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CLUBELO_API_BASE_URL = "http://api.clubelo.com"

FOOTBALL_DATA_MAX_FREE_LEAGUES: dict[str, str] = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "National League",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "I1": "Serie A",
    "I2": "Serie B",
    "SP1": "La Liga",
    "SP2": "Segunda Division",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Belgian Pro League",
    "P1": "Primeira Liga",
    "T1": "Super Lig",
    "G1": "Greek Super League",
    "ARG": "Argentina Primera Division",
    "AUT": "Austrian Bundesliga",
    "BRA": "Brazil Serie A",
    "CHN": "Chinese Super League",
    "DNK": "Danish Superliga",
    "FIN": "Finnish Veikkausliiga",
    "IRL": "League of Ireland Premier",
    "JPN": "J1 League",
    "MEX": "Liga MX",
    "NOR": "Norwegian Eliteserien",
    "POL": "Polish Ekstraklasa",
    "ROU": "Romanian Liga 1",
    "RUS": "Russian Premier League",
    "SWE": "Swedish Allsvenskan",
    "SWZ": "Swiss Super League",
    "USA": "MLS",
}


@dataclass(frozen=True)
class SimulationDataSourceSpec:
    source_id: str
    source_url: str
    sport: str
    entities: tuple[str, ...]
    fetch_mode: str
    requires_auth: bool
    license_notes: str
    default_enabled: bool
    status: str
    coverage_expectation: str


SIM_DATA_SOURCE_REGISTRY: dict[str, SimulationDataSourceSpec] = {
    "football_data": SimulationDataSourceSpec(
        source_id="football_data",
        source_url="https://www.football-data.co.uk/data.php",
        sport="football",
        entities=("matches", "team_match_stats", "market_reference"),
        fetch_mode="csv_http",
        requires_auth=False,
        license_notes="Free public football CSVs; store raw snapshots and source URL for reproducibility.",
        default_enabled=True,
        status="enabled",
        coverage_expectation="Broad league/season match-level coverage; player and event data unavailable.",
    ),
    "statsbomb_open_data": SimulationDataSourceSpec(
        source_id="statsbomb_open_data",
        source_url="https://github.com/statsbomb/open-data",
        sport="football",
        entities=("competitions", "matches", "lineups", "events"),
        fetch_mode="github_json",
        requires_auth=False,
        license_notes="Open data with attribution requirements; coverage is competition-limited.",
        default_enabled=False,
        status="registered_coverage_limited",
        coverage_expectation="High-quality event and lineup data where competitions are available.",
    ),
    "openfootball": SimulationDataSourceSpec(
        source_id="openfootball",
        source_url="https://github.com/openfootball/football.json",
        sport="football",
        entities=("competitions", "fixtures", "results", "team_aliases"),
        fetch_mode="github_json",
        requires_auth=False,
        license_notes="Open fixture/result metadata; useful for mapping, limited advanced stats.",
        default_enabled=False,
        status="registered_mapping_only",
        coverage_expectation="Fixture/result metadata for selected competitions.",
    ),
    "clubelo": SimulationDataSourceSpec(
        source_id="clubelo",
        source_url="http://clubelo.com/",
        sport="football",
        entities=("team_ratings",),
        fetch_mode="csv_http",
        requires_auth=False,
        license_notes="Public rating data; activate only after license/terms audit for intended use.",
        default_enabled=False,
        status="enabled_mapping_audited",
        coverage_expectation="Historical team strength signal if terms are acceptable.",
    ),
    "worldfootballr": SimulationDataSourceSpec(
        source_id="worldfootballr",
        source_url="https://jaseziv.github.io/worldfootballR/",
        sport="football",
        entities=("matches", "players", "events"),
        fetch_mode="quarantine_scraper",
        requires_auth=False,
        license_notes="Optional fragile connector; disabled until stability and terms are audited.",
        default_enabled=False,
        status="quarantine_disabled",
        coverage_expectation="Potentially broad coverage, but not part of default benchmark.",
    ),
    "soccerdata": SimulationDataSourceSpec(
        source_id="soccerdata",
        source_url="https://soccerdata.readthedocs.io/",
        sport="football",
        entities=("matches", "players", "events", "team_stats"),
        fetch_mode="quarantine_scraper",
        requires_auth=False,
        license_notes="Optional fragile connector; disabled until stability and terms are audited.",
        default_enabled=False,
        status="quarantine_disabled",
        coverage_expectation="Potentially useful wrappers, but not default-enabled.",
    ),
    "fbref_understat_quarantine": SimulationDataSourceSpec(
        source_id="fbref_understat_quarantine",
        source_url="https://fbref.com/",
        sport="football",
        entities=("team_stats", "player_stats", "xg"),
        fetch_mode="quarantine_scraper",
        requires_auth=False,
        license_notes="No aggressive scraping; connector remains disabled unless terms and rate limits are safe.",
        default_enabled=False,
        status="quarantine_disabled",
        coverage_expectation="Potential xG/player richness after explicit audit.",
    ),
}


def get_sim_data_source_spec(source_id: str) -> SimulationDataSourceSpec:
    try:
        return SIM_DATA_SOURCE_REGISTRY[str(source_id)]
    except KeyError as exc:
        raise ValueError(f"Fuente de simulacion no declarada: {source_id}") from exc


def default_football_sim_db_path(settings: Settings) -> Path:
    return settings.paths.data_dir / FOOTBALL_SIM_DATABASE_FILENAME


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")


def _stable_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).strip().lower())
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_") or "unknown"


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(_stable_slug(str(part)) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _content_hash(content: str | bytes) -> str:
    raw = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(raw).hexdigest()


def _sim_output_dir(settings: Settings) -> Path:
    path = settings.paths.outputs_dir / SIM_OUTPUT_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_football_sim_db(path: Path | str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sim_raw_payloads (
            content_hash TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_key TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            payload_format TEXT NOT NULL,
            content_text TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            fetch_status TEXT NOT NULL,
            failure_reason TEXT,
            license_status TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sim_teams (
            team_id TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sim_team_aliases (
            alias_id TEXT PRIMARY KEY,
            team_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            alias_name TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            mapping_status TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sim_competitions (
            competition_key TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            competition_id TEXT NOT NULL,
            season_id TEXT NOT NULL,
            country_name TEXT,
            competition_name TEXT NOT NULL,
            season_name TEXT,
            gender TEXT,
            match_available TEXT,
            match_available_360 TEXT
        );

        CREATE TABLE IF NOT EXISTS sim_players (
            player_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_player_id TEXT,
            player_name TEXT NOT NULL,
            team_id TEXT,
            canonical_name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sim_match_source_links (
            link_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_match_id TEXT NOT NULL,
            match_id TEXT NOT NULL,
            mapping_status TEXT NOT NULL,
            confidence REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sim_team_ratings (
            rating_id TEXT PRIMARY KEY,
            team_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            rating_date TEXT NOT NULL,
            rating_value REAL NOT NULL,
            rating_type TEXT NOT NULL,
            mapping_status TEXT NOT NULL,
            source_team_name TEXT
        );

        CREATE TABLE IF NOT EXISTS sim_matches (
            match_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_match_key TEXT NOT NULL,
            match_date TEXT NOT NULL,
            league_code TEXT NOT NULL,
            league_name TEXT NOT NULL,
            season TEXT NOT NULL,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            home_team_name TEXT NOT NULL,
            away_team_name TEXT NOT NULL,
            home_goals REAL,
            away_goals REAL,
            result_code TEXT,
            outcome TEXT,
            home_shots REAL,
            away_shots REAL,
            home_shots_target REAL,
            away_shots_target REAL,
            home_corners REAL,
            away_corners REAL,
            home_yellow_cards REAL,
            away_yellow_cards REAL,
            home_red_cards REAL,
            away_red_cards REAL,
            odds_home REAL,
            odds_draw REAL,
            odds_away REAL,
            source_url TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sim_lineups (
            lineup_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            team_id TEXT NOT NULL,
            player_id TEXT,
            player_name TEXT,
            known_before_match INTEGER NOT NULL DEFAULT 0,
            source_id TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sim_events (
            event_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            team_id TEXT,
            player_id TEXT,
            event_type TEXT,
            minute REAL,
            source_id TEXT NOT NULL,
            known_before_match INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS sim_gold_features (
            match_id TEXT PRIMARY KEY,
            as_of_time TEXT NOT NULL,
            match_start_time TEXT NOT NULL,
            known_before_match INTEGER NOT NULL,
            league_code TEXT NOT NULL,
            season TEXT NOT NULL,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            home_team_name TEXT NOT NULL,
            away_team_name TEXT NOT NULL,
            feature_role TEXT NOT NULL,
            feature_family_set TEXT NOT NULL,
            home_matches_played_pre REAL,
            away_matches_played_pre REAL,
            home_points_last_5 REAL,
            away_points_last_5 REAL,
            home_points_last_10 REAL,
            away_points_last_10 REAL,
            home_points_per_match_last_5 REAL,
            away_points_per_match_last_5 REAL,
            home_points_per_match_last_10 REAL,
            away_points_per_match_last_10 REAL,
            points_form_diff_5 REAL,
            points_form_diff_10 REAL,
            home_goals_for_avg_last_5 REAL,
            away_goals_for_avg_last_5 REAL,
            home_goals_against_avg_last_5 REAL,
            away_goals_against_avg_last_5 REAL,
            home_internal_elo_pre REAL,
            away_internal_elo_pre REAL,
            internal_elo_diff REAL,
            home_rest_days REAL,
            away_rest_days REAL,
            rest_advantage REAL,
            home_matches_last_7d REAL,
            away_matches_last_7d REAL,
            home_matches_last_14d REAL,
            away_matches_last_14d REAL,
            market_reference_present INTEGER,
            odds_home REAL,
            odds_draw REAL,
            odds_away REAL,
            market_prob_home REAL,
            market_prob_draw REAL,
            market_prob_away REAL,
            home_clubelo_pre REAL,
            away_clubelo_pre REAL,
            clubelo_diff REAL,
            clubelo_mapping_status TEXT,
            home_statsbomb_event_count_avg_last_5 REAL,
            away_statsbomb_event_count_avg_last_5 REAL,
            home_statsbomb_shot_count_avg_last_5 REAL,
            away_statsbomb_shot_count_avg_last_5 REAL
        );
        """
    )
    _ensure_columns(
        connection,
        "sim_gold_features",
        {
            "home_clubelo_pre": "REAL",
            "away_clubelo_pre": "REAL",
            "clubelo_diff": "REAL",
            "clubelo_mapping_status": "TEXT",
            "home_statsbomb_event_count_avg_last_5": "REAL",
            "away_statsbomb_event_count_avg_last_5": "REAL",
            "home_statsbomb_shot_count_avg_last_5": "REAL",
            "away_statsbomb_shot_count_avg_last_5": "REAL",
        },
    )
    connection.commit()
    return connection


def _ensure_columns(connection: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    for column, definition in columns.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def discover_sim_data_sources(settings: Settings, db_path: Path | str | None = None) -> tuple[dict[str, Any], dict[str, Path]]:
    output_dir = _sim_output_dir(settings)
    database_path = Path(db_path) if db_path else default_football_sim_db_path(settings)
    connection = init_football_sim_db(database_path)
    try:
        payload = _build_sim_data_report_payload(settings=settings, connection=connection, database_path=database_path)
    finally:
        connection.close()

    artifacts = _write_sim_data_reports(output_dir, payload)
    return payload["summary"], artifacts


def collect_sim_data_source(
    settings: Settings,
    source_id: str,
    leagues: tuple[str, ...] | list[str],
    seasons: tuple[str, ...] | list[str],
    db_path: Path | str | None = None,
    football_data_loader: Callable[[str, str], pd.DataFrame] | None = None,
    profile: str | None = None,
    seasons_back: int | None = None,
    probe_missing: bool = False,
    teams: str = "mapped",
    statsbomb_json_loader: Callable[[str], Any] | None = None,
    clubelo_loader: Callable[[str], pd.DataFrame] | None = None,
    max_statsbomb_matches: int | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    spec = get_sim_data_source_spec(source_id)
    output_dir = _sim_output_dir(settings)
    database_path = Path(db_path) if db_path else default_football_sim_db_path(settings)
    connection = init_football_sim_db(database_path)
    fetched_at = _utcnow().isoformat()
    inserted = 0
    failures: list[dict[str, str]] = []

    try:
        if spec.source_id == "football_data":
            inserted, failures = _collect_football_data_raw(
                connection=connection,
                spec=spec,
                leagues=tuple(leagues),
                seasons=tuple(seasons),
                profile=profile,
                seasons_back=seasons_back,
                fetched_at=fetched_at,
                football_data_loader=football_data_loader,
            )
        elif spec.source_id == "statsbomb_open_data":
            inserted, failures = _collect_statsbomb_raw(
                connection=connection,
                spec=spec,
                profile=profile or "open_data_all",
                fetched_at=fetched_at,
                json_loader=statsbomb_json_loader,
                max_matches=max_statsbomb_matches,
            )
        elif spec.source_id == "clubelo":
            inserted, failures = _collect_clubelo_raw(
                connection=connection,
                spec=spec,
                fetched_at=fetched_at,
                teams=teams,
                clubelo_loader=clubelo_loader,
            )
        else:
            failures.append(
                {"source_id": spec.source_id, "source_key": "", "failure_reason": f"connector_status={spec.status}"}
            )
        if probe_missing:
            _store_fetch_failures(connection, spec.source_id, failures, fetched_at)
        connection.commit()
        payload = _build_sim_data_report_payload(settings=settings, connection=connection, database_path=database_path)
    finally:
        connection.close()

    summary = dict(payload["summary"])
    summary.update(
        {
            "operation": "collect_sim_data",
            "source_id": spec.source_id,
            "requested_leagues": list(leagues),
            "requested_seasons": list(seasons),
            "profile": profile,
            "seasons_back": seasons_back,
            "probe_missing": bool(probe_missing),
            "raw_payloads_inserted": inserted,
            "failures": failures,
            "picks_emitidos": 0,
            "global_roi_actionable": False,
        }
    )
    payload["summary"] = summary
    artifacts = _write_sim_data_reports(output_dir, payload)
    return summary, artifacts


def _collect_football_data_raw(
    connection: sqlite3.Connection,
    spec: SimulationDataSourceSpec,
    leagues: tuple[str, ...],
    seasons: tuple[str, ...],
    profile: str | None,
    seasons_back: int | None,
    fetched_at: str,
    football_data_loader: Callable[[str, str], pd.DataFrame] | None,
) -> tuple[int, list[dict[str, str]]]:
    selected_leagues = tuple(FOOTBALL_DATA_MAX_FREE_LEAGUES) if profile == "max_free_v1" else tuple(leagues)
    selected_seasons = _season_codes_back(seasons_back) if seasons_back else tuple(seasons)
    client = FootballDataClient()
    inserted = 0
    failures: list[dict[str, str]] = []

    for league in selected_leagues:
        for season in selected_seasons:
            source_key = f"{league}/{season}"
            try:
                frame = (
                    football_data_loader(str(league), str(season))
                    if football_data_loader is not None
                    else client.load_one(str(league), str(season))
                )
                if frame.empty:
                    raise ValueError("empty_payload")
                if "league_name" not in frame.columns:
                    frame["league_name"] = FOOTBALL_DATA_MAX_FREE_LEAGUES.get(str(league), LEAGUE_NAMES.get(str(league), str(league)))
                csv_text = frame.to_csv(index=False)
                source_url = str(frame.get("source_url", pd.Series([spec.source_url])).dropna().iloc[0])
                _insert_raw_payload(
                    connection=connection,
                    spec=spec,
                    source_url=source_url,
                    source_key=source_key,
                    fetched_at=fetched_at,
                    schema_version="football_data_csv_v1",
                    payload_format="csv",
                    content_text=csv_text,
                    row_count=int(len(frame)),
                    license_status="registered_free_public",
                )
                inserted += 1
            except Exception as exc:  # pragma: no cover - depends on network/source availability
                failures.append({"source_id": spec.source_id, "source_key": source_key, "failure_reason": str(exc)})
    return inserted, failures


def _collect_statsbomb_raw(
    connection: sqlite3.Connection,
    spec: SimulationDataSourceSpec,
    profile: str,
    fetched_at: str,
    json_loader: Callable[[str], Any] | None,
    max_matches: int | None,
) -> tuple[int, list[dict[str, str]]]:
    inserted = 0
    failures: list[dict[str, str]] = []
    try:
        competitions = _load_statsbomb_json("competitions.json", json_loader=json_loader)
        _insert_json_payload(
            connection=connection,
            spec=spec,
            source_url=f"{STATSBOMB_RAW_BASE_URL}/competitions.json",
            source_key="competitions",
            fetched_at=fetched_at,
            schema_version="statsbomb_competitions_json_v1",
            payload=competitions,
            license_status="statsbomb_open_data_attribution_required",
        )
        inserted += 1
    except Exception as exc:
        return inserted, [{"source_id": spec.source_id, "source_key": "competitions", "failure_reason": str(exc)}]

    match_ids: list[str] = []
    for competition in competitions if isinstance(competitions, list) else []:
        competition_id = str(competition.get("competition_id", ""))
        season_id = str(competition.get("season_id", ""))
        if not competition_id or not season_id:
            continue
        source_key = f"matches/{competition_id}/{season_id}"
        try:
            matches = _load_statsbomb_json(f"matches/{competition_id}/{season_id}.json", json_loader=json_loader)
            _insert_json_payload(
                connection=connection,
                spec=spec,
                source_url=f"{STATSBOMB_RAW_BASE_URL}/matches/{competition_id}/{season_id}.json",
                source_key=source_key,
                fetched_at=fetched_at,
                schema_version="statsbomb_matches_json_v1",
                payload=matches,
                license_status="statsbomb_open_data_attribution_required",
            )
            inserted += 1
            for match in matches if isinstance(matches, list) else []:
                match_id = str(match.get("match_id", ""))
                if match_id:
                    match_ids.append(match_id)
        except Exception as exc:
            failures.append({"source_id": spec.source_id, "source_key": source_key, "failure_reason": str(exc)})

    selected_match_ids = match_ids[:max_matches] if max_matches else match_ids
    if profile not in {"open_data_all", "open_data_matches_only"}:
        selected_match_ids = selected_match_ids[:25]
    if profile == "open_data_matches_only":
        return inserted, failures

    for match_id in selected_match_ids:
        for folder, schema_version in (
            ("lineups", "statsbomb_lineups_json_v1"),
            ("events", "statsbomb_events_json_v1"),
            ("three-sixty", "statsbomb_360_json_v1"),
        ):
            source_key = f"{folder}/{match_id}"
            try:
                payload = _load_statsbomb_json(f"{folder}/{match_id}.json", json_loader=json_loader)
                _insert_json_payload(
                    connection=connection,
                    spec=spec,
                    source_url=f"{STATSBOMB_RAW_BASE_URL}/{folder}/{match_id}.json",
                    source_key=source_key,
                    fetched_at=fetched_at,
                    schema_version=schema_version,
                    payload=payload,
                    license_status="statsbomb_open_data_attribution_required",
                )
                inserted += 1
            except Exception as exc:
                if folder != "three-sixty":
                    failures.append({"source_id": spec.source_id, "source_key": source_key, "failure_reason": str(exc)})
    return inserted, failures


def _collect_clubelo_raw(
    connection: sqlite3.Connection,
    spec: SimulationDataSourceSpec,
    fetched_at: str,
    teams: str,
    clubelo_loader: Callable[[str], pd.DataFrame] | None,
) -> tuple[int, list[dict[str, str]]]:
    rows = connection.execute(
        "SELECT team_id, canonical_name FROM sim_teams ORDER BY canonical_name"
    ).fetchall()
    if teams != "mapped":
        rows = []
    inserted = 0
    failures: list[dict[str, str]] = []
    if not rows:
        return 0, [{"source_id": spec.source_id, "source_key": "teams:mapped", "failure_reason": "no_mapped_teams"}]
    session = requests.Session()
    session.headers.update({"User-Agent": "predicciones-football-sim/0.1"})

    for row in rows:
        team_id = str(row["team_id"])
        team_name = str(row["canonical_name"])
        source_key = f"clubelo/{team_id}/{team_name}"
        try:
            frame = clubelo_loader(team_name) if clubelo_loader else _download_clubelo_team(session, team_name)
            if frame.empty:
                raise ValueError("empty_payload")
            frame["team_id"] = team_id
            frame["canonical_name"] = team_name
            csv_text = frame.to_csv(index=False)
            _insert_raw_payload(
                connection=connection,
                spec=spec,
                source_url=f"{CLUBELO_API_BASE_URL}/{quote(_clubelo_candidate_names(team_name)[0])}",
                source_key=source_key,
                fetched_at=fetched_at,
                schema_version="clubelo_csv_v1",
                payload_format="csv",
                content_text=csv_text,
                row_count=int(len(frame)),
                license_status="registered_public_license_audit_required",
            )
            inserted += 1
        except Exception as exc:  # pragma: no cover - depends on network/source availability
            failures.append({"source_id": spec.source_id, "source_key": source_key, "failure_reason": str(exc)})
    return inserted, failures


def _insert_raw_payload(
    connection: sqlite3.Connection,
    spec: SimulationDataSourceSpec,
    source_url: str,
    source_key: str,
    fetched_at: str,
    schema_version: str,
    payload_format: str,
    content_text: str,
    row_count: int,
    license_status: str,
    fetch_status: str = "success",
    failure_reason: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO sim_raw_payloads (
            content_hash, source_id, source_url, source_key, fetched_at,
            schema_version, payload_format, content_text, row_count,
            fetch_status, failure_reason, license_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _content_hash(content_text),
            spec.source_id,
            source_url,
            source_key,
            fetched_at,
            schema_version,
            payload_format,
            content_text,
            int(row_count),
            fetch_status,
            failure_reason,
            license_status,
        ),
    )


def _insert_json_payload(
    connection: sqlite3.Connection,
    spec: SimulationDataSourceSpec,
    source_url: str,
    source_key: str,
    fetched_at: str,
    schema_version: str,
    payload: Any,
    license_status: str,
) -> None:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    row_count = len(payload) if isinstance(payload, list) else 1
    _insert_raw_payload(
        connection=connection,
        spec=spec,
        source_url=source_url,
        source_key=source_key,
        fetched_at=fetched_at,
        schema_version=schema_version,
        payload_format="json",
        content_text=text,
        row_count=row_count,
        license_status=license_status,
    )


def _store_fetch_failures(
    connection: sqlite3.Connection,
    source_id: str,
    failures: list[dict[str, str]],
    fetched_at: str,
) -> None:
    spec = get_sim_data_source_spec(source_id)
    for failure in failures:
        source_key = failure.get("source_key", "")
        content = json.dumps(failure, sort_keys=True)
        _insert_raw_payload(
            connection=connection,
            spec=spec,
            source_url=spec.source_url,
            source_key=f"failed/{source_key}",
            fetched_at=fetched_at,
            schema_version="fetch_failure_v1",
            payload_format="json",
            content_text=content,
            row_count=0,
            license_status="not_applicable",
            fetch_status="fetch_failed",
            failure_reason=failure.get("failure_reason"),
        )


def _load_statsbomb_json(path: str, json_loader: Callable[[str], Any] | None = None) -> Any:
    if json_loader is not None:
        return json_loader(path)
    response = requests.get(f"{STATSBOMB_RAW_BASE_URL}/{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _download_clubelo_team(session: requests.Session, team_name: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for candidate in _clubelo_candidate_names(team_name):
        try:
            response = session.get(f"{CLUBELO_API_BASE_URL}/{quote(candidate)}", timeout=30)
            response.raise_for_status()
            frame = pd.read_csv(io.StringIO(response.text))
            if not frame.empty and "Elo" in frame.columns:
                frame["clubelo_query"] = candidate
                return frame
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(last_error) if last_error else "clubelo_mapping_failed")


def _clubelo_candidate_names(team_name: str) -> list[str]:
    base = normalize_team_name(team_name)
    stripped = re.sub(r"\b(fc|cf|afc|sc|club)\b", "", base, flags=re.IGNORECASE)
    candidates = [
        base.replace(" ", ""),
        stripped.replace(" ", ""),
        _stable_slug(base).replace("_", ""),
    ]
    return [candidate for idx, candidate in enumerate(candidates) if candidate and candidate not in candidates[:idx]]


def _season_codes_back(count: int | None) -> tuple[str, ...]:
    if not count or count <= 0:
        return ()
    now = pd.Timestamp.utcnow()
    start_year = int(now.year if now.month >= 7 else now.year - 1)
    seasons = []
    for offset in range(int(count)):
        year = start_year - offset
        seasons.append(f"{year % 100:02d}{(year + 1) % 100:02d}")
    return tuple(seasons)


def normalize_sim_data(settings: Settings, db_path: Path | str | None = None) -> tuple[dict[str, Any], dict[str, Path]]:
    output_dir = _sim_output_dir(settings)
    database_path = Path(db_path) if db_path else default_football_sim_db_path(settings)
    connection = init_football_sim_db(database_path)
    now = _utcnow().isoformat()
    failures: list[dict[str, str]] = []

    try:
        raw_rows = connection.execute(
            """
            SELECT source_id, source_url, source_key, schema_version, payload_format, content_text
            FROM sim_raw_payloads
            WHERE fetch_status = 'success'
            ORDER BY source_id, source_key
            """
        ).fetchall()
        raw_rows = sorted(raw_rows, key=_normalization_priority)
        frames: list[pd.DataFrame] = []
        for row in raw_rows:
            try:
                if row["source_id"] == "football_data" and row["payload_format"] == "csv":
                    frame = pd.read_csv(io.StringIO(row["content_text"]))
                    frame["source_id"] = row["source_id"]
                    if "source_url" not in frame.columns:
                        frame["source_url"] = row["source_url"]
                    frames.append(frame)
                elif row["source_id"] == "statsbomb_open_data" and row["payload_format"] == "json":
                    _normalize_statsbomb_payload(connection, row, now)
                elif row["source_id"] == "clubelo" and row["payload_format"] == "csv":
                    _normalize_clubelo_payload(connection, row)
            except Exception as exc:
                failures.append(
                    {
                        "source_id": row["source_id"],
                        "source_key": row["source_key"],
                        "failure_reason": str(exc),
                    }
                )

        if frames:
            raw_matches = pd.concat(frames, ignore_index=True)
            canonical = canonicalize_matches(raw_matches, require_results=True)
            _replace_silver_from_canonical(connection, canonical, now)
        connection.commit()
        payload = _build_sim_data_report_payload(settings=settings, connection=connection, database_path=database_path)
    finally:
        connection.close()

    summary = dict(payload["summary"])
    summary.update(
        {
            "operation": "normalize_sim_data",
            "normalization_failures": failures,
            "picks_emitidos": 0,
            "global_roi_actionable": False,
        }
    )
    payload["summary"] = summary
    artifacts = _write_sim_data_reports(output_dir, payload)
    return summary, artifacts


def _replace_silver_from_canonical(connection: sqlite3.Connection, canonical: pd.DataFrame, updated_at: str) -> None:
    connection.execute("DELETE FROM sim_matches WHERE source_id = 'football_data'")
    connection.execute("DELETE FROM sim_team_aliases WHERE source_id = 'football_data'")

    team_rows: dict[str, dict[str, Any]] = {}
    alias_rows: dict[str, dict[str, Any]] = {}
    match_rows: list[tuple[Any, ...]] = []

    for row in canonical.itertuples(index=False):
        home_name = normalize_team_name(row.HomeTeam)
        away_name = normalize_team_name(row.AwayTeam)
        home_id = _stable_id("team", home_name)
        away_id = _stable_id("team", away_name)
        for team_id, team_name in ((home_id, home_name), (away_id, away_name)):
            team_rows[team_id] = {
                "team_id": team_id,
                "canonical_name": team_name,
                "created_at": updated_at,
            }
            alias_id = _stable_id("alias", "football_data", team_name)
            alias_rows[alias_id] = {
                "alias_id": alias_id,
                "team_id": team_id,
                "source_id": "football_data",
                "alias_name": team_name,
                "canonical_name": team_name,
                "confidence": 1.0,
                "mapping_status": "active",
            }

        match_date = pd.Timestamp(row.Date).date().isoformat()
        match_id = _stable_id("match", row.league_code, row.season, match_date, home_name, away_name)
        source_key = f"{row.league_code}/{row.season}/{match_date}/{home_name}/{away_name}"
        match_rows.append(
            (
                match_id,
                "football_data",
                source_key,
                match_date,
                str(row.league_code),
                str(row.league_name),
                str(row.season),
                home_id,
                away_id,
                home_name,
                away_name,
                _to_float(getattr(row, "FTHG", np.nan)),
                _to_float(getattr(row, "FTAG", np.nan)),
                str(getattr(row, "FTR", "")),
                str(getattr(row, "outcome", "")),
                _to_float(getattr(row, "HS", np.nan)),
                _to_float(getattr(row, "AS", np.nan)),
                _to_float(getattr(row, "HST", np.nan)),
                _to_float(getattr(row, "AST", np.nan)),
                _to_float(getattr(row, "HC", np.nan)),
                _to_float(getattr(row, "AC", np.nan)),
                _to_float(getattr(row, "HY", np.nan)),
                _to_float(getattr(row, "AY", np.nan)),
                _to_float(getattr(row, "HR", np.nan)),
                _to_float(getattr(row, "AR", np.nan)),
                _to_float(getattr(row, "B365H", np.nan)),
                _to_float(getattr(row, "B365D", np.nan)),
                _to_float(getattr(row, "B365A", np.nan)),
                str(getattr(row, "source_url", "")),
                updated_at,
            )
        )

    connection.executemany(
        "INSERT OR REPLACE INTO sim_teams (team_id, canonical_name, created_at) VALUES (:team_id, :canonical_name, :created_at)",
        list(team_rows.values()),
    )
    connection.executemany(
        """
        INSERT OR REPLACE INTO sim_team_aliases (
            alias_id, team_id, source_id, alias_name, canonical_name, confidence, mapping_status
        ) VALUES (:alias_id, :team_id, :source_id, :alias_name, :canonical_name, :confidence, :mapping_status)
        """,
        list(alias_rows.values()),
    )
    connection.executemany(
        """
        INSERT OR REPLACE INTO sim_matches (
            match_id, source_id, source_match_key, match_date, league_code, league_name, season,
            home_team_id, away_team_id, home_team_name, away_team_name, home_goals, away_goals,
            result_code, outcome, home_shots, away_shots, home_shots_target, away_shots_target,
            home_corners, away_corners, home_yellow_cards, away_yellow_cards, home_red_cards,
            away_red_cards, odds_home, odds_draw, odds_away, source_url, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        match_rows,
    )
    connection.commit()


def _normalize_statsbomb_payload(connection: sqlite3.Connection, row: sqlite3.Row, updated_at: str) -> None:
    payload = json.loads(row["content_text"])
    source_key = str(row["source_key"])
    if source_key == "competitions":
        competition_rows = []
        for item in payload if isinstance(payload, list) else []:
            competition_id = str(item.get("competition_id", ""))
            season_id = str(item.get("season_id", ""))
            if not competition_id or not season_id:
                continue
            competition_rows.append(
                {
                    "competition_key": _stable_id("comp", "statsbomb", competition_id, season_id),
                    "source_id": "statsbomb_open_data",
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "country_name": item.get("country_name"),
                    "competition_name": item.get("competition_name") or f"StatsBomb {competition_id}",
                    "season_name": item.get("season_name"),
                    "gender": item.get("competition_gender"),
                    "match_available": item.get("match_available"),
                    "match_available_360": item.get("match_available_360"),
                }
            )
        connection.executemany(
            """
            INSERT OR REPLACE INTO sim_competitions (
                competition_key, source_id, competition_id, season_id, country_name,
                competition_name, season_name, gender, match_available, match_available_360
            ) VALUES (
                :competition_key, :source_id, :competition_id, :season_id, :country_name,
                :competition_name, :season_name, :gender, :match_available, :match_available_360
            )
            """,
            competition_rows,
        )
        return

    if source_key.startswith("matches/"):
        parts = source_key.split("/")
        competition_id = parts[1] if len(parts) > 1 else ""
        season_id = parts[2] if len(parts) > 2 else ""
        comp = connection.execute(
            "SELECT * FROM sim_competitions WHERE competition_id = ? AND season_id = ?",
            (competition_id, season_id),
        ).fetchone()
        competition_name = comp["competition_name"] if comp else f"StatsBomb {competition_id}"
        season_name = comp["season_name"] if comp else f"SB_{season_id}"
        country_name = comp["country_name"] if comp else ""
        for item in payload if isinstance(payload, list) else []:
            source_match_id = str(item.get("match_id", ""))
            home = item.get("home_team") or {}
            away = item.get("away_team") or {}
            home_name = normalize_team_name(home.get("home_team_name") or home.get("name") or home.get("team_name") or "")
            away_name = normalize_team_name(away.get("away_team_name") or away.get("name") or away.get("team_name") or "")
            match_date = str(item.get("match_date") or "")[:10]
            if not source_match_id or not home_name or not away_name or not match_date:
                continue
            home_id = _upsert_team_alias(connection, "statsbomb_open_data", home_name, updated_at)
            away_id = _upsert_team_alias(connection, "statsbomb_open_data", away_name, updated_at)
            match_id = _match_id_for_source_match(
                connection=connection,
                source_match_id=source_match_id,
                match_date=match_date,
                home_name=home_name,
                away_name=away_name,
                fallback_league=f"SB_{competition_id}",
                season=str(season_name or f"SB_{season_id}"),
            )
            source_match_key = f"statsbomb/{source_match_id}"
            connection.execute(
                """
                INSERT OR REPLACE INTO sim_matches (
                    match_id, source_id, source_match_key, match_date, league_code, league_name, season,
                    home_team_id, away_team_id, home_team_name, away_team_name, home_goals, away_goals,
                    result_code, outcome, source_url, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    "statsbomb_open_data",
                    source_match_key,
                    match_date,
                    f"SB_{competition_id}",
                    str(competition_name),
                    str(season_name or f"SB_{season_id}"),
                    home_id,
                    away_id,
                    home_name,
                    away_name,
                    _to_float(item.get("home_score")),
                    _to_float(item.get("away_score")),
                    _result_code_from_goals(item.get("home_score"), item.get("away_score")),
                    _outcome_from_goals(item.get("home_score"), item.get("away_score")),
                    row["source_url"],
                    updated_at,
                ),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO sim_match_source_links (
                    link_id, source_id, source_match_id, match_id, mapping_status, confidence
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _stable_id("link", "statsbomb", source_match_id),
                    "statsbomb_open_data",
                    source_match_id,
                    match_id,
                    "active_source_match",
                    1.0,
                ),
            )
        return

    if source_key.startswith("lineups/"):
        source_match_id = source_key.split("/")[-1]
        match_id = _match_id_from_source_link(connection, "statsbomb_open_data", source_match_id)
        if not match_id:
            return
        for team_item in payload if isinstance(payload, list) else []:
            team_name = normalize_team_name(team_item.get("team_name") or team_item.get("team", {}).get("name") or "")
            team_id = _upsert_team_alias(connection, "statsbomb_open_data", team_name, updated_at) if team_name else ""
            for player in team_item.get("lineup", []) if isinstance(team_item.get("lineup", []), list) else []:
                player_name = normalize_team_name(player.get("player_name") or player.get("name") or "")
                source_player_id = str(player.get("player_id") or "")
                player_id = _upsert_player(connection, "statsbomb_open_data", source_player_id, player_name, team_id)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO sim_lineups (
                        lineup_id, match_id, team_id, player_id, player_name, known_before_match, source_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _stable_id("lineup", "statsbomb", match_id, team_id, player_id),
                        match_id,
                        team_id,
                        player_id,
                        player_name,
                        0,
                        "statsbomb_open_data",
                    ),
                )
        return

    if source_key.startswith("events/"):
        source_match_id = source_key.split("/")[-1]
        match_id = _match_id_from_source_link(connection, "statsbomb_open_data", source_match_id)
        if not match_id:
            return
        for item in payload if isinstance(payload, list) else []:
            event_source_id = str(item.get("id") or item.get("index") or "")
            team_name = normalize_team_name((item.get("team") or {}).get("name") or "")
            player_name = normalize_team_name((item.get("player") or {}).get("name") or "")
            team_id = _upsert_team_alias(connection, "statsbomb_open_data", team_name, updated_at) if team_name else None
            source_player_id = str((item.get("player") or {}).get("id") or "")
            player_id = _upsert_player(connection, "statsbomb_open_data", source_player_id, player_name, team_id) if player_name else None
            event_type = (item.get("type") or {}).get("name") if isinstance(item.get("type"), dict) else item.get("type")
            connection.execute(
                """
                INSERT OR REPLACE INTO sim_events (
                    event_id, match_id, team_id, player_id, event_type, minute, source_id, known_before_match
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _stable_id("event", "statsbomb", match_id, event_source_id),
                    match_id,
                    team_id,
                    player_id,
                    str(event_type or ""),
                    _to_float(item.get("minute")),
                    "statsbomb_open_data",
                    0,
                ),
            )


def _normalization_priority(row: sqlite3.Row) -> tuple[int, str, str]:
    source_id = str(row["source_id"])
    source_key = str(row["source_key"])
    if source_id == "football_data":
        return (0, source_id, source_key)
    if source_id == "statsbomb_open_data" and source_key == "competitions":
        return (1, source_id, source_key)
    if source_id == "statsbomb_open_data" and source_key.startswith("matches/"):
        return (2, source_id, source_key)
    if source_id == "statsbomb_open_data":
        return (3, source_id, source_key)
    if source_id == "clubelo":
        return (4, source_id, source_key)
    return (9, source_id, source_key)


def _normalize_clubelo_payload(connection: sqlite3.Connection, row: sqlite3.Row) -> None:
    frame = pd.read_csv(io.StringIO(row["content_text"]))
    if frame.empty or "Elo" not in frame.columns:
        return
    team_id = str(frame.get("team_id", pd.Series([""])).iloc[0])
    canonical_name = str(frame.get("canonical_name", pd.Series([""])).iloc[0])
    rows = []
    for _, item in frame.iterrows():
        date_value = item.get("From") if "From" in frame.columns else item.get("Date")
        rating_date = pd.to_datetime(date_value, errors="coerce")
        elo = _to_float(item.get("Elo"))
        if not team_id or pd.isna(rating_date) or not np.isfinite(elo):
            continue
        source_team_name = str(item.get("Club", canonical_name))
        rows.append(
            {
                "rating_id": _stable_id("rating", "clubelo", team_id, rating_date.date().isoformat()),
                "team_id": team_id,
                "source_id": "clubelo",
                "rating_date": rating_date.date().isoformat(),
                "rating_value": float(elo),
                "rating_type": "clubelo",
                "mapping_status": "active" if source_team_name else "clubelo_mapping_failed",
                "source_team_name": source_team_name,
            }
        )
    connection.executemany(
        """
        INSERT OR REPLACE INTO sim_team_ratings (
            rating_id, team_id, source_id, rating_date, rating_value,
            rating_type, mapping_status, source_team_name
        ) VALUES (
            :rating_id, :team_id, :source_id, :rating_date, :rating_value,
            :rating_type, :mapping_status, :source_team_name
        )
        """,
        rows,
    )


def _upsert_team_alias(connection: sqlite3.Connection, source_id: str, team_name: str, updated_at: str) -> str:
    canonical = normalize_team_name(team_name)
    team_id = _stable_id("team", canonical)
    connection.execute(
        "INSERT OR REPLACE INTO sim_teams (team_id, canonical_name, created_at) VALUES (?, ?, ?)",
        (team_id, canonical, updated_at),
    )
    connection.execute(
        """
        INSERT OR REPLACE INTO sim_team_aliases (
            alias_id, team_id, source_id, alias_name, canonical_name, confidence, mapping_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (_stable_id("alias", source_id, canonical), team_id, source_id, canonical, canonical, 1.0, "active"),
    )
    return team_id


def _upsert_player(
    connection: sqlite3.Connection,
    source_id: str,
    source_player_id: str,
    player_name: str,
    team_id: str | None,
) -> str:
    player_id = _stable_id("player", source_id, source_player_id or player_name)
    connection.execute(
        """
        INSERT OR REPLACE INTO sim_players (
            player_id, source_id, source_player_id, player_name, team_id, canonical_name
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (player_id, source_id, source_player_id, player_name, team_id, normalize_team_name(player_name)),
    )
    return player_id


def _match_id_from_source_link(connection: sqlite3.Connection, source_id: str, source_match_id: str) -> str | None:
    row = connection.execute(
        "SELECT match_id FROM sim_match_source_links WHERE source_id = ? AND source_match_id = ?",
        (source_id, source_match_id),
    ).fetchone()
    return str(row["match_id"]) if row else None


def _match_id_for_source_match(
    connection: sqlite3.Connection,
    source_match_id: str,
    match_date: str,
    home_name: str,
    away_name: str,
    fallback_league: str,
    season: str,
) -> str:
    existing = connection.execute(
        """
        SELECT match_id FROM sim_matches
        WHERE match_date = ? AND home_team_name = ? AND away_team_name = ?
        LIMIT 1
        """,
        (match_date, home_name, away_name),
    ).fetchone()
    if existing:
        return str(existing["match_id"])
    return _stable_id("match", fallback_league, season, match_date, home_name, away_name, source_match_id)


def _result_code_from_goals(home: Any, away: Any) -> str:
    home_goals = _to_float(home)
    away_goals = _to_float(away)
    if not np.isfinite(home_goals) or not np.isfinite(away_goals):
        return ""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _outcome_from_goals(home: Any, away: Any) -> str:
    return {"H": "home", "A": "away", "D": "draw"}.get(_result_code_from_goals(home, away), "")


def build_sim_features(
    settings: Settings,
    as_of_date: str | None = None,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    output_dir = _sim_output_dir(settings)
    database_path = Path(db_path) if db_path else default_football_sim_db_path(settings)
    connection = init_football_sim_db(database_path)

    try:
        matches = pd.read_sql_query("SELECT * FROM sim_matches ORDER BY match_date, league_code, home_team_name", connection)
        ratings = pd.read_sql_query("SELECT * FROM sim_team_ratings ORDER BY rating_date", connection)
        events = pd.read_sql_query("SELECT * FROM sim_events", connection)
        if as_of_date and not matches.empty:
            cutoff = pd.Timestamp(as_of_date).date().isoformat()
            matches = matches[matches["match_date"] <= cutoff].copy()
        features = _build_gold_feature_frame(matches, ratings=ratings, events=events)
        connection.execute("DELETE FROM sim_gold_features")
        if not features.empty:
            features.to_sql("sim_gold_features", connection, if_exists="append", index=False)
        connection.commit()
        payload = _build_sim_data_report_payload(settings=settings, connection=connection, database_path=database_path)
    finally:
        connection.close()

    summary = dict(payload["summary"])
    summary.update(
        {
            "operation": "build_sim_features",
            "as_of_date": as_of_date,
            "gold_feature_rows": int(len(features)),
            "picks_emitidos": 0,
            "global_roi_actionable": False,
        }
    )
    payload["summary"] = summary
    artifacts = _write_sim_data_reports(output_dir, payload)
    return summary, artifacts


def _build_gold_feature_frame(
    matches: pd.DataFrame,
    ratings: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()

    working = matches.copy()
    working["match_date"] = pd.to_datetime(working["match_date"], errors="coerce")
    working = working.dropna(subset=["match_date", "home_team_id", "away_team_id"]).sort_values(
        ["match_date", "league_code", "home_team_name", "away_team_name"]
    )

    team_history: dict[str, list[dict[str, Any]]] = {}
    event_history = _statsbomb_event_history(matches=working, events=events if events is not None else pd.DataFrame())
    ratings_by_team = _ratings_by_team(ratings if ratings is not None else pd.DataFrame())
    elo: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    families = ("team_form", "team_strength", "schedule_context", "market_reference", "tactical_event_profile")

    for match in working.itertuples(index=False):
        match_start = pd.Timestamp(match.match_date).normalize() + pd.Timedelta(hours=15)
        as_of_time = match_start - pd.Timedelta(minutes=45)
        home_id = str(match.home_team_id)
        away_id = str(match.away_team_id)
        home_history = team_history.get(home_id, [])
        away_history = team_history.get(away_id, [])
        home_elo = float(elo.get(home_id, 1500.0))
        away_elo = float(elo.get(away_id, 1500.0))
        home_clubelo, home_clubelo_status = _rating_before(ratings_by_team, home_id, as_of_time)
        away_clubelo, away_clubelo_status = _rating_before(ratings_by_team, away_id, as_of_time)
        clubelo_status = (
            "active"
            if home_clubelo_status == "active" and away_clubelo_status == "active"
            else "clubelo_mapping_failed"
        )
        home_event_history = _events_before(event_history.get(home_id, []), match_start)
        away_event_history = _events_before(event_history.get(away_id, []), match_start)
        odds = [_to_float(getattr(match, "odds_home", np.nan)), _to_float(getattr(match, "odds_draw", np.nan)), _to_float(getattr(match, "odds_away", np.nan))]
        market_probs = _market_probs_from_odds(odds)

        row = {
            "match_id": match.match_id,
            "as_of_time": as_of_time.isoformat(),
            "match_start_time": match_start.isoformat(),
            "known_before_match": 1,
            "league_code": match.league_code,
            "season": match.season,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_name": match.home_team_name,
            "away_team_name": match.away_team_name,
            "feature_role": "simulation_ready_without_market_reference",
            "feature_family_set": json.dumps(list(families), ensure_ascii=True),
            "home_matches_played_pre": len(home_history),
            "away_matches_played_pre": len(away_history),
            "home_points_last_5": _sum_recent(home_history, "points", 5),
            "away_points_last_5": _sum_recent(away_history, "points", 5),
            "home_points_last_10": _sum_recent(home_history, "points", 10),
            "away_points_last_10": _sum_recent(away_history, "points", 10),
            "home_points_per_match_last_5": _mean_recent(home_history, "points", 5),
            "away_points_per_match_last_5": _mean_recent(away_history, "points", 5),
            "home_points_per_match_last_10": _mean_recent(home_history, "points", 10),
            "away_points_per_match_last_10": _mean_recent(away_history, "points", 10),
            "points_form_diff_5": _mean_recent(home_history, "points", 5) - _mean_recent(away_history, "points", 5),
            "points_form_diff_10": _mean_recent(home_history, "points", 10) - _mean_recent(away_history, "points", 10),
            "home_goals_for_avg_last_5": _mean_recent(home_history, "goals_for", 5),
            "away_goals_for_avg_last_5": _mean_recent(away_history, "goals_for", 5),
            "home_goals_against_avg_last_5": _mean_recent(home_history, "goals_against", 5),
            "away_goals_against_avg_last_5": _mean_recent(away_history, "goals_against", 5),
            "home_internal_elo_pre": home_elo,
            "away_internal_elo_pre": away_elo,
            "internal_elo_diff": home_elo - away_elo,
            "home_rest_days": _rest_days(home_history, match_start),
            "away_rest_days": _rest_days(away_history, match_start),
            "rest_advantage": _rest_days(home_history, match_start) - _rest_days(away_history, match_start),
            "home_matches_last_7d": _matches_within_days(home_history, match_start, 7),
            "away_matches_last_7d": _matches_within_days(away_history, match_start, 7),
            "home_matches_last_14d": _matches_within_days(home_history, match_start, 14),
            "away_matches_last_14d": _matches_within_days(away_history, match_start, 14),
            "market_reference_present": int(all(np.isfinite(value) and value > 0 for value in odds)),
            "odds_home": odds[0],
            "odds_draw": odds[1],
            "odds_away": odds[2],
            "market_prob_home": market_probs[0],
            "market_prob_draw": market_probs[1],
            "market_prob_away": market_probs[2],
            "home_clubelo_pre": home_clubelo if np.isfinite(home_clubelo) else home_elo,
            "away_clubelo_pre": away_clubelo if np.isfinite(away_clubelo) else away_elo,
            "clubelo_diff": (home_clubelo if np.isfinite(home_clubelo) else home_elo)
            - (away_clubelo if np.isfinite(away_clubelo) else away_elo),
            "clubelo_mapping_status": clubelo_status,
            "home_statsbomb_event_count_avg_last_5": _mean_recent(home_event_history, "events", 5),
            "away_statsbomb_event_count_avg_last_5": _mean_recent(away_event_history, "events", 5),
            "home_statsbomb_shot_count_avg_last_5": _mean_recent(home_event_history, "shots", 5),
            "away_statsbomb_shot_count_avg_last_5": _mean_recent(away_event_history, "shots", 5),
        }
        rows.append(row)

        home_goals = _to_float(getattr(match, "home_goals", np.nan))
        away_goals = _to_float(getattr(match, "away_goals", np.nan))
        if np.isfinite(home_goals) and np.isfinite(away_goals):
            if home_goals > away_goals:
                home_points, away_points = 3.0, 0.0
                home_score = 1.0
            elif home_goals < away_goals:
                home_points, away_points = 0.0, 3.0
                home_score = 0.0
            else:
                home_points, away_points = 1.0, 1.0
                home_score = 0.5
            away_score = 1.0 - home_score
            expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))
            expected_away = 1.0 - expected_home
            k = 20.0
            elo[home_id] = home_elo + k * (home_score - expected_home)
            elo[away_id] = away_elo + k * (away_score - expected_away)
            team_history.setdefault(home_id, []).append(
                {
                    "date": match_start,
                    "points": home_points,
                    "goals_for": home_goals,
                    "goals_against": away_goals,
                }
            )
            team_history.setdefault(away_id, []).append(
                {
                    "date": match_start,
                    "points": away_points,
                    "goals_for": away_goals,
                    "goals_against": home_goals,
                }
            )

    return pd.DataFrame(rows)


def _ratings_by_team(ratings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if ratings.empty or "team_id" not in ratings.columns:
        return {}
    working = ratings.copy()
    working["rating_date"] = pd.to_datetime(working["rating_date"], errors="coerce")
    working = working.dropna(subset=["rating_date", "rating_value"])
    return {str(team_id): group.sort_values("rating_date") for team_id, group in working.groupby("team_id")}


def _rating_before(ratings_by_team: dict[str, pd.DataFrame], team_id: str, as_of_time: pd.Timestamp) -> tuple[float, str]:
    frame = ratings_by_team.get(str(team_id))
    if frame is None or frame.empty:
        return float("nan"), "clubelo_mapping_failed"
    eligible = frame[frame["rating_date"] <= pd.Timestamp(as_of_time)]
    if eligible.empty:
        return float("nan"), "clubelo_mapping_failed"
    value = _to_float(eligible.iloc[-1]["rating_value"])
    return value, "active" if np.isfinite(value) else "clubelo_mapping_failed"


def _statsbomb_event_history(matches: pd.DataFrame, events: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if events.empty or matches.empty:
        return {}
    match_dates = matches.set_index("match_id")["match_date"].to_dict()
    history: dict[str, dict[str, dict[str, Any]]] = {}
    for event in events.itertuples(index=False):
        team_id = str(getattr(event, "team_id", "") or "")
        match_id = str(getattr(event, "match_id", "") or "")
        if not team_id or not match_id or match_id not in match_dates:
            continue
        by_team = history.setdefault(team_id, {})
        item = by_team.setdefault(match_id, {"date": pd.Timestamp(match_dates[match_id]), "events": 0.0, "shots": 0.0})
        item["events"] += 1.0
        if str(getattr(event, "event_type", "")).lower() == "shot":
            item["shots"] += 1.0
    return {
        team_id: sorted(items.values(), key=lambda value: pd.Timestamp(value["date"]))
        for team_id, items in history.items()
    }


def _events_before(history: list[dict[str, Any]], match_start: pd.Timestamp) -> list[dict[str, Any]]:
    return [item for item in history if pd.Timestamp(item.get("date")) < match_start]


def report_sim_data(settings: Settings, db_path: Path | str | None = None) -> tuple[dict[str, Any], str, dict[str, Path]]:
    output_dir = _sim_output_dir(settings)
    database_path = Path(db_path) if db_path else default_football_sim_db_path(settings)
    connection = init_football_sim_db(database_path)
    try:
        payload = _build_sim_data_report_payload(settings=settings, connection=connection, database_path=database_path)
    finally:
        connection.close()
    artifacts = _write_sim_data_reports(output_dir, payload)
    summary = payload["summary"]
    text = "\n".join(
        [
            "Football sim data report",
            f"- database: {database_path}",
            f"- raw_payloads: {summary['raw_payloads']}",
            f"- matches: {summary['matches']}",
            f"- teams: {summary['teams']}",
            f"- gold_feature_rows: {summary['gold_feature_rows']}",
            f"- match_level_status: {summary['match_level_status']}",
            f"- player_adjusted_status: {summary['player_adjusted_status']}",
            f"- event_level_status: {summary['event_level_status']}",
            "- picks_emitidos: 0",
            "- global_roi_actionable: false",
        ]
    )
    return summary, text, artifacts


def export_sim_training_dataset(
    settings: Settings,
    db_path: Path | str | None = None,
    exclude_market_reference: bool = True,
) -> tuple[dict[str, Any], dict[str, Path]]:
    output_dir = _sim_output_dir(settings)
    database_path = Path(db_path) if db_path else default_football_sim_db_path(settings)
    connection = init_football_sim_db(database_path)
    try:
        features = pd.read_sql_query("SELECT * FROM sim_gold_features", connection)
        matches = pd.read_sql_query(
            """
            SELECT match_id, match_date, home_goals, away_goals, outcome
            FROM sim_matches
            """,
            connection,
        )
        payload = _build_sim_data_report_payload(settings=settings, connection=connection, database_path=database_path)
    finally:
        connection.close()

    dataset = features.merge(matches, on="match_id", how="left", suffixes=("", "_target"))
    if not dataset.empty:
        dataset["match_date"] = pd.to_datetime(dataset["match_date"], errors="coerce")
        dataset["home_goals"] = pd.to_numeric(dataset["home_goals"], errors="coerce")
        dataset["away_goals"] = pd.to_numeric(dataset["away_goals"], errors="coerce")
        dataset = dataset.dropna(subset=["home_goals", "away_goals", "match_date"]).copy()
        dataset["total_goals"] = dataset["home_goals"] + dataset["away_goals"]
        dataset["btts"] = ((dataset["home_goals"] > 0) & (dataset["away_goals"] > 0)).astype(int)
        dataset = dataset.sort_values("match_date").reset_index(drop=True)
        dataset["split"] = _temporal_training_splits(len(dataset))
    if exclude_market_reference and not dataset.empty:
        dataset = dataset.drop(columns=[column for column in _market_reference_columns(dataset) if column in dataset.columns])

    dataset_path = output_dir / "simulation_training_dataset.csv"
    manifest_path = output_dir / "simulation_training_manifest.json"
    splits_path = output_dir / "simulation_training_splits.json"
    dataset.to_csv(dataset_path, index=False)

    target_columns = ["home_goals", "away_goals", "outcome", "total_goals", "btts"]
    metadata_columns = {
        "match_id",
        "match_date",
        "match_start_time",
        "as_of_time",
        "known_before_match",
        "league_code",
        "season",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "split",
        "feature_role",
        "feature_family_set",
        *target_columns,
    }
    feature_columns = [column for column in dataset.columns if column not in metadata_columns]
    split_counts = dataset["split"].value_counts().to_dict() if "split" in dataset.columns else {}
    manifest = {
        "generated_at": _utcnow().isoformat(),
        "database_path": str(database_path),
        "dataset_path": str(dataset_path),
        "rows": int(len(dataset)),
        "exclude_market_reference": bool(exclude_market_reference),
        "target_columns": target_columns,
        "feature_columns": feature_columns,
        "market_reference_columns_excluded": _market_reference_columns(features) if exclude_market_reference else [],
        "global_roi_actionable": False,
        "picks_emitidos": 0,
    }
    splits_payload = {
        "generated_at": _utcnow().isoformat(),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "split_policy": "temporal_60_20_20_train_dev_locked_holdout",
    }
    _write_json(manifest_path, manifest)
    _write_json(splits_path, splits_payload)
    artifacts = _write_sim_data_reports(output_dir, payload)
    artifacts.update(
        {
            "simulation_training_dataset": dataset_path,
            "simulation_training_manifest": manifest_path,
            "simulation_training_splits": splits_path,
        }
    )
    summary = dict(payload["summary"])
    summary.update(
        {
            "operation": "export_sim_training_dataset",
            "training_rows": int(len(dataset)),
            "feature_columns": len(feature_columns),
            "exclude_market_reference": bool(exclude_market_reference),
            "picks_emitidos": 0,
            "global_roi_actionable": False,
        }
    )
    return summary, artifacts


def _temporal_training_splits(rows: int) -> list[str]:
    train_end = int(rows * 0.6)
    dev_end = int(rows * 0.8)
    return [
        "train" if idx < train_end else "dev" if idx < dev_end else "locked_holdout"
        for idx in range(rows)
    ]


def _market_reference_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("odds_", "market_prob_")
    explicit = {"market_reference_present"}
    return [column for column in frame.columns if column.startswith(prefixes) or column in explicit]


def _build_sim_data_report_payload(
    settings: Settings,
    connection: sqlite3.Connection,
    database_path: Path,
) -> dict[str, Any]:
    raw_payloads = _count_table(connection, "sim_raw_payloads")
    teams = _count_table(connection, "sim_teams")
    matches = _count_table(connection, "sim_matches")
    lineups = _count_table(connection, "sim_lineups")
    events = _count_table(connection, "sim_events")
    features = _count_table(connection, "sim_gold_features")
    league_seasons = _league_season_counts(connection)
    source_counts = _source_counts(connection)
    known_lineup_coverage = _known_lineup_coverage(connection)
    football_data_coverage_matrix = _football_data_coverage_matrix(connection)
    source_column_coverage = _source_column_coverage(connection)
    statsbomb_coverage = _statsbomb_coverage_report(connection)
    clubelo_mapping = _clubelo_mapping_report(connection)

    match_level_status = "ready" if matches > 0 else "blocked_no_matches"
    player_adjusted_status = "ready" if known_lineup_coverage >= 0.70 and lineups > 0 else "blocked_lineup_coverage_low"
    event_level_status = "ready" if events > 0 else "blocked_event_coverage_missing"
    summary = {
        "database_path": str(database_path),
        "raw_payloads": raw_payloads,
        "teams": teams,
        "matches": matches,
        "lineups": lineups,
        "events": events,
        "gold_feature_rows": features,
        "match_level_status": match_level_status,
        "player_adjusted_status": player_adjusted_status,
        "event_level_status": event_level_status,
        "picks_emitidos": 0,
        "global_roi_actionable": False,
    }

    source_manifest = {
        "generated_at": _utcnow().isoformat(),
        "sources": [asdict(spec) for spec in SIM_DATA_SOURCE_REGISTRY.values()],
        "default_enabled_sources": [
            spec.source_id for spec in SIM_DATA_SOURCE_REGISTRY.values() if spec.default_enabled
        ],
        "quarantine_policy": "Fragile or scraping-based connectors stay disabled until explicit coverage, stability, and terms audit.",
    }
    coverage_report = {
        "generated_at": _utcnow().isoformat(),
        "database_path": str(database_path),
        "summary": summary,
        "source_counts": source_counts,
        "league_season_counts": league_seasons,
        "source_column_coverage": source_column_coverage,
        "readiness": {
            "match_level_simulator": match_level_status,
            "player_adjusted_simulator": player_adjusted_status,
            "event_level_simulator": event_level_status,
        },
    }
    entity_report = {
        "generated_at": _utcnow().isoformat(),
        "teams": teams,
        "aliases": _count_table(connection, "sim_team_aliases"),
        "mapping_status_counts": _mapping_status_counts(connection),
        "fuzzy_suggestions_auto_activated": 0,
        "entity_resolution_policy": "Only deterministic source aliases are active in v1; fuzzy matches must be audited first.",
    }
    leakage_report = {
        "generated_at": _utcnow().isoformat(),
        "leakage_violations": _feature_leakage_violations(connection),
        "rules": [
            "Gold feature rows use only previous matches before the current match_start_time.",
            "Real lineups/events are excluded from predictive features unless known_before_match=1.",
            "Market reference columns are marked as reference-only and not part of the first simulator training contract.",
        ],
        "market_reference_training_enabled": False,
    }
    feature_manifest = {
        "generated_at": _utcnow().isoformat(),
        "feature_rows": features,
        "families": {
            "team_form": {"status": "ready" if features else "blocked_no_features", "train_allowed": True},
            "team_strength": {"status": "ready" if features else "blocked_no_features", "train_allowed": True},
            "schedule_context": {"status": "ready" if features else "blocked_no_features", "train_allowed": True},
            "clubelo": {"status": "ready" if clubelo_mapping["mapped_teams"] else "blocked_mapping_missing", "train_allowed": True},
            "market_reference": {"status": "reference_only", "training_enabled": False, "train_allowed": False},
            "player_availability": {"status": player_adjusted_status, "train_allowed": player_adjusted_status == "ready"},
            "player_form": {"status": "blocked_player_data_missing", "train_allowed": False},
            "tactical_event_profile": {"status": event_level_status, "train_allowed": event_level_status == "ready"},
        },
    }
    dataset_summary = {
        "generated_at": _utcnow().isoformat(),
        "summary": summary,
        "league_season_counts": league_seasons,
        "next_model_contract": "football_sim_poisson_v1_or_monte_carlo_can_train_only_from gold_sim_features with market_reference columns excluded by default.",
    }
    return {
        "summary": summary,
        "source_license_manifest": source_manifest,
        "source_coverage_report": coverage_report,
        "entity_resolution_report": entity_report,
        "leakage_audit_report": leakage_report,
        "simulation_feature_manifest": feature_manifest,
        "simulation_dataset_summary": dataset_summary,
        "football_data_coverage_matrix": football_data_coverage_matrix,
        "statsbomb_coverage_report": statsbomb_coverage,
        "clubelo_mapping_report": clubelo_mapping,
        "source_column_coverage_report": {"generated_at": _utcnow().isoformat(), "rows": source_column_coverage},
    }


def _write_sim_data_reports(output_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    artifacts = {
        "source_coverage_report": output_dir / "source_coverage_report.json",
        "source_license_manifest": output_dir / "source_license_manifest.json",
        "entity_resolution_report": output_dir / "entity_resolution_report.json",
        "leakage_audit_report": output_dir / "leakage_audit_report.json",
        "simulation_feature_manifest": output_dir / "simulation_feature_manifest.json",
        "simulation_dataset_summary": output_dir / "simulation_dataset_summary.json",
        "football_data_coverage_matrix": output_dir / "football_data_coverage_matrix.csv",
        "statsbomb_coverage_report": output_dir / "statsbomb_coverage_report.json",
        "clubelo_mapping_report": output_dir / "clubelo_mapping_report.json",
        "source_column_coverage_report": output_dir / "source_column_coverage_report.json",
    }
    for key, path in artifacts.items():
        if key == "football_data_coverage_matrix":
            pd.DataFrame(payload.get(key, [])).to_csv(path, index=False)
        else:
            _write_json(path, payload[key])
    return artifacts


def _count_table(connection: sqlite3.Connection, table: str) -> int:
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.Error:
        return 0


def _source_counts(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT source_id, fetch_status, COUNT(*) AS payloads, SUM(row_count) AS rows
            FROM sim_raw_payloads
            GROUP BY source_id, fetch_status
            ORDER BY source_id, fetch_status
            """
        ).fetchall()
    ]


def _football_data_coverage_matrix(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT league_code, league_name, season,
               COUNT(*) AS matches,
               SUM(CASE WHEN odds_home IS NOT NULL AND odds_draw IS NOT NULL AND odds_away IS NOT NULL THEN 1 ELSE 0 END) AS odds_rows,
               SUM(CASE WHEN home_shots IS NOT NULL AND away_shots IS NOT NULL THEN 1 ELSE 0 END) AS shot_stat_rows,
               SUM(CASE WHEN home_corners IS NOT NULL AND away_corners IS NOT NULL THEN 1 ELSE 0 END) AS corner_rows
        FROM sim_matches
        WHERE source_id = 'football_data'
        GROUP BY league_code, league_name, season
        ORDER BY league_code, season
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _source_column_coverage(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT source_id, source_key, payload_format, content_text FROM sim_raw_payloads WHERE fetch_status = 'success'"
    ).fetchall()
    coverage: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row["payload_format"] != "csv":
            continue
        try:
            frame = pd.read_csv(io.StringIO(row["content_text"]))
        except Exception:
            continue
        for column in frame.columns:
            key = (str(row["source_id"]), str(column))
            item = coverage.setdefault(
                key,
                {"source_id": row["source_id"], "column": column, "payloads": 0, "rows": 0, "non_null_rows": 0},
            )
            item["payloads"] += 1
            item["rows"] += int(len(frame))
            item["non_null_rows"] += int(frame[column].notna().sum())
    for item in coverage.values():
        item["non_null_rate"] = float(item["non_null_rows"] / item["rows"]) if item["rows"] else 0.0
    return sorted(coverage.values(), key=lambda value: (value["source_id"], value["column"]))


def _statsbomb_coverage_report(connection: sqlite3.Connection) -> dict[str, Any]:
    competitions = _count_table(connection, "sim_competitions")
    rows = connection.execute(
        """
        SELECT COUNT(*) AS matches
        FROM sim_matches
        WHERE source_id = 'statsbomb_open_data'
        """
    ).fetchone()
    linked = connection.execute(
        "SELECT COUNT(*) AS links FROM sim_match_source_links WHERE source_id = 'statsbomb_open_data'"
    ).fetchone()
    lineup_matches = connection.execute(
        "SELECT COUNT(DISTINCT match_id) AS rows FROM sim_lineups WHERE source_id = 'statsbomb_open_data'"
    ).fetchone()
    event_matches = connection.execute(
        "SELECT COUNT(DISTINCT match_id) AS rows FROM sim_events WHERE source_id = 'statsbomb_open_data'"
    ).fetchone()
    return {
        "generated_at": _utcnow().isoformat(),
        "competitions": competitions,
        "matches": int(rows["matches"] or 0),
        "match_links": int(linked["links"] or 0),
        "lineup_matches": int(lineup_matches["rows"] or 0),
        "event_matches": int(event_matches["rows"] or 0),
        "coverage_status": "ready" if int(event_matches["rows"] or 0) else "coverage_limited_or_not_collected",
    }


def _clubelo_mapping_report(connection: sqlite3.Connection) -> dict[str, Any]:
    teams = _count_table(connection, "sim_teams")
    mapped = connection.execute(
        "SELECT COUNT(DISTINCT team_id) AS teams FROM sim_team_ratings WHERE source_id = 'clubelo'"
    ).fetchone()
    ratings = _count_table(connection, "sim_team_ratings")
    mapped_teams = int(mapped["teams"] or 0)
    return {
        "generated_at": _utcnow().isoformat(),
        "teams": teams,
        "mapped_teams": mapped_teams,
        "unmapped_teams": max(0, teams - mapped_teams),
        "ratings": ratings,
        "mapping_rate": float(mapped_teams / teams) if teams else 0.0,
        "coverage_status": "ready" if mapped_teams else "mapping_missing",
    }


def _league_season_counts(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT league_code, league_name, season, COUNT(*) AS matches
            FROM sim_matches
            GROUP BY league_code, league_name, season
            ORDER BY league_code, season
            """
        ).fetchall()
    ]


def _mapping_status_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        "SELECT mapping_status, COUNT(*) AS rows FROM sim_team_aliases GROUP BY mapping_status"
    ).fetchall()
    return {str(row["mapping_status"]): int(row["rows"]) for row in rows}


def _known_lineup_coverage(connection: sqlite3.Connection) -> float:
    matches = _count_table(connection, "sim_matches")
    if matches <= 0:
        return 0.0
    rows = connection.execute(
        "SELECT COUNT(DISTINCT match_id) AS known_matches FROM sim_lineups WHERE known_before_match = 1"
    ).fetchone()
    known_matches = int(rows["known_matches"] or 0)
    return known_matches / matches


def _feature_leakage_violations(connection: sqlite3.Connection) -> int:
    rows = connection.execute(
        """
        SELECT COUNT(*) AS violations
        FROM sim_gold_features
        WHERE datetime(as_of_time) >= datetime(match_start_time) OR known_before_match != 1
        """
    ).fetchone()
    return int(rows["violations"] or 0)


def _to_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def _sum_recent(history: list[dict[str, Any]], key: str, window: int) -> float:
    values = [float(item.get(key, 0.0)) for item in history[-window:]]
    return float(sum(values)) if values else 0.0


def _mean_recent(history: list[dict[str, Any]], key: str, window: int) -> float:
    values = [float(item.get(key, 0.0)) for item in history[-window:]]
    return float(sum(values) / len(values)) if values else 0.0


def _rest_days(history: list[dict[str, Any]], match_start: pd.Timestamp) -> float:
    if not history:
        return float("nan")
    previous = pd.Timestamp(history[-1]["date"])
    return float((match_start - previous).total_seconds() / 86400.0)


def _matches_within_days(history: list[dict[str, Any]], match_start: pd.Timestamp, days: int) -> float:
    cutoff = match_start - pd.Timedelta(days=days)
    return float(sum(pd.Timestamp(item["date"]) >= cutoff for item in history))


def _market_probs_from_odds(odds: list[float]) -> tuple[float, float, float]:
    if not all(np.isfinite(value) and value > 0 for value in odds):
        return (float("nan"), float("nan"), float("nan"))
    implied = np.array([1.0 / value for value in odds], dtype=float)
    total = float(implied.sum())
    if total <= 0:
        return (float("nan"), float("nan"), float("nan"))
    probs = implied / total
    return (float(probs[0]), float(probs[1]), float(probs[2]))


def supported_football_data_leagues() -> dict[str, str]:
    return dict(LEAGUE_NAMES)
