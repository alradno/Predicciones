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


def _load_model_payload(settings: Settings, model_path: Path | str | None = None) -> tuple[Path, dict[str, Any]]:
    if model_path is None:
        for pointer_name in ("latest_niche_model.txt", "latest_model.txt"):
            pointer = settings.paths.outputs_dir / pointer_name
            if pointer.exists():
                model_path = Path(pointer.read_text(encoding="utf-8").strip())
                break
    artifact_path = Path(model_path) if model_path else None
    if artifact_path is None or not artifact_path.exists():
        raise FileNotFoundError("No encuentro un modelo entrenado para el backtest retro de Polymarket.")
    return artifact_path, joblib.load(artifact_path)


def _history_matches(settings: Settings, payload: dict[str, Any], matches: pd.DataFrame | None = None) -> pd.DataFrame:
    history = canonicalize_matches(matches if matches is not None else payload["history_matches"], require_results=True)
    history = history[history["league_code"].astype(str).isin(settings.polymarket.supported_leagues)].copy()
    if history.empty:
        raise ValueError("No hay partidos historicos con resultado en las ligas soportadas.")
    if "league_name" not in history.columns:
        history["league_name"] = history["league_code"].map(LEAGUE_NAMES).fillna(history["league_code"])
    return history.sort_values(["Date", "league_code", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def _history_window(settings: Settings, matches: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    if settings.polymarket.historical_backfill_start:
        start = pd.Timestamp(settings.polymarket.historical_backfill_start, tz="UTC")
    else:
        start = pd.Timestamp(matches["Date"].min(), tz="UTC")
    if settings.polymarket.historical_backfill_end:
        end = pd.Timestamp(settings.polymarket.historical_backfill_end, tz="UTC")
    else:
        end = pd.Timestamp(matches["Date"].max(), tz="UTC")
    return start.normalize(), end.normalize()


def _chunk_bounds(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    rows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    delta = pd.to_timedelta(max(1, chunk_days), unit="D")
    while cursor <= end:
        chunk_end = min(cursor + delta - pd.to_timedelta(1, unit="ns"), end)
        rows.append((cursor, chunk_end))
        cursor = chunk_end + pd.to_timedelta(1, unit="ns")
    return rows


DRAW_QUESTION_PATTERN = re.compile(
    r"^\s*will\s+(?P<home>.+?)\s+vs\.?\s+(?P<away>.+?)\s+end\s+in\s+a\s+draw(?:\s*\?)?\s*$",
    re.IGNORECASE,
)
DRAW_QUESTION_ALT_PATTERN = re.compile(
    r"^\s*will\s+(?P<home>.+?)\s+vs\.?\s+(?P<away>.+?)\s+end\s+in\s+draw(?:\s*\?)?\s*$",
    re.IGNORECASE,
)
DRAW_BETWEEN_PATTERN = re.compile(
    r"^\s*will\s+the\s+match\s+between\s+(?P<home>.+?)\s+and\s+(?P<away>.+?)\s+end\s+in\s+(?:a\s+)?draw(?:\s*\?)?\s*$",
    re.IGNORECASE,
)
DRAW_MATCH_PATTERN = re.compile(
    r"^\s*will\s+the\s+(?P<home>.+?)\s+vs\.?\s+(?P<away>.+?)\s+match\s+be\s+a?\s*draw(?:\s*\?)?\s*$",
    re.IGNORECASE,
)
BEAT_QUESTION_PATTERN = re.compile(
    r"^\s*will\s+(?P<winner>.+?)\s+beat\s+(?P<loser>.+?)\s*\?\s*$",
    re.IGNORECASE,
)
WIN_VS_PATTERN = re.compile(
    r"^\s*will\s+(?P<winner>.+?)\s+win\s+vs\.?\s+(?P<loser>.+?)\s*\?\s*$",
    re.IGNORECASE,
)
WIN_ON_DATE_PATTERN = re.compile(
    r"^\s*will\s+(?P<winner>.+?)\s+win\s+on\s+\d{4}-\d{2}-\d{2}\s*\?\s*$",
    re.IGNORECASE,
)
COMBINE_GOALS_PATTERN = re.compile(
    r"^\s*will\s+(?P<home>.+?)\s+and\s+(?P<away>.+?)\s+combine\s+for\s+.+\?\s*$",
    re.IGNORECASE,
)
WHO_WILL_WIN_GAME_PATTERN = re.compile(
    r"^\s*(?:.+?:\s*)?who\s+will\s+win\s+the\s+(?P<home>.+?)\s+vs\.?\s+(?P<away>.+?)\s+game\s+on\s+.+\?\s*$",
    re.IGNORECASE,
)
SLUG_FIXTURE_PATTERN = re.compile(
    r"(?P<home>[a-z0-9]+(?:-[a-z0-9]+)*)-vs-(?P<away>[a-z0-9]+(?:-[a-z0-9]+)*)",
    re.IGNORECASE,
)
DATE_TOKEN_PATTERN = re.compile(r"^\d{1,4}$")
SLUG_PREFIX_TOKENS = {
    "epl",
    "premier",
    "league",
    "premier-league",
    "lal",
    "la",
    "liga",
    "laliga",
    "bun",
    "bundesliga",
}
TRUE_1X2_SHAPES = {
    "true_1x2_leg_home",
    "true_1x2_leg_draw",
    "true_1x2_leg_away",
}


def _fixture_key(league_code: str, match_date: pd.Timestamp, home_team: str, away_team: str) -> str:
    return "|".join(
        [
            str(league_code),
            pd.Timestamp(match_date).strftime("%Y-%m-%d"),
            pm_shadow._normalize_team_key(home_team),
            pm_shadow._normalize_team_key(away_team),
        ]
    )


def _undirected_fixture_key(league_code: str, match_date: pd.Timestamp, home_team: str, away_team: str) -> str:
    normalized = sorted([pm_shadow._normalize_team_key(home_team), pm_shadow._normalize_team_key(away_team)])
    return "|".join([str(league_code), pd.Timestamp(match_date).strftime("%Y-%m-%d"), *normalized])


def _market_start_time(event: dict[str, Any], market: dict[str, Any]) -> pd.Timestamp | pd.NaT:
    timestamp = pm_shadow._parse_timestamp(
        market.get("gameStartTime")
        or market.get("startTime")
        or event.get("startTime")
        or event.get("gameStartTime")
        or event.get("eventDate")
    )
    return timestamp.normalize() if not pd.isna(timestamp) else pd.NaT


def _parse_draw_question(question: str) -> tuple[str, str] | None:
    value = str(question).strip()
    match = (
        DRAW_QUESTION_PATTERN.match(value)
        or DRAW_QUESTION_ALT_PATTERN.match(value)
        or DRAW_BETWEEN_PATTERN.match(value)
        or DRAW_MATCH_PATTERN.match(value)
    )
    if not match:
        return None
    return normalize_team_name(match.group("home")), normalize_team_name(match.group("away"))


def _parse_beat_question(question: str) -> tuple[str, str] | None:
    match = BEAT_QUESTION_PATTERN.match(str(question).strip())
    if not match:
        return None
    return normalize_team_name(match.group("winner")), normalize_team_name(match.group("loser"))


def _parse_win_vs_question(question: str) -> tuple[str, str] | None:
    match = WIN_VS_PATTERN.match(str(question).strip())
    if not match:
        return None
    return normalize_team_name(match.group("winner")), normalize_team_name(match.group("loser"))


def _parse_win_on_date_question(question: str) -> str | None:
    match = WIN_ON_DATE_PATTERN.match(str(question).strip())
    if not match:
        return None
    return normalize_team_name(match.group("winner"))


def _parse_combine_goals_question(question: str) -> tuple[str, str] | None:
    match = COMBINE_GOALS_PATTERN.match(str(question).strip())
    if not match:
        return None
    return normalize_team_name(match.group("home")), normalize_team_name(match.group("away"))


def _parse_who_will_win_game(question: str) -> tuple[str, str] | None:
    match = WHO_WILL_WIN_GAME_PATTERN.match(str(question).strip())
    if not match:
        return None
    return normalize_team_name(match.group("home")), normalize_team_name(match.group("away"))


def _trim_slug_tokens(value: str) -> list[str]:
    tokens = [token for token in str(value).strip().lower().split("-") if token]
    while tokens and tokens[0] in SLUG_PREFIX_TOKENS:
        tokens = tokens[1:]
    while tokens and DATE_TOKEN_PATTERN.match(tokens[-1]):
        tokens = tokens[:-1]
    return tokens


def _parse_slug_fixture(value: str) -> tuple[str, str] | None:
    match = SLUG_FIXTURE_PATTERN.search(str(value).strip().lower())
    if not match:
        return None
    home_tokens = _trim_slug_tokens(match.group("home"))
    away_tokens = _trim_slug_tokens(match.group("away"))
    if not home_tokens or not away_tokens:
        return None
    return normalize_team_name(" ".join(home_tokens)), normalize_team_name(" ".join(away_tokens))


def _team_label_score(label: str, team_name: str, league_code: str | None) -> float:
    normalized_label = pm_shadow._normalize_team_key(label)
    if not normalized_label:
        return 0.0
    variants = pm_shadow._team_alias_variants(team_name, league_code)
    if normalized_label in variants:
        return 1.0
    compact = normalized_label.replace(" ", "")
    abbreviations: set[str] = set()
    for variant in variants:
        tokens = variant.split()
        if tokens:
            abbreviations.add("".join(token[0] for token in tokens[:3]))
            abbreviations.add(tokens[-1][:3])
            abbreviations.add(variant.replace(" ", "")[:3])
    if compact in abbreviations or normalized_label in abbreviations:
        return 1.0
    return pm_shadow._team_match_score(label, team_name, league_code)


def _extract_fixture_context(
    event: dict[str, Any],
    market: dict[str, Any],
) -> tuple[str | None, str | None, str, str]:
    question = str(market.get("question", "")).strip()
    group_item = str(market.get("groupItemTitle", "")).strip()
    event_title = str(event.get("title", "")).strip()
    market_slug = str(market.get("slug", "")).strip()
    event_slug = str(event.get("slug", "")).strip()

    draw_question = _parse_draw_question(question)
    if draw_question is not None:
        return draw_question[0], draw_question[1], "question", "draw_question"
    beat_question = _parse_beat_question(question)
    if beat_question is not None:
        return beat_question[0], beat_question[1], "question", "beat_question"
    win_vs_question = _parse_win_vs_question(question)
    if win_vs_question is not None:
        return win_vs_question[0], win_vs_question[1], "question", "win_vs_question"
    who_will_win = _parse_who_will_win_game(question)
    if who_will_win is not None:
        return who_will_win[0], who_will_win[1], "question", "who_will_win_game"
    who_will_win = _parse_who_will_win_game(event_title)
    if who_will_win is not None:
        return who_will_win[0], who_will_win[1], "event.title", "who_will_win_game"
    match_title = pm_shadow._parse_match_title(question)
    if match_title is not None:
        return match_title[0], match_title[1], "question", "match_title"
    match_title = pm_shadow._parse_match_title(group_item)
    if match_title is not None:
        return match_title[0], match_title[1], "groupItemTitle", "match_title"
    match_title = pm_shadow._parse_match_title(event_title)
    if match_title is not None:
        return match_title[0], match_title[1], "event.title", "match_title"
    slug_fixture = _parse_slug_fixture(market_slug)
    if slug_fixture is not None:
        return slug_fixture[0], slug_fixture[1], "market.slug", "slug_fixture"
    slug_fixture = _parse_slug_fixture(event_slug)
    if slug_fixture is not None:
        return slug_fixture[0], slug_fixture[1], "event.slug", "slug_fixture"
    combine_fixture = _parse_combine_goals_question(question)
    if combine_fixture is not None:
        return combine_fixture[0], combine_fixture[1], "question", "combine_goals"
    beat_question = _parse_beat_question(question)
    if beat_question is not None:
        return beat_question[0], beat_question[1], "question", "beat_question"
    win_vs_question = _parse_win_vs_question(question)
    if win_vs_question is not None:
        return win_vs_question[0], win_vs_question[1], "question", "win_vs_question"
    return None, None, "", "unparsed"


def _classify_market_shape(
    event: dict[str, Any],
    market: dict[str, Any],
    league_code: str,
    home_team_raw: str | None,
    away_team_raw: str | None,
) -> tuple[str, str | None]:
    question = str(market.get("question", "")).strip()
    outcomes = [str(item) for item in pm_shadow._json_list(market.get("outcomes"))]
    normalized_outcomes = [item.strip().lower() for item in outcomes]

    if normalized_outcomes == ["yes", "no"]:
        if _parse_draw_question(question) is not None or ("draw" in question.lower() and home_team_raw and away_team_raw):
            return "true_1x2_leg_draw", OUTCOME_DRAW
        beat_question = _parse_beat_question(question)
        if beat_question is not None:
            winner, _ = beat_question
            if home_team_raw and away_team_raw:
                home_score = pm_shadow._team_match_score(winner, home_team_raw, league_code)
                away_score = pm_shadow._team_match_score(winner, away_team_raw, league_code)
                return (
                    "true_1x2_leg_home" if home_score >= away_score else "true_1x2_leg_away",
                    OUTCOME_HOME if home_score >= away_score else OUTCOME_AWAY,
                )
            return "true_1x2_leg_home", OUTCOME_HOME
        win_vs = _parse_win_vs_question(question)
        if win_vs is not None:
            winner, _ = win_vs
            if home_team_raw and away_team_raw:
                home_score = pm_shadow._team_match_score(winner, home_team_raw, league_code)
                away_score = pm_shadow._team_match_score(winner, away_team_raw, league_code)
                return (
                    "true_1x2_leg_home" if home_score >= away_score else "true_1x2_leg_away",
                    OUTCOME_HOME if home_score >= away_score else OUTCOME_AWAY,
                )
        win_on_date = _parse_win_on_date_question(question)
        if win_on_date is not None and home_team_raw and away_team_raw:
            home_score = pm_shadow._team_match_score(win_on_date, home_team_raw, league_code)
            away_score = pm_shadow._team_match_score(win_on_date, away_team_raw, league_code)
            if max(home_score, away_score) > 0.72:
                return (
                    "true_1x2_leg_home" if home_score >= away_score else "true_1x2_leg_away",
                    OUTCOME_HOME if home_score >= away_score else OUTCOME_AWAY,
                )
        if _parse_combine_goals_question(question) is not None:
            return "other_match_market", None

    if len(outcomes) == 2 and any("draw" in item.lower() for item in outcomes):
        non_draw = next((item for item in outcomes if "draw" not in item.lower()), "")
        if home_team_raw and away_team_raw:
            home_score = _team_label_score(non_draw, home_team_raw, league_code)
            away_score = _team_label_score(non_draw, away_team_raw, league_code)
            if max(home_score, away_score) > 0.72:
                return (
                    "binary_home_vs_away_draw" if home_score >= away_score else "binary_away_vs_home_draw",
                    OUTCOME_HOME if home_score >= away_score else OUTCOME_AWAY,
                )
        return "other_match_market", None

    return "other_match_market", None


def _history_fixture_frame(matches: pd.DataFrame) -> pd.DataFrame:
    fixtures = matches[
        ["match_id", "Date", "league_code", "league_name", "season", "HomeTeam", "AwayTeam"]
    ].drop_duplicates().copy()
    fixtures["match_date"] = pd.to_datetime(fixtures["Date"], utc=True).dt.normalize()
    fixtures["fixture_key"] = fixtures.apply(
        lambda row: _fixture_key(str(row["league_code"]), pd.Timestamp(row["match_date"]), str(row["HomeTeam"]), str(row["AwayTeam"])),
        axis=1,
    )
    fixtures["undirected_fixture_key"] = fixtures.apply(
        lambda row: _undirected_fixture_key(
            str(row["league_code"]),
            pd.Timestamp(row["match_date"]),
            str(row["HomeTeam"]),
            str(row["AwayTeam"]),
        ),
        axis=1,
    )
    return fixtures


def _match_market_to_fixture(
    settings: Settings,
    market_row: dict[str, Any],
    fixture_rows: pd.DataFrame,
) -> dict[str, Any]:
    league_code = str(market_row["league_code"])
    home_team_raw = str(market_row.get("home_team_raw") or "")
    away_team_raw = str(market_row.get("away_team_raw") or "")
    match_date = market_row.get("match_date")
    if not home_team_raw or not away_team_raw or pd.isna(match_date):
        return {
            "classification_status": "unmatched",
            "classification_reason": "fixture_not_parseable",
        }

    candidates = fixture_rows[fixture_rows["league_code"].astype(str).eq(league_code)].copy()
    if candidates.empty:
        return {
            "classification_status": "out_of_scope",
            "classification_reason": "league_without_history_rows",
        }

    date_value = pd.Timestamp(match_date).normalize()
    date_window = pd.to_timedelta(max(2, settings.polymarket.historical_chunk_days // 3), unit="D")

    def _score_candidates(frame: pd.DataFrame, reference_date: pd.Timestamp) -> list[dict[str, Any]]:
        scored_rows: list[dict[str, Any]] = []
        for fixture in frame.itertuples(index=False):
            direct_home = pm_shadow._team_match_score(home_team_raw, str(fixture.HomeTeam), league_code)
            direct_away = pm_shadow._team_match_score(away_team_raw, str(fixture.AwayTeam), league_code)
            reverse_home = pm_shadow._team_match_score(home_team_raw, str(fixture.AwayTeam), league_code)
            reverse_away = pm_shadow._team_match_score(away_team_raw, str(fixture.HomeTeam), league_code)
            direct_score = (direct_home + direct_away) / 2.0
            reverse_score = (reverse_home + reverse_away) / 2.0
            if direct_score >= reverse_score:
                orientation = "direct"
                score = direct_score
                home_score = direct_home
                away_score = direct_away
            else:
                orientation = "mirror"
                score = reverse_score
                home_score = reverse_home
                away_score = reverse_away
            delta_days = abs(int((pd.Timestamp(fixture.match_date) - reference_date).days))
            scored_rows.append(
                {
                    "match_id": str(fixture.match_id),
                    "fixture_key": str(fixture.fixture_key),
                    "undirected_fixture_key": str(fixture.undirected_fixture_key),
                    "league_code": str(fixture.league_code),
                    "league_name": str(fixture.league_name),
                    "season": str(fixture.season),
                    "match_date": pd.Timestamp(fixture.match_date),
                    "home_team": str(fixture.HomeTeam),
                    "away_team": str(fixture.AwayTeam),
                    "orientation": orientation,
                    "score": score,
                    "home_score": home_score,
                    "away_score": away_score,
                    "delta_days": delta_days,
                }
            )
        return sorted(scored_rows, key=lambda item: (item["score"], -item["delta_days"]), reverse=True)

    window_candidates = candidates[
        candidates["match_date"].between(date_value - date_window, date_value + date_window, inclusive="both")
    ].copy()
    scored = _score_candidates(window_candidates if not window_candidates.empty else candidates.copy(), date_value)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    no_window_match = window_candidates.empty
    if best["score"] < settings.polymarket.mapping_score_threshold and not no_window_match:
        broad_scored = _score_candidates(candidates.copy(), date_value)
        broad_best = broad_scored[0]
        broad_second = broad_scored[1] if len(broad_scored) > 1 else None
        if (
            broad_best["score"] >= 0.999
            and broad_best["delta_days"] <= 120
            and (
                broad_second is None
                or broad_second["score"] < 0.999
                or broad_second["delta_days"] > broad_best["delta_days"]
            )
        ):
            best = broad_best
            second = broad_second
            no_window_match = True
    if no_window_match and best["score"] < 0.999:
        return {
            "classification_status": "unmatched",
            "classification_reason": "no_fixture_within_date_window",
        }
    if no_window_match and best["delta_days"] > 120:
        return {
            "classification_status": "unmatched",
            "classification_reason": "no_fixture_within_date_window",
        }
    if best["score"] < settings.polymarket.mapping_score_threshold:
        return {
            "classification_status": "unmatched",
            "classification_reason": "best_fuzzy_score_below_threshold",
            "best_score": float(best["score"]),
        }
    if second is not None and (
        (best["score"] - second["score"]) < settings.polymarket.mapping_score_gap
        and best["delta_days"] == second["delta_days"]
    ):
        return {
            "classification_status": "ambiguous",
            "classification_reason": "fixture_score_gap_too_small",
            "best_score": float(best["score"]),
        }
    return {
        "classification_status": "matched",
        "classification_reason": "best_unique_fixture_match" if not no_window_match else "best_unique_fixture_match_outside_window",
        **best,
    }


def _flatten_event_rows(
    settings: Settings,
    events_by_slug: dict[str, dict[str, Any]],
    updated_at: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_slug, event in sorted(events_by_slug.items()):
        sport_code, league_code, league_name = pm_shadow._infer_league_from_event(event_slug, event)
        if not sport_code or league_code not in settings.polymarket.supported_leagues:
            continue
        game_start = pm_shadow._event_game_start(event)
        rows.append(
            {
                "event_slug": str(event_slug),
                "event_id": str(event.get("id", "")),
                "series_slug": str(event.get("seriesSlug", "")),
                "title": str(event.get("title", "")),
                "start_date": pm_shadow._iso_timestamp(game_start) if not pd.isna(game_start) else "",
                "game_start_time": pm_shadow._iso_timestamp(game_start) if not pd.isna(game_start) else "",
                "event_date": pm_shadow._iso_timestamp(pm_shadow._parse_timestamp(event.get("eventDate")))
                if event.get("eventDate")
                else "",
                "league_code": league_code,
                "league_name": league_name,
                "sport_code": sport_code,
                "closed": int(bool(event.get("closed", False))),
                "archived": int(bool(event.get("archived", False))),
                "raw_json": pm_shadow._clean_json(event),
                "updated_at": updated_at,
            }
        )
    return pd.DataFrame(rows)


def _flatten_market_rows(
    settings: Settings,
    events_by_slug: dict[str, dict[str, Any]],
    updated_at: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_slug, event in sorted(events_by_slug.items()):
        sport_code, league_code, league_name = pm_shadow._infer_league_from_event(event_slug, event)
        if not sport_code or league_code not in settings.polymarket.supported_leagues:
            continue
        for market in event.get("markets", []):
            market_id = str(market.get("id", "")).strip()
            if not market_id:
                continue
            game_start = pm_shadow._parse_timestamp(
                market.get("gameStartTime")
                or market.get("startTime")
                or event.get("startTime")
                or event.get("gameStartTime")
                or event.get("eventDate")
            )
            rows.append(
                {
                    "market_id": market_id,
                    "event_id": str(event.get("id", "")),
                    "event_slug": str(event_slug),
                    "event_title": str(event.get("title", "")),
                    "series_slug": str(event.get("seriesSlug", "")),
                    "market_slug": str(market.get("slug", "")),
                    "question": str(market.get("question", "")),
                    "group_item_title": str(market.get("groupItemTitle", "")),
                    "league_code": league_code,
                    "league_name": league_name,
                    "sport_code": sport_code,
                    "game_start_time": pm_shadow._iso_timestamp(game_start) if not pd.isna(game_start) else "",
                    "closed": int(bool(market.get("closed", False))),
                    "archived": int(bool(event.get("archived", False))),
                    "active": int(bool(market.get("active", False))),
                    "accepting_orders": int(bool(market.get("acceptingOrders", market.get("active", False)))),
                    "sports_market_type": str(market.get("sportsMarketType", "")),
                    "outcomes_json": pm_shadow._clean_json(pm_shadow._json_list(market.get("outcomes"))),
                    "outcome_prices_json": pm_shadow._clean_json(pm_shadow._json_list(market.get("outcomePrices"))),
                    "clob_token_ids_json": pm_shadow._clean_json(pm_shadow._json_list(market.get("clobTokenIds"))),
                    "fees_enabled": int(bool(market.get("feesEnabled", False))),
                    "fee_rate": float(PolymarketGammaClient.extract_fee_rate(market)),
                    "raw_json": pm_shadow._clean_json(market),
                    "updated_at": updated_at,
                }
            )
    return pd.DataFrame(rows)


def _fetch_closed_soccer_events(
    settings: Settings,
    gamma: PolymarketGammaClient,
    matches: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    start, end = _history_window(settings, matches)
    min_date = start - pd.to_timedelta(2, unit="D")
    max_date = end + pd.to_timedelta(2, unit="D")
    events_by_slug: dict[str, dict[str, Any]] = {}
    page_rows: list[dict[str, Any]] = []
    cursor: str | None = None
    page_index = 0
    entered_window = False

    while True:
        events, next_cursor = gamma.list_events_keyset(limit=200, after_cursor=cursor, closed=True)
        if not events:
            break
        page_index += 1
        page_starts: list[pd.Timestamp] = []
        matched = 0
        for event in events:
            slug = str(event.get("slug", "")).strip()
            if not slug:
                continue
            start_time = pm_shadow._event_game_start(event)
            if pd.isna(start_time):
                continue
            page_starts.append(start_time)
            if start_time < min_date or start_time > max_date:
                continue
            sport_code, league_code, _ = pm_shadow._infer_league_from_event(slug, event)
            if not sport_code or league_code not in settings.polymarket.supported_leagues:
                continue
            events_by_slug[slug] = event
            matched += 1
            entered_window = True
        page_rows.append(
            {
                "page_index": page_index,
                "cursor": cursor or "",
                "rows": len(events),
                "matched_rows": matched,
                "page_min_start": pm_shadow._iso_timestamp(min(page_starts)) if page_starts else "",
                "page_max_start": pm_shadow._iso_timestamp(max(page_starts)) if page_starts else "",
                "next_cursor": next_cursor or "",
            }
        )
        if not next_cursor or next_cursor == cursor:
            break
        if page_starts and entered_window and min(page_starts) > max_date:
            break
        cursor = next_cursor

    return events_by_slug, page_rows


def _history_team_lookup(matches: pd.DataFrame) -> dict[str, list[str]]:
    return pm_shadow._history_team_lookup(matches)


def _team_alias_rows(matches: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    updated_at = pm_shadow._iso_timestamp()
    alias_rows: list[dict[str, Any]] = []
    history_lookup = _history_team_lookup(matches)

    for league_code, teams in history_lookup.items():
        for team in teams:
            team_name = normalize_team_name(team)
            for alias_key in sorted(pm_shadow._team_alias_variants(team_name, str(league_code))):
                alias_rows.append(
                    {
                        "alias_id": f"{league_code}:{team_name}:{alias_key}",
                        "league_code": str(league_code),
                        "canonical_team": team_name,
                        "alias_text": team_name,
                        "alias_key": alias_key,
                        "source": "history_matches",
                        "score": 1.0,
                        "updated_at": updated_at,
                    }
                )

    if not groups.empty:
        for row in groups.itertuples(index=False):
            candidates = history_lookup.get(str(row.league_code), [])
            for team_name in (str(row.home_team), str(row.away_team)):
                canonical = pm_shadow._resolve_team_alias(team_name, candidates, league_code=str(row.league_code))
                score = pm_shadow._team_match_score(team_name, canonical, str(row.league_code))
                variant_keys = pm_shadow._team_alias_variants(team_name, str(row.league_code))
                variant_keys.update(pm_shadow._team_alias_variants(canonical, str(row.league_code)))
                for alias_key in sorted(variant_keys):
                    alias_rows.append(
                        {
                            "alias_id": f"{row.league_code}:{canonical}:{alias_key}",
                            "league_code": str(row.league_code),
                            "canonical_team": normalize_team_name(canonical),
                            "alias_text": normalize_team_name(team_name),
                            "alias_key": alias_key,
                            "source": "polymarket_group",
                            "score": score,
                            "updated_at": updated_at,
                        }
                    )

    return pd.DataFrame(alias_rows).drop_duplicates(subset=["alias_id"]).reset_index(drop=True)


def _alias_sets(alias_frame: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    lookup: dict[str, dict[str, set[str]]] = {}
    if alias_frame.empty:
        return lookup
    for row in alias_frame.itertuples(index=False):
        league_lookup = lookup.setdefault(str(row.league_code), {})
        alias_set = league_lookup.setdefault(str(row.canonical_team), set())
        alias_set.add(str(row.alias_key))
    return lookup


def _best_alias_score(name: str, alias_keys: set[str]) -> tuple[float, str]:
    fixture_key = pm_shadow._normalize_team_key(name)
    if fixture_key in alias_keys:
        return 1.0, fixture_key
    best_score = 0.0
    best_alias = ""
    for alias_key in alias_keys:
        score = __import__("difflib").SequenceMatcher(None, fixture_key, alias_key).ratio()
        if score > best_score:
            best_score = score
            best_alias = alias_key
    return best_score, best_alias


def _classify_market_candidates(
    settings: Settings,
    history_matches: pd.DataFrame,
    raw_markets: pd.DataFrame,
    events_by_slug: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if raw_markets.empty:
        return pd.DataFrame()
    fixture_rows = _history_fixture_frame(history_matches)
    rows: list[dict[str, Any]] = []
    for raw in raw_markets.to_dict(orient="records"):
        event = events_by_slug.get(str(raw.get("event_slug", "")), {})
        market_payload = json.loads(str(raw.get("raw_json", "{}")))
        home_raw, away_raw, parse_source, parse_method = _extract_fixture_context(event, market_payload)
        match_date = _market_start_time(event, market_payload)
        market_shape, market_role = _classify_market_shape(
            event=event,
            market=market_payload,
            league_code=str(raw["league_code"]),
            home_team_raw=home_raw,
            away_team_raw=away_raw,
        )
        mapping = _match_market_to_fixture(
            settings=settings,
            market_row={
                "league_code": str(raw["league_code"]),
                "home_team_raw": home_raw,
                "away_team_raw": away_raw,
                "match_date": match_date,
            },
            fixture_rows=fixture_rows,
        )
        classification_status = str(mapping.get("classification_status", "unmatched"))
        classification_reason = str(mapping.get("classification_reason", "unmatched"))
        final_role = market_role
        fixture_key = ""
        undirected_key = ""
        match_id = ""
        canonical_home = normalize_team_name(home_raw or "")
        canonical_away = normalize_team_name(away_raw or "")
        match_date_value = match_date

        if classification_status == "matched":
            fixture_key = str(mapping["fixture_key"])
            undirected_key = str(mapping["undirected_fixture_key"])
            match_id = str(mapping["match_id"])
            canonical_home = str(mapping["home_team"])
            canonical_away = str(mapping["away_team"])
            match_date_value = pd.Timestamp(mapping["match_date"])
            if market_shape in TRUE_1X2_SHAPES or market_shape.startswith("binary_"):
                if str(mapping.get("orientation")) == "mirror" and final_role in {OUTCOME_HOME, OUTCOME_AWAY}:
                    final_role = OUTCOME_AWAY if final_role == OUTCOME_HOME else OUTCOME_HOME
            classification_status = "candidate_true_1x2" if market_shape in TRUE_1X2_SHAPES else "non_1x2"
            classification_reason = (
                "matched_true_1x2_candidate" if market_shape in TRUE_1X2_SHAPES else f"matched_{market_shape}"
            )
        elif classification_status == "ambiguous":
            classification_status = "ambiguous"
        elif classification_status == "out_of_scope":
            classification_status = "out_of_scope"
        else:
            classification_status = "non_1x2" if market_shape == "other_match_market" else "unmatched"
            if market_shape == "other_match_market" and classification_reason == "fixture_not_parseable":
                classification_reason = "unmatched_other_match_market"

        quality_rank = 0.0
        quality_rank += 10.0 if market_shape in TRUE_1X2_SHAPES else 0.0
        quality_rank += {
            "question": 4.0,
            "groupItemTitle": 3.0,
            "event.title": 2.0,
            "market.slug": 1.0,
            "event.slug": 0.5,
        }.get(parse_source, 0.0)
        quality_rank += {
            "draw_question": 2.0,
            "beat_question": 1.5,
            "win_vs_question": 1.5,
            "who_will_win_game": 1.25,
            "match_title": 1.0,
            "slug_fixture": 0.5,
        }.get(parse_method, 0.0)
        rows.append(
            {
                **raw,
                "match_id": match_id,
                "match_date": match_date_value,
                "fixture_key": fixture_key,
                "undirected_fixture_key": undirected_key,
                "home_team_raw": normalize_team_name(home_raw or ""),
                "away_team_raw": normalize_team_name(away_raw or ""),
                "home_team_canonical": canonical_home,
                "away_team_canonical": canonical_away,
                "parse_source": parse_source,
                "parse_method": parse_method,
                "market_shape": market_shape,
                "market_role": final_role or "",
                "market_quality_rank": quality_rank,
                "classification_status": classification_status,
                "classification_reason": classification_reason,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True, errors="coerce")
        frame["game_start_time"] = pd.to_datetime(frame["game_start_time"], utc=True, errors="coerce")
    return frame


def _build_fixture_groups_v2(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    audit = candidates[
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
            "updated_at",
        ]
    ].copy()
    audit["group_key"] = ""

    group_rows: list[dict[str, Any]] = []
    grouped = candidates[candidates["classification_status"].astype(str).eq("candidate_true_1x2")].copy()
    if grouped.empty:
        return pd.DataFrame(group_rows), audit

    for match_id, group in grouped.groupby("match_id", observed=True):
        duplicates: list[str] = []
        selected_by_role: dict[str, pd.Series] = {}
        for role in pm_shadow.GROUP_ROLE_ORDER:
            role_rows = group[group["market_role"].astype(str).eq(role)].copy()
            if role_rows.empty:
                continue
            role_rows = role_rows.sort_values(
                ["market_quality_rank", "game_start_time", "market_id"],
                ascending=[False, True, True],
            )
            selected_by_role[role] = role_rows.iloc[0]
            duplicates.extend(role_rows.iloc[1:]["market_id"].astype(str).tolist())

        base_row = group.sort_values(["market_quality_rank", "game_start_time"], ascending=[False, True]).iloc[0]
        group_key = str(base_row["fixture_key"])
        status = "complete_group" if set(selected_by_role) == set(pm_shadow.GROUP_ROLE_ORDER) else "partial_group"
        group_reason = "true_1x2_complete" if status == "complete_group" else "missing_true_1x2_leg"
        group_rows.append(
            {
                "group_key": group_key,
                "fixture_key": str(base_row["fixture_key"]),
                "undirected_fixture_key": str(base_row["undirected_fixture_key"]),
                "event_slug": str(base_row["event_slug"]),
                "event_title": str(base_row["event_title"]),
                "league_code": str(base_row["league_code"]),
                "league_name": str(base_row["league_name"]),
                "sport_code": str(base_row["sport_code"]),
                "match_id": str(match_id),
                "match_date": pm_shadow._iso_timestamp(pd.Timestamp(base_row["match_date"])),
                "game_start_time": pm_shadow._iso_timestamp(pd.Timestamp(base_row["game_start_time"])),
                "home_team": str(base_row["home_team_canonical"]),
                "away_team": str(base_row["away_team_canonical"]),
                "home_market_id": str(selected_by_role.get(OUTCOME_HOME, {}).get("market_id", "")) if OUTCOME_HOME in selected_by_role else "",
                "draw_market_id": str(selected_by_role.get(OUTCOME_DRAW, {}).get("market_id", "")) if OUTCOME_DRAW in selected_by_role else "",
                "away_market_id": str(selected_by_role.get(OUTCOME_AWAY, {}).get("market_id", "")) if OUTCOME_AWAY in selected_by_role else "",
                "group_status": status,
                "group_reason": group_reason,
                "source_event_slugs_json": pm_shadow._clean_json(sorted(group["event_slug"].astype(str).unique().tolist())),
                "source_market_ids_json": pm_shadow._clean_json(sorted(group["market_id"].astype(str).tolist())),
                "duplicate_market_ids_json": pm_shadow._clean_json(sorted(duplicates)),
                "updated_at": str(base_row["updated_at"]),
            }
        )

        for role, selected in selected_by_role.items():
            audit.loc[audit["market_id"].astype(str).eq(str(selected["market_id"])), "group_key"] = group_key
            if status == "complete_group":
                audit.loc[audit["market_id"].astype(str).eq(str(selected["market_id"])), "classification_status"] = "1x2_counted"
                audit.loc[audit["market_id"].astype(str).eq(str(selected["market_id"])), "classification_reason"] = "selected_true_1x2_leg"
            else:
                audit.loc[audit["market_id"].astype(str).eq(str(selected["market_id"])), "classification_status"] = "partial_1x2"
                audit.loc[audit["market_id"].astype(str).eq(str(selected["market_id"])), "classification_reason"] = "incomplete_true_1x2_group"
        for duplicate_market_id in duplicates:
            audit.loc[audit["market_id"].astype(str).eq(str(duplicate_market_id)), "group_key"] = group_key
            audit.loc[audit["market_id"].astype(str).eq(str(duplicate_market_id)), "classification_status"] = "duplicate"
            audit.loc[audit["market_id"].astype(str).eq(str(duplicate_market_id)), "classification_reason"] = "duplicate_true_1x2_leg"

    return pd.DataFrame(group_rows), audit


def _legacy_tables_from_v2(
    candidates: pd.DataFrame,
    groups_v2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty or groups_v2.empty:
        return pd.DataFrame(), pd.DataFrame()
    complete_groups = groups_v2[groups_v2["group_status"].astype(str).eq("complete_group")].copy()
    if complete_groups.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected_market_ids = {
        str(market_id)
        for row in complete_groups.itertuples(index=False)
        for market_id in (row.home_market_id, row.draw_market_id, row.away_market_id)
        if str(market_id)
    }
    selected = candidates[candidates["market_id"].astype(str).isin(selected_market_ids)].copy()
    selected = selected.merge(
        complete_groups[
            [
                "match_id",
                "group_key",
                "home_market_id",
                "draw_market_id",
                "away_market_id",
                "home_team",
                "away_team",
                "game_start_time",
                "league_code",
                "league_name",
                "sport_code",
                "event_slug",
                "event_title",
                "source_market_ids_json",
                "updated_at",
            ]
        ],
        on="match_id",
        how="left",
        suffixes=("", "_group"),
    )
    catalog_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        token_ids = pm_shadow._json_list(row.clob_token_ids_json)
        catalog_rows.append(
            {
                "market_id": str(row.market_id),
                "event_id": str(row.event_id),
                "event_slug": str(row.event_slug_group or row.event_slug),
                "event_title": str(row.event_title_group or row.event_title),
                "market_slug": str(row.market_slug),
                "question": str(row.question),
                "league_code": str(row.league_code_group or row.league_code),
                "league_name": str(row.league_name_group or row.league_name),
                "sport_code": str(row.sport_code_group or row.sport_code),
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "market_role": str(row.market_role),
                "game_start_time": pm_shadow._iso_timestamp(pd.Timestamp(row.game_start_time_group or row.game_start_time)),
                "yes_token_id": str(token_ids[0]) if token_ids else "",
                "no_token_id": str(token_ids[1]) if len(token_ids) > 1 else "",
                "fees_enabled": int(row.fees_enabled),
                "fee_rate": float(row.fee_rate),
                "status": pm_shadow._status_label(bool(row.active), bool(row.closed), bool(row.accepting_orders)),
                "active": int(row.active),
                "closed": int(row.closed),
                "accepting_orders": int(row.accepting_orders),
                "raw_json": str(row.raw_json),
                "updated_at": str(row.updated_at_group or row.updated_at),
            }
        )
    legacy_groups = complete_groups.rename(
        columns={
            "group_status": "mapping_status",
            "group_reason": "mapping_reason",
            "source_market_ids_json": "raw_market_ids_json",
        }
    )[
        [
            "group_key",
            "event_slug",
            "event_title",
            "league_code",
            "league_name",
            "sport_code",
            "match_id",
            "home_team",
            "away_team",
            "game_start_time",
            "home_market_id",
            "draw_market_id",
            "away_market_id",
            "mapping_status",
            "mapping_reason",
            "raw_market_ids_json",
            "updated_at",
        ]
    ].copy()
    legacy_groups["mapping_status"] = "complete"
    return pd.DataFrame(catalog_rows), legacy_groups


__all__ = [name for name in globals() if not name.startswith("__")]
