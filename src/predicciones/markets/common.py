from __future__ import annotations

import difflib
import json
import re
import unicodedata
from datetime import UTC
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME, OUTCOME_ORDER
from ..data_sources import LEAGUE_NAMES

LEAGUE_TO_SPORT = {
    "E0": "epl",
    "SP1": "lal",
    "D1": "bun",
}
SPORT_TO_LEAGUE = {value: key for key, value in LEAGUE_TO_SPORT.items()}
GROUP_ROLE_ORDER = (OUTCOME_HOME, OUTCOME_DRAW, OUTCOME_AWAY)
TEAM_SUFFIX_PATTERN = re.compile(r"\b(fc|cf|afc|ac|sc|calcio|club)\b", re.IGNORECASE)
MATCH_TITLE_PATTERN = re.compile(r"^\s*(?P<home>.+?)\s+vs\.?\s+(?P<away>.+?)\s*$", re.IGNORECASE)
QUALITY_TIER_RANK = {"resolution_only": 0, "history_proxy": 1, "history_exact": 2}
SOURCE_MODE_RETRO = "retro_approx"
SOURCE_MODE_FORWARD = "forward_t45m"
BUNDLE_STATUS_PROVISIONAL = "provisional"
BUNDLE_STATUS_PROMOTABLE = "promotable_for_forward"
PRICE_PROVENANCE_EXACT = "exact"
PRICE_PROVENANCE_PROXY = "proxy"
PRICE_PROVENANCE_RESOLUTION_ONLY = "resolution_only"
VALIDATION_STAGE_RETRO = "retro"
VALIDATION_STAGE_SHADOW = "shadow"
VALIDATION_STAGE_LIVE = "live"
BUNDLE_READINESS_VALIDATED = "validated"
COVERAGE_STATUS_LIMITED = "coverage_limited"
COVERAGE_STATUS_READY = "coverage_ready"
SOURCE_MODE_TO_VALIDATION_STAGE = {
    SOURCE_MODE_RETRO: VALIDATION_STAGE_RETRO,
    SOURCE_MODE_FORWARD: VALIDATION_STAGE_SHADOW,
    VALIDATION_STAGE_LIVE: VALIDATION_STAGE_LIVE,
}
QUALITY_TIER_TO_PRICE_PROVENANCE = {
    "history_exact": PRICE_PROVENANCE_EXACT,
    "history_proxy": PRICE_PROVENANCE_PROXY,
    "resolution_only": PRICE_PROVENANCE_RESOLUTION_ONLY,
}
SOURCE_TYPE_TO_PRICE_PROVENANCE = {
    "polymarket_checkpoint": PRICE_PROVENANCE_EXACT,
    "polymarket_history_local": PRICE_PROVENANCE_EXACT,
    "polymarket_history_remote": PRICE_PROVENANCE_EXACT,
    "bookmaker_proxy": PRICE_PROVENANCE_PROXY,
    "resolution_only": PRICE_PROVENANCE_RESOLUTION_ONLY,
}
PRICE_PROVENANCE_ORDER = {
    PRICE_PROVENANCE_EXACT: 0,
    PRICE_PROVENANCE_PROXY: 1,
    PRICE_PROVENANCE_RESOLUTION_ONLY: 2,
}
BUNDLE_READINESS_ORDER = {
    BUNDLE_STATUS_PROVISIONAL: 0,
    BUNDLE_STATUS_PROMOTABLE: 1,
    BUNDLE_READINESS_VALIDATED: 2,
}
LEGACY_SPORT_ALIASES = {
    "epl": ("epl", "premier league", "premier-league"),
    "lal": ("la liga", "laliga", "la-liga"),
    "bun": ("bundesliga",),
}
OUT_OF_SCOPE_EVENT_KEYWORDS = (
    "euro 2024",
    "euros",
    "euro ",
    "copa america",
    "olympic",
    "olympics",
    "uefa super cup",
    "super cup",
    "champions league",
    "europa league",
    "conference league",
    "nations league",
    "world cup",
    "club world cup",
    "copa del rey",
    "to advance",
    "fa cup",
    "carabao cup",
    "efl cup",
    "coppa italia",
)
GENERIC_PREFIX_TOKENS = {
    "fc",
    "cf",
    "afc",
    "sc",
    "ac",
    "ud",
    "cd",
    "ca",
    "rc",
    "rcd",
    "sv",
    "tsg",
    "vfl",
    "vfb",
    "fsv",
}
LEAGUE_TEAM_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "E0": {
        "Arsenal": ("arsenal",),
        "Aston Villa": ("aston villa", "villa"),
        "Bournemouth": ("bournemouth", "afc bournemouth"),
        "Brentford": ("brentford",),
        "Brighton": ("brighton", "brighton hove albion", "brighton and hove albion"),
        "Burnley": ("burnley",),
        "Chelsea": ("chelsea",),
        "Crystal Palace": ("crystal palace", "palace"),
        "Everton": ("everton",),
        "Fulham": ("fulham",),
        "Ipswich": ("ipswich", "ipswich town"),
        "Leeds": ("leeds", "leeds united"),
        "Leicester": ("leicester", "leicester city"),
        "Liverpool": ("liverpool",),
        "Luton": ("luton", "luton town"),
        "Man City": ("man city", "manchester city", "manchester city fc"),
        "Man United": ("man united", "man utd", "manchester united", "manchester utd", "manchester united fc"),
        "Newcastle": ("newcastle", "newcastle united"),
        "Nott'm Forest": ("nott m forest", "nottingham forest", "nottingham forest fc", "forest"),
        "Sheffield United": ("sheffield united", "sheff utd", "sheff united"),
        "Southampton": ("southampton",),
        "Tottenham": ("tottenham", "tottenham hotspur", "spurs"),
        "West Ham": ("west ham", "west ham united"),
        "Wolves": ("wolves", "wolverhampton", "wolverhampton wanderers"),
    },
    "SP1": {
        "Alaves": ("alaves", "deportivo alaves"),
        "Almeria": ("almeria", "ud almeria"),
        "Ath Bilbao": ("ath bilbao", "athletic bilbao", "athletic club", "athletic club bilbao"),
        "Ath Madrid": ("ath madrid", "atletico madrid", "atletico de madrid", "atleti"),
        "Barcelona": ("barcelona", "fc barcelona", "barca"),
        "Betis": ("betis", "real betis"),
        "Cadiz": ("cadiz", "cadiz cf"),
        "Celta": ("celta", "celta vigo", "real club celta de vigo", "rc celta"),
        "Elche": ("elche", "elche cf"),
        "Espanol": ("espanol", "espanyol", "rcd espanyol"),
        "Getafe": ("getafe", "getafe cf"),
        "Girona": ("girona", "girona fc"),
        "Granada": ("granada", "granada cf"),
        "Las Palmas": ("las palmas", "ud las palmas"),
        "Leganes": ("leganes", "cd leganes"),
        "Mallorca": ("mallorca", "rcd mallorca"),
        "Osasuna": ("osasuna", "ca osasuna"),
        "Real Madrid": ("real madrid", "real madrid cf"),
        "Sevilla": ("sevilla", "sevilla fc"),
        "Sociedad": ("sociedad", "real sociedad"),
        "Valencia": ("valencia", "valencia cf"),
        "Valladolid": ("valladolid", "real valladolid"),
        "Vallecano": ("vallecano", "rayo vallecano"),
        "Villarreal": ("villarreal", "villarreal cf"),
    },
    "D1": {
        "Augsburg": ("augsburg", "fc augsburg"),
        "Bayern Munich": ("bayern munich", "bayern", "bayern munchen", "fc bayern munich", "fc bayern munchen"),
        "Bochum": ("bochum", "vfl bochum"),
        "Darmstadt": ("darmstadt", "darmstadt 98", "sv darmstadt 98"),
        "Dortmund": ("dortmund", "borussia dortmund", "bvb"),
        "Ein Frankfurt": ("ein frankfurt", "eintracht frankfurt"),
        "FC Koln": ("fc koln", "koln", "1 fc koln", "cologne"),
        "Freiburg": ("freiburg", "sc freiburg"),
        "Heidenheim": ("heidenheim", "fc heidenheim", "1 fc heidenheim"),
        "Hertha": ("hertha", "hertha berlin", "hertha bsc"),
        "Hoffenheim": ("hoffenheim", "tsg hoffenheim"),
        "Holstein Kiel": ("holstein kiel", "kiel"),
        "Leverkusen": ("leverkusen", "bayer leverkusen", "bayer 04 leverkusen", "b04"),
        "M'gladbach": ("m gladbach", "mgladbach", "monchengladbach", "borussia monchengladbach"),
        "Mainz": ("mainz", "mainz 05", "1 fsv mainz 05"),
        "RB Leipzig": ("rb leipzig", "rasenballsport leipzig", "leipzig"),
        "Schalke 04": ("schalke 04", "schalke", "fc schalke 04"),
        "St Pauli": ("st pauli", "fc st pauli", "fc st pauli 1910", "sankt pauli"),
        "Stuttgart": ("stuttgart", "vfb stuttgart"),
        "Union Berlin": ("union berlin", "1 fc union berlin"),
        "Werder Bremen": ("werder bremen", "bremen", "sv werder bremen"),
        "Wolfsburg": ("wolfsburg", "vfl wolfsburg"),
    },
}


def normalize_team_name(name: str) -> str:
    value = unicodedata.normalize("NFKC", str(name).strip())
    value = value.replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", value)


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz=UTC)


def _iso_timestamp(value: Any | None = None) -> str:
    timestamp = pd.Timestamp(value if value is not None else _utcnow())
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.isoformat()


def _parse_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, "", pd.NaT):
        return pd.NaT
    if isinstance(value, (int, float)) or (isinstance(value, str) and str(value).isdigit()):
        numeric = int(value)
        unit = "ms" if numeric > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True)
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(UTC)
    else:
        parsed = parsed.tz_convert(UTC)
    return parsed


def _clean_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _normalize_team_key(name: str) -> str:
    value = normalize_team_name(name).lower()
    value = _strip_accents(value)
    value = TEAM_SUFFIX_PATTERN.sub(" ", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _manual_alias_family(league_code: str | None, name: str) -> set[str]:
    if not league_code:
        return set()
    key = _normalize_team_key(name)
    for canonical_team, aliases in LEAGUE_TEAM_ALIASES.get(str(league_code), {}).items():
        family = {_normalize_team_key(canonical_team)}
        family.update(_normalize_team_key(alias) for alias in aliases)
        if key in family:
            return family
    return set()


def _generic_team_variants(name: str) -> set[str]:
    base = _normalize_team_key(name)
    if not base:
        return set()
    variants = {base}
    tokens = base.split()
    while tokens and tokens[0] in GENERIC_PREFIX_TOKENS:
        tokens = tokens[1:]
        if tokens:
            variants.add(" ".join(tokens))
    phrase_replacements = {
        "man city": ("manchester city",),
        "man united": ("manchester united", "man utd"),
        "man utd": ("man united", "manchester united"),
        "nott m forest": ("nottingham forest",),
        "wolves": ("wolverhampton", "wolverhampton wanderers"),
        "wolverhampton": ("wolves", "wolverhampton wanderers"),
        "wolverhampton wanderers": ("wolves", "wolverhampton"),
        "espanol": ("espanyol",),
        "espanyol": ("espanol",),
        "celta": ("celta vigo",),
        "celta vigo": ("celta",),
        "ath madrid": ("atletico madrid",),
        "atletico madrid": ("ath madrid",),
        "ath bilbao": ("athletic bilbao", "athletic club"),
        "athletic bilbao": ("ath bilbao", "athletic club"),
        "athletic club": ("ath bilbao", "athletic bilbao"),
        "ein frankfurt": ("eintracht frankfurt",),
        "eintracht frankfurt": ("ein frankfurt",),
        "bayern": ("bayern munich", "bayern munchen"),
        "bayern munich": ("bayern", "bayern munchen"),
        "bayern munchen": ("bayern", "bayern munich"),
        "fc koln": ("koln", "1 fc koln", "cologne"),
        "koln": ("fc koln", "1 fc koln", "cologne"),
        "m gladbach": ("monchengladbach", "mgladbach"),
        "monchengladbach": ("m gladbach", "mgladbach"),
        "mgladbach": ("m gladbach", "monchengladbach"),
        "st pauli": ("sankt pauli", "fc st pauli"),
        "sankt pauli": ("st pauli", "fc st pauli"),
        "leverkusen": ("bayer leverkusen", "bayer 04 leverkusen"),
        "bayer leverkusen": ("leverkusen", "bayer 04 leverkusen"),
        "sociedad": ("real sociedad",),
        "betis": ("real betis",),
        "valladolid": ("real valladolid",),
    }
    for variant in list(variants):
        for replacement in phrase_replacements.get(variant, ()):
            normalized = _normalize_team_key(replacement)
            if normalized:
                variants.add(normalized)
    return {variant for variant in variants if variant}


def _team_alias_variants(name: str, league_code: str | None = None) -> set[str]:
    variants = _generic_team_variants(name)
    manual_family = _manual_alias_family(league_code, name)
    if manual_family:
        variants.update(manual_family)
    return variants


def _team_match_score(name: str, candidate: str, league_code: str | None = None) -> float:
    left = _team_alias_variants(name, league_code)
    right = _team_alias_variants(candidate, league_code)
    if not left or not right:
        return 0.0
    if left & right:
        return 1.0
    return max(difflib.SequenceMatcher(None, left_key, right_key).ratio() for left_key in left for right_key in right)


def _best_team_alias_match(name: str, candidates: list[str], league_code: str | None = None) -> tuple[float, str]:
    if not candidates:
        return 0.0, normalize_team_name(name)
    scored = sorted((_team_match_score(name, candidate, league_code), candidate) for candidate in candidates)
    return scored[-1]


def _parse_match_title(title: str) -> tuple[str, str] | None:
    value = str(title).strip()
    if ":" in value and " vs" in value.lower():
        _, suffix = value.split(":", 1)
        if " vs" in suffix.lower():
            value = suffix.strip()
    match = MATCH_TITLE_PATTERN.match(value)
    if not match:
        return None
    return normalize_team_name(match.group("home")), normalize_team_name(match.group("away"))


def _infer_league_from_event(event_slug: str, event: dict[str, Any]) -> tuple[str, str, str] | tuple[None, None, None]:
    title = str(event.get("title", "")).strip()
    series_slug = str(event.get("seriesSlug", "")).strip().lower()
    slug = str(event_slug).strip().lower()
    tag_labels = [
        re.sub(r"[^a-z0-9]+", " ", _strip_accents(str(item.get("label", "")).lower())).strip()
        for item in event.get("tags", [])
        if isinstance(item, dict)
    ]
    tag_slugs = [
        re.sub(r"[^a-z0-9]+", " ", _strip_accents(str(item.get("slug", "")).lower())).strip()
        for item in event.get("tags", [])
        if isinstance(item, dict)
    ]
    title_text = re.sub(r"[^a-z0-9]+", " ", _strip_accents(title.lower())).strip()
    scope_texts = [title_text, slug.replace("-", " "), series_slug.replace("-", " "), *tag_labels, *tag_slugs]
    if series_slug not in SPORT_TO_LEAGUE and any(
        keyword in text for keyword in OUT_OF_SCOPE_EVENT_KEYWORDS for text in scope_texts if text
    ):
        return None, None, None

    if series_slug in SPORT_TO_LEAGUE:
        league_code = SPORT_TO_LEAGUE.get(series_slug, series_slug.upper())
        league_name = LEAGUE_NAMES.get(league_code, league_code)
        return series_slug, league_code, league_name

    for sport_code, aliases in LEGACY_SPORT_ALIASES.items():
        for alias in aliases:
            alias_text = re.sub(r"[^a-z0-9]+", " ", _strip_accents(alias.lower())).strip()
            alias_slug = alias_text.replace(" ", "-")
            if title_text == alias_text or title_text.startswith(f"{alias_text} "):
                league_code = SPORT_TO_LEAGUE.get(sport_code, sport_code.upper())
                league_name = LEAGUE_NAMES.get(league_code, league_code)
                return sport_code, league_code, league_name
            if slug == alias_slug or slug.startswith(f"{alias_slug}-"):
                league_code = SPORT_TO_LEAGUE.get(sport_code, sport_code.upper())
                league_name = LEAGUE_NAMES.get(league_code, league_code)
                return sport_code, league_code, league_name
    return None, None, None


def _status_label(active: bool, closed: bool, accepting_orders: bool) -> str:
    if closed:
        return "closed"
    if active and accepting_orders:
        return "open"
    if active:
        return "active_no_orders"
    return "inactive"


def _event_game_start(event: dict[str, Any]) -> pd.Timestamp | pd.NaT:
    timestamps = []
    direct_start = _parse_timestamp(event.get("startTime") or event.get("gameStartTime") or event.get("eventDate"))
    if not pd.isna(direct_start):
        timestamps.append(direct_start)
    for market in event.get("markets", []):
        game_start = market.get("gameStartTime")
        if not game_start:
            continue
        timestamps.append(_parse_timestamp(game_start))
    if not timestamps:
        return pd.NaT
    return min(timestamps)


def _infer_market_role(question: str, home_team: str, away_team: str) -> str | None:
    question_key = _normalize_team_key(question)
    home_key = _normalize_team_key(home_team)
    away_key = _normalize_team_key(away_team)
    if "draw" in question_key:
        return OUTCOME_DRAW
    if home_key and question_key.startswith(f"will {home_key} "):
        return OUTCOME_HOME
    if away_key and question_key.startswith(f"will {away_key} "):
        return OUTCOME_AWAY
    if home_key and home_key in question_key and any(token in question_key for token in ("win", "beat", "beats", "defeat", "defeats")):
        return OUTCOME_HOME
    if away_key and away_key in question_key and any(token in question_key for token in ("win", "beat", "beats", "defeat", "defeats")):
        return OUTCOME_AWAY
    return None


def _order_levels(raw_levels: list[dict[str, Any]], ascending: bool) -> list[dict[str, float]]:
    levels = [
        {"price": float(item["price"]), "size": float(item["size"])}
        for item in raw_levels or []
        if item.get("price") is not None and item.get("size") is not None
    ]
    return sorted(levels, key=lambda item: item["price"], reverse=not ascending)


def _event_within_window(event: dict[str, Any], now: pd.Timestamp, window_hours: float) -> bool:
    game_start = _event_game_start(event)
    if pd.isna(game_start):
        return False
    window_end = now + pd.to_timedelta(window_hours, unit="h")
    return now <= game_start <= window_end


def _history_team_lookup(history_matches: pd.DataFrame) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    unique_rows = pd.concat(
        [
            history_matches[["league_code", "HomeTeam"]].rename(columns={"HomeTeam": "team"}),
            history_matches[["league_code", "AwayTeam"]].rename(columns={"AwayTeam": "team"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    for league_code, group in unique_rows.groupby("league_code", observed=True):
        lookup[str(league_code)] = sorted(group["team"].astype(str).unique().tolist())
    return lookup


def _resolve_team_alias(name: str, candidates: list[str], league_code: str | None = None) -> str:
    if not candidates:
        return normalize_team_name(name)
    exact_key = _normalize_team_key(name)
    for candidate in candidates:
        if exact_key in _team_alias_variants(candidate, league_code):
            return candidate
    best_score, best_match = _best_team_alias_match(name, candidates, league_code)
    return best_match if best_score >= 0.72 else normalize_team_name(name)


def _quality_at_or_above(value: str, minimum: str) -> bool:
    return QUALITY_TIER_RANK.get(str(value), -1) >= QUALITY_TIER_RANK.get(str(minimum), -1)


def _source_mode_validation_stage(source_mode: str) -> str:
    return SOURCE_MODE_TO_VALIDATION_STAGE.get(str(source_mode), str(source_mode).strip() or VALIDATION_STAGE_LIVE)


def _price_provenance_from_quality_tier(quality_tier: str) -> str:
    return QUALITY_TIER_TO_PRICE_PROVENANCE.get(str(quality_tier), PRICE_PROVENANCE_RESOLUTION_ONLY)


def _price_provenance_from_source_type(source_type: str) -> str:
    return SOURCE_TYPE_TO_PRICE_PROVENANCE.get(str(source_type), PRICE_PROVENANCE_RESOLUTION_ONLY)


def _bundle_readiness_from_status(bundle_status: str) -> str:
    status = str(bundle_status).strip()
    if not status:
        return BUNDLE_STATUS_PROVISIONAL
    if status == BUNDLE_READINESS_VALIDATED:
        return BUNDLE_READINESS_VALIDATED
    return status


def _dominant_label(counts: Mapping[str, Any] | None, preferred_order: tuple[str, ...], fallback: str) -> str:
    normalized = {
        str(label): int(value)
        for label, value in (counts or {}).items()
        if value is not None and int(value) > 0
    }
    if not normalized:
        return fallback
    order = {label: index for index, label in enumerate(preferred_order)}
    return sorted(normalized.items(), key=lambda item: (-item[1], order.get(item[0], 99), item[0]))[0][0]


def build_polymarket_lifecycle_summary(
    *,
    source_mode: str,
    bundle_status: str = BUNDLE_STATUS_PROVISIONAL,
    price_provenance_counts: Mapping[str, Any] | None = None,
    validation_stage: str | None = None,
    provenance_counts_label: str = "price_provenance_counts",
) -> dict[str, Any]:
    counts = {
        str(label): int(value)
        for label, value in (price_provenance_counts or {}).items()
        if value is not None and int(value) > 0
    }
    stage = str(validation_stage).strip() if validation_stage is not None else _source_mode_validation_stage(source_mode)
    if not stage:
        stage = VALIDATION_STAGE_LIVE
    dominant_price_provenance = _dominant_label(
        counts,
        (
            PRICE_PROVENANCE_EXACT,
            PRICE_PROVENANCE_PROXY,
            PRICE_PROVENANCE_RESOLUTION_ONLY,
        ),
        PRICE_PROVENANCE_RESOLUTION_ONLY if counts else PRICE_PROVENANCE_EXACT,
    )
    readiness = _bundle_readiness_from_status(bundle_status)
    lifecycle = {
        "source_mode": str(source_mode),
        "validation_stage": stage,
        "price_provenance": dominant_price_provenance,
        provenance_counts_label: counts,
        "bundle_status": readiness,
        "bundle_readiness": readiness,
        "bundle_ready": readiness in {BUNDLE_STATUS_PROMOTABLE, BUNDLE_READINESS_VALIDATED},
        "lifecycle_label": f"{stage} / {dominant_price_provenance} / {readiness}",
    }
    return lifecycle


def format_polymarket_lifecycle_label(lifecycle: Mapping[str, Any] | None) -> str:
    if not lifecycle:
        return "unknown / unknown / unknown"
    validation_stage = str(lifecycle.get("validation_stage", "unknown"))
    price_provenance = str(lifecycle.get("price_provenance", "unknown"))
    bundle_readiness = str(lifecycle.get("bundle_readiness", lifecycle.get("bundle_status", "unknown")))
    return f"{validation_stage} / {price_provenance} / {bundle_readiness}"


_SKIP_REASON_TO_BLOCKER = {
    "no_market_group": "mapping",
    "incomplete_market_group": "mapping",
    "market_not_in_catalog": "mapping",
    "no_book_checkpoint": "coverage",
    "stale_book": "coverage",
    "no_ask_ladder": "coverage",
    "policy_rejected": "policy",
}
_BOOK_AGE_BUCKETS = (
    (5 * 60, "0-5m"),
    (15 * 60, "5-15m"),
    (30 * 60, "15-30m"),
    (45 * 60, "30-45m"),
    (60 * 60, "45-60m"),
)
_BOOK_AGE_BUCKET_ORDER = {label: index for index, (_, label) in enumerate(_BOOK_AGE_BUCKETS)}
_BOOK_AGE_BUCKET_ORDER["60m+"] = len(_BOOK_AGE_BUCKET_ORDER)


def _skip_reason_blocker(reason: str) -> str:
    reason = str(reason).strip()
    return _SKIP_REASON_TO_BLOCKER.get(reason, "other") if reason else "none"


def _book_age_bucket(age_seconds: Any) -> str:
    if age_seconds is None or pd.isna(age_seconds):
        return "missing"
    try:
        age_seconds = float(age_seconds)
    except (TypeError, ValueError):
        return "missing"
    for threshold, label in _BOOK_AGE_BUCKETS:
        if age_seconds <= threshold:
            return label
    return "60m+"
