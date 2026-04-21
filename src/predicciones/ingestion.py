from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import MARKET_ODDS_COLUMNS, RESULT_CODE_TO_OUTCOME
from .data_sources import DownloadResult, FootballDataClient

NUMERIC_COLUMNS = [
    "FTHG",
    "FTAG",
    "HTHG",
    "HTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
    "B365H",
    "B365D",
    "B365A",
]


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    matches_path: Path
    market_odds_path: Path
    market_snapshots_path: Path
    feature_rows_path: Path
    manifest_path: Path


def normalize_team_name(name: str) -> str:
    value = unicodedata.normalize("NFKC", str(name).strip())
    value = value.replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", value)


def infer_season_code(date_value: pd.Timestamp) -> str:
    year = int(date_value.year)
    month = int(date_value.month)
    start_year = year if month >= 7 else year - 1
    end_year = (start_year + 1) % 100
    return f"{start_year % 100:02d}{end_year:02d}"


def canonicalize_matches(raw_matches: pd.DataFrame, require_results: bool = True) -> pd.DataFrame:
    matches = raw_matches.copy()

    if "Date" not in matches.columns:
        raise ValueError("Falta la columna Date en el dataset.")

    matches["Date"] = pd.to_datetime(matches["Date"], format="mixed", dayfirst=True, errors="coerce")
    matches["HomeTeam"] = matches["HomeTeam"].map(normalize_team_name)
    matches["AwayTeam"] = matches["AwayTeam"].map(normalize_team_name)
    matches["league_code"] = matches["league_code"].astype(str).str.strip()
    matches["league_name"] = matches.get("league_name", matches["league_code"]).fillna(matches["league_code"])
    matches["league_name"] = matches["league_name"].astype(str).str.strip()

    if "season" not in matches.columns:
        matches["season"] = matches["Date"].map(infer_season_code)
    else:
        matches["season"] = matches["season"].fillna("").astype(str).str.strip()
        empty_mask = matches["season"].eq("")
        matches.loc[empty_mask, "season"] = matches.loc[empty_mask, "Date"].map(infer_season_code)

    for column in NUMERIC_COLUMNS:
        if column in matches.columns:
            matches[column] = pd.to_numeric(matches[column], errors="coerce")

    essential = ["Date", "league_code", "HomeTeam", "AwayTeam"]
    if require_results:
        essential.append("FTR")

    matches = matches.dropna(subset=essential).copy()

    if require_results:
        matches = matches[matches["FTR"].isin(RESULT_CODE_TO_OUTCOME)].copy()
        matches["outcome"] = matches["FTR"].map(RESULT_CODE_TO_OUTCOME)
    else:
        matches["outcome"] = matches.get("FTR", pd.Series(index=matches.index, dtype=object)).map(RESULT_CODE_TO_OUTCOME)

    matches["source_url"] = matches.get("source_url", pd.Series("", index=matches.index)).fillna("")

    matches = (
        matches.sort_values(["Date", "league_code", "HomeTeam", "AwayTeam", "season"])
        .drop_duplicates(subset=["Date", "league_code", "HomeTeam", "AwayTeam"], keep="last")
        .reset_index(drop=True)
    )
    matches["match_id"] = np.arange(len(matches), dtype=int)
    return matches


def build_market_odds(matches: pd.DataFrame) -> pd.DataFrame:
    market = matches[["match_id", "Date", "league_code", "season", "HomeTeam", "AwayTeam"]].copy()
    market["odds_home"] = matches.get("B365H")
    market["odds_draw"] = matches.get("B365D")
    market["odds_away"] = matches.get("B365A")

    if set(MARKET_ODDS_COLUMNS).issubset(matches.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            market["implied_home_raw"] = 1.0 / matches["B365H"]
            market["implied_draw_raw"] = 1.0 / matches["B365D"]
            market["implied_away_raw"] = 1.0 / matches["B365A"]

        total = market[["implied_home_raw", "implied_draw_raw", "implied_away_raw"]].sum(axis=1)
        market["bookmaker_margin"] = total - 1.0
        market["market_prob_home"] = market["implied_home_raw"] / total
        market["market_prob_draw"] = market["implied_draw_raw"] / total
        market["market_prob_away"] = market["implied_away_raw"] / total
    else:
        market["bookmaker_margin"] = np.nan
        market["market_prob_home"] = np.nan
        market["market_prob_draw"] = np.nan
        market["market_prob_away"] = np.nan

    return market


def build_dataset_paths(root: Path, dataset_name: str) -> DatasetPaths:
    dataset_root = root / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    return DatasetPaths(
        root=dataset_root,
        matches_path=dataset_root / "matches.csv",
        market_odds_path=dataset_root / "market_odds.csv",
        market_snapshots_path=dataset_root / "market_snapshots.csv",
        feature_rows_path=dataset_root / "feature_rows.csv",
        manifest_path=dataset_root / "manifest.json",
    )


def load_matches(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])


def download_historical_matches(leagues: list[str] | tuple[str, ...], seasons: list[str] | tuple[str, ...]) -> DownloadResult:
    client = FootballDataClient()
    return client.download(leagues=leagues, seasons=seasons)
