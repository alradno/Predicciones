from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .contracts import CaptureOddsResult, METADATA_COLUMNS
from .ingestion import normalize_team_name


def _default_kickoff(date_series: pd.Series, default_hour: int) -> pd.Series:
    normalized = pd.to_datetime(date_series).dt.normalize()
    return normalized + pd.to_timedelta(default_hour, unit="h")


def _categorize_time_bucket(minutes: float, edges: tuple[int, ...]) -> str:
    if np.isnan(minutes):
        return "unknown"
    start = 0
    for edge in edges:
        if minutes <= edge:
            return f"{start}-{edge}m"
        start = edge
    return f">{edges[-1]}m"


def _season_phase(date_value: pd.Timestamp) -> str:
    month = int(pd.Timestamp(date_value).month)
    if month in {7, 8, 9, 10}:
        return "early"
    if month in {11, 12, 1}:
        return "mid"
    return "late"


def _odds_band(odds: float) -> str:
    if pd.isna(odds):
        return "unknown"
    if odds <= 1.5:
        return "<=1.5"
    if odds <= 2.0:
        return "1.5-2.0"
    if odds <= 3.0:
        return "2.0-3.0"
    if odds <= 5.0:
        return "3.0-5.0"
    return ">5.0"


def _edge_band(edge: float) -> str:
    if pd.isna(edge):
        return "unknown"
    if edge < 0.02:
        return "<0.02"
    if edge < 0.05:
        return "0.02-0.05"
    if edge < 0.08:
        return "0.05-0.08"
    return ">=0.08"


def _prepare_snapshot_frame(snapshots: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if snapshots.empty:
        return snapshots.copy()

    frame = snapshots.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    if "kickoff_time" in frame.columns:
        frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], errors="coerce")
    else:
        frame["kickoff_time"] = pd.NaT
    missing_kickoff = frame["kickoff_time"].isna()
    frame.loc[missing_kickoff, "kickoff_time"] = _default_kickoff(frame.loc[missing_kickoff, "Date"], settings.snapshot.default_kickoff_hour)

    if "snapshot_time" in frame.columns:
        frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], errors="coerce")
    else:
        frame["snapshot_time"] = pd.NaT
    missing_snapshot = frame["snapshot_time"].isna()
    frame.loc[missing_snapshot, "snapshot_time"] = frame.loc[missing_snapshot, "kickoff_time"] - pd.to_timedelta(
        settings.snapshot.closing_proxy_capture_minutes,
        unit="m",
    )

    for outcome in ["home", "draw", "away"]:
        frame[f"odds_{outcome}"] = pd.to_numeric(frame.get(f"odds_{outcome}"), errors="coerce")

    implied_columns: list[str] = []
    for outcome in ["home", "draw", "away"]:
        implied = f"implied_{outcome}_raw"
        if implied not in frame.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                frame[implied] = 1.0 / frame[f"odds_{outcome}"]
        implied_columns.append(implied)

    total = frame[implied_columns].sum(axis=1)
    frame["bookmaker_margin"] = total - 1.0
    for outcome in ["home", "draw", "away"]:
        frame[f"market_prob_{outcome}"] = np.where(total > 0, frame[f"implied_{outcome}_raw"] / total, np.nan)

    frame["source_name"] = frame.get("source_name", pd.Series("bet365_closing_proxy", index=frame.index)).fillna("bet365_closing_proxy")
    frame["source_type"] = frame.get("source_type", pd.Series("closing_proxy", index=frame.index)).fillna("closing_proxy")
    frame["liquidity"] = pd.to_numeric(frame.get("liquidity", pd.Series(np.nan, index=frame.index)), errors="coerce").fillna(np.inf)
    frame["time_to_kickoff_minutes"] = (
        (frame["kickoff_time"] - frame["snapshot_time"]).dt.total_seconds().div(60.0).clip(lower=0.0)
    )
    frame["time_bucket"] = frame["time_to_kickoff_minutes"].map(
        lambda value: _categorize_time_bucket(float(value), settings.snapshot.time_bucket_edges)
    )

    frame = frame.sort_values(["match_id", "source_name", "snapshot_time"]).reset_index(drop=True)
    group_keys = ["match_id", "source_name"]
    for outcome in ["home", "draw", "away"]:
        opening_prob = frame.groupby(group_keys, observed=True)[f"market_prob_{outcome}"].transform("first")
        delta = frame[f"market_prob_{outcome}"] - opening_prob
        regime = np.where(
            delta > settings.snapshot.movement_threshold,
            "steam",
            np.where(delta < -settings.snapshot.movement_threshold, "drift", "flat"),
        )
        frame[f"movement_delta_{outcome}"] = delta
        frame[f"movement_regime_{outcome}"] = regime

    return frame


def build_market_snapshots(
    market_odds: pd.DataFrame,
    settings: Settings,
    source_name: str = "bet365_closing_proxy",
    source_type: str = "closing_proxy",
) -> pd.DataFrame:
    frame = market_odds.copy()
    if "league_name" not in frame.columns:
        frame["league_name"] = frame["league_code"]
    frame["kickoff_time"] = _default_kickoff(frame["Date"], settings.snapshot.default_kickoff_hour)
    frame["snapshot_time"] = frame["kickoff_time"] - pd.to_timedelta(settings.snapshot.closing_proxy_capture_minutes, unit="m")
    frame["source_name"] = source_name
    frame["source_type"] = source_type
    frame["liquidity"] = np.inf
    return _prepare_snapshot_frame(frame, settings)


def load_snapshot_file(path: Path | str, reference_matches: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "match_id" not in raw.columns:
        required = {"Date", "league_code", "HomeTeam", "AwayTeam"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"El snapshot {path} no trae match_id ni claves suficientes: {sorted(missing)}")
        working = raw.copy()
        working["Date"] = pd.to_datetime(working["Date"], format="mixed", dayfirst=True, errors="coerce")
        working["HomeTeam"] = working["HomeTeam"].map(normalize_team_name)
        working["AwayTeam"] = working["AwayTeam"].map(normalize_team_name)
        merged = working.merge(
            reference_matches[list(METADATA_COLUMNS)],
            on=["Date", "league_code", "HomeTeam", "AwayTeam"],
            how="left",
            suffixes=("", "_ref"),
        )
        raw = merged
        raw["league_name"] = raw.get("league_name", raw.get("league_name_ref", raw["league_code"]))
        raw["season"] = raw.get("season", raw.get("season_ref"))
    else:
        raw = raw.merge(
            reference_matches[list(METADATA_COLUMNS)],
            on=["match_id"],
            how="left",
            suffixes=("", "_ref"),
        )
        for column in METADATA_COLUMNS:
            if column == "match_id":
                continue
            if column not in raw.columns or raw[column].isna().all():
                raw[column] = raw[f"{column}_ref"]

    return _prepare_snapshot_frame(raw, settings)


def capture_odds_for_dataset(
    dataset_dir: Path,
    settings: Settings,
    reference_matches: pd.DataFrame,
    market_odds: pd.DataFrame,
    snapshot_files: list[Path] | None = None,
) -> CaptureOddsResult:
    proxy = build_market_snapshots(market_odds=market_odds, settings=settings)
    extra_frames = [load_snapshot_file(path, reference_matches=reference_matches, settings=settings) for path in snapshot_files or []]
    snapshots = pd.concat([proxy, *extra_frames], ignore_index=True, sort=False) if extra_frames else proxy
    snapshots = _prepare_snapshot_frame(snapshots, settings)
    snapshots_path = dataset_dir / "market_snapshots.csv"
    snapshots.to_csv(snapshots_path, index=False)
    summary = {
        "rows": int(len(snapshots)),
        "sources": sorted(snapshots["source_name"].astype(str).unique().tolist()),
        "source_types": sorted(snapshots["source_type"].astype(str).unique().tolist()),
        "proxy_rows": int(snapshots["source_type"].eq("closing_proxy").sum()),
    }
    return CaptureOddsResult(dataset_dir=dataset_dir, snapshots_path=snapshots_path, snapshots=snapshots, summary=summary)
