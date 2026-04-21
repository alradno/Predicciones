from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


OUTCOME_AWAY = "away"
OUTCOME_DRAW = "draw"
OUTCOME_HOME = "home"
OUTCOME_ORDER = (OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME)
OUTCOME_TO_TARGET = {OUTCOME_AWAY: 0, OUTCOME_DRAW: 1, OUTCOME_HOME: 2}
TARGET_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_TARGET.items()}
RESULT_CODE_TO_OUTCOME = {"A": OUTCOME_AWAY, "D": OUTCOME_DRAW, "H": OUTCOME_HOME}
OUTCOME_TO_RESULT_CODE = {value: key for key, value in RESULT_CODE_TO_OUTCOME.items()}

MARKET_ODDS_COLUMNS = ("B365H", "B365D", "B365A")
METADATA_COLUMNS = (
    "match_id",
    "Date",
    "league_code",
    "league_name",
    "season",
    "HomeTeam",
    "AwayTeam",
)


@dataclass(frozen=True)
class DataBundle:
    matches: pd.DataFrame
    market_odds: pd.DataFrame
    feature_rows: pd.DataFrame
    market_snapshots: pd.DataFrame | None = None


@dataclass(frozen=True)
class IngestSummary:
    rows_downloaded: int
    rows_canonical: int
    failures: list[str]
    artifacts: dict[str, Path]
    bundle: DataBundle


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path


@dataclass(frozen=True)
class FoldWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    test_rows: int


@dataclass(frozen=True)
class FoldArtifacts:
    window: FoldWindow
    predictions: pd.DataFrame
    bets: pd.DataFrame
    metrics: dict[str, Any]
    policy: dict[str, Any]


@dataclass(frozen=True)
class BacktestResult:
    run: RunContext
    predictions: pd.DataFrame
    bets: pd.DataFrame
    fold_results: list[FoldArtifacts]
    summary: dict[str, Any]
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class FinalTrainingResult:
    run: RunContext
    model_path: Path
    feature_importance_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class PredictionResult:
    run: RunContext
    predictions: pd.DataFrame
    bets: pd.DataFrame
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class CaptureOddsResult:
    dataset_dir: Path
    snapshots_path: Path
    snapshots: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class NetBacktestResult:
    run: RunContext
    predictions: pd.DataFrame
    candidate_rows: pd.DataFrame
    execution_rows: pd.DataFrame
    net_bet_rows: pd.DataFrame
    niches: pd.DataFrame
    summary: dict[str, Any]
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class NicheTrainingResult:
    run: RunContext
    model_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class ShadowRunResult:
    run: RunContext
    predictions: pd.DataFrame
    candidate_rows: pd.DataFrame
    planned_bets: pd.DataFrame
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class PolymarketCollectResult:
    database_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class PolymarketHistoryBackfillResult:
    run: RunContext
    database_path: Path
    summary_path: Path
    summary: dict[str, Any]
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class PolymarketCoverageAuditResult:
    run: RunContext
    database_path: Path
    summary_path: Path
    summary: dict[str, Any]
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class PolymarketShadowResult:
    run: RunContext
    database_path: Path
    decisions: pd.DataFrame
    fills: pd.DataFrame
    mappings: pd.DataFrame
    summary: dict[str, Any]
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class PolymarketRetroResult:
    run: RunContext
    database_path: Path | None
    candidates: pd.DataFrame
    decisions: pd.DataFrame
    fills: pd.DataFrame
    mappings: pd.DataFrame
    mapping_audit: pd.DataFrame
    summary: dict[str, Any]
    coverage_summary: dict[str, Any]
    artifacts: dict[str, Path]
    policy_bundle_path: Path | None = None


@dataclass(frozen=True)
class MultiMarketDiscoveryResult:
    run: RunContext
    database_path: Path
    catalog: pd.DataFrame
    coverage: pd.DataFrame
    summary: dict[str, Any]
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class MultiMarketCaptureResult:
    run: RunContext
    database_path: Path
    catalog: pd.DataFrame
    checkpoints: pd.DataFrame
    blockers: pd.DataFrame
    summary: dict[str, Any]
    artifacts: dict[str, Path]
