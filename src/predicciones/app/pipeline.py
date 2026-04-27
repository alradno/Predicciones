from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Settings
from ..football.sim_data import (
    build_sim_features,
    collect_sim_data_source,
    default_football_sim_db_path,
    discover_sim_data_sources,
    export_sim_training_dataset,
    normalize_sim_data,
    report_sim_data,
)
from ..football.simulator import train_football_simulator
from ..football.model_ops import train_lane_model_candidate, validate_lane_model_candidate
from ..lanes.evaluation import evaluate_forward_lane
from ..lanes.runtime import (
    build_market_lane_predictions,
    capture_multi_market,
    default_multi_market_db_path,
    report_market_lane,
    report_sport_merge,
    run_market_lane,
)
from ..markets.execution import (
    run_trade_cycle,
    run_trade_reconciliation,
    update_trade_kill_switch,
)
from ..markets.trading import TradeMode


def capture_market_raw_pipeline(
    settings: Settings,
    db_path: Path | str | None = None,
    stream_seconds: int = 0,
    lane_id: str | None = None,
    max_markets: int | None = None,
    max_events: int | None = None,
    capture_priority: str = "missing_or_stale",
    freshness_seconds: int = 3600,
    max_stale_age_seconds: int | None = None,
    include_policy_ready_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return capture_multi_market(
        settings=settings,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
        stream_seconds=stream_seconds,
        lane_id=lane_id,
        max_markets=max_markets,
        max_events=max_events,
        capture_priority=capture_priority,
        freshness_seconds=freshness_seconds,
        max_stale_age_seconds=max_stale_age_seconds,
        include_policy_ready_only=include_policy_ready_only,
    )


def build_market_lane_predictions_pipeline(
    settings: Settings,
    lane_id: str,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return build_market_lane_predictions(
        settings=settings,
        lane_id=lane_id,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
        model_path=Path(model_path) if model_path else None,
    )


def run_market_lane_pipeline(
    settings: Settings,
    lane_id: str,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return run_market_lane(
        settings=settings,
        lane_id=lane_id,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
    )


def evaluate_forward_lane_pipeline(
    settings: Settings,
    lane_id: str,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return evaluate_forward_lane(
        settings=settings,
        lane_id=lane_id,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
    )


def report_market_lane_pipeline(
    settings: Settings,
    lane_id: str,
    run_dir: Path | str | None = None,
) -> tuple[dict[str, Any], str]:
    return report_market_lane(settings=settings, lane_id=lane_id, run_dir=run_dir)


def report_sport_merge_pipeline(
    settings: Settings,
    sport: str,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return report_sport_merge(
        settings=settings,
        sport=sport,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
    )


def discover_sim_data_sources_pipeline(
    settings: Settings,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return discover_sim_data_sources(
        settings=settings,
        db_path=Path(db_path) if db_path else default_football_sim_db_path(settings),
    )


def collect_sim_data_pipeline(
    settings: Settings,
    source_id: str,
    leagues: list[str] | tuple[str, ...],
    seasons: list[str] | tuple[str, ...],
    db_path: Path | str | None = None,
    profile: str | None = None,
    seasons_back: int | None = None,
    probe_missing: bool = False,
    teams: str = "mapped",
    max_statsbomb_matches: int | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return collect_sim_data_source(
        settings=settings,
        source_id=source_id,
        leagues=leagues,
        seasons=seasons,
        db_path=Path(db_path) if db_path else default_football_sim_db_path(settings),
        profile=profile,
        seasons_back=seasons_back,
        probe_missing=probe_missing,
        teams=teams,
        max_statsbomb_matches=max_statsbomb_matches,
    )


def normalize_sim_data_pipeline(
    settings: Settings,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return normalize_sim_data(
        settings=settings,
        db_path=Path(db_path) if db_path else default_football_sim_db_path(settings),
    )


def build_sim_features_pipeline(
    settings: Settings,
    as_of_date: str | None = None,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return build_sim_features(
        settings=settings,
        as_of_date=as_of_date,
        db_path=Path(db_path) if db_path else default_football_sim_db_path(settings),
    )


def report_sim_data_pipeline(
    settings: Settings,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], str, dict[str, Path]]:
    return report_sim_data(
        settings=settings,
        db_path=Path(db_path) if db_path else default_football_sim_db_path(settings),
    )


def export_sim_training_dataset_pipeline(
    settings: Settings,
    db_path: Path | str | None = None,
    exclude_market_reference: bool = True,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return export_sim_training_dataset(
        settings=settings,
        db_path=Path(db_path) if db_path else default_football_sim_db_path(settings),
        exclude_market_reference=exclude_market_reference,
    )


def train_football_sim_pipeline(
    settings: Settings,
    dataset_path: Path | str | None = None,
    model_name: str = "football_sim_poisson_v1",
    exclude_market_reference: bool = True,
) -> tuple[dict[str, Any], dict[str, Path]]:
    result = train_football_simulator(
        settings=settings,
        dataset_path=Path(dataset_path) if dataset_path else None,
        model_name=model_name,
        exclude_market_reference=exclude_market_reference,
    )
    return result.summary, result.artifacts


def train_lane_model_candidate_pipeline(
    settings: Settings,
    lane_id: str,
    variant: str,
    dataset_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return train_lane_model_candidate(
        settings=settings,
        lane_id=lane_id,
        variant=variant,
        dataset_path=Path(dataset_path) if dataset_path else None,
    )


def validate_lane_model_candidate_pipeline(
    settings: Settings,
    lane_id: str,
    candidate: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return validate_lane_model_candidate(settings=settings, lane_id=lane_id, candidate=candidate)


def trade_paper_pipeline(
    settings: Settings,
    lane_id: str,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return run_trade_cycle(
        settings=settings,
        lane_id=lane_id,
        mode=TradeMode.paper,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
    )


def trade_live_pipeline(
    settings: Settings,
    lane_id: str,
    mode: str,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    if mode != "micro":
        raise ValueError("Only micro live mode is supported.")
    return run_trade_cycle(
        settings=settings,
        lane_id=lane_id,
        mode=TradeMode.live,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
    )


def trade_reconcile_pipeline(
    settings: Settings,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return run_trade_reconciliation(
        settings=settings,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
    )


def trade_kill_switch_pipeline(
    settings: Settings,
    *,
    enabled: bool,
    reason: str = "",
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return update_trade_kill_switch(
        settings=settings,
        enabled=enabled,
        reason=reason,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
    )


def archive_legacy_1x2_pipeline(settings: Settings) -> tuple[dict[str, Any], str]:
    manifest_path = settings.paths.root / "docs" / "archive" / "football_1x2_canonical.md"
    summary = {
        "lane_id": "football_1x2_canonical",
        "mode": "reference_only",
        "manifest_path": str(manifest_path),
        "active_workflow": False,
        "policy_tuning_allowed": False,
    }
    text = (
        "football_1x2_canonical is archived as reference_only. "
        f"Manifest: {manifest_path}"
    )
    return summary, text
