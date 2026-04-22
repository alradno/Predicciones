from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .backtest import _temporal_subsets, run_backtest
from .config import Settings
from .core.promotion_state import (
    ForwardSampleInputs,
    ForwardSampleThresholds,
    SampleStatus,
    classify_forward_sample,
)
from .contracts import (
    BacktestResult,
    CaptureOddsResult,
    DataBundle,
    FinalTrainingResult,
    IngestSummary,
    MultiMarketCaptureResult,
    MultiMarketDiscoveryResult,
    NetBacktestResult,
    NicheTrainingResult,
    PolymarketCollectResult,
    PolymarketCoverageAuditResult,
    PolymarketHistoryBackfillResult,
    PolymarketRetroResult,
    PolymarketShadowResult,
    PredictionResult,
    ShadowRunResult,
)
from .dataset import build_feature_rows, build_fixture_feature_rows, model_feature_columns
from .data_sources import PolymarketGammaClient
from .football_sim_data import (
    build_sim_features,
    collect_sim_data_source,
    default_football_sim_db_path,
    discover_sim_data_sources,
    export_sim_training_dataset,
    normalize_sim_data,
    report_sim_data,
)
from .football_simulator import train_football_simulator
from .ingestion import (
    build_dataset_paths,
    build_market_odds,
    canonicalize_matches,
    download_historical_matches,
    load_matches,
)
from .models import (
    OutcomeCalibrator,
    build_goal_model,
    estimate_feature_importance,
    fit_dixon_coles_rho,
    fit_goal_model,
    outcome_probabilities_from_lambdas,
)
from .multi_market import (
    GLOBAL_FOOTBALL_1X2_LEAGUES,
    build_market_lane_predictions,
    capture_multi_market,
    create_market_lane_policy,
    default_multi_market_db_path,
    discover_multi_market,
    get_market_lane_spec,
    latest_multi_market_run,
    report_multi_market,
    report_market_lane,
    report_sport_merge,
    run_market_lane,
)
from .reporting import create_run_context, format_summary, save_backtest_artifacts, save_prediction_artifacts
from .polymarket_retro import (
    audit_polymarket_coverage,
    backfill_polymarket_history,
    backtest_polymarket_retro,
    report_polymarket_retro,
    tune_polymarket_policy,
)
from .polymarket_shadow import collect_polymarket, default_polymarket_db_path, report_polymarket, shadow_polymarket
from .research import (
    capture_odds_for_dataset,
    discover_niches,
    format_net_summary,
    load_research_run,
    run_net_backtest,
    shadow_run_from_bundle,
    train_niche_bundle,
)
from .strategy import add_edge_columns, optimize_policy, select_bets


def _dataset_name(leagues: list[str] | tuple[str, ...], seasons: list[str] | tuple[str, ...]) -> str:
    league_part = "-".join(leagues)
    season_part = "-".join(seasons)
    return f"dataset_{league_part}_{season_part}"


def _latest_pointer(path: Path, value: str | None = None) -> str | None:
    if value is not None:
        path.write_text(value, encoding="utf-8")
        return value
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


POLICY_WRITE_BLOCKED_SAMPLE_MESSAGE = (
    "Policy write blocked: lane sample_status is {sample_status}. "
    "Forward sample must be sample_ready before policy changes are allowed."
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_latest_polymarket_forward_sample(settings: Settings) -> dict[str, Any] | None:
    pointer = settings.paths.outputs_dir / "latest_polymarket_shadow.txt"
    run_dir_value = _latest_pointer(pointer)
    if not run_dir_value:
        return None
    run_dir = Path(run_dir_value)
    candidates = [
        run_dir / "forward_sample_report.json",
        run_dir / "shadow_summary.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "shadow_summary.json":
            payload = payload.get("forward_sample", {}) or {}
        if payload:
            return payload
    return None


def _polymarket_lane_sample_state(settings: Settings) -> dict[str, Any]:
    report = _load_latest_polymarket_forward_sample(settings)
    if not report:
        decision = classify_forward_sample(
            ForwardSampleInputs(valid_forward_decisions=0, settled_decisions=0, fresh_book_rate=1.0)
        )
        return {
            "sample_status": decision.sample_status.value,
            "sample_blockers": list(decision.sample_blockers),
            "source": "missing_latest_polymarket_shadow",
        }

    cumulative = report.get("cumulative", {}) or report
    target_valid = _safe_int(cumulative.get("target_valid_forward_decisions", report.get("target_valid_forward_decisions")), 100)
    target_settled = _safe_int(cumulative.get("target_settled_decisions", report.get("target_settled_decisions")), 40)
    target_fresh = _safe_float(cumulative.get("target_fresh_book_rate", report.get("target_fresh_book_rate")), 0.80)
    decision = classify_forward_sample(
        ForwardSampleInputs(
            valid_forward_decisions=_safe_int(cumulative.get("valid_forward_decisions")),
            settled_decisions=_safe_int(
                cumulative.get("settled_unique_decisions", cumulative.get("settled_decisions"))
            ),
            fresh_book_rate=_safe_float(cumulative.get("fresh_book_rate")),
            selected_candidates=_safe_int(cumulative.get("policy_selected_decisions")),
        ),
        ForwardSampleThresholds(
            min_valid_forward_decisions=target_valid,
            min_settled_decisions=target_settled,
            min_fresh_book_rate=target_fresh,
        ),
    )
    return {
        "sample_status": decision.sample_status.value,
        "sample_blockers": list(decision.sample_blockers),
        "source": "latest_polymarket_shadow",
        "valid_forward_decisions": _safe_int(cumulative.get("valid_forward_decisions")),
        "settled_unique_decisions": _safe_int(
            cumulative.get("settled_unique_decisions", cumulative.get("settled_decisions"))
        ),
        "fresh_book_rate": _safe_float(cumulative.get("fresh_book_rate")),
    }


def _retro_result_uses_history_proxy(result: PolymarketRetroResult) -> bool:
    summary = result.summary
    if bool(summary.get("history_proxy_used")):
        return True
    if str(summary.get("minimum_quality_tier", "")).strip() == "history_proxy":
        return True
    if str(summary.get("price_provenance", "")).strip() in {"proxy", "history_proxy"}:
        return True
    counts = summary.get("price_provenance_counts", {}) or {}
    return _safe_int(counts.get("proxy")) > 0 or _safe_int(counts.get("history_proxy")) > 0


def _set_policy_write_state(
    result: PolymarketRetroResult,
    *,
    diagnostic_only: bool,
    policy_written: bool,
    blocked_reason: str = "",
    sample_state: dict[str, Any] | None = None,
) -> None:
    result.summary["diagnostic_only"] = bool(diagnostic_only)
    result.summary["policy_written"] = bool(policy_written)
    result.summary["policy_write_blocked_reason"] = str(blocked_reason)
    if sample_state is not None:
        result.summary["policy_write_sample_state"] = sample_state
    summary_path = result.artifacts.get("retro_shadow_summary")
    if summary_path:
        summary_path.write_text(json.dumps(result.summary, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _ensure_policy_write_allowed(settings: Settings, result: PolymarketRetroResult) -> bool:
    sample_state = _polymarket_lane_sample_state(settings)
    sample_status = str(sample_state.get("sample_status", SampleStatus.collecting_forward_sample.value))
    if sample_status != SampleStatus.sample_ready.value:
        _set_policy_write_state(
            result,
            diagnostic_only=True,
            policy_written=False,
            blocked_reason="lane_sample_not_ready",
            sample_state=sample_state,
        )
        raise RuntimeError(POLICY_WRITE_BLOCKED_SAMPLE_MESSAGE.format(sample_status=sample_status))
    if _retro_result_uses_history_proxy(result):
        _set_policy_write_state(
            result,
            diagnostic_only=True,
            policy_written=False,
            blocked_reason="history_proxy_never_promotes_policy",
            sample_state=sample_state,
        )
        return False
    return True


def ingest_pipeline(
    settings: Settings,
    leagues: list[str] | tuple[str, ...],
    seasons: list[str] | tuple[str, ...],
    dataset_name: str | None = None,
) -> IngestSummary:
    dataset_name = dataset_name or _dataset_name(leagues, seasons)
    paths = build_dataset_paths(settings.paths.data_dir, dataset_name)
    download = download_historical_matches(leagues=leagues, seasons=seasons)
    matches = canonicalize_matches(download.matches, require_results=True)
    market_odds = build_market_odds(matches)
    market_snapshots = capture_odds_for_dataset(
        dataset_dir=paths.root,
        settings=settings,
        reference_matches=matches,
        market_odds=market_odds,
    ).snapshots
    feature_rows = build_feature_rows(matches, rolling_window=settings.backtest.rolling_window)

    matches.to_csv(paths.matches_path, index=False)
    market_odds.to_csv(paths.market_odds_path, index=False)
    market_snapshots.to_csv(paths.market_snapshots_path, index=False)
    feature_rows.to_csv(paths.feature_rows_path, index=False)
    paths.manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "leagues": list(leagues),
                "seasons": list(seasons),
                "rows_downloaded": len(download.matches),
                "rows_canonical": len(matches),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _latest_pointer(settings.paths.data_dir / "latest_dataset.txt", str(paths.root))

    return IngestSummary(
        rows_downloaded=len(download.matches),
        rows_canonical=len(matches),
        failures=download.failures,
        artifacts={
            "dataset_dir": paths.root,
            "matches": paths.matches_path,
            "market_odds": paths.market_odds_path,
            "market_snapshots": paths.market_snapshots_path,
            "feature_rows": paths.feature_rows_path,
            "manifest": paths.manifest_path,
        },
        bundle=DataBundle(matches=matches, market_odds=market_odds, feature_rows=feature_rows, market_snapshots=market_snapshots),
    )


def load_dataset_bundle(settings: Settings, dataset_dir: Path | str | None = None) -> DataBundle:
    latest = _latest_pointer(settings.paths.data_dir / "latest_dataset.txt")
    dataset_root = Path(dataset_dir) if dataset_dir else (Path(latest) if latest else None)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError("No hay dataset disponible. Ejecuta primero `predicciones ingest`.")

    matches = load_matches(dataset_root / "matches.csv")
    market_odds = pd.read_csv(dataset_root / "market_odds.csv", parse_dates=["Date"])
    feature_rows = pd.read_csv(dataset_root / "feature_rows.csv", parse_dates=["Date"])
    snapshots_path = dataset_root / "market_snapshots.csv"
    market_snapshots = pd.read_csv(
        snapshots_path,
        parse_dates=["Date", "kickoff_time", "snapshot_time"],
    ) if snapshots_path.exists() else None
    return DataBundle(matches=matches, market_odds=market_odds, feature_rows=feature_rows, market_snapshots=market_snapshots)


def _combined_dataset(bundle: DataBundle) -> pd.DataFrame:
    return bundle.feature_rows.merge(
        bundle.market_odds,
        on=["match_id", "Date", "league_code", "season", "HomeTeam", "AwayTeam"],
        how="left",
    )


def backtest_pipeline(
    settings: Settings,
    dataset_dir: Path | str | None = None,
    leagues: list[str] | tuple[str, ...] | None = None,
    seasons: list[str] | tuple[str, ...] | None = None,
) -> BacktestResult:
    if dataset_dir is None:
        if _latest_pointer(settings.paths.data_dir / "latest_dataset.txt") is None:
            ingest_pipeline(
                settings=settings,
                leagues=tuple(leagues or settings.default_leagues),
                seasons=tuple(seasons or settings.default_seasons),
            )
    bundle = load_dataset_bundle(settings, dataset_dir=dataset_dir)
    dataset = _combined_dataset(bundle)

    run = create_run_context(settings.paths.runs_dir, "backtest")
    result = run_backtest(dataset=dataset, config=settings.backtest, run=run)

    feature_columns = model_feature_columns(bundle.feature_rows)
    final_model = build_goal_model(bundle.feature_rows[feature_columns])
    fit_goal_model(final_model, bundle.feature_rows[feature_columns], bundle.feature_rows["home_goals"], bundle.feature_rows["away_goals"])
    importance = estimate_feature_importance(
        final_model,
        bundle.feature_rows[feature_columns],
        bundle.feature_rows["home_goals"],
        bundle.feature_rows["away_goals"],
    )

    artifacts = save_backtest_artifacts(
        result=result,
        feature_importance=importance,
        benchmarks_dir=settings.paths.benchmarks_dir,
        benchmark_dir_name=settings.benchmark_dir_name,
    )
    _latest_pointer(settings.paths.outputs_dir / "latest_backtest.txt", str(run.run_dir))

    return BacktestResult(
        run=result.run,
        predictions=result.predictions,
        bets=result.bets,
        fold_results=result.fold_results,
        summary=result.summary,
        artifacts=artifacts,
    )


def capture_odds_pipeline(
    settings: Settings,
    dataset_dir: Path | str | None = None,
    snapshot_files: list[Path] | None = None,
) -> CaptureOddsResult:
    bundle = load_dataset_bundle(settings, dataset_dir=dataset_dir)
    latest = _latest_pointer(settings.paths.data_dir / "latest_dataset.txt")
    target_dir = Path(dataset_dir) if dataset_dir else Path(latest or "")
    if not target_dir.exists():
        raise FileNotFoundError("No encuentro el dataset para capturar snapshots.")
    return capture_odds_for_dataset(
        dataset_dir=target_dir,
        settings=settings,
        reference_matches=bundle.matches,
        market_odds=bundle.market_odds,
        snapshot_files=snapshot_files,
    )


def backtest_net_pipeline(
    settings: Settings,
    dataset_dir: Path | str | None = None,
    snapshot_files: list[Path] | None = None,
    leagues: list[str] | tuple[str, ...] | None = None,
    seasons: list[str] | tuple[str, ...] | None = None,
) -> NetBacktestResult:
    if dataset_dir is None and _latest_pointer(settings.paths.data_dir / "latest_dataset.txt") is None:
        ingest_pipeline(
            settings=settings,
            leagues=tuple(leagues or settings.default_leagues),
            seasons=tuple(seasons or settings.default_seasons),
        )
    bundle = load_dataset_bundle(settings, dataset_dir=dataset_dir)
    if bundle.market_snapshots is None or snapshot_files:
        capture = capture_odds_pipeline(settings, dataset_dir=dataset_dir, snapshot_files=snapshot_files)
        bundle = DataBundle(
            matches=bundle.matches,
            market_odds=bundle.market_odds,
            feature_rows=bundle.feature_rows,
            market_snapshots=capture.snapshots,
        )

    run = create_run_context(settings.paths.runs_dir, "backtest_net")
    result = run_net_backtest(
        dataset=_combined_dataset(bundle),
        snapshots=bundle.market_snapshots if bundle.market_snapshots is not None else pd.DataFrame(),
        settings=settings,
        run=run,
    )
    _latest_pointer(settings.paths.outputs_dir / "latest_backtest_net.txt", str(run.run_dir))
    return result


def discover_niches_pipeline(
    settings: Settings,
    run_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, str]:
    root = Path(run_dir) if run_dir else Path(_latest_pointer(settings.paths.outputs_dir / "latest_backtest_net.txt") or "")
    if not root.exists():
        raise FileNotFoundError("No encuentro un backtest neto previo para descubrir nichos.")
    execution_rows = pd.read_csv(root / "execution_rows.csv", parse_dates=["Date", "kickoff_time", "decision_time", "snapshot_time"])
    niches = discover_niches(execution_rows[execution_rows["fold_segment"] == "train"], settings=settings)
    niches_path = root / "niches_discovery.csv"
    niches.to_csv(niches_path, index=False)
    return niches, str(niches_path)


def train_final_pipeline(
    settings: Settings,
    dataset_dir: Path | str | None = None,
    leagues: list[str] | tuple[str, ...] | None = None,
    seasons: list[str] | tuple[str, ...] | None = None,
) -> FinalTrainingResult:
    if dataset_dir is None:
        if _latest_pointer(settings.paths.data_dir / "latest_dataset.txt") is None:
            ingest_pipeline(
                settings=settings,
                leagues=tuple(leagues or settings.default_leagues),
                seasons=tuple(seasons or settings.default_seasons),
            )
    bundle = load_dataset_bundle(settings, dataset_dir=dataset_dir)
    dataset = _combined_dataset(bundle).sort_values(["Date", "match_id"]).reset_index(drop=True)
    feature_columns = model_feature_columns(bundle.feature_rows)
    subtrain, calibration, policy_rows = _temporal_subsets(dataset, settings.backtest)

    model = build_goal_model(subtrain[feature_columns])
    fit_goal_model(model, subtrain[feature_columns], subtrain["home_goals"], subtrain["away_goals"])
    calibration_home, calibration_away = model.predict_lambdas(calibration[feature_columns])
    rho = fit_dixon_coles_rho(
        calibration_home,
        calibration_away,
        calibration["home_goals"].to_numpy(),
        calibration["away_goals"].to_numpy(),
        settings.backtest.dixon_coles_bounds,
    )
    calibration_raw = outcome_probabilities_from_lambdas(
        calibration_home,
        calibration_away,
        rho=rho,
        max_goals=settings.backtest.max_poisson_goals,
    )
    calibrator = OutcomeCalibrator(epsilon=settings.backtest.calibrator_epsilon).fit(
        calibration_raw,
        calibration["target"].to_numpy(),
    )

    policy_home, policy_away = model.predict_lambdas(policy_rows[feature_columns])
    policy_raw = outcome_probabilities_from_lambdas(
        policy_home,
        policy_away,
        rho=rho,
        max_goals=settings.backtest.max_poisson_goals,
    )
    policy_calibrated = calibrator.transform(policy_raw)
    policy_frame = _build_prediction_rows(policy_rows, policy_home, policy_away, policy_raw, policy_calibrated)
    tuned_policy = optimize_policy(
        add_edge_columns(policy_frame),
        edge_thresholds=settings.backtest.policy_search.edge_thresholds,
        ev_thresholds=settings.backtest.policy_search.ev_thresholds,
        min_odds_options=settings.backtest.policy_search.min_odds_options,
        max_odds_options=settings.backtest.policy_search.max_odds_options,
        min_bets=settings.backtest.policy_search.min_bets,
        kelly_fraction=settings.backtest.policy_search.max_kelly_fraction,
    )

    final_model = build_goal_model(bundle.feature_rows[feature_columns])
    fit_goal_model(final_model, bundle.feature_rows[feature_columns], bundle.feature_rows["home_goals"], bundle.feature_rows["away_goals"])
    importance = estimate_feature_importance(
        final_model,
        bundle.feature_rows[feature_columns],
        bundle.feature_rows["home_goals"],
        bundle.feature_rows["away_goals"],
    )

    run = create_run_context(settings.paths.models_dir, "train_final")
    model_path = run.run_dir / "model_bundle.joblib"
    feature_importance_path = run.run_dir / "feature_importance.csv"
    summary_path = run.run_dir / "summary.json"
    feature_importance_path.write_text(importance.to_csv(index=False), encoding="utf-8")
    summary = {
        "rows": int(len(bundle.feature_rows)),
        "policy": tuned_policy.to_dict(),
        "rho": rho,
        "feature_columns": feature_columns,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model": final_model,
            "calibrator": calibrator,
            "rho": rho,
            "policy": tuned_policy.to_dict(),
            "feature_columns": feature_columns,
            "history_matches": bundle.matches,
            "rolling_window": settings.backtest.rolling_window,
            "max_poisson_goals": settings.backtest.max_poisson_goals,
        },
        model_path,
    )
    _latest_pointer(settings.paths.outputs_dir / "latest_model.txt", str(model_path))
    return FinalTrainingResult(
        run=run,
        model_path=model_path,
        feature_importance_path=feature_importance_path,
        summary_path=summary_path,
        summary=summary,
    )


def _load_or_build_lane_dataset(
    settings: Settings,
    lane_id: str,
    leagues: tuple[str, ...],
    seasons: tuple[str, ...],
    dataset_dir: Path | str | None = None,
    dataset_name: str | None = None,
) -> tuple[DataBundle, Path, dict[str, Any]]:
    if dataset_dir is not None:
        root = Path(dataset_dir)
        return load_dataset_bundle(settings, dataset_dir=root), root, {
            "dataset_source": "existing_dataset_dir",
            "dataset_dir": str(root),
            "latest_dataset_pointer_updated": False,
        }

    dataset_name = dataset_name or f"dataset_{lane_id}_{'-'.join(leagues)}_{'-'.join(seasons)}"
    paths = build_dataset_paths(settings.paths.data_dir, dataset_name)
    if paths.manifest_path.exists() and paths.matches_path.exists() and paths.feature_rows_path.exists() and paths.market_odds_path.exists():
        return load_dataset_bundle(settings, dataset_dir=paths.root), paths.root, {
            "dataset_source": "existing_lane_dataset",
            "dataset_dir": str(paths.root),
            "latest_dataset_pointer_updated": False,
        }

    download = download_historical_matches(leagues=leagues, seasons=seasons)
    matches = canonicalize_matches(download.matches, require_results=True)
    market_odds = build_market_odds(matches)
    market_snapshots = capture_odds_for_dataset(
        dataset_dir=paths.root,
        settings=settings,
        reference_matches=matches,
        market_odds=market_odds,
    ).snapshots
    feature_rows = build_feature_rows(matches, rolling_window=settings.backtest.rolling_window)

    matches.to_csv(paths.matches_path, index=False)
    market_odds.to_csv(paths.market_odds_path, index=False)
    market_snapshots.to_csv(paths.market_snapshots_path, index=False)
    feature_rows.to_csv(paths.feature_rows_path, index=False)
    paths.manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "lane_id": lane_id,
                "leagues": list(leagues),
                "seasons": list(seasons),
                "rows_downloaded": len(download.matches),
                "rows_canonical": len(matches),
                "failures": download.failures,
                "latest_dataset_pointer_updated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return (
        DataBundle(matches=matches, market_odds=market_odds, feature_rows=feature_rows, market_snapshots=market_snapshots),
        paths.root,
        {
            "dataset_source": "built_lane_dataset",
            "dataset_dir": str(paths.root),
            "rows_downloaded": len(download.matches),
            "rows_canonical": len(matches),
            "download_failures": download.failures,
            "latest_dataset_pointer_updated": False,
        },
    )


def train_market_lane_model_pipeline(
    settings: Settings,
    lane_id: str,
    dataset_dir: Path | str | None = None,
    leagues: list[str] | tuple[str, ...] | None = None,
    seasons: list[str] | tuple[str, ...] | None = None,
    dataset_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    spec = get_market_lane_spec(lane_id)
    if spec.sport != "football" or lane_id not in {"football_1x2_global", "football_goals_core"}:
        raise ValueError(f"El carril {lane_id} no tiene entrenamiento de modelo de futbol v1.")

    lane_dir = settings.paths.outputs_dir / "lanes" / lane_id
    model_root = lane_dir / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    target_leagues = tuple(leagues or GLOBAL_FOOTBALL_1X2_LEAGUES)
    target_seasons = tuple(seasons or settings.default_seasons)
    bundle, resolved_dataset_dir, dataset_summary = _load_or_build_lane_dataset(
        settings=settings,
        lane_id=lane_id,
        leagues=target_leagues,
        seasons=target_seasons,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
    )
    actual_leagues = sorted(bundle.matches["league_code"].dropna().astype(str).unique().tolist()) if "league_code" in bundle.matches.columns else list(target_leagues)
    actual_seasons = sorted(bundle.matches["season"].dropna().astype(str).unique().tolist(), reverse=True) if "season" in bundle.matches.columns else list(target_seasons)
    dataset = _combined_dataset(bundle).sort_values(["Date", "match_id"]).reset_index(drop=True)
    feature_columns = model_feature_columns(bundle.feature_rows)
    subtrain, calibration, _ = _temporal_subsets(dataset, settings.backtest)

    model = build_goal_model(subtrain[feature_columns])
    fit_goal_model(model, subtrain[feature_columns], subtrain["home_goals"], subtrain["away_goals"])
    calibration_home, calibration_away = model.predict_lambdas(calibration[feature_columns])
    rho = fit_dixon_coles_rho(
        calibration_home,
        calibration_away,
        calibration["home_goals"].to_numpy(),
        calibration["away_goals"].to_numpy(),
        settings.backtest.dixon_coles_bounds,
    )

    final_model = build_goal_model(bundle.feature_rows[feature_columns])
    fit_goal_model(final_model, bundle.feature_rows[feature_columns], bundle.feature_rows["home_goals"], bundle.feature_rows["away_goals"])
    importance = estimate_feature_importance(
        final_model,
        bundle.feature_rows[feature_columns],
        bundle.feature_rows["home_goals"],
        bundle.feature_rows["away_goals"],
    )

    run = create_run_context(model_root, "train_lane_model")
    model_path = run.run_dir / "model_bundle.joblib"
    feature_importance_path = run.run_dir / "feature_importance.csv"
    summary_path = run.run_dir / "summary.json"
    manifest_path = lane_dir / "lane_model_manifest.json"
    latest_lane_model_path = lane_dir / "latest_model.txt"
    feature_importance_path.write_text(importance.to_csv(index=False), encoding="utf-8")
    summary = {
        "lane_id": lane_id,
        "benchmark_id": spec.benchmark_id,
        "sport": spec.sport,
        "model_mode": spec.model_mode,
        "leagues": actual_leagues,
        "seasons": actual_seasons,
        "requested_leagues": list(target_leagues),
        "requested_seasons": list(target_seasons),
        "rows": int(len(bundle.feature_rows)),
        "matches": int(len(bundle.matches)),
        "rho": rho,
        "feature_columns": feature_columns,
        "dataset_dir": str(resolved_dataset_dir),
        "model_path": str(model_path),
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        "latest_model_pointer_updated": False,
        **dataset_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model": final_model,
            "rho": rho,
            "feature_columns": feature_columns,
            "history_matches": bundle.matches,
            "rolling_window": settings.backtest.rolling_window,
            "max_poisson_goals": settings.backtest.max_poisson_goals,
            "model_variant": f"{lane_id}_lane_goal_model",
            "lane_id": lane_id,
            "benchmark_id": spec.benchmark_id,
            "policy_reoptimized": False,
            "global_roi_actionable": False,
        },
        model_path,
    )
    latest_lane_model_path.write_text(str(model_path), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, {
        "model": model_path,
        "summary": summary_path,
        "feature_importance": feature_importance_path,
        "lane_model_manifest": manifest_path,
        "latest_lane_model": latest_lane_model_path,
        "dataset_dir": resolved_dataset_dir,
    }


def train_niche_pipeline(
    settings: Settings,
    dataset_dir: Path | str | None = None,
    research_run_dir: Path | str | None = None,
    leagues: list[str] | tuple[str, ...] | None = None,
    seasons: list[str] | tuple[str, ...] | None = None,
) -> NicheTrainingResult:
    if research_run_dir is None:
        latest = _latest_pointer(settings.paths.outputs_dir / "latest_backtest_net.txt")
        if latest is None:
            backtest_net_pipeline(
                settings=settings,
                dataset_dir=dataset_dir,
                leagues=tuple(leagues or settings.default_leagues),
                seasons=tuple(seasons or settings.default_seasons),
            )
            latest = _latest_pointer(settings.paths.outputs_dir / "latest_backtest_net.txt")
        research_run_dir = Path(latest or "")

    bundle = load_dataset_bundle(settings, dataset_dir=dataset_dir)
    result = train_niche_bundle(
        settings=settings,
        bundle_matches=bundle.matches,
        feature_rows=bundle.feature_rows,
        research_run_dir=Path(research_run_dir),
    )
    _latest_pointer(settings.paths.outputs_dir / "latest_niche_model.txt", str(result.model_path))
    return result


def predict_pipeline(
    settings: Settings,
    fixtures_path: Path | str,
    model_path: Path | str | None = None,
) -> PredictionResult:
    artifact_path = Path(model_path) if model_path else Path(_latest_pointer(settings.paths.outputs_dir / "latest_model.txt") or "")
    if not artifact_path.exists():
        raise FileNotFoundError("No hay modelo final entrenado. Ejecuta primero `predicciones train-final`.")

    payload = joblib.load(artifact_path)
    history_matches = payload["history_matches"]
    fixtures = canonicalize_matches(pd.read_csv(fixtures_path), require_results=False)
    fixture_features = build_fixture_feature_rows(
        history_matches=history_matches,
        fixtures=fixtures,
        rolling_window=int(payload["rolling_window"]),
    )
    market = build_market_odds(fixtures)
    dataset = fixture_features.merge(
        market,
        on=["match_id", "Date", "league_code", "season", "HomeTeam", "AwayTeam"],
        how="left",
    )

    model = payload["model"]
    calibrator = payload["calibrator"]
    rho = float(payload["rho"])
    feature_columns = payload["feature_columns"]
    lambda_home, lambda_away = model.predict_lambdas(dataset[feature_columns])
    raw_probabilities = outcome_probabilities_from_lambdas(
        lambda_home,
        lambda_away,
        rho=rho,
        max_goals=int(payload["max_poisson_goals"]),
    )
    calibrated_probabilities = calibrator.transform(raw_probabilities)
    prediction_rows = _build_prediction_rows(dataset, lambda_home, lambda_away, raw_probabilities, calibrated_probabilities)
    prediction_rows["raw_prediction"] = [("away", "draw", "home")[index] for index in raw_probabilities.argmax(axis=1)]
    prediction_rows["calibrated_prediction"] = [("away", "draw", "home")[index] for index in calibrated_probabilities.argmax(axis=1)]
    prediction_rows = add_edge_columns(prediction_rows)
    bets = select_bets(prediction_rows, policy=_policy_from_payload(payload["policy"]))

    run = create_run_context(settings.paths.runs_dir, "predict")
    result = PredictionResult(run=run, predictions=prediction_rows, bets=bets, artifacts={})
    artifacts = save_prediction_artifacts(result)
    return PredictionResult(run=run, predictions=prediction_rows, bets=bets, artifacts=artifacts)


def shadow_run_pipeline(
    settings: Settings,
    fixtures_path: Path | str,
    model_path: Path | str | None = None,
    snapshot_files: list[Path] | None = None,
) -> ShadowRunResult:
    artifact_path = Path(model_path) if model_path else Path(_latest_pointer(settings.paths.outputs_dir / "latest_niche_model.txt") or "")
    if not artifact_path.exists():
        raise FileNotFoundError("No hay niche model entrenado. Ejecuta primero `predicciones train-niche`.")
    result = shadow_run_from_bundle(
        settings=settings,
        fixtures_path=Path(fixtures_path),
        model_path=artifact_path,
        snapshot_files=snapshot_files,
    )
    _latest_pointer(settings.paths.outputs_dir / "latest_shadow.txt", str(result.run.run_dir))
    return result


def collect_polymarket_pipeline(
    settings: Settings,
    db_path: Path | str | None = None,
    stream_seconds: int = 0,
) -> PolymarketCollectResult:
    result = collect_polymarket(
        settings=settings,
        db_path=Path(db_path) if db_path else default_polymarket_db_path(settings),
        stream_seconds=stream_seconds,
    )
    _latest_pointer(settings.paths.outputs_dir / "latest_polymarket_collect.txt", str(result.summary_path))
    return result


def discover_multi_market_pipeline(
    settings: Settings,
    db_path: Path | str | None = None,
    limit_per_query: int = 50,
) -> MultiMarketDiscoveryResult:
    result = discover_multi_market(
        settings=settings,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
        limit_per_query=limit_per_query,
    )
    _latest_pointer(settings.paths.outputs_dir / "latest_multi_market_discovery.txt", str(result.run.run_dir))
    return result


def discover_market_raw_pipeline(
    settings: Settings,
    db_path: Path | str | None = None,
    limit_per_query: int = 50,
) -> MultiMarketDiscoveryResult:
    return discover_multi_market_pipeline(settings=settings, db_path=db_path, limit_per_query=limit_per_query)


def capture_multi_market_pipeline(
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
) -> MultiMarketCaptureResult:
    result = capture_multi_market(
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
    _latest_pointer(settings.paths.outputs_dir / "latest_multi_market_capture.txt", str(result.run.run_dir))
    return result


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
) -> MultiMarketCaptureResult:
    return capture_multi_market_pipeline(
        settings=settings,
        db_path=db_path,
        stream_seconds=stream_seconds,
        lane_id=lane_id,
        max_markets=max_markets,
        max_events=max_events,
        capture_priority=capture_priority,
        freshness_seconds=freshness_seconds,
        max_stale_age_seconds=max_stale_age_seconds,
        include_policy_ready_only=include_policy_ready_only,
    )


def run_market_lane_pipeline(
    settings: Settings,
    lane_id: str,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    summary, artifacts = run_market_lane(
        settings=settings,
        lane_id=lane_id,
        db_path=Path(db_path) if db_path else default_multi_market_db_path(settings),
    )
    _latest_pointer(settings.paths.outputs_dir / f"latest_market_lane_{lane_id}.txt", str(artifacts["run_sample_report"].parent))
    return summary, artifacts


def build_market_lane_predictions_pipeline(
    settings: Settings,
    lane_id: str,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    summary, artifacts = build_market_lane_predictions(
        settings=settings,
        lane_id=lane_id,
        db_path=Path(db_path) if db_path else None,
        model_path=Path(model_path) if model_path else None,
    )
    _latest_pointer(settings.paths.outputs_dir / f"latest_market_lane_predictions_{lane_id}.txt", str(artifacts["model_prediction_report"]))
    return summary, artifacts


def create_market_lane_policy_pipeline(
    settings: Settings,
    lane_id: str,
    overwrite: bool = False,
) -> tuple[dict[str, Any], dict[str, Path]]:
    return create_market_lane_policy(
        settings=settings,
        lane_id=lane_id,
        overwrite=overwrite,
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


def report_multi_market_pipeline(
    settings: Settings,
    run_dir: Path | str | None = None,
) -> tuple[dict[str, Any], str]:
    root = Path(run_dir) if run_dir else latest_multi_market_run(settings)
    if root is None:
        raise FileNotFoundError("No encuentro un run multi-market previo para reportar.")
    return report_multi_market(root)


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
        leagues=tuple(leagues),
        seasons=tuple(seasons),
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


def backfill_polymarket_history_pipeline(
    settings: Settings,
    dataset_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
) -> PolymarketHistoryBackfillResult:
    if dataset_dir is not None or _latest_pointer(settings.paths.data_dir / "latest_dataset.txt") is not None:
        bundle = load_dataset_bundle(settings, dataset_dir=dataset_dir)
        history_matches = bundle.matches
    else:
        artifact_path = Path(model_path) if model_path else None
        if artifact_path is None or not artifact_path.exists():
            for pointer_name in ("latest_niche_model.txt", "latest_model.txt"):
                pointer = settings.paths.outputs_dir / pointer_name
                if pointer.exists():
                    artifact_path = Path(pointer.read_text(encoding="utf-8").strip())
                    break
        if artifact_path is None or not artifact_path.exists():
            raise FileNotFoundError("No encuentro dataset ni modelo para derivar el historico del backfill.")
        payload = joblib.load(artifact_path)
        history_matches = payload["history_matches"]
    result = backfill_polymarket_history(
        settings=settings,
        history_matches=history_matches,
        db_path=Path(db_path) if db_path else None,
    )
    _latest_pointer(settings.paths.outputs_dir / "latest_polymarket_backfill.txt", str(result.summary_path))
    return result


def backtest_polymarket_retro_pipeline(
    settings: Settings,
    dataset_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
    policy_bundle_path: Path | str | None = None,
) -> PolymarketRetroResult:
    bundle = load_dataset_bundle(settings, dataset_dir=dataset_dir) if dataset_dir else None
    result = backtest_polymarket_retro(
        settings=settings,
        history_matches=bundle.matches if bundle is not None else None,
        db_path=Path(db_path) if db_path else None,
        model_path=Path(model_path) if model_path else None,
        policy_bundle_path=Path(policy_bundle_path) if policy_bundle_path else None,
    )
    _set_policy_write_state(result, diagnostic_only=True, policy_written=False)
    _latest_pointer(settings.paths.outputs_dir / "latest_polymarket_retro.txt", str(result.run.run_dir))
    return result


def audit_polymarket_coverage_pipeline(
    settings: Settings,
    dataset_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
) -> PolymarketCoverageAuditResult:
    if dataset_dir is not None or _latest_pointer(settings.paths.data_dir / "latest_dataset.txt") is not None:
        bundle = load_dataset_bundle(settings, dataset_dir=dataset_dir)
        history_matches = bundle.matches
    else:
        artifact_path = Path(model_path) if model_path else None
        if artifact_path is None or not artifact_path.exists():
            for pointer_name in ("latest_niche_model.txt", "latest_model.txt"):
                pointer = settings.paths.outputs_dir / pointer_name
                if pointer.exists():
                    artifact_path = Path(pointer.read_text(encoding="utf-8").strip())
                    break
        if artifact_path is None or not artifact_path.exists():
            raise FileNotFoundError("No encuentro dataset ni modelo para auditar la cobertura historica de Polymarket.")
        payload = joblib.load(artifact_path)
        history_matches = payload["history_matches"]
    result = audit_polymarket_coverage(
        settings=settings,
        history_matches=history_matches,
        db_path=Path(db_path) if db_path else None,
    )
    _latest_pointer(settings.paths.outputs_dir / "latest_polymarket_coverage_audit.txt", str(result.summary_path))
    return result


def tune_polymarket_policy_pipeline(
    settings: Settings,
    dataset_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
    write_policy: bool = False,
) -> PolymarketRetroResult:
    bundle = load_dataset_bundle(settings, dataset_dir=dataset_dir) if dataset_dir else None
    result = tune_polymarket_policy(
        settings=settings,
        history_matches=bundle.matches if bundle is not None else None,
        db_path=Path(db_path) if db_path else None,
        model_path=Path(model_path) if model_path else None,
    )
    _set_policy_write_state(result, diagnostic_only=not write_policy, policy_written=False)
    if write_policy and _ensure_policy_write_allowed(settings, result):
        if result.summary.get("bundle_status") == "promotable_for_forward" and result.policy_bundle_path:
            _latest_pointer(settings.paths.outputs_dir / "latest_polymarket_policy.txt", str(result.policy_bundle_path))
            _latest_pointer(
                settings.paths.outputs_dir / "latest_polymarket_policy_provisional.txt",
                str(result.policy_bundle_path),
            )
            _set_policy_write_state(result, diagnostic_only=False, policy_written=True)
        elif result.policy_bundle_path:
            _set_policy_write_state(
                result,
                diagnostic_only=True,
                policy_written=False,
                blocked_reason="bundle_status_not_promotable",
            )
    _latest_pointer(settings.paths.outputs_dir / "latest_polymarket_retro.txt", str(result.run.run_dir))
    return result


def shadow_polymarket_pipeline(
    settings: Settings,
    fixtures_path: Path | str | None = None,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
    policy_bundle_path: Path | str | None = None,
) -> PolymarketShadowResult:
    result = shadow_polymarket(
        settings=settings,
        db_path=Path(db_path) if db_path else default_polymarket_db_path(settings),
        fixtures_path=Path(fixtures_path) if fixtures_path else None,
        model_path=Path(model_path) if model_path else None,
        policy_bundle_path=Path(policy_bundle_path) if policy_bundle_path else None,
    )
    _latest_pointer(settings.paths.outputs_dir / "latest_polymarket_shadow.txt", str(result.run.run_dir))
    return result


def report_pipeline(run_dir: Path | str | None = None) -> tuple[dict[str, Any], str]:
    root = Path(run_dir) if run_dir else None
    if root is None:
        candidates = [
            Path.cwd() / "outputs" / "latest_backtest.txt",
            Path(__file__).resolve().parents[2] / "outputs" / "latest_backtest.txt",
        ]
        latest = next((path.read_text(encoding="utf-8").strip() for path in candidates if path.exists()), None)
        root = Path(latest) if latest else None
    if root is None or not root.exists():
        raise FileNotFoundError("No encuentro un run para reportar.")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return summary, format_summary(summary)


def promotion_report_pipeline(run_dir: Path | str | None = None) -> tuple[dict[str, Any], str]:
    root = Path(run_dir) if run_dir else None
    if root is None:
        candidates = [
            Path.cwd() / "outputs" / "latest_backtest_net.txt",
            Path(__file__).resolve().parents[2] / "outputs" / "latest_backtest_net.txt",
        ]
        latest = next((path.read_text(encoding="utf-8").strip() for path in candidates if path.exists()), None)
        root = Path(latest) if latest else None
    if root is None or not root.exists():
        raise FileNotFoundError("No encuentro un backtest neto para leer el promotion report.")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    report = summary.get("promotion_report", {})
    truth = summary.get("experimental_truth", {})
    signal_quality = truth.get("signal_quality", {})
    policy_effect = truth.get("policy_effect", {})
    execution_viability = truth.get("execution_viability", {})
    gate = report.get("gate", {})
    lines = [
        f"- Predice mejor: {signal_quality.get('beats_market', False)}",
        f"- La policy suma: {policy_effect.get('positive_holdout_roi', False)}",
        f"- Ejecucion viable: {execution_viability.get('forward_ready', False)}",
        f"- Stage 0 signal gate: {report.get('stage_0_signal_gate', {}).get('passed', False)}",
        f"- Stage 1 discovery: {report.get('stage_1_discovery_complete', report.get('stage_1_research', {})).get('passed', False)}",
        f"- Stage 2 holdout: {report.get('stage_2_outer_holdout', report.get('stage_2_holdout', {})).get('passed', False)}",
        f"- Stage 3 shadow: {report.get('stage_3_shadow', {}).get('passed', False)}",
        f"- Stage 4 limited live: {report.get('stage_4_limited_live', {}).get('passed', False)}",
        f"- Promotion gate stage: {gate.get('stage', 'stage_0_signal_gate')}",
        "- Blockers:",
    ]
    for blocker in report.get("blockers", []):
        lines.append(f"  - {blocker}")
    return report, "\n".join(lines)


def report_polymarket_pipeline(settings: Settings, run_dir: Path | str | None = None) -> tuple[dict[str, Any], str]:
    root = Path(run_dir) if run_dir else Path(_latest_pointer(settings.paths.outputs_dir / "latest_polymarket_shadow.txt") or "")
    if not root.exists():
        raise FileNotFoundError("No encuentro un shadow run de Polymarket para reportar.")
    return report_polymarket(root)


def report_polymarket_retro_pipeline(settings: Settings, run_dir: Path | str | None = None) -> tuple[dict[str, Any], str]:
    root = Path(run_dir) if run_dir else Path(_latest_pointer(settings.paths.outputs_dir / "latest_polymarket_retro.txt") or "")
    if not root.exists():
        raise FileNotFoundError("No encuentro un run retrospectivo de Polymarket para reportar.")
    return report_polymarket_retro(root)


def search_polymarket(query: str, limit: int = 10) -> list[dict]:
    client = PolymarketGammaClient()
    return client.search_markets(query=query, limit=limit)


def _build_prediction_rows(
    rows: pd.DataFrame,
    lambda_home,
    lambda_away,
    raw_probabilities,
    calibrated_probabilities,
) -> pd.DataFrame:
    prediction_rows = rows[
        [
            "match_id",
            "Date",
            "league_code",
            "league_name",
            "season",
            "HomeTeam",
            "AwayTeam",
            "odds_home",
            "odds_draw",
            "odds_away",
            "market_prob_home",
            "market_prob_draw",
            "market_prob_away",
        ]
    ].copy()
    prediction_rows["actual_outcome"] = rows.get("outcome", pd.Series(pd.NA, index=rows.index))
    prediction_rows["expected_goals_home"] = lambda_home
    prediction_rows["expected_goals_away"] = lambda_away
    for index, outcome in enumerate(("away", "draw", "home")):
        prediction_rows[f"prob_{outcome}_raw"] = raw_probabilities[:, index]
        prediction_rows[f"prob_{outcome}_calibrated"] = calibrated_probabilities[:, index]
    return prediction_rows


def _policy_from_payload(raw: dict[str, Any]):
    from .strategy import BetPolicy

    return BetPolicy(
        edge_threshold=float(raw["edge_threshold"]),
        ev_threshold=float(raw["ev_threshold"]),
        min_odds=float(raw["min_odds"]),
        max_odds=float(raw["max_odds"]),
        kelly_fraction=float(raw.get("kelly_fraction", 0.25)),
    )
