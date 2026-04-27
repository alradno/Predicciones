from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _split_csv_env(name: str, default: str) -> tuple[str, ...]:
    value = os.getenv(name, default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    outputs_dir: Path
    runs_dir: Path
    models_dir: Path
    benchmarks_dir: Path


@dataclass(frozen=True)
class PolicySearchSpace:
    edge_thresholds: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)
    ev_thresholds: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05)
    min_odds_options: tuple[float, ...] = (1.2, 1.5, 2.0)
    max_odds_options: tuple[float, ...] = (4.0, 6.0, 10.0)
    top_quantiles: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    min_bets: int = 20
    max_kelly_fraction: float = 0.25


@dataclass(frozen=True)
class SnapshotConfig:
    default_kickoff_hour: int = 15
    closing_proxy_capture_minutes: int = 45
    time_bucket_edges: tuple[int, ...] = (30, 120, 360)
    movement_threshold: float = 0.03


@dataclass(frozen=True)
class ExecutionConfig:
    decision_minutes_before_kickoff: int = 45
    max_quote_age_minutes: int = 360
    slippage_rate: float = 0.01
    commission_rate: float = 0.0
    max_stake: float = 1.0
    min_liquidity: float = 0.0
    allow_closing_proxy_for_research: bool = True
    allow_closing_proxy_for_promotion: bool = False


@dataclass(frozen=True)
class ResearchConfig:
    holdout_windows: int = 3
    min_segment_bets: int = 60
    min_segment_folds: int = 3
    max_segment_dimensions: int = 3
    positive_fold_ratio: float = 0.55
    positive_penalty_weight: float = 0.10
    drawdown_weight: float = 0.20
    generalization_gap_weight: float = 0.10
    stage1_min_roi: float = 0.08
    stage1_min_bets: int = 300
    stage2_min_roi: float = 0.20
    stage2_min_bets: int = 150
    max_drawdown_units: float = 50.0
    flat_stake: float = 1.0
    provisional_edge_threshold: float = 0.02
    provisional_ev_threshold: float = 0.0
    regional_source_strategy: str = "regional_best_with_global_fallback"
    regional_source_fields: tuple[str, ...] = ("league_code", "odds_band", "edge_band_calibrated", "time_bucket")
    regional_source_min_rows: int = 75
    regional_source_min_log_loss_delta: float = 0.005
    regional_source_require_non_worse_brier: bool = True
    candidate_top_niches: int = 20
    candidate_top_policies: int = 25
    clv_coverage_min: float = 0.60
    mean_clv_min: float = 0.0
    observed_fill_rate_min: float = 0.25
    fill_adjusted_ev_min: float = 0.0


@dataclass(frozen=True)
class PolymarketConfig:
    database_filename: str = "polymarket_shadow.sqlite"
    supported_leagues: tuple[str, ...] = ("E0", "SP1", "D1")
    supported_sports: tuple[str, ...] = ("epl", "lal", "bun")
    discovery_queries: tuple[str, ...] = ("Premier League", "La Liga", "Bundesliga")
    discovery_window_hours: int = 72
    poll_interval_seconds: int = 300
    checkpoint_interval_seconds: int = 15
    book_freshness_seconds: int = 5
    decision_offset_minutes: int = 45
    decision_book_freshness_seconds: int = 15
    decision_capture_lead_seconds: int = 15
    decision_capture_retry_attempts: int = 3
    decision_capture_retry_delay_seconds: float = 0.75
    slippage_cushion: float = 0.01
    notionals_ladder: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0)
    league_match_tolerance_minutes: int = 180
    historical_backfill_start: str = ""
    historical_backfill_end: str = ""
    historical_chunk_days: int = 30
    historical_price_mode: str = "approx"
    historical_quality_min: str = "history_proxy"
    historical_proxy_haircut: float = 0.03
    historical_min_markets: int = 20
    historical_tuning_quality_min: str = "history_proxy"
    policy_objective: str = "roi_drawdown"
    mapping_score_threshold: float = 0.85
    mapping_score_gap: float = 0.03
    retro_min_mapped_matches: int = 200
    retro_min_selected_predictions: int = 100
    market_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    sports_ws_url: str = "wss://sports-api.polymarket.com/ws"


@dataclass(frozen=True)
class TradingConfig:
    venue: str = "disabled"
    live_enabled: bool = False
    paper_enabled: bool = True
    bankroll_usdc: float | None = None
    jurisdiction_confirmed: bool = False
    allow_vpn_bypass: bool = False
    max_order_bankroll_fraction: float = 0.0025
    max_daily_bankroll_fraction: float = 0.01
    max_open_bankroll_fraction: float = 0.03
    max_quote_age_seconds: int = 15
    paper_max_notional: float = 1.0
    default_order_size: float = 1.0
    order_time_in_force: str = "GTD"


@dataclass(frozen=True)
class BacktestConfig:
    rolling_window: int = 8
    min_train_matches: int = 500
    min_train_days: int = 365
    test_window_days: int = 28
    calibration_matches: int = 150
    policy_matches: int = 150
    min_subtrain_matches: int = 250
    max_poisson_goals: int = 10
    calibrator_epsilon: float = 1e-6
    dixon_coles_bounds: tuple[float, float] = (-0.15, 0.15)
    policy_search: PolicySearchSpace = field(default_factory=PolicySearchSpace)


@dataclass(frozen=True)
class Settings:
    paths: ProjectPaths
    anthropic_api_key: str | None
    claude_model: str
    default_leagues: tuple[str, ...]
    default_seasons: tuple[str, ...]
    benchmark_dir_name: str
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    trade: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def _paths_to_dict(paths: ProjectPaths) -> dict[str, str]:
    return {
        "root": str(paths.root),
        "data_dir": str(paths.data_dir),
        "outputs_dir": str(paths.outputs_dir),
        "runs_dir": str(paths.runs_dir),
        "models_dir": str(paths.models_dir),
        "benchmarks_dir": str(paths.benchmarks_dir),
    }


def _settings_dict(settings: Settings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["paths"] = _paths_to_dict(settings.paths)
    return payload


def _coerce_policy_search(raw: dict[str, Any]) -> PolicySearchSpace:
    defaults = PolicySearchSpace()
    return PolicySearchSpace(
        edge_thresholds=tuple(raw.get("edge_thresholds", defaults.edge_thresholds)),
        ev_thresholds=tuple(raw.get("ev_thresholds", defaults.ev_thresholds)),
        min_odds_options=tuple(raw.get("min_odds_options", defaults.min_odds_options)),
        max_odds_options=tuple(raw.get("max_odds_options", defaults.max_odds_options)),
        top_quantiles=tuple(raw.get("top_quantiles", defaults.top_quantiles)),
        min_bets=int(raw.get("min_bets", defaults.min_bets)),
        max_kelly_fraction=float(raw.get("max_kelly_fraction", defaults.max_kelly_fraction)),
    )


def _coerce_snapshot(raw: dict[str, Any]) -> SnapshotConfig:
    defaults = SnapshotConfig()
    return SnapshotConfig(
        default_kickoff_hour=int(raw.get("default_kickoff_hour", defaults.default_kickoff_hour)),
        closing_proxy_capture_minutes=int(raw.get("closing_proxy_capture_minutes", defaults.closing_proxy_capture_minutes)),
        time_bucket_edges=tuple(raw.get("time_bucket_edges", defaults.time_bucket_edges)),
        movement_threshold=float(raw.get("movement_threshold", defaults.movement_threshold)),
    )


def _coerce_execution(raw: dict[str, Any]) -> ExecutionConfig:
    defaults = ExecutionConfig()
    return ExecutionConfig(
        decision_minutes_before_kickoff=int(raw.get("decision_minutes_before_kickoff", defaults.decision_minutes_before_kickoff)),
        max_quote_age_minutes=int(raw.get("max_quote_age_minutes", defaults.max_quote_age_minutes)),
        slippage_rate=float(raw.get("slippage_rate", defaults.slippage_rate)),
        commission_rate=float(raw.get("commission_rate", defaults.commission_rate)),
        max_stake=float(raw.get("max_stake", defaults.max_stake)),
        min_liquidity=float(raw.get("min_liquidity", defaults.min_liquidity)),
        allow_closing_proxy_for_research=bool(raw.get("allow_closing_proxy_for_research", defaults.allow_closing_proxy_for_research)),
        allow_closing_proxy_for_promotion=bool(raw.get("allow_closing_proxy_for_promotion", defaults.allow_closing_proxy_for_promotion)),
    )


def _coerce_research(raw: dict[str, Any]) -> ResearchConfig:
    defaults = ResearchConfig()
    return ResearchConfig(
        holdout_windows=int(raw.get("holdout_windows", defaults.holdout_windows)),
        min_segment_bets=int(raw.get("min_segment_bets", defaults.min_segment_bets)),
        min_segment_folds=int(raw.get("min_segment_folds", defaults.min_segment_folds)),
        max_segment_dimensions=int(raw.get("max_segment_dimensions", defaults.max_segment_dimensions)),
        positive_fold_ratio=float(raw.get("positive_fold_ratio", defaults.positive_fold_ratio)),
        positive_penalty_weight=float(raw.get("positive_penalty_weight", defaults.positive_penalty_weight)),
        drawdown_weight=float(raw.get("drawdown_weight", defaults.drawdown_weight)),
        generalization_gap_weight=float(raw.get("generalization_gap_weight", defaults.generalization_gap_weight)),
        stage1_min_roi=float(raw.get("stage1_min_roi", defaults.stage1_min_roi)),
        stage1_min_bets=int(raw.get("stage1_min_bets", defaults.stage1_min_bets)),
        stage2_min_roi=float(raw.get("stage2_min_roi", defaults.stage2_min_roi)),
        stage2_min_bets=int(raw.get("stage2_min_bets", defaults.stage2_min_bets)),
        max_drawdown_units=float(raw.get("max_drawdown_units", defaults.max_drawdown_units)),
        flat_stake=float(raw.get("flat_stake", defaults.flat_stake)),
        provisional_edge_threshold=float(raw.get("provisional_edge_threshold", defaults.provisional_edge_threshold)),
        provisional_ev_threshold=float(raw.get("provisional_ev_threshold", defaults.provisional_ev_threshold)),
        regional_source_strategy=str(raw.get("regional_source_strategy", defaults.regional_source_strategy)),
        regional_source_fields=tuple(raw.get("regional_source_fields", defaults.regional_source_fields)),
        regional_source_min_rows=int(raw.get("regional_source_min_rows", defaults.regional_source_min_rows)),
        regional_source_min_log_loss_delta=float(
            raw.get("regional_source_min_log_loss_delta", defaults.regional_source_min_log_loss_delta)
        ),
        regional_source_require_non_worse_brier=bool(
            raw.get("regional_source_require_non_worse_brier", defaults.regional_source_require_non_worse_brier)
        ),
        candidate_top_niches=int(raw.get("candidate_top_niches", defaults.candidate_top_niches)),
        candidate_top_policies=int(raw.get("candidate_top_policies", defaults.candidate_top_policies)),
        clv_coverage_min=float(raw.get("clv_coverage_min", defaults.clv_coverage_min)),
        mean_clv_min=float(raw.get("mean_clv_min", defaults.mean_clv_min)),
        observed_fill_rate_min=float(raw.get("observed_fill_rate_min", defaults.observed_fill_rate_min)),
        fill_adjusted_ev_min=float(raw.get("fill_adjusted_ev_min", defaults.fill_adjusted_ev_min)),
    )


def _coerce_polymarket(raw: dict[str, Any]) -> PolymarketConfig:
    defaults = PolymarketConfig()
    return PolymarketConfig(
        database_filename=str(raw.get("database_filename", defaults.database_filename)),
        supported_leagues=tuple(raw.get("supported_leagues", defaults.supported_leagues)),
        supported_sports=tuple(raw.get("supported_sports", defaults.supported_sports)),
        discovery_queries=tuple(raw.get("discovery_queries", defaults.discovery_queries)),
        discovery_window_hours=int(raw.get("discovery_window_hours", defaults.discovery_window_hours)),
        poll_interval_seconds=int(raw.get("poll_interval_seconds", defaults.poll_interval_seconds)),
        checkpoint_interval_seconds=int(raw.get("checkpoint_interval_seconds", defaults.checkpoint_interval_seconds)),
        book_freshness_seconds=int(raw.get("book_freshness_seconds", defaults.book_freshness_seconds)),
        decision_offset_minutes=int(raw.get("decision_offset_minutes", defaults.decision_offset_minutes)),
        decision_book_freshness_seconds=int(
            raw.get("decision_book_freshness_seconds", defaults.decision_book_freshness_seconds)
        ),
        decision_capture_lead_seconds=int(
            raw.get("decision_capture_lead_seconds", defaults.decision_capture_lead_seconds)
        ),
        decision_capture_retry_attempts=int(
            raw.get("decision_capture_retry_attempts", defaults.decision_capture_retry_attempts)
        ),
        decision_capture_retry_delay_seconds=float(
            raw.get(
                "decision_capture_retry_delay_seconds",
                defaults.decision_capture_retry_delay_seconds,
            )
        ),
        slippage_cushion=float(raw.get("slippage_cushion", defaults.slippage_cushion)),
        notionals_ladder=tuple(raw.get("notionals_ladder", defaults.notionals_ladder)),
        league_match_tolerance_minutes=int(
            raw.get("league_match_tolerance_minutes", defaults.league_match_tolerance_minutes)
        ),
        historical_backfill_start=str(raw.get("historical_backfill_start", defaults.historical_backfill_start)),
        historical_backfill_end=str(raw.get("historical_backfill_end", defaults.historical_backfill_end)),
        historical_chunk_days=int(raw.get("historical_chunk_days", defaults.historical_chunk_days)),
        historical_price_mode=str(raw.get("historical_price_mode", defaults.historical_price_mode)),
        historical_quality_min=str(raw.get("historical_quality_min", defaults.historical_quality_min)),
        historical_proxy_haircut=float(raw.get("historical_proxy_haircut", defaults.historical_proxy_haircut)),
        historical_min_markets=int(raw.get("historical_min_markets", defaults.historical_min_markets)),
        historical_tuning_quality_min=str(
            raw.get("historical_tuning_quality_min", defaults.historical_tuning_quality_min)
        ),
        policy_objective=str(raw.get("policy_objective", defaults.policy_objective)),
        mapping_score_threshold=float(raw.get("mapping_score_threshold", defaults.mapping_score_threshold)),
        mapping_score_gap=float(raw.get("mapping_score_gap", defaults.mapping_score_gap)),
        retro_min_mapped_matches=int(raw.get("retro_min_mapped_matches", defaults.retro_min_mapped_matches)),
        retro_min_selected_predictions=int(
            raw.get("retro_min_selected_predictions", defaults.retro_min_selected_predictions)
        ),
        market_ws_url=str(raw.get("market_ws_url", defaults.market_ws_url)),
        sports_ws_url=str(raw.get("sports_ws_url", defaults.sports_ws_url)),
    )


def _coerce_trade(raw: dict[str, Any]) -> TradingConfig:
    defaults = TradingConfig()
    return TradingConfig(
        venue=str(raw.get("venue", defaults.venue)),
        live_enabled=bool(raw.get("live_enabled", defaults.live_enabled)),
        paper_enabled=bool(raw.get("paper_enabled", defaults.paper_enabled)),
        bankroll_usdc=_optional_float(raw.get("bankroll_usdc", defaults.bankroll_usdc)),
        jurisdiction_confirmed=bool(raw.get("jurisdiction_confirmed", defaults.jurisdiction_confirmed)),
        allow_vpn_bypass=bool(raw.get("allow_vpn_bypass", defaults.allow_vpn_bypass)),
        max_order_bankroll_fraction=float(
            raw.get("max_order_bankroll_fraction", defaults.max_order_bankroll_fraction)
        ),
        max_daily_bankroll_fraction=float(
            raw.get("max_daily_bankroll_fraction", defaults.max_daily_bankroll_fraction)
        ),
        max_open_bankroll_fraction=float(
            raw.get("max_open_bankroll_fraction", defaults.max_open_bankroll_fraction)
        ),
        max_quote_age_seconds=int(raw.get("max_quote_age_seconds", defaults.max_quote_age_seconds)),
        paper_max_notional=float(raw.get("paper_max_notional", defaults.paper_max_notional)),
        default_order_size=float(raw.get("default_order_size", defaults.default_order_size)),
        order_time_in_force=str(raw.get("order_time_in_force", defaults.order_time_in_force)),
    )


def _coerce_backtest(raw: dict[str, Any]) -> BacktestConfig:
    defaults = BacktestConfig()
    return BacktestConfig(
        rolling_window=int(raw.get("rolling_window", defaults.rolling_window)),
        min_train_matches=int(raw.get("min_train_matches", defaults.min_train_matches)),
        min_train_days=int(raw.get("min_train_days", defaults.min_train_days)),
        test_window_days=int(raw.get("test_window_days", defaults.test_window_days)),
        calibration_matches=int(raw.get("calibration_matches", defaults.calibration_matches)),
        policy_matches=int(raw.get("policy_matches", defaults.policy_matches)),
        min_subtrain_matches=int(raw.get("min_subtrain_matches", defaults.min_subtrain_matches)),
        max_poisson_goals=int(raw.get("max_poisson_goals", defaults.max_poisson_goals)),
        calibrator_epsilon=float(raw.get("calibrator_epsilon", defaults.calibrator_epsilon)),
        dixon_coles_bounds=tuple(raw.get("dixon_coles_bounds", defaults.dixon_coles_bounds)),
        policy_search=_coerce_policy_search(raw.get("policy_search", {})),
    )


def load_settings(root: Path | None = None, config_path: Path | str | None = None) -> Settings:
    root = root or Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")

    paths = ProjectPaths(
        root=root,
        data_dir=root / "data",
        outputs_dir=root / "outputs",
        runs_dir=root / "outputs" / "runs",
        models_dir=root / "outputs" / "models",
        benchmarks_dir=root / "benchmarks",
    )

    for path in (paths.data_dir, paths.outputs_dir, paths.runs_dir, paths.models_dir, paths.benchmarks_dir):
        path.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        paths=paths,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        default_leagues=_split_csv_env("DEFAULT_LEAGUES", "E0,SP1,D1,I1,F1"),
        default_seasons=_split_csv_env("DEFAULT_SEASONS", "2425,2324,2223"),
        benchmark_dir_name=os.getenv("BENCHMARK_DIR_NAME", "legacy_v1"),
        snapshot=SnapshotConfig(
            default_kickoff_hour=int(os.getenv("DEFAULT_KICKOFF_HOUR", str(SnapshotConfig().default_kickoff_hour))),
            closing_proxy_capture_minutes=int(
                os.getenv("CLOSING_PROXY_CAPTURE_MINUTES", str(SnapshotConfig().closing_proxy_capture_minutes))
            ),
        ),
        execution=ExecutionConfig(
            decision_minutes_before_kickoff=int(
                os.getenv("DECISION_MINUTES_BEFORE_KICKOFF", str(ExecutionConfig().decision_minutes_before_kickoff))
            ),
            slippage_rate=float(os.getenv("SLIPPAGE_RATE", str(ExecutionConfig().slippage_rate))),
            commission_rate=float(os.getenv("COMMISSION_RATE", str(ExecutionConfig().commission_rate))),
        ),
        polymarket=PolymarketConfig(
            database_filename=os.getenv("POLYMARKET_DB_FILENAME", PolymarketConfig().database_filename),
            discovery_window_hours=int(
                os.getenv("POLYMARKET_DISCOVERY_WINDOW_HOURS", str(PolymarketConfig().discovery_window_hours))
            ),
            poll_interval_seconds=int(
                os.getenv("POLYMARKET_POLL_INTERVAL_SECONDS", str(PolymarketConfig().poll_interval_seconds))
            ),
            checkpoint_interval_seconds=int(
                os.getenv(
                    "POLYMARKET_CHECKPOINT_INTERVAL_SECONDS",
                    str(PolymarketConfig().checkpoint_interval_seconds),
                )
            ),
            book_freshness_seconds=int(
                os.getenv("POLYMARKET_BOOK_FRESHNESS_SECONDS", str(PolymarketConfig().book_freshness_seconds))
            ),
            decision_offset_minutes=int(
                os.getenv("POLYMARKET_DECISION_OFFSET_MINUTES", str(PolymarketConfig().decision_offset_minutes))
            ),
            decision_book_freshness_seconds=int(
                os.getenv(
                    "POLYMARKET_DECISION_BOOK_FRESHNESS_SECONDS",
                    str(PolymarketConfig().decision_book_freshness_seconds),
                )
            ),
            decision_capture_lead_seconds=int(
                os.getenv(
                    "POLYMARKET_DECISION_CAPTURE_LEAD_SECONDS",
                    str(PolymarketConfig().decision_capture_lead_seconds),
                )
            ),
            decision_capture_retry_attempts=int(
                os.getenv(
                    "POLYMARKET_DECISION_CAPTURE_RETRY_ATTEMPTS",
                    str(PolymarketConfig().decision_capture_retry_attempts),
                )
            ),
            decision_capture_retry_delay_seconds=float(
                os.getenv(
                    "POLYMARKET_DECISION_CAPTURE_RETRY_DELAY_SECONDS",
                    str(PolymarketConfig().decision_capture_retry_delay_seconds),
                )
            ),
            slippage_cushion=float(
                os.getenv("POLYMARKET_SLIPPAGE_CUSHION", str(PolymarketConfig().slippage_cushion))
            ),
            historical_backfill_start=os.getenv(
                "POLYMARKET_HISTORICAL_BACKFILL_START",
                PolymarketConfig().historical_backfill_start,
            ),
            historical_backfill_end=os.getenv(
                "POLYMARKET_HISTORICAL_BACKFILL_END",
                PolymarketConfig().historical_backfill_end,
            ),
            historical_chunk_days=int(
                os.getenv(
                    "POLYMARKET_HISTORICAL_CHUNK_DAYS",
                    str(PolymarketConfig().historical_chunk_days),
                )
            ),
            historical_price_mode=os.getenv(
                "POLYMARKET_HISTORICAL_PRICE_MODE",
                PolymarketConfig().historical_price_mode,
            ),
            historical_quality_min=os.getenv(
                "POLYMARKET_HISTORICAL_QUALITY_MIN",
                PolymarketConfig().historical_quality_min,
            ),
            historical_proxy_haircut=float(
                os.getenv(
                    "POLYMARKET_HISTORICAL_PROXY_HAIRCUT",
                    str(PolymarketConfig().historical_proxy_haircut),
                )
            ),
            historical_min_markets=int(
                os.getenv(
                    "POLYMARKET_HISTORICAL_MIN_MARKETS",
                    str(PolymarketConfig().historical_min_markets),
                )
            ),
            historical_tuning_quality_min=os.getenv(
                "POLYMARKET_HISTORICAL_TUNING_QUALITY_MIN",
                PolymarketConfig().historical_tuning_quality_min,
            ),
            policy_objective=os.getenv(
                "POLYMARKET_POLICY_OBJECTIVE",
                PolymarketConfig().policy_objective,
            ),
            mapping_score_threshold=float(
                os.getenv(
                    "POLYMARKET_MAPPING_SCORE_THRESHOLD",
                    str(PolymarketConfig().mapping_score_threshold),
                )
            ),
            mapping_score_gap=float(
                os.getenv(
                    "POLYMARKET_MAPPING_SCORE_GAP",
                    str(PolymarketConfig().mapping_score_gap),
                )
            ),
            retro_min_mapped_matches=int(
                os.getenv(
                    "POLYMARKET_RETRO_MIN_MAPPED_MATCHES",
                    str(PolymarketConfig().retro_min_mapped_matches),
                )
            ),
            retro_min_selected_predictions=int(
                os.getenv(
                    "POLYMARKET_RETRO_MIN_SELECTED_PREDICTIONS",
                    str(PolymarketConfig().retro_min_selected_predictions),
                )
            ),
        ),
        trade=TradingConfig(
            venue=os.getenv("PREDICCIONES_TRADE_VENUE", TradingConfig().venue),
            live_enabled=_env_bool("PREDICCIONES_TRADE_LIVE_ENABLED", TradingConfig().live_enabled),
            paper_enabled=_env_bool("PREDICCIONES_TRADE_PAPER_ENABLED", TradingConfig().paper_enabled),
            bankroll_usdc=_optional_float(os.getenv("PREDICCIONES_TRADE_BANKROLL_USDC")),
            jurisdiction_confirmed=_env_bool(
                "PREDICCIONES_TRADE_JURISDICTION_CONFIRMED",
                TradingConfig().jurisdiction_confirmed,
            ),
            allow_vpn_bypass=_env_bool("PREDICCIONES_TRADE_ALLOW_VPN_BYPASS", TradingConfig().allow_vpn_bypass),
            max_order_bankroll_fraction=float(
                os.getenv(
                    "PREDICCIONES_TRADE_MAX_ORDER_BANKROLL_FRACTION",
                    str(TradingConfig().max_order_bankroll_fraction),
                )
            ),
            max_daily_bankroll_fraction=float(
                os.getenv(
                    "PREDICCIONES_TRADE_MAX_DAILY_BANKROLL_FRACTION",
                    str(TradingConfig().max_daily_bankroll_fraction),
                )
            ),
            max_open_bankroll_fraction=float(
                os.getenv(
                    "PREDICCIONES_TRADE_MAX_OPEN_BANKROLL_FRACTION",
                    str(TradingConfig().max_open_bankroll_fraction),
                )
            ),
            max_quote_age_seconds=int(
                os.getenv("PREDICCIONES_TRADE_MAX_QUOTE_AGE_SECONDS", str(TradingConfig().max_quote_age_seconds))
            ),
            paper_max_notional=float(
                os.getenv("PREDICCIONES_TRADE_PAPER_MAX_NOTIONAL", str(TradingConfig().paper_max_notional))
            ),
            default_order_size=float(
                os.getenv("PREDICCIONES_TRADE_DEFAULT_ORDER_SIZE", str(TradingConfig().default_order_size))
            ),
            order_time_in_force=os.getenv("PREDICCIONES_TRADE_ORDER_TIME_IN_FORCE", TradingConfig().order_time_in_force),
        ),
        backtest=BacktestConfig(
            rolling_window=int(os.getenv("ROLLING_WINDOW", str(BacktestConfig().rolling_window))),
        ),
    )

    if config_path:
        config_file = Path(config_path)
        override = json.loads(config_file.read_text(encoding="utf-8"))
        payload = _merge_dict(_settings_dict(settings), override)
        settings = replace(
            settings,
            anthropic_api_key=payload.get("anthropic_api_key") or settings.anthropic_api_key,
            claude_model=payload.get("claude_model", settings.claude_model),
            default_leagues=tuple(payload.get("default_leagues", settings.default_leagues)),
            default_seasons=tuple(payload.get("default_seasons", settings.default_seasons)),
            benchmark_dir_name=payload.get("benchmark_dir_name", settings.benchmark_dir_name),
            snapshot=_coerce_snapshot(payload.get("snapshot", {})),
            execution=_coerce_execution(payload.get("execution", {})),
            research=_coerce_research(payload.get("research", {})),
            polymarket=_coerce_polymarket(payload.get("polymarket", {})),
            trade=_coerce_trade(payload.get("trade", {})),
            backtest=_coerce_backtest(payload.get("backtest", {})),
        )

    return settings
