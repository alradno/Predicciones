from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .contracts import BacktestResult, FinalTrainingResult, PredictionResult, RunContext
from .evaluation import ExecutionSummary, PolicySummary, SignalSummary


def create_run_context(base_dir: Path, prefix: str) -> RunContext:
    run_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(run_id=run_id, run_dir=run_dir)


def compare_with_legacy(benchmarks_dir: Path, benchmark_dir_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    legacy_summary_path = benchmarks_dir / benchmark_dir_name / "metrics.json"
    if not legacy_summary_path.exists():
        return {}

    legacy = json.loads(legacy_summary_path.read_text(encoding="utf-8"))
    return {
        "legacy_accuracy": legacy.get("accuracy"),
        "legacy_log_loss": legacy.get("log_loss"),
        "new_baseline_accuracy": summary.get("baseline", {}).get("accuracy"),
        "new_strategy_roi": summary.get("strategy", {}).get("roi"),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")


_HIGHER_IS_BETTER = {"accuracy", "roi", "profit", "yield", "win_rate", "bets", "executed"}
_LOWER_IS_BETTER = {"log_loss", "brier_score", "max_drawdown"}


def _metric_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _metric_delta(candidate: dict[str, Any], reference: dict[str, Any], metric: str) -> float | None:
    candidate_value = _metric_value(candidate, metric)
    reference_value = _metric_value(reference, metric)
    if candidate_value is None or reference_value is None:
        return None
    if metric in _LOWER_IS_BETTER:
        return float(reference_value - candidate_value)
    return float(candidate_value - reference_value)


def _pairwise_comparison(
    candidate_name: str,
    candidate: dict[str, Any],
    reference_name: str,
    reference: dict[str, Any],
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "candidate": candidate_name,
        "reference": reference_name,
        "verdict": "not_comparable",
    }
    wins = 0
    losses = 0
    evaluated = 0
    for metric in metrics:
        delta = _metric_delta(candidate, reference, metric)
        if delta is None:
            continue
        evaluated += 1
        comparison[f"{metric}_delta"] = delta
        comparison[f"{metric}_better"] = delta > 0
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
    if evaluated == 0:
        return comparison
    if wins == evaluated:
        comparison["verdict"] = "better"
    elif losses == evaluated:
        comparison["verdict"] = "worse"
    elif wins > losses:
        comparison["verdict"] = "mixed_positive"
    elif losses > wins:
        comparison["verdict"] = "mixed_negative"
    else:
        comparison["verdict"] = "mixed_flat"
    return comparison


def _winner_by_metric(named_payloads: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
    winner_name: str | None = None
    winner_value: float | None = None
    for name, payload in named_payloads.items():
        value = _metric_value(payload, metric)
        if value is None:
            continue
        if winner_value is None:
            winner_name = name
            winner_value = value
            continue
        if metric in _LOWER_IS_BETTER and value < winner_value:
            winner_name = name
            winner_value = value
        elif metric not in _LOWER_IS_BETTER and value > winner_value:
            winner_name = name
            winner_value = value
    return {"metric": metric, "winner": winner_name, "value": winner_value}


def _ensure_signal_summary(summary: dict[str, Any]) -> SignalSummary:
    nested = summary.get("signal_summary")
    if nested is not None:
        return SignalSummary.ensure(nested)
    return SignalSummary.from_dict(
        {
            "selected_probability_source": summary.get("selected_probability_source"),
            "baseline": summary.get("baseline", {}),
            "model_raw": summary.get("model_raw", {}),
            "model_calibrated": summary.get("model_calibrated", {}),
            "feature_sanitization": summary.get("feature_sanitization", {}),
        }
    )


def _ensure_policy_summary(summary: dict[str, Any]) -> PolicySummary:
    nested = summary.get("policy_summary")
    if nested is not None:
        return PolicySummary.ensure(nested)
    return PolicySummary.from_dict(
        {
            "selected_policy": summary.get("selected_policy", {}),
            "training_metrics": summary.get("training_overall", {}),
            "holdout_metrics": summary.get("holdout_overall", summary.get("strategy", {})),
            "training_niche_metrics": summary.get("training_niche", {}),
            "holdout_niche_metrics": summary.get("holdout_niche", {}),
            "objective": {},
        }
    )


def _ensure_execution_summary(summary: dict[str, Any]) -> ExecutionSummary:
    nested = summary.get("execution_summary")
    if nested is not None:
        return ExecutionSummary.ensure(nested)
    return ExecutionSummary.from_dict(
        {
            "coverage": summary.get("execution_assumptions", {}),
            "fills": {
                "training_overall": summary.get("training_overall", {}),
                "holdout_overall": summary.get("holdout_overall", {}),
                "strategy": summary.get("strategy", {}),
            },
        }
    )


def build_backtest_truth_summary(summary: dict[str, Any]) -> dict[str, Any]:
    signal = _ensure_signal_summary(summary)
    policy = _ensure_policy_summary(summary)
    baseline = signal.baseline_metrics
    model_raw = signal.raw_metrics
    model_calibrated = signal.calibrated_metrics
    strategy = policy.holdout_metrics or summary.get("strategy", {})

    raw_vs_market = _pairwise_comparison("model_raw", model_raw, "baseline", baseline, ("accuracy", "log_loss", "brier_score"))
    calibrated_vs_raw = _pairwise_comparison(
        "model_calibrated",
        model_calibrated,
        "model_raw",
        model_raw,
        ("accuracy", "log_loss", "brier_score"),
    )
    strategy_vs_market = _pairwise_comparison("strategy", strategy, "baseline", baseline, ("roi", "profit"))
    strategy_vs_raw = _pairwise_comparison("strategy", strategy, "model_raw", model_raw, ("roi", "profit"))

    raw_beats_market = bool(raw_vs_market.get("accuracy_better")) and bool(raw_vs_market.get("log_loss_better"))
    calibration_helps = bool(calibrated_vs_raw.get("log_loss_better")) and calibrated_vs_raw.get("accuracy_delta", 0.0) >= 0.0
    policy_positive = (_metric_value(strategy, "roi") or 0.0) > 0.0 and (_metric_value(strategy, "profit") or 0.0) > 0.0

    flags: list[str] = []
    if raw_vs_market.get("accuracy_delta", 0.0) <= 0.0:
        flags.append("market_accuracy_still_ahead")
    if raw_vs_market.get("log_loss_delta", 0.0) <= 0.0:
        flags.append("market_log_loss_still_ahead")
    if not calibration_helps:
        flags.append("calibration_not_helping")
    if calibrated_vs_raw.get("accuracy_delta", 0.0) < 0.0:
        flags.append("calibration_hurts_accuracy")
    if not policy_positive:
        flags.append("policy_roi_non_positive")
    if strategy_vs_market.get("roi_delta", 0.0) <= 0.0:
        flags.append("policy_not_beating_market_roi")

    if raw_beats_market and policy_positive:
        verdict = "encouraging"
        headline = "El modelo ya supera al baseline de mercado y la policy monetiza la senal."
    elif raw_beats_market and not policy_positive:
        verdict = "signal_without_monetization"
        headline = "Las probabilidades mejoran frente al mercado, pero la policy todavia no convierte esa senal en beneficio."
    elif not raw_beats_market and policy_positive:
        verdict = "mixed"
        headline = "La policy encuentra valor positivo, pero la ventaja del modelo frente al mercado todavia no es robusta."
    else:
        verdict = "not_ready"
        headline = "El baseline de mercado sigue dominando o la policy pierde dinero; la ventaja experimental aun no esta demostrada."

    return {
        "overall_verdict": verdict,
        "headline": headline,
        "selected_probability_source": signal.selected_probability_source or ("calibrated" if calibration_helps else "raw"),
        "probability_leaderboard": {
            "accuracy": _winner_by_metric(
                {"baseline": baseline, "model_raw": model_raw, "model_calibrated": model_calibrated},
                "accuracy",
            ),
            "log_loss": _winner_by_metric(
                {"baseline": baseline, "model_raw": model_raw, "model_calibrated": model_calibrated},
                "log_loss",
            ),
            "brier_score": _winner_by_metric(
                {"baseline": baseline, "model_raw": model_raw, "model_calibrated": model_calibrated},
                "brier_score",
            ),
        },
        "comparisons": {
            "raw_vs_market": raw_vs_market,
            "calibrated_vs_raw": calibrated_vs_raw,
            "strategy_vs_market": strategy_vs_market,
            "strategy_vs_raw": strategy_vs_raw,
        },
        "signal_quality": {
            "beats_market": raw_beats_market,
            "calibration_helps": calibration_helps,
        },
        "policy_effect": {
            "positive_roi": policy_positive,
            "beats_market_roi": (strategy_vs_market.get("roi_delta", 0.0) or 0.0) > 0.0,
        },
        "execution_viability": {
            "forward_ready": False,
            "status": "not_applicable",
        },
        "flags": flags,
    }


def build_research_truth_summary(summary: dict[str, Any]) -> dict[str, Any]:
    signal = _ensure_signal_summary(summary)
    policy = _ensure_policy_summary(summary)
    execution = _ensure_execution_summary(summary)
    baseline = signal.baseline_metrics
    model_raw = signal.raw_metrics
    model_calibrated = signal.calibrated_metrics
    probability_decision = signal.extra.get("probability_decision", summary.get("probability_decision", {}))
    promotion_report = summary.get("promotion_report", {})
    training_niche = policy.extra.get("training_niche_metrics", summary.get("training_niche", {}))
    holdout_niche = policy.extra.get("holdout_niche_metrics", summary.get("holdout_niche", {}))
    holdout_overall = policy.holdout_metrics or summary.get("holdout_overall", {})
    quality_gate = signal.quality_gate or {}

    raw_vs_market = _pairwise_comparison("model_raw", model_raw, "baseline", baseline, ("accuracy", "log_loss", "brier_score"))
    calibrated_vs_raw = _pairwise_comparison(
        "model_calibrated",
        model_calibrated,
        "model_raw",
        model_raw,
        ("accuracy", "log_loss", "brier_score"),
    )
    stage0 = promotion_report.get("stage_0_signal_gate", {})
    stage1 = promotion_report.get("stage_1_discovery_complete", promotion_report.get("stage_1_research", {}))
    stage2 = promotion_report.get("stage_2_outer_holdout", promotion_report.get("stage_2_holdout", {}))
    stage3 = promotion_report.get("stage_3_shadow", {})
    stage4 = promotion_report.get("stage_4_limited_live", {})
    proxy_blocked = bool(stage2.get("proxy_blocked"))
    stage1_passed = bool(stage1.get("passed"))
    stage2_passed = bool(stage2.get("passed"))
    stage3_passed = bool(stage3.get("passed"))
    stage4_passed = bool(stage4.get("passed"))
    signal_beats_market = bool(quality_gate.get("passed", False)) or (
        bool(raw_vs_market.get("log_loss_better")) and bool(raw_vs_market.get("brier_score_better"))
    )
    stage0_passed = bool(stage0.get("passed")) if stage0 else signal_beats_market
    policy_positive = (_metric_value(holdout_overall, "roi") or 0.0) > 0.0
    conservative_score = float(policy.conservative_score_breakdown.get("score", 0.0) or 0.0)
    clv_summary = execution.clv_summary or {}
    fill_adjusted_ev_summary = execution.fill_adjusted_ev_summary or {}
    fill_model_summary = execution.fill_model_summary or {}
    execution_fill_rate = float(fill_model_summary.get("overall_fill_rate", holdout_overall.get("fill_rate", 0.0)) or 0.0)
    execution_viable = (
        float(fill_adjusted_ev_summary.get("mean", 0.0) or 0.0) >= 0.0
        and float(clv_summary.get("mean_clv", 0.0) or 0.0) >= 0.0
        and float(clv_summary.get("coverage", 0.0) or 0.0) >= 0.60
        and execution_fill_rate >= 0.25
    )

    flags: list[str] = []
    if not stage0_passed:
        flags.append("signal_gate_failed")
    if proxy_blocked:
        flags.append("proxy_blocked")
    if not stage1_passed:
        flags.append("stage1_not_ready")
    if not stage2_passed:
        flags.append("stage2_not_ready")
    if not stage3_passed:
        flags.append("shadow_not_validated")
    if (signal.selected_probability_source or summary.get("selected_probability_source", "raw")) == "raw":
        flags.append("calibration_not_selected")

    if not stage0_passed:
        verdict = "signal_gate_failed"
        headline = "La senal todavia no supera el quality gate probabilistico frente al baseline."
    elif proxy_blocked:
        verdict = "blocked_by_proxies"
        headline = "El nicho sigue bloqueado por proxies; la evidencia todavia no cuenta como forward real."
    elif stage2_passed and not stage3_passed:
        verdict = "holdout_ready_shadow_pending"
        headline = "El holdout ya supera Stage 2, pero todavia falta validar shadow y luego live."
    elif stage1_passed and not stage2_passed:
        verdict = "research_only"
        headline = "Hay senal en research, pero el holdout aun no confirma una ventaja robusta."
    elif stage3_passed and not stage4_passed:
        verdict = "shadow_ready_live_pending"
        headline = "El shadow ya valida ejecucion, pero todavia falta limited live."
    else:
        verdict = "not_ready"
        headline = "La evidencia actual no justifica promocion; primero hay que consolidar training, holdout y shadow."

    return {
        "overall_verdict": verdict,
        "headline": headline,
        "selected_probability_source": signal.selected_probability_source or summary.get("selected_probability_source", "raw"),
        "selected_source_strategy": signal.selected_source_strategy or signal.extra.get("selected_source_strategy", "global_best"),
        "probability_leaderboard": {
            "accuracy": _winner_by_metric(
                {"baseline": baseline, "model_raw": model_raw, "model_calibrated": model_calibrated},
                "accuracy",
            ),
            "log_loss": _winner_by_metric(
                {"baseline": baseline, "model_raw": model_raw, "model_calibrated": model_calibrated},
                "log_loss",
            ),
        },
        "comparisons": {
            "raw_vs_market": raw_vs_market,
            "calibrated_vs_raw": calibrated_vs_raw,
        },
        "probability_decision": probability_decision,
        "execution_snapshot": {
            "training_niche_roi": _metric_value(training_niche, "roi"),
            "holdout_niche_roi": _metric_value(holdout_niche, "roi"),
            "training_niche_bets": _metric_value(training_niche, "executed"),
            "holdout_niche_bets": _metric_value(holdout_niche, "executed"),
        },
        "signal_quality": {
            "beats_market": signal_beats_market,
            "calibration_helps": bool(calibrated_vs_raw.get("log_loss_better")) and calibrated_vs_raw.get("accuracy_delta", 0.0) >= 0.0,
            "quality_gate_passed": stage0_passed,
            "quality_gate": quality_gate,
        },
        "policy_effect": {
            "positive_holdout_roi": policy_positive,
            "conservative_score_positive": conservative_score > 0.0,
            "conservative_score": conservative_score,
            "stage1_passed": stage1_passed,
            "stage2_passed": stage2_passed,
        },
        "execution_viability": {
            "forward_ready": stage2_passed and not proxy_blocked and stage3_passed,
            "execution_gate_passed": execution_viable,
            "proxy_blocked": proxy_blocked,
            "shadow_passed": stage3_passed,
            "live_passed": stage4_passed,
            "clv_summary": clv_summary,
            "fill_adjusted_ev_summary": fill_adjusted_ev_summary,
            "fill_model_summary": fill_model_summary,
            "coverage": execution.coverage,
        },
        "promotion_readiness": {
            "stage0_passed": stage0_passed,
            "stage1_passed": stage1_passed,
            "stage2_passed": stage2_passed,
            "shadow_passed": stage3_passed,
            "live_passed": stage4_passed,
            "proxy_blocked": proxy_blocked,
            "blockers": promotion_report.get("blockers", []),
        },
        "flags": flags,
    }


def _metric_breakdown(predictions: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in predictions.groupby(group_column):
        rows.append(
            {
                group_column: value,
                "matches": int(len(group)),
                "raw_accuracy": float((group["raw_prediction"] == group["actual_outcome"]).mean()),
                "calibrated_accuracy": float((group["calibrated_prediction"] == group["actual_outcome"]).mean()),
                "baseline_accuracy": float((group["baseline_prediction"] == group["actual_outcome"]).mean()),
            }
        )
    return rows


def _odds_bin_breakdown(bets: pd.DataFrame) -> list[dict[str, Any]]:
    if bets.empty:
        return []

    working = bets.copy()
    working["odds_bin"] = pd.cut(
        working["selection_odds"],
        bins=[0.0, 1.5, 2.0, 3.0, 5.0, np.inf],
        labels=["<=1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", ">5.0"],
        include_lowest=True,
    )
    rows: list[dict[str, Any]] = []
    for odds_bin, group in working.groupby("odds_bin", observed=False):
        if len(group) == 0:
            continue
        rows.append(
            {
                "odds_bin": str(odds_bin),
                "bets": int(len(group)),
                "profit": float(group["flat_profit"].sum()),
                "roi": float(group["flat_profit"].mean()),
            }
        )
    return rows


def _plot_calibration(predictions: pd.DataFrame, target: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for axis, outcome in zip(axes, ("away", "draw", "home")):
        y_true = (predictions["actual_outcome"] == outcome).astype(int)
        prob_raw = predictions[f"prob_{outcome}_raw"].to_numpy()
        prob_cal = predictions[f"prob_{outcome}_calibrated"].to_numpy()
        raw_true, raw_pred = calibration_curve(y_true, prob_raw, n_bins=10, strategy="quantile")
        cal_true, cal_pred = calibration_curve(y_true, prob_cal, n_bins=10, strategy="quantile")
        axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
        axis.plot(raw_pred, raw_true, marker="o", label="raw")
        axis.plot(cal_pred, cal_true, marker="s", label="calibrated")
        axis.set_title(outcome)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Observed")
        axis.legend()

    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_bank_curve(bets: pd.DataFrame, target: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if bets.empty:
        ax.text(0.5, 0.5, "No bets selected", ha="center", va="center")
    else:
        working = bets.sort_values(["Date", "match_id"]).reset_index(drop=True)
        bank = working["flat_profit"].cumsum()
        ax.plot(bank.index, bank.values, linewidth=2, color="#1f77b4")
        ax.set_title("Flat Stake Bank Curve")
        ax.set_xlabel("Bet number")
        ax.set_ylabel("Profit units")
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(importance: pd.DataFrame, target: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if importance.empty:
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
    else:
        top = importance.head(15).sort_values("importance", ascending=True)
        ax.barh(top["feature"], top["importance"], color="#2a6f97")
        ax.set_title("Permutation Importance")
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_backtest_artifacts(
    result: BacktestResult,
    feature_importance: pd.DataFrame,
    benchmarks_dir: Path,
    benchmark_dir_name: str,
) -> dict[str, Path]:
    run_dir = result.run.run_dir
    predictions_path = run_dir / "prediction_rows.csv"
    bets_path = run_dir / "bet_rows.csv"
    summary_path = run_dir / "summary.json"
    calibration_path = run_dir / "calibration_curves.png"
    bank_curve_path = run_dir / "bank_curve.png"
    feature_importance_path = run_dir / "feature_importance.csv"
    feature_importance_plot = run_dir / "feature_importance.png"
    fold_metrics_path = run_dir / "fold_metrics.json"

    result.predictions.to_csv(predictions_path, index=False)
    result.bets.to_csv(bets_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    _plot_calibration(result.predictions, calibration_path)
    _plot_bank_curve(result.bets[result.bets["strategy_name"] == "edge_policy"], bank_curve_path)
    _plot_feature_importance(feature_importance, feature_importance_plot)

    summary_payload = dict(result.summary)
    summary_payload["by_league"] = _metric_breakdown(result.predictions, "league_code")
    summary_payload["by_season"] = _metric_breakdown(result.predictions, "season")
    strategy_bets = result.bets[result.bets["strategy_name"] == "edge_policy"]
    summary_payload["strategy_by_outcome"] = [
        {
            "selection": value,
            "bets": int(len(group)),
            "profit": float(group["flat_profit"].sum()),
            "roi": float(group["flat_profit"].mean()),
            "win_rate": float(group["won"].fillna(0).mean()),
        }
        for value, group in strategy_bets.groupby("selection", observed=True)
    ]
    summary_payload["strategy_by_odds_bin"] = _odds_bin_breakdown(strategy_bets)
    summary_payload["legacy_comparison"] = compare_with_legacy(benchmarks_dir, benchmark_dir_name, summary_payload)
    _save_json(summary_path, summary_payload)
    _save_json(fold_metrics_path, {"fold_metrics": summary_payload["fold_metrics"]})

    return {
        "predictions": predictions_path,
        "bets": bets_path,
        "summary": summary_path,
        "calibration": calibration_path,
        "bank_curve": bank_curve_path,
        "feature_importance": feature_importance_path,
        "feature_importance_plot": feature_importance_plot,
        "fold_metrics": fold_metrics_path,
    }


def save_final_training_artifacts(
    result: FinalTrainingResult,
    feature_importance: pd.DataFrame,
) -> dict[str, Path]:
    _plot_feature_importance(feature_importance, result.run.run_dir / "feature_importance.png")
    return {
        "model": result.model_path,
        "summary": result.summary_path,
        "feature_importance": result.feature_importance_path,
    }


def save_prediction_artifacts(result: PredictionResult) -> dict[str, Path]:
    predictions_path = result.run.run_dir / "predictions.csv"
    bets_path = result.run.run_dir / "bets.csv"
    result.predictions.to_csv(predictions_path, index=False)
    result.bets.to_csv(bets_path, index=False)
    return {"predictions": predictions_path, "bets": bets_path}


def format_summary(summary: dict[str, Any]) -> str:
    if "holdout_overall" in summary:
        signal = _ensure_signal_summary(summary)
        policy = _ensure_policy_summary(summary)
        execution = _ensure_execution_summary(summary)
        holdout = policy.holdout_metrics or summary.get("holdout_overall", {})
        niche = policy.extra.get("holdout_niche_metrics", summary.get("holdout_niche", {}))
        promotion = summary.get("promotion_report", {})
        truth = summary.get("experimental_truth") or build_research_truth_summary(summary)
        return (
            f"- Veredicto: {truth.get('headline', '')}\n"
            f"- Predice mejor: {truth.get('signal_quality', {}).get('beats_market', False)}\n"
            f"- La policy suma: {truth.get('policy_effect', {}).get('positive_holdout_roi', False)}\n"
            f"- Ejecucion viable: {truth.get('execution_viability', {}).get('forward_ready', False)}\n"
            f"- Signal gate: {truth.get('signal_quality', {}).get('quality_gate_passed', False)}\n"
            f"- Fuente probabilistica elegida: {signal.selected_probability_source or summary.get('selected_probability_source', 'raw')}\n"
            f"- Estrategia de fuente: {signal.selected_source_strategy or summary.get('selected_source_strategy', 'global_best')}\n"
            f"- Holdout net bets: {holdout.get('executed', 0)}\n"
            f"- Holdout net ROI: {holdout.get('roi', 0.0):.4f}\n"
            f"- Niche holdout ROI: {niche.get('roi', 0.0):.4f}\n"
            f"- CLV medio: {execution.clv_summary.get('mean_clv', 0.0):.4f}\n"
            f"- Proxy blocked: {execution.coverage.get('proxy_blocked_for_promotion', False)}\n"
            f"- Stage 2 listo: {promotion.get('stage_2_outer_holdout', promotion.get('stage_2_holdout', {})).get('passed', False)}"
        )
    signal = _ensure_signal_summary(summary)
    policy = _ensure_policy_summary(summary)
    baseline = signal.baseline_metrics
    strategy = policy.holdout_metrics or summary.get("strategy", {})
    truth = summary.get("honest_diagnostics") or build_backtest_truth_summary(summary)
    return (
        f"- Veredicto: {truth.get('headline', '')}\n"
        f"- Predice mejor: {truth.get('signal_quality', {}).get('beats_market', False)}\n"
        f"- La policy suma: {truth.get('policy_effect', {}).get('positive_roi', False)}\n"
        f"- Ejecucion viable: n/a\n"
        f"- Fuente probabilistica recomendada: {truth.get('selected_probability_source', 'raw')}\n"
        f"- Baseline accuracy: {baseline.get('accuracy', 0.0):.4f}\n"
        f"- Goal model raw accuracy: {signal.raw_metrics.get('accuracy', 0.0):.4f}\n"
        f"- Strategy bets: {strategy.get('bets', 0)}\n"
        f"- Strategy ROI: {strategy.get('roi', 0.0):.4f}\n"
        f"- Strategy profit: {strategy.get('profit', 0.0):.2f}"
    )
