from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import _probability_metrics
from .contracts import OUTCOME_ORDER
from .evaluation import ExecutionSummary, PolicySummary, SignalSummary
from .models import predicted_outcomes
from .reporting import build_research_truth_summary
from .research_niches import build_promotion_report
from .research_candidates import _probability_array
from .strategy import BetPolicy


def _summary_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return dict(payload)
    return {"value": value}


def _uses_proxy(rows: pd.DataFrame) -> bool:
    if rows.empty or "source_type" not in rows.columns:
        return False
    return bool(rows["source_type"].astype(str).eq("closing_proxy").any())


def _source_type_counts(rows: pd.DataFrame) -> dict[str, int]:
    if rows.empty or "source_type" not in rows.columns:
        return {}
    counts = rows["source_type"].fillna("").astype(str).value_counts()
    return {str(index): int(value) for index, value in counts.items() if str(index)}


def _build_research_summary(
    predictions: pd.DataFrame,
    training_execution: pd.DataFrame,
    holdout_execution: pd.DataFrame,
    training_niche_execution: pd.DataFrame,
    holdout_niche_execution: pd.DataFrame,
    probability_decision: dict[str, Any],
    selected_probability_source: str,
    selected_policy: BetPolicy,
    niches: pd.DataFrame,
    settings: Any,
    inner_fold_assignments: pd.DataFrame | None = None,
    policy_ranking: pd.DataFrame | None = None,
    combination_ranking: pd.DataFrame | None = None,
    execution_quality: dict[str, Any] | None = None,
    top_niche_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outer_holdout = predictions[predictions["fold_segment"].astype(str).eq("holdout")].copy()
    actual = outer_holdout["actual_target"].to_numpy()
    baseline_probs = outer_holdout[[f"market_prob_{outcome}" for outcome in OUTCOME_ORDER]].to_numpy()
    raw_probs = _probability_array(outer_holdout, "raw")
    calibrated_probs = _probability_array(outer_holdout, "calibrated")
    effective_probs = _probability_array(outer_holdout, "effective") if all(
        f"prob_{outcome}_effective" in outer_holdout.columns for outcome in OUTCOME_ORDER
    ) else _probability_array(outer_holdout, probability_decision.get("global_best_source", "raw"))
    probability_decision_payload = _summary_payload(probability_decision)
    selected_policy_payload = selected_policy.to_dict()
    training_overall = summarize_execution(training_execution)
    holdout_overall = summarize_execution(holdout_execution)
    training_niche = summarize_execution(training_niche_execution)
    holdout_niche = summarize_execution(holdout_niche_execution)
    selected_source_strategy = str(probability_decision_payload.get("selected_source_strategy", "global_best"))
    global_best_source = str(probability_decision_payload.get("global_best_source", selected_probability_source))
    training_proxy = _uses_proxy(training_niche_execution)
    holdout_proxy = _uses_proxy(holdout_niche_execution)
    execution_quality = execution_quality or {}
    selected_metrics = _probability_metrics(actual, effective_probs, predicted_outcomes(effective_probs))
    baseline_metrics = _probability_metrics(
        actual,
        baseline_probs,
        np.array([OUTCOME_ORDER.index(value) for value in outer_holdout["baseline_prediction"]]),
    )
    quality_gate = {
        "passed": bool(
            selected_metrics.get("log_loss", float("inf")) < baseline_metrics.get("log_loss", float("inf"))
            and selected_metrics.get("brier_score", float("inf")) < baseline_metrics.get("brier_score", float("inf"))
        ),
        "reason": (
            "La fuente seleccionada supera al baseline en log_loss y brier_score sobre outer holdout."
            if (
                selected_metrics.get("log_loss", float("inf")) < baseline_metrics.get("log_loss", float("inf"))
                and selected_metrics.get("brier_score", float("inf")) < baseline_metrics.get("brier_score", float("inf"))
            )
            else "La fuente seleccionada no supera al baseline en log_loss y brier_score sobre outer holdout."
        ),
        "selected_source": selected_probability_source,
        "selected_source_strategy": selected_source_strategy,
        "global_best_source": global_best_source,
        "selected_metrics": selected_metrics,
        "baseline_metrics": baseline_metrics,
    }
    execution_assumptions = {
        "decision_minutes_before_kickoff": settings.execution.decision_minutes_before_kickoff,
        "max_quote_age_minutes": settings.execution.max_quote_age_minutes,
        "slippage_rate": settings.execution.slippage_rate,
        "commission_rate": settings.execution.commission_rate,
        "max_stake": settings.execution.max_stake,
        "min_liquidity": settings.execution.min_liquidity,
        "allow_closing_proxy_for_research": settings.execution.allow_closing_proxy_for_research,
        "allow_closing_proxy_for_promotion": settings.execution.allow_closing_proxy_for_promotion,
    }

    signal_summary = SignalSummary(
        selected_probability_source=selected_probability_source,
        selected_source_strategy=selected_source_strategy,
        baseline_metrics=baseline_metrics,
        raw_metrics=_probability_metrics(actual, raw_probs, predicted_outcomes(raw_probs)),
        calibrated_metrics=_probability_metrics(actual, calibrated_probs, predicted_outcomes(calibrated_probs)),
        quality_gate=quality_gate,
        regional_diagnostics=dict(probability_decision_payload.get("regional_diagnostics", {})),
        verdict=None,
        extra={
            "rows_scored": int(len(predictions)),
            "training_rows": int((predictions["fold_segment"] == "train").sum()),
            "holdout_rows": int((predictions["fold_segment"] == "holdout").sum()),
            "probability_decision": probability_decision_payload,
            "selected_metrics": selected_metrics,
        },
    ).to_dict()

    policy_summary = PolicySummary(
        selected_policy=selected_policy_payload,
        training_metrics=training_overall,
        holdout_metrics=holdout_overall,
        objective={
            "name": "roi_first_with_quality_gate",
            "probability_source": selected_probability_source,
        },
        inner_validation={
            "outer_holdout_rows": int(len(outer_holdout)),
            "inner_fold_assignments": [] if inner_fold_assignments is None else inner_fold_assignments.to_dict(orient="records"),
        },
        conservative_score_breakdown={}
        if combination_ranking is None or combination_ranking.empty
        else {
            key: value
            for key, value in combination_ranking.iloc[0].to_dict().items()
            if key
            in {
                "score",
                "raw_roi",
                "shrunken_roi",
                "fold_roi_mean",
                "fold_roi_std",
                "roi_lcb_80",
                "max_drawdown_norm",
                "generalization_gap",
            }
        },
        candidate_rankings={
            "policy_candidates": [] if policy_ranking is None else policy_ranking.to_dict(orient="records"),
            "selection_candidates": [] if combination_ranking is None else combination_ranking.to_dict(orient="records"),
        },
        verdict=None,
        extra={
            "training_niche_metrics": training_niche,
            "holdout_niche_metrics": holdout_niche,
            "top_niche": top_niche_filter,
            "top_niche_row": None if niches.empty else niches.iloc[0].to_dict(),
            "niche_count": int(len(niches)),
        },
    ).to_dict()

    execution_summary = ExecutionSummary(
        coverage={
            "training_proxy_used": training_proxy,
            "holdout_proxy_used": holdout_proxy,
            "proxy_blocked_for_promotion": training_proxy or holdout_proxy,
            "training_executed": int(training_overall.get("executed", 0)),
            "holdout_executed": int(holdout_overall.get("executed", 0)),
            "training_rejected": int(training_overall.get("rejected", 0)),
            "holdout_rejected": int(holdout_overall.get("rejected", 0)),
        },
        blockers={
            "training_proxy_used": training_proxy,
            "holdout_proxy_used": holdout_proxy,
        },
        fills={
            "training_overall": training_overall,
            "holdout_overall": holdout_overall,
            "training_niche": training_niche,
            "holdout_niche": holdout_niche,
        },
        decisions={
            "execution_assumptions": execution_assumptions,
            "selected_probability_source": selected_probability_source,
            "training_source_type_counts": _source_type_counts(training_execution),
            "holdout_source_type_counts": _source_type_counts(holdout_execution),
        },
        clv_summary=dict(execution_quality.get("clv_summary", {})),
        fill_model_summary=dict(execution_quality.get("fill_model_summary", {})),
        fill_adjusted_ev_summary=dict(execution_quality.get("fill_adjusted_ev_summary", {})),
        sizing_summary=dict(execution_quality.get("sizing_summary", {})),
        verdict=None,
    ).to_dict()

    summary = {
        "rows_scored": int(len(predictions)),
        "training_rows": int((predictions["fold_segment"] == "train").sum()),
        "holdout_rows": int((predictions["fold_segment"] == "holdout").sum()),
        "selected_probability_source": selected_probability_source,
        "selected_source_strategy": selected_source_strategy,
        "probability_decision": probability_decision_payload,
        "selected_policy": selected_policy_payload,
        "baseline": signal_summary["baseline_metrics"],
        "model_raw": signal_summary["raw_metrics"],
        "model_calibrated": signal_summary["calibrated_metrics"],
        "model_selected": selected_metrics,
        "training_overall": training_overall,
        "holdout_overall": holdout_overall,
        "training_niche": training_niche,
        "holdout_niche": holdout_niche,
        "top_niche": top_niche_filter,
        "top_niche_row": None if niches.empty else niches.iloc[0].to_dict(),
        "niche_count": int(len(niches)),
        "execution_assumptions": execution_assumptions,
        "inner_fold_assignments": [] if inner_fold_assignments is None else inner_fold_assignments.to_dict(orient="records"),
        "signal_summary": signal_summary,
        "policy_summary": policy_summary,
        "execution_summary": execution_summary,
    }
    summary["promotion_report"] = build_promotion_report(
        training_niche_execution=training_niche_execution,
        holdout_niche_execution=holdout_niche_execution,
        settings=settings,
        signal_summary=signal_summary,
        policy_summary=policy_summary,
        execution_summary=execution_summary,
    )
    summary["experimental_truth"] = build_research_truth_summary(summary)
    return summary


def format_net_summary(summary: dict[str, Any]) -> str:
    signal = SignalSummary.ensure(summary.get("signal_summary") or summary)
    policy = PolicySummary.ensure(summary.get("policy_summary") or summary)
    execution = ExecutionSummary.ensure(summary.get("execution_summary"))
    holdout = policy.holdout_metrics or summary.get("holdout_overall", {})
    niche = policy.extra.get("holdout_niche_metrics", summary.get("holdout_niche", {}))
    promotion = summary.get("promotion_report", {})
    truth = summary.get("experimental_truth") or build_research_truth_summary(summary)
    signal_beats_market = truth.get("signal_quality", {}).get("beats_market", False)
    policy_positive = truth.get("policy_effect", {}).get("positive_holdout_roi", False)
    execution_ready = truth.get("execution_viability", {}).get("forward_ready", False)
    return (
        f"- Veredicto: {truth.get('headline', '')}\n"
        f"- Predice mejor: {signal_beats_market}\n"
        f"- La policy suma: {policy_positive}\n"
        f"- Ejecucion viable: {execution_ready}\n"
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


def load_research_run(run_dir: Path | str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    root = Path(run_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(root / "prediction_rows.csv", parse_dates=["Date", "kickoff_time"])
    candidate_rows = pd.read_csv(root / "candidate_rows.csv", parse_dates=["Date", "kickoff_time", "decision_time", "snapshot_time"])
    return summary, predictions, candidate_rows


def summarize_execution(rows: pd.DataFrame, profit_column: str = "net_profit") -> dict[str, float]:
    if rows.empty:
        return {
            "bets": 0,
            "wins": 0,
            "profit": 0.0,
            "roi": 0.0,
            "yield": 0.0,
            "max_drawdown": 0.0,
            "executed": 0,
            "rejected": 0,
            "stake": 0.0,
        }

    executed = rows[rows["execution_status"] == "executed"].copy()
    if executed.empty:
        return {
            "bets": int(len(rows)),
            "wins": 0,
            "profit": 0.0,
            "roi": 0.0,
            "yield": 0.0,
            "max_drawdown": 0.0,
            "executed": 0,
            "rejected": int(len(rows)),
            "stake": 0.0,
        }

    profits = executed[profit_column].dropna().astype(float)
    cumulative = profits.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    stake = float(executed["accepted_stake"].sum())
    return {
        "bets": int(len(rows)),
        "wins": int(executed.get("won", pd.Series(0, index=executed.index)).fillna(0).eq(1).sum()),
        "profit": float(profits.sum()),
        "roi": float(profits.sum() / stake) if stake > 0 else 0.0,
        "yield": float(profits.mean()) if not profits.empty else 0.0,
        "max_drawdown": float(drawdown.max()) if not drawdown.empty else 0.0,
        "executed": int(len(executed)),
        "rejected": int(len(rows) - len(executed)),
        "stake": stake,
    }
