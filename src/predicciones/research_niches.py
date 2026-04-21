from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import pandas as pd

from .config import Settings
from .evaluation import ExecutionSummary, PolicySummary, PromotionGate, SignalSummary
from .research_candidates import _roi_by_fold, summarize_execution
from .selection_scoring import conservative_score_breakdown, multiple_testing_penalty


def discover_niches(execution_rows: pd.DataFrame, settings: Settings, top_n: int | None = None) -> pd.DataFrame:
    executed = execution_rows[execution_rows["execution_status"] == "executed"].copy()
    if executed.empty:
        return pd.DataFrame()
    if top_n is None:
        top_n = int(settings.research.candidate_top_niches)

    segment_columns = [
        "league_code",
        "selection",
        "odds_band",
        "time_bucket",
        "season",
        "season_phase",
        "movement_regime",
        "edge_band_calibrated",
    ]
    rows: list[dict[str, Any]] = []
    tested_group_counts: dict[tuple[str, ...], int] = {}
    for size in range(1, settings.research.max_segment_dimensions + 1):
        for combo in combinations(segment_columns, size):
            tested_group_counts[tuple(combo)] = 0
            for values, group in executed.groupby(list(combo), dropna=False, observed=True):
                tested_group_counts[tuple(combo)] += 1
                if not isinstance(values, tuple):
                    values = (values,)
                if len(group) < settings.research.min_segment_bets:
                    continue
                fold_count = int(group["fold_id"].nunique()) if "fold_id" in group.columns else 1
                if fold_count < settings.research.min_segment_folds:
                    continue

                metrics = summarize_execution(group)
                fold_roi = _roi_by_fold(group)
                positive_ratio = float((fold_roi.fillna(-1.0) >= 0.0).mean()) if not fold_roi.empty else 0.0
                if positive_ratio < settings.research.positive_fold_ratio:
                    continue

                monthly_profit = group.groupby(group["Date"].dt.to_period("M"), observed=True)["net_profit"].sum()
                total_profit = float(group["net_profit"].sum())
                max_fold_profit_share = 1.0
                if total_profit > 0 and not fold_roi.empty:
                    fold_profit = group.groupby("fold_id", observed=True)["net_profit"].sum()
                    max_fold_profit_share = float(fold_profit.max() / total_profit)
                max_month_profit_share = 1.0
                if total_profit > 0 and not monthly_profit.empty:
                    max_month_profit_share = float(monthly_profit.max() / total_profit)
                if max_fold_profit_share > 0.7 or max_month_profit_share > 0.7:
                    continue

                ordered = group.sort_values(["Date", "match_id"]).reset_index(drop=True)
                tail_size = max(1, len(ordered) // 3)
                recent = ordered.tail(tail_size)
                recent_roi = _safe_ratio(float(recent["net_profit"].sum()), float(recent["accepted_stake"].sum()))
                testing_penalty = multiple_testing_penalty(
                    dimension_count=len(combo),
                    tested_segments=tested_group_counts[tuple(combo)],
                )
                breakdown = conservative_score_breakdown(
                    metrics,
                    fold_roi=fold_roi,
                    reference_roi=recent_roi,
                    positive_fold_target=settings.research.positive_fold_ratio,
                    prior_bets=settings.research.min_segment_bets,
                    multiple_testing_penalty=testing_penalty,
                    drawdown_weight=settings.research.drawdown_weight,
                    generalization_gap_weight=settings.research.generalization_gap_weight,
                    positive_penalty_weight=settings.research.positive_penalty_weight,
                )

                filters = {column: value for column, value in zip(combo, values)}
                rows.append(
                    {
                        "dimensions": "|".join(combo),
                        "filters_json": json.dumps(filters, ensure_ascii=True, default=str),
                        "bets": metrics["executed"],
                        "profit": metrics["profit"],
                        "roi": metrics["roi"],
                        "raw_roi": breakdown["raw_roi"],
                        "shrunken_roi": breakdown["shrunken_roi"],
                        "yield": metrics["yield"],
                        "max_drawdown": metrics["max_drawdown"],
                        "max_drawdown_norm": breakdown["max_drawdown_norm"],
                        "folds": fold_count,
                        "positive_fold_ratio": positive_ratio,
                        "recent_roi": recent_roi,
                        "fold_roi_mean": breakdown["fold_roi_mean"],
                        "fold_roi_std": breakdown["fold_roi_std"],
                        "roi_lcb_80": breakdown["roi_lcb_80"],
                        "generalization_gap": breakdown["generalization_gap"],
                        "multiple_testing_penalty": breakdown["multiple_testing_penalty"],
                        "max_fold_profit_share": max_fold_profit_share,
                        "max_month_profit_share": max_month_profit_share,
                        "score": breakdown["score"],
                    }
                )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["score", "generalization_gap", "bets"], ascending=[False, True, False])
        .head(top_n)
        .reset_index(drop=True)
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _segment_filter_from_row(row: pd.Series | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return json.loads(str(row["filters_json"]))


def _coerce_summary_payload(value: Any) -> dict[str, Any]:
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


def _gate_status_from_summary(value: Any, default_reason: str) -> tuple[bool, str, dict[str, Any]]:
    payload = _coerce_summary_payload(value)
    if not payload:
        return False, default_reason, {}

    passed = payload.get("passed")
    if passed is None:
        status = str(payload.get("status", "")).strip().lower()
        readiness = str(payload.get("bundle_readiness", "")).strip().lower()
        stage = str(payload.get("validation_stage", "")).strip().lower()
        if status in {"passed", "validated", "ready", "approved"}:
            passed = True
        elif readiness in {"validated", "promotable", "ready"}:
            passed = True
        elif stage == "live" and int(payload.get("settled_bets", 0) or 0) > 0:
            passed = True
        else:
            passed = False

    reason = str(payload.get("reason") or payload.get("headline") or payload.get("message") or default_reason)
    return bool(passed), reason, payload


def _signal_gate_from_summary(value: Any) -> tuple[bool, str, dict[str, Any]]:
    if value is None:
        return True, "No se proporciono signal summary; se asume gate superado por compatibilidad.", {}
    signal = SignalSummary.ensure(value)
    payload = signal.to_dict()
    quality_gate = dict(payload.get("quality_gate", {}))
    if quality_gate:
        passed = bool(quality_gate.get("passed"))
        reason = str(quality_gate.get("reason") or quality_gate.get("headline") or "Signal gate evaluado.")
        return passed, reason, payload
    baseline = signal.baseline_metrics
    selected_source = str(signal.selected_probability_source or "raw")
    selected_metrics = signal.calibrated_metrics if selected_source == "calibrated" else signal.raw_metrics
    baseline_log_loss = float(baseline.get("log_loss", float("inf")))
    baseline_brier = float(baseline.get("brier_score", float("inf")))
    selected_log_loss = float(selected_metrics.get("log_loss", float("inf")))
    selected_brier = float(selected_metrics.get("brier_score", float("inf")))
    passed = selected_log_loss < baseline_log_loss and selected_brier < baseline_brier
    if passed:
        reason = "La fuente seleccionada mejora log_loss y brier_score frente al baseline."
    else:
        reason = "La fuente seleccionada no supera al baseline en log_loss y brier_score."
    return passed, reason, payload


def _execution_gate_from_summary(payload: dict[str, Any], settings: Settings) -> tuple[bool, str]:
    fill_adjusted = dict(payload.get("fill_adjusted_ev_summary", payload.get("fill_adjusted_ev", {})))
    clv_summary = dict(payload.get("clv_summary", payload.get("clv", {})))
    fill_rate = float(payload.get("fill_rate", payload.get("observed_fill_rate", 0.0)) or 0.0)
    if "observed_fill_rate" in payload:
        fill_rate = float(payload.get("observed_fill_rate", 0.0) or 0.0)
    elif "fill_model_summary" in payload:
        fill_rate = float(dict(payload.get("fill_model_summary", {})).get("overall_fill_rate", fill_rate) or fill_rate)
    fill_adjusted_mean = float(fill_adjusted.get("mean", 0.0) or 0.0)
    mean_clv = float(clv_summary.get("mean_clv", 0.0) or 0.0)
    clv_coverage = float(clv_summary.get("coverage", 0.0) or 0.0)
    if fill_adjusted_mean < settings.research.fill_adjusted_ev_min:
        return False, "La ejecucion no supera el gate de fill-adjusted EV."
    if mean_clv < settings.research.mean_clv_min:
        return False, "La ejecucion no supera el gate de CLV medio."
    if clv_coverage < settings.research.clv_coverage_min:
        return False, "La ejecucion no tiene suficiente cobertura CLV."
    if fill_rate < settings.research.observed_fill_rate_min:
        return False, "La ejecucion no alcanza fill-rate suficiente."
    return True, "La ejecucion supera fill-adjusted EV, CLV y fill-rate."


def build_promotion_report(
    training_niche_execution: pd.DataFrame,
    holdout_niche_execution: pd.DataFrame,
    settings: Settings,
    signal_summary: SignalSummary | dict[str, Any] | None = None,
    policy_summary: PolicySummary | dict[str, Any] | None = None,
    execution_summary: ExecutionSummary | dict[str, Any] | None = None,
    shadow_summary: dict[str, Any] | None = None,
    live_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    training_metrics = summarize_execution(training_niche_execution)
    holdout_metrics = summarize_execution(holdout_niche_execution)
    training_fold_roi = _roi_by_fold(training_niche_execution)
    positive_fold_ratio = float((training_fold_roi.fillna(-1.0) >= 0.0).mean()) if not training_fold_roi.empty else 0.0
    training_proxy = bool(
        training_niche_execution["source_type"].eq("closing_proxy").any()
    ) if not training_niche_execution.empty else False
    holdout_proxy = bool(
        holdout_niche_execution["source_type"].eq("closing_proxy").any()
    ) if not holdout_niche_execution.empty else False

    signal_pass, signal_reason, signal_payload = _signal_gate_from_summary(signal_summary)
    stage1_pass = (
        training_metrics["roi"] >= settings.research.stage1_min_roi
        and training_metrics["executed"] >= settings.research.stage1_min_bets
        and positive_fold_ratio >= settings.research.positive_fold_ratio
    )
    stage2_pass = (
        stage1_pass
        and holdout_metrics["roi"] >= settings.research.stage2_min_roi
        and holdout_metrics["executed"] >= settings.research.stage2_min_bets
        and holdout_metrics["max_drawdown"] <= settings.research.max_drawdown_units
    )
    stage2_promotable = stage2_pass and not (training_proxy or holdout_proxy)

    shadow_pass, shadow_reason, shadow_payload = _gate_status_from_summary(shadow_summary, "No hay shadow run validado todavia.")
    live_pass, live_reason, live_payload = _gate_status_from_summary(live_summary, "No hay historial live con ejecucion real.")
    if shadow_payload and any(
        key in shadow_payload for key in ("fill_adjusted_ev_summary", "clv_summary", "fill_model_summary", "observed_fill_rate", "fill_rate")
    ):
        shadow_pass, shadow_reason = _execution_gate_from_summary(shadow_payload, settings)
    if live_payload and any(
        key in live_payload for key in ("fill_adjusted_ev_summary", "clv_summary", "fill_model_summary", "observed_fill_rate", "fill_rate")
    ):
        live_pass, live_reason = _execution_gate_from_summary(live_payload, settings)

    blockers: list[str] = []
    if not signal_pass:
        blockers.append(signal_reason)
    if training_proxy or holdout_proxy:
        blockers.append("Se estan usando closing proxies; no cuentan como ROI real para promocion.")
    if not stage1_pass:
        blockers.append("El nicho no supera los criterios minimos de investigacion Stage 1.")
    if stage1_pass and not stage2_pass:
        blockers.append("El holdout forward aun no demuestra el 20% neto exigido para Stage 2.")
    if not shadow_pass:
        blockers.append(shadow_reason)
    if not live_pass:
        blockers.append(live_reason)

    required_inputs: list[str] = []
    if not signal_pass:
        required_inputs.append("outer_holdout_signal_quality")
    if training_proxy or holdout_proxy:
        required_inputs.append("forward_quotes_without_proxy")
    if not shadow_pass:
        required_inputs.append("shadow_validation")
    if not live_pass:
        required_inputs.append("limited_live_validation")

    if not signal_pass:
        gate_stage = "stage_0_signal_gate"
    elif not stage1_pass:
        gate_stage = "stage_1_discovery_complete"
    elif not stage2_promotable:
        gate_stage = "stage_2_outer_holdout"
    elif not shadow_pass:
        gate_stage = "stage_3_shadow"
    elif not live_pass:
        gate_stage = "stage_4_limited_live"
    else:
        gate_stage = "automation_ready"

    gate = PromotionGate(
        passed=signal_pass and stage2_promotable and shadow_pass and live_pass,
        stage=gate_stage,
        blockers=blockers,
        required_inputs=required_inputs,
        reason="Promotion gate computed from separated signal/policy/execution evidence.",
        evidence={
            "signal_summary": signal_payload,
            "policy_summary": PolicySummary.ensure(policy_summary).to_dict() if policy_summary is not None else {},
            "execution_summary": ExecutionSummary.ensure(execution_summary).to_dict() if execution_summary is not None else {},
            "shadow_summary": shadow_payload,
            "live_summary": live_payload,
        },
    )

    return {
        "stage_0_signal_gate": {
            "passed": signal_pass,
            "reason": signal_reason,
            "quality_gate": signal_payload.get("quality_gate", {}),
        },
        "stage_1_research": {
            "passed": stage1_pass,
            "metrics": training_metrics,
            "positive_fold_ratio": positive_fold_ratio,
        },
        "stage_1_discovery_complete": {
            "passed": stage1_pass,
            "metrics": training_metrics,
            "positive_fold_ratio": positive_fold_ratio,
        },
        "stage_2_holdout": {
            "passed": stage2_promotable,
            "metrics": holdout_metrics,
            "proxy_blocked": training_proxy or holdout_proxy,
        },
        "stage_2_outer_holdout": {
            "passed": stage2_promotable,
            "metrics": holdout_metrics,
            "proxy_blocked": training_proxy or holdout_proxy,
        },
        "stage_3_shadow": {"passed": shadow_pass, "reason": shadow_reason},
        "stage_4_limited_live": {"passed": live_pass, "reason": live_reason},
        "automation_ready": bool(gate.passed),
        "blockers": blockers,
        "required_inputs": required_inputs,
        "gate": gate.to_dict(),
    }
