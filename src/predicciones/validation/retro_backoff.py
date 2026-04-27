"""Confidence and backoff helpers for the Polymarket retro pipeline.

This module is intentionally self-contained so the main retro pipeline can
import it cleanly later without pulling additional project state.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

CONFIDENCE_WEIGHT_SHRINKAGE = 0.30
CONFIDENCE_WEIGHT_ADEQUACY = 0.30
CONFIDENCE_WEIGHT_STABILITY = 0.25
CONFIDENCE_WEIGHT_SEASON = 0.15
CONFIDENCE_BACKOFF_WEIGHT_CONFIDENCE = 0.45
CONFIDENCE_BACKOFF_WEIGHT_DIVERGENCE = 0.25
CONFIDENCE_VOLATILITY_SCALE = 1.5
CONFIDENCE_SHARE_THRESHOLDS = (0.9, 0.75, 0.5)
CONFIDENCE_BACKOFF_BUCKETS = 10

__all__ = [
    "CONFIDENCE_BACKOFF_BUCKETS",
    "CONFIDENCE_SHARE_THRESHOLDS",
    "CONFIDENCE_VOLATILITY_SCALE",
    "CONFIDENCE_WEIGHT_ADEQUACY",
    "CONFIDENCE_WEIGHT_SEASON",
    "CONFIDENCE_WEIGHT_SHRINKAGE",
    "CONFIDENCE_WEIGHT_STABILITY",
    "backoff_inactive_reason",
    "blend_probabilities",
    "build_confidence_backoff_report",
    "confidence_backoff_deciles",
    "confidence_score_summary",
    "confidence_scores",
]


def _float_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _resolve_signal_series(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    resolved = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        resolved = resolved.where(resolved.notna(), values)
    return resolved


def _coerce_probability_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with one row per observation.")
    return array


def confidence_scores(
    rows: pd.DataFrame,
    variant_raw: np.ndarray | None = None,
    v1_raw: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a confidence score in [0, 1] for each row.

    The score combines shrinkage, long-horizon adequacy, stability and season
    context. Divergence to V1 is handled separately in the backoff weight.
    """

    shrinkage_frame = pd.DataFrame(
        {
            "home_shrinkage_ratio_overall": _float_series(rows, "home_shrinkage_ratio_overall"),
            "away_shrinkage_ratio_overall": _float_series(rows, "away_shrinkage_ratio_overall"),
            "home_shrinkage_ratio_side": _float_series(rows, "home_shrinkage_ratio_side"),
            "away_shrinkage_ratio_side": _float_series(rows, "away_shrinkage_ratio_side"),
        },
        index=rows.index,
    )
    adequacy_frame = pd.DataFrame(
        {
            "home_long_sample_ratio_overall": _resolve_signal_series(rows, ("home_long_sample_ratio_overall", "home_sample_coverage_ratio_overall")),
            "away_long_sample_ratio_overall": _resolve_signal_series(rows, ("away_long_sample_ratio_overall", "away_sample_coverage_ratio_overall")),
            "home_long_sample_ratio_side": _resolve_signal_series(rows, ("home_long_sample_ratio_side", "home_sample_coverage_ratio_side")),
            "away_long_sample_ratio_side": _resolve_signal_series(rows, ("away_long_sample_ratio_side", "away_sample_coverage_ratio_side")),
        },
        index=rows.index,
    )
    volatility_frame = pd.DataFrame(
        {
            "home_opponent_strength_volatility_10": _resolve_signal_series(
                rows,
                ("home_opponent_strength_volatility_20", "home_opponent_strength_volatility_10"),
            ),
            "away_opponent_strength_volatility_10": _resolve_signal_series(
                rows,
                ("away_opponent_strength_volatility_20", "away_opponent_strength_volatility_10"),
            ),
            "home_goals_for_volatility_10": _resolve_signal_series(rows, ("home_goal_volatility_20", "home_goals_for_volatility_20", "home_goals_for_volatility_10")),
            "away_goals_for_volatility_10": _resolve_signal_series(rows, ("away_goal_volatility_20", "away_goals_for_volatility_20", "away_goals_for_volatility_10")),
            "home_goals_against_volatility_10": _resolve_signal_series(
                rows,
                ("home_goals_against_volatility_20", "home_goals_against_volatility_10"),
            ),
            "away_goals_against_volatility_10": _resolve_signal_series(
                rows,
                ("away_goals_against_volatility_20", "away_goals_against_volatility_10"),
            ),
        },
        index=rows.index,
    )

    shrinkage_conf = shrinkage_frame.mean(axis=1, skipna=True)
    adequacy_conf = adequacy_frame.mean(axis=1, skipna=True)
    stability_penalty = (volatility_frame.mean(axis=1, skipna=True) / CONFIDENCE_VOLATILITY_SCALE).clip(lower=0.0, upper=1.0)
    stability_conf = 1.0 - stability_penalty.fillna(1.0)
    season_conf = _resolve_signal_series(rows, ("season_progress_pct",)).fillna(0.25).clip(lower=0.25, upper=1.0)

    confidence = (
        (CONFIDENCE_WEIGHT_SHRINKAGE * shrinkage_conf.fillna(0.0))
        + (CONFIDENCE_WEIGHT_ADEQUACY * adequacy_conf.fillna(0.0))
        + (CONFIDENCE_WEIGHT_STABILITY * stability_conf.fillna(0.0))
        + (CONFIDENCE_WEIGHT_SEASON * season_conf.fillna(0.25))
    )
    return np.clip(confidence.to_numpy(dtype=float), 0.0, 1.0)


def blend_probabilities(
    primary: np.ndarray,
    fallback: np.ndarray,
    confidence_scores: np.ndarray,
    divergence_scores: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Blend primary and fallback probabilities using confidence weights."""

    primary_array = _coerce_probability_array(primary, name="primary")
    fallback_array = _coerce_probability_array(fallback, name="fallback")
    confidence_array = np.asarray(confidence_scores, dtype=float)
    if confidence_array.ndim != 1:
        raise ValueError("confidence_scores must be a 1D array.")
    if primary_array.shape != fallback_array.shape:
        raise ValueError("primary and fallback probabilities must have the same shape.")
    if primary_array.shape[0] != confidence_array.shape[0]:
        raise ValueError("confidence_scores must contain one value per observation.")
    if divergence_scores is None:
        divergence_array = np.zeros_like(confidence_array)
    else:
        divergence_array = np.asarray(divergence_scores, dtype=float)
        if divergence_array.ndim != 1 or divergence_array.shape[0] != confidence_array.shape[0]:
            raise ValueError("divergence_scores must contain one value per observation.")
    backoff_weight = np.clip(
        (CONFIDENCE_BACKOFF_WEIGHT_CONFIDENCE * confidence_array)
        - (CONFIDENCE_BACKOFF_WEIGHT_DIVERGENCE * divergence_array),
        0.0,
        1.0,
    )
    return (backoff_weight[:, None] * primary_array) + ((1.0 - backoff_weight[:, None]) * fallback_array)


def backoff_inactive_reason(
    backoff_variant: bool,
    confidence_scores: np.ndarray | Sequence[float],
    divergence_scores: np.ndarray | Sequence[float],
) -> str | None:
    """Return the reason why backoff is inactive, or None if it is active."""

    if not backoff_variant:
        return "not_backoff_variant"

    confidence = np.asarray(confidence_scores, dtype=float)
    divergence = np.asarray(divergence_scores, dtype=float)

    if confidence.size == 0 or not np.isfinite(confidence).any():
        return "missing_confidence_scores"
    if divergence.size == 0 or not np.isfinite(divergence).any() or np.allclose(np.nan_to_num(divergence, nan=0.0), 0.0, atol=1e-12):
        return "probabilities_identical_to_v1"
    if np.all(np.isclose(np.nan_to_num(confidence, nan=1.0), 1.0, atol=1e-9)):
        return "backoff_not_engaged"
    return None


def confidence_score_summary(
    scores: np.ndarray | Sequence[float],
    *,
    applied: bool,
    variant_supports_backoff: bool | None = None,
    mean_divergence_to_v1: float | None = None,
    backoff_inactive_reason: str | None = None,
    backoff_not_engaged: bool | None = None,
    thresholds: Sequence[float] = CONFIDENCE_SHARE_THRESHOLDS,
) -> dict[str, Any]:
    """Summarize confidence score distribution and key backoff flags."""

    array = np.asarray(scores, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        summary = {
            "applied": bool(applied),
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "spread": 0.0,
        }
    else:
        summary = {
            "applied": bool(applied),
            "count": int(finite.size),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=0)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "p10": float(np.quantile(finite, 0.10)),
            "p25": float(np.quantile(finite, 0.25)),
            "p75": float(np.quantile(finite, 0.75)),
            "p90": float(np.quantile(finite, 0.90)),
            "max": float(np.max(finite)),
            "spread": float(np.max(finite) - np.min(finite)),
        }
    for threshold in thresholds:
        key = f"share_below_{str(threshold).replace('.', '_')}"
        if finite.size == 0:
            summary[key] = 0.0
        else:
            summary[key] = float(np.mean(finite < float(threshold)))
    summary["variant_supports_backoff"] = bool(variant_supports_backoff) if variant_supports_backoff is not None else bool(applied)
    summary["mean_divergence_to_v1"] = float(mean_divergence_to_v1) if mean_divergence_to_v1 is not None else 0.0
    summary["backoff_inactive_reason"] = backoff_inactive_reason
    summary["backoff_not_engaged"] = bool(backoff_not_engaged) if backoff_not_engaged is not None else False
    summary["degenerate_constant_score"] = bool(finite.size > 0 and np.isclose(summary["spread"], 0.0, atol=1e-12))
    return summary


def _build_trade_frame(
    selected: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    score_column: str,
    divergence_column: str,
    win_columns: Sequence[str] = ("won", "win", "is_win"),
    pnl_columns: Sequence[str] = ("net_profit", "pnl", "net_pnl", "profit"),
    cost_columns: Sequence[str] = ("cost_basis", "total_cost", "notional", "spend"),
) -> pd.DataFrame:
    selected_frame = selected.reset_index(drop=True).copy()
    fill_frame = fills.reset_index(drop=True).copy()
    if len(selected_frame) != len(fill_frame):
        if "decision_id" in selected_frame.columns and "decision_id" in fill_frame.columns:
            selected_keep = selected_frame[[column for column in selected_frame.columns if column not in fill_frame.columns or column == "decision_id"]].copy()
            merged = fill_frame.merge(selected_keep, on="decision_id", how="left", suffixes=("", "_selected"))
            selected_frame = merged
            fill_frame = merged
        else:
            aligned_length = min(len(selected_frame), len(fill_frame))
            selected_frame = selected_frame.iloc[:aligned_length].copy()
            fill_frame = fill_frame.iloc[:aligned_length].copy()
    frame = pd.DataFrame(index=range(len(selected_frame)))
    frame["confidence_score"] = _float_series(selected_frame, score_column).to_numpy(dtype=float)
    frame["confidence_divergence_to_v1"] = _float_series(selected_frame, divergence_column).to_numpy(dtype=float)

    win_series = None
    for candidate in win_columns:
        if candidate in fill_frame.columns:
            win_series = pd.to_numeric(fill_frame[candidate], errors="coerce")
            break
    if win_series is None:
        pnl_series = None
        for candidate in pnl_columns:
            if candidate in fill_frame.columns:
                pnl_series = pd.to_numeric(fill_frame[candidate], errors="coerce")
                break
        if pnl_series is not None:
            win_series = (pnl_series.fillna(0.0) > 0.0).astype(float)
        else:
            win_series = pd.Series(0.0, index=fill_frame.index, dtype=float)
    frame["won"] = pd.to_numeric(win_series, errors="coerce").fillna(0.0).astype(float).to_numpy(dtype=float)

    pnl_series = None
    for candidate in pnl_columns:
        if candidate in fill_frame.columns:
            pnl_series = pd.to_numeric(fill_frame[candidate], errors="coerce")
            break
    if pnl_series is None:
        pnl_series = pd.Series(0.0, index=fill_frame.index, dtype=float)
    frame["net_pnl"] = pnl_series.fillna(0.0).astype(float).to_numpy(dtype=float)

    cost_series = None
    for candidate in cost_columns:
        if candidate in fill_frame.columns:
            cost_series = pd.to_numeric(fill_frame[candidate], errors="coerce")
            break
    if cost_series is None:
        cost_series = pd.Series(0.0, index=fill_frame.index, dtype=float)
    frame["total_cost"] = cost_series.fillna(0.0).astype(float).to_numpy(dtype=float)
    return frame


def _summary_scores_from_frame(
    frame: pd.DataFrame,
    *,
    score_column: str,
    divergence_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    score_series = _resolve_signal_series(frame, (score_column, "selection_confidence_score", "policy_confidence_score"))
    divergence_series = _resolve_signal_series(
        frame,
        (divergence_column, "selection_confidence_divergence_to_v1", "policy_confidence_divergence_to_v1"),
    )
    scores = score_series.dropna().to_numpy(dtype=float)
    divergences = divergence_series.dropna().to_numpy(dtype=float)
    return scores, divergences


def confidence_backoff_deciles(
    selected: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    score_column: str = "confidence_score",
    divergence_column: str = "confidence_divergence_to_v1",
    bucket_count: int = CONFIDENCE_BACKOFF_BUCKETS,
) -> list[dict[str, Any]]:
    """Return a confidence-decile report for selected trades."""

    trade_frame = _build_trade_frame(
        selected,
        fills,
        score_column=score_column,
        divergence_column=divergence_column,
    )
    if trade_frame.empty:
        return []

    ordered = trade_frame.sort_values("confidence_score", ascending=True, kind="mergesort").reset_index(drop=True)
    splits = np.array_split(np.arange(len(ordered)), min(bucket_count, len(ordered)))
    deciles: list[dict[str, Any]] = []
    for idx, split in enumerate(splits, start=1):
        if split.size == 0:
            continue
        bucket = ordered.iloc[split]
        scores = pd.to_numeric(bucket["confidence_score"], errors="coerce").dropna()
        divergences = pd.to_numeric(bucket["confidence_divergence_to_v1"], errors="coerce").dropna()
        total_cost = float(pd.to_numeric(bucket["total_cost"], errors="coerce").fillna(0.0).sum())
        net_pnl = float(pd.to_numeric(bucket["net_pnl"], errors="coerce").fillna(0.0).sum())
        wins = float(pd.to_numeric(bucket["won"], errors="coerce").fillna(0.0).sum())
        bets = int(len(bucket))
        deciles.append(
            {
                "bucket": f"d{idx}",
                "bets": bets,
                "wins": int(round(wins)),
                "net_pnl": net_pnl,
                "total_cost": total_cost,
                "mean_confidence_score": float(scores.mean()) if not scores.empty else 0.0,
                "mean_divergence_to_v1": float(divergences.mean()) if not divergences.empty else 0.0,
                "hit_rate": float(wins / bets) if bets > 0 else 0.0,
                "net_roi": float(net_pnl / total_cost) if total_cost > 0 else 0.0,
            }
        )
    return deciles


def build_confidence_backoff_report(
    combined_predictions: pd.DataFrame,
    selected: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    backoff_variant: bool,
    applied: bool | None = None,
    score_column: str = "confidence_score",
    divergence_column: str = "confidence_divergence_to_v1",
    bucket_count: int = CONFIDENCE_BACKOFF_BUCKETS,
) -> dict[str, Any]:
    """Build a confidence/backoff report with a summary and decile breakdown."""

    if combined_predictions.empty:
        scores = np.array([], dtype=float)
        divergences = np.array([], dtype=float)
    else:
        scores, divergences = _summary_scores_from_frame(
            combined_predictions,
            score_column=score_column,
            divergence_column=divergence_column,
        )
        if (
            scores.size == 0
            or not np.isfinite(scores).any()
            or np.isclose(float(np.nanmax(scores) - np.nanmin(scores)), 0.0, atol=1e-12)
            or np.all(np.isclose(np.nan_to_num(scores, nan=1.0), 1.0, atol=1e-9))
        ) and not selected.empty:
            fallback_scores, fallback_divergences = _summary_scores_from_frame(
                selected,
                score_column=score_column,
                divergence_column=divergence_column,
            )
            if fallback_scores.size > 0 and np.isfinite(fallback_scores).any():
                scores = fallback_scores
                divergences = fallback_divergences

    if applied is None:
        applied = bool(backoff_variant and scores.size > 0 and not np.all(np.isclose(scores, 1.0, atol=1e-9)))

    inactive_reason = backoff_inactive_reason(backoff_variant, scores, divergences)
    summary = confidence_score_summary(
        scores,
        applied=applied,
        variant_supports_backoff=backoff_variant,
        mean_divergence_to_v1=float(divergences.mean()) if divergences.size > 0 else 0.0,
        backoff_inactive_reason=inactive_reason,
        backoff_not_engaged=bool(backoff_variant and not applied),
    )
    deciles = confidence_backoff_deciles(
        selected,
        fills,
        score_column=score_column,
        divergence_column=divergence_column,
        bucket_count=bucket_count,
    )
    return {
        "summary": summary,
        "deciles": deciles,
    }
