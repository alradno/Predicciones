from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd


INNER_ROLE_DISCOVERY = "discovery"
INNER_ROLE_POLICY_TUNE = "policy_tune"
INNER_ROLE_SELECTION_VALIDATE = "selection_validate"
INNER_ROLE_OUTER_HOLDOUT = "outer_holdout"
INNER_ROLE_UNUSED = "unused"
DEFAULT_INNER_ROLE_ORDER = (
    INNER_ROLE_DISCOVERY,
    INNER_ROLE_POLICY_TUNE,
    INNER_ROLE_SELECTION_VALIDATE,
)
DEFAULT_REGIONAL_SOURCE_FIELDS = (
    "league_code",
    "odds_band",
    "edge_band_calibrated",
    "time_bucket",
)
LCB_Z_80 = float(NormalDist().inv_cdf(0.80))


@dataclass(frozen=True)
class InnerFoldSplit:
    assignments: pd.DataFrame
    discovery_folds: tuple[int, ...]
    policy_tune_folds: tuple[int, ...]
    selection_validate_folds: tuple[int, ...]


def _sorted_fold_ids(frame: pd.DataFrame, *, outer_segment_column: str, fold_column: str) -> list[int]:
    train = frame[frame[outer_segment_column].astype(str).eq("train")].copy()
    if train.empty:
        return []
    folds = pd.to_numeric(train[fold_column], errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(folds)


def _role_counts(n_folds: int) -> tuple[int, int, int]:
    if n_folds < 3:
        raise ValueError("Se necesitan al menos 3 folds de train para crear discovery/policy_tune/selection_validate.")
    if n_folds == 3:
        return (1, 1, 1)
    if n_folds == 4:
        return (2, 1, 1)
    discovery = int(round(n_folds * 0.60))
    discovery = min(max(discovery, 1), n_folds - 2)
    remaining = n_folds - discovery
    policy = max(1, remaining // 2)
    selection = remaining - policy
    if selection < 1:
        policy = max(1, policy - 1)
        selection = n_folds - discovery - policy
    if selection < 1:
        discovery = max(1, discovery - 1)
        remaining = n_folds - discovery
        policy = max(1, remaining // 2)
        selection = remaining - policy
    if discovery < 1 or policy < 1 or selection < 1:
        raise ValueError("No se pudo construir un split interior valido para ROI-first.")
    return discovery, policy, selection


def build_inner_fold_split(
    frame: pd.DataFrame,
    *,
    outer_segment_column: str = "fold_segment",
    fold_column: str = "fold_id",
) -> InnerFoldSplit:
    fold_ids = _sorted_fold_ids(frame, outer_segment_column=outer_segment_column, fold_column=fold_column)
    discovery_count, policy_count, selection_count = _role_counts(len(fold_ids))
    discovery_folds = tuple(fold_ids[:discovery_count])
    policy_folds = tuple(fold_ids[discovery_count : discovery_count + policy_count])
    selection_folds = tuple(fold_ids[discovery_count + policy_count : discovery_count + policy_count + selection_count])
    assignments = pd.DataFrame(
        [
            {"fold_id": int(fold_id), "inner_role": INNER_ROLE_DISCOVERY}
            for fold_id in discovery_folds
        ]
        + [
            {"fold_id": int(fold_id), "inner_role": INNER_ROLE_POLICY_TUNE}
            for fold_id in policy_folds
        ]
        + [
            {"fold_id": int(fold_id), "inner_role": INNER_ROLE_SELECTION_VALIDATE}
            for fold_id in selection_folds
        ]
    )
    return InnerFoldSplit(
        assignments=assignments,
        discovery_folds=discovery_folds,
        policy_tune_folds=policy_folds,
        selection_validate_folds=selection_folds,
    )


def assign_inner_validation_roles(
    frame: pd.DataFrame,
    *,
    outer_segment_column: str = "fold_segment",
    fold_column: str = "fold_id",
    role_column: str = "inner_role",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        empty = frame.copy()
        empty[role_column] = INNER_ROLE_UNUSED
        return empty, pd.DataFrame(columns=["fold_id", role_column])
    split = build_inner_fold_split(frame, outer_segment_column=outer_segment_column, fold_column=fold_column)
    output = frame.copy()
    output[role_column] = INNER_ROLE_UNUSED
    output.loc[output[outer_segment_column].astype(str).eq("holdout"), role_column] = INNER_ROLE_OUTER_HOLDOUT
    for assignment in split.assignments.itertuples(index=False):
        mask = output[outer_segment_column].astype(str).eq("train") & pd.to_numeric(output[fold_column], errors="coerce").eq(
            int(assignment.fold_id)
        )
        output.loc[mask, role_column] = str(assignment.inner_role)
    return output, split.assignments.copy()


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def shrink_roi(raw_roi: float, bets: float, prior_bets: float) -> float:
    prior = max(float(prior_bets), 0.0)
    total = max(float(bets), 0.0)
    if total <= 0.0:
        return 0.0
    return float(raw_roi * (total / (total + prior))) if (total + prior) > 0 else float(raw_roi)


def roi_lcb_80(fold_roi: pd.Series | Iterable[float] | None) -> float:
    stats = fold_roi_statistics(fold_roi)
    if stats["count"] <= 0:
        return 0.0
    return float(stats["roi_lcb_80"])


def fold_roi_statistics(fold_roi: pd.Series | Iterable[float] | None) -> dict[str, float]:
    series = pd.Series(dtype=float) if fold_roi is None else pd.to_numeric(pd.Series(list(fold_roi) if not isinstance(fold_roi, pd.Series) else fold_roi), errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {
            "count": 0.0,
            "fold_roi_mean": 0.0,
            "fold_roi_std": 0.0,
            "positive_fold_ratio": 0.0,
            "roi_lcb_80": 0.0,
        }
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    lcb = mean - (LCB_Z_80 * std / np.sqrt(len(series))) if len(series) > 1 else mean
    positive_ratio = float((series >= 0.0).mean())
    return {
        "count": float(len(series)),
        "fold_roi_mean": mean,
        "fold_roi_std": std,
        "positive_fold_ratio": positive_ratio,
        "roi_lcb_80": float(lcb),
    }


def normalized_drawdown(metrics: dict[str, Any]) -> float:
    stake = max(float(metrics.get("stake", 0.0)), 1.0)
    return float(metrics.get("max_drawdown", 0.0)) / stake


def conservative_score_breakdown(
    metrics: dict[str, Any],
    *,
    fold_roi: pd.Series | Iterable[float] | None = None,
    reference_roi: float | None = None,
    positive_fold_target: float = 0.0,
    prior_bets: float = 0.0,
    multiple_testing_penalty: float = 0.0,
    drawdown_weight: float = 0.20,
    generalization_gap_weight: float = 0.10,
    positive_penalty_weight: float = 0.10,
) -> dict[str, float]:
    executed = float(metrics.get("executed", metrics.get("bets", 0.0)) or 0.0)
    raw_roi = float(metrics.get("roi", 0.0))
    roi_stats = fold_roi_statistics(fold_roi)
    shrunken = shrink_roi(raw_roi, executed, prior_bets)
    drawdown_norm = normalized_drawdown(metrics)
    positive_gap = max(float(positive_fold_target) - float(roi_stats["positive_fold_ratio"]), 0.0)
    if reference_roi is None:
        generalization_gap = 0.0
    else:
        generalization_gap = abs(float(roi_stats["fold_roi_mean"]) - float(reference_roi))
    score = (
        float(roi_stats["roi_lcb_80"])
        - (drawdown_weight * drawdown_norm)
        - (generalization_gap_weight * generalization_gap)
        - (positive_penalty_weight * positive_gap)
        - float(multiple_testing_penalty)
    )
    return {
        "raw_roi": raw_roi,
        "shrunken_roi": shrunken,
        "fold_roi_mean": float(roi_stats["fold_roi_mean"]),
        "fold_roi_std": float(roi_stats["fold_roi_std"]),
        "roi_lcb_80": float(roi_stats["roi_lcb_80"]),
        "max_drawdown_norm": drawdown_norm,
        "generalization_gap": float(generalization_gap),
        "multiple_testing_penalty": float(multiple_testing_penalty),
        "positive_fold_ratio": float(roi_stats["positive_fold_ratio"]),
        "positive_fold_gap": float(positive_gap),
        "score": float(score),
    }


def multiple_testing_penalty(*, dimension_count: int, tested_segments: int) -> float:
    dimension_penalty = max(int(dimension_count) - 1, 0) * 0.01
    search_penalty = np.log1p(max(int(tested_segments), 0)) * 0.0025
    return float(dimension_penalty + search_penalty)
