from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REGIONAL_ADJUSTMENT_NONE = "none"
REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK = "oof_region_shrink"
REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK = "oof_region_soft_shrink"

_ODDS_BANDS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 1.5, "<1.5"),
    (1.5, 2.0, "1.5-2"),
    (2.0, 3.0, "2-3"),
    (3.0, 6.0, "3-6"),
    (6.0, None, "6+"),
)


@dataclass(frozen=True)
class RegionalAdjustmentModel:
    adjustment_mode: str
    probability_source: str
    global_factor: float
    prior_strength: float
    min_rows: int
    training_scope: str
    training_rows: int
    source_rows: int
    factors: dict[str, dict[str, float]]


def _resolve_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _odds_band(odds: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(odds, errors="coerce")
    output = pd.Series("unknown", index=numeric.index, dtype="object")
    valid = numeric.notna() & numeric.gt(0.0)
    for lower, upper, label in _ODDS_BANDS:
        if upper is None:
            mask = valid & numeric.ge(lower)
        else:
            mask = valid & numeric.ge(lower) & numeric.lt(upper)
        output.loc[mask] = label
    return output


def _region_key(frame: pd.DataFrame) -> pd.Series:
    outcome = frame.get("selection", pd.Series("", index=frame.index, dtype="object")).astype(str)
    odds_band = _odds_band(frame.get("quoted_odds", pd.Series(np.nan, index=frame.index, dtype=float)))
    return outcome + "|" + odds_band.astype(str)


def _policy_region_mask(frame: pd.DataFrame, policy: Any | None) -> pd.Series | None:
    if policy is None:
        return None
    required_columns = {"quote_status", "quoted_odds", "policy_edge", "policy_ev"}
    if not required_columns.issubset(frame.columns):
        return None

    odds = _resolve_series(frame, "quoted_odds")
    edge = _resolve_series(frame, "policy_edge")
    ev = _resolve_series(frame, "policy_ev")
    mask = frame["quote_status"].fillna("").astype(str).eq("eligible")
    mask &= odds.ge(float(getattr(policy, "min_odds", 0.0)))
    if str(getattr(policy, "scope_name", "")).startswith("mid_odds"):
        mask &= odds.lt(float(getattr(policy, "max_odds", np.inf)))
    else:
        mask &= odds.le(float(getattr(policy, "max_odds", np.inf)))
    mask &= edge.ge(float(getattr(policy, "edge_threshold", np.inf)))
    mask &= ev.ge(float(getattr(policy, "ev_threshold", np.inf)))

    allowed_leagues = tuple(getattr(policy, "allowed_leagues", ()) or ())
    if allowed_leagues:
        if "league_code" not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        mask &= frame["league_code"].fillna("").astype(str).isin(allowed_leagues)

    allowed_outcomes = tuple(getattr(policy, "allowed_outcomes", ()) or ())
    if allowed_outcomes:
        if "selection" not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        mask &= frame["selection"].fillna("").astype(str).isin(allowed_outcomes)
    return mask.fillna(False).astype(bool)


def fit_regional_adjustment_model(
    candidate_rows: pd.DataFrame,
    *,
    probability_source: str,
    adjustment_mode: str = REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK,
    policy: Any | None = None,
    prior_strength: float = 12.0,
    min_rows: int = 12,
) -> RegionalAdjustmentModel | None:
    prob_column = f"model_prob_{probability_source}"
    if prob_column not in candidate_rows.columns:
        return None
    working = candidate_rows.copy()
    working["__prob__"] = _resolve_series(working, prob_column)
    if "won" in working.columns:
        won = _resolve_series(working, "won")
    elif {"selection", "actual_outcome"}.issubset(working.columns):
        won = working["selection"].fillna("").astype(str).eq(working["actual_outcome"].fillna("").astype(str)).astype(float)
    else:
        won = pd.Series(np.nan, index=working.index, dtype=float)
    working["__won__"] = won
    working = working[working["__prob__"].notna() & working["__won__"].notna() & working["__prob__"].gt(0.0)].copy()
    if working.empty:
        return None
    source_rows = int(len(working))
    training_scope = "all_eligible"
    policy_mask = _policy_region_mask(working, policy)
    if policy_mask is not None:
        policy_rows = int(policy_mask.sum())
        if policy_rows >= max(int(min_rows), 12):
            working = working.loc[policy_mask].copy()
            training_scope = "base_policy_region"
        else:
            training_scope = "all_eligible_fallback"
    working["__region_key__"] = _region_key(working)
    total_expected = float(working["__prob__"].sum())
    total_wins = float(working["__won__"].sum())
    if total_expected <= 0.0:
        return None
    global_factor = float((total_wins + prior_strength) / (total_expected + prior_strength))
    global_factor = float(np.clip(global_factor, 0.55, 1.20))
    grouped = (
        working.groupby("__region_key__", observed=True, dropna=False)
        .agg(
            rows=("__prob__", "size"),
            expected=("__prob__", "sum"),
            wins=("__won__", "sum"),
        )
        .reset_index()
    )
    factors: dict[str, dict[str, float]] = {}
    for _, row in grouped.iterrows():
        rows = int(row["rows"])
        expected = float(row["expected"])
        wins = float(row["wins"])
        if rows < min_rows or expected <= 0.0:
            factor = global_factor
        else:
            factor = float((wins + (prior_strength * global_factor)) / (expected + prior_strength))
        factors[str(row["__region_key__"])] = {
            "rows": rows,
            "expected_wins": expected,
            "actual_wins": wins,
            "factor": float(np.clip(factor, 0.55, 1.20)),
            "support": float(np.clip(rows / (rows + prior_strength), 0.0, 1.0)),
        }
    return RegionalAdjustmentModel(
        adjustment_mode=str(adjustment_mode or REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK),
        probability_source=str(probability_source),
        global_factor=global_factor,
        prior_strength=float(prior_strength),
        min_rows=int(min_rows),
        training_scope=training_scope,
        training_rows=int(len(working)),
        source_rows=source_rows,
        factors=factors,
    )


def apply_regional_adjustment(
    candidate_rows: pd.DataFrame,
    model: RegionalAdjustmentModel | None,
) -> pd.DataFrame:
    output = candidate_rows.copy()
    if model is None:
        output["regional_adjustment_mode"] = REGIONAL_ADJUSTMENT_NONE
        return output
    prob_column = f"model_prob_{model.probability_source}"
    if prob_column not in output.columns:
        output["regional_adjustment_mode"] = REGIONAL_ADJUSTMENT_NONE
        return output
    key = _region_key(output)
    output["regional_adjustment_key"] = key
    factor_map = {region: payload["factor"] for region, payload in model.factors.items()}
    support_map = {region: payload["support"] for region, payload in model.factors.items()}
    output["regional_probability_multiplier"] = key.map(factor_map).fillna(model.global_factor).astype(float)
    if model.adjustment_mode == REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK:
        multiplier = output["regional_probability_multiplier"].to_numpy(dtype=float)
        softened = np.where(
            multiplier < 1.0,
            1.0 - (0.35 * (1.0 - multiplier)),
            1.0 + (0.25 * (multiplier - 1.0)),
        )
        output["regional_probability_multiplier"] = np.clip(softened, 0.85, 1.08)
    output["regional_adjustment_support"] = key.map(support_map).fillna(0.0).astype(float)
    base_prob = _resolve_series(output, prob_column).fillna(0.0)
    adjusted_prob = np.clip(base_prob.to_numpy(dtype=float) * output["regional_probability_multiplier"].to_numpy(dtype=float), 1e-6, 0.999)
    output["regional_adjusted_prob"] = adjusted_prob
    quoted_odds = _resolve_series(output, "quoted_odds").replace(0.0, np.nan)
    top_ask = _resolve_series(output, "top_ask", default=np.nan)
    if top_ask.isna().all():
        top_ask = (1.0 / quoted_odds).replace([np.inf, -np.inf], np.nan)
    output["regional_adjusted_edge"] = adjusted_prob - top_ask.fillna(0.0).to_numpy(dtype=float)
    output["regional_adjusted_ev"] = (adjusted_prob * quoted_odds.fillna(0.0).to_numpy(dtype=float)) - 1.0
    output["regional_calibration_confidence"] = np.clip(
        0.50 + (0.50 * output["regional_adjustment_support"].to_numpy(dtype=float)),
        0.0,
        1.0,
    )
    output["regional_adjustment_mode"] = model.adjustment_mode
    return output


def crossfit_regional_adjustment(
    oof_candidates: pd.DataFrame,
    dev_candidates: pd.DataFrame,
    holdout_candidates: pd.DataFrame,
    *,
    probability_source: str,
    fold_column: str = "oof_fold",
    adjustment_mode: str = REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK,
    policy: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    oof_output = oof_candidates.copy()
    if fold_column not in oof_output.columns or oof_output.empty:
        model = fit_regional_adjustment_model(
            oof_output,
            probability_source=probability_source,
            adjustment_mode=adjustment_mode,
            policy=policy,
        )
        adjusted_oof = apply_regional_adjustment(oof_output, model)
        adjusted_dev = apply_regional_adjustment(dev_candidates, model)
        adjusted_holdout = apply_regional_adjustment(holdout_candidates, model)
        diagnostics = {
            "mode": adjustment_mode if model is not None else REGIONAL_ADJUSTMENT_NONE,
            "global_factor": None if model is None else model.global_factor,
            "training_scope": None if model is None else model.training_scope,
            "training_rows": 0 if model is None else model.training_rows,
            "source_rows": 0 if model is None else model.source_rows,
            "regions": {} if model is None else model.factors,
            "crossfit_folds": [],
        }
        return adjusted_oof, adjusted_dev, adjusted_holdout, diagnostics
    crossfit_folds: list[dict[str, Any]] = []
    adjusted_parts: list[pd.DataFrame] = []
    unique_folds = [fold for fold in oof_output[fold_column].dropna().unique().tolist()]
    full_model = fit_regional_adjustment_model(
        oof_output,
        probability_source=probability_source,
        adjustment_mode=adjustment_mode,
        policy=policy,
    )
    for fold in unique_folds:
        train = oof_output[oof_output[fold_column] != fold].copy()
        test = oof_output[oof_output[fold_column] == fold].copy()
        fold_model = fit_regional_adjustment_model(
            train,
            probability_source=probability_source,
            adjustment_mode=adjustment_mode,
            policy=policy,
        )
        adjusted_parts.append(apply_regional_adjustment(test, fold_model))
        crossfit_folds.append(
            {
                "fold": str(fold),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "global_factor": None if fold_model is None else fold_model.global_factor,
                "region_count": 0 if fold_model is None else len(fold_model.factors),
                "training_scope": None if fold_model is None else fold_model.training_scope,
                "training_rows": 0 if fold_model is None else fold_model.training_rows,
                "source_rows": 0 if fold_model is None else fold_model.source_rows,
            }
        )
    if adjusted_parts:
        adjusted_oof = pd.concat(adjusted_parts, axis=0).sort_index().reset_index(drop=True)
    else:
        adjusted_oof = apply_regional_adjustment(oof_output, full_model)
    adjusted_dev = apply_regional_adjustment(dev_candidates, full_model)
    adjusted_holdout = apply_regional_adjustment(holdout_candidates, full_model)
    diagnostics = {
        "mode": adjustment_mode if full_model is not None else REGIONAL_ADJUSTMENT_NONE,
        "global_factor": None if full_model is None else full_model.global_factor,
        "training_scope": None if full_model is None else full_model.training_scope,
        "training_rows": 0 if full_model is None else full_model.training_rows,
        "source_rows": 0 if full_model is None else full_model.source_rows,
        "regions": {} if full_model is None else full_model.factors,
        "crossfit_folds": crossfit_folds,
    }
    return adjusted_oof, adjusted_dev, adjusted_holdout, diagnostics


__all__ = [
    "REGIONAL_ADJUSTMENT_NONE",
    "REGIONAL_ADJUSTMENT_OOF_REGION_SHRINK",
    "REGIONAL_ADJUSTMENT_OOF_REGION_SOFT_SHRINK",
    "RegionalAdjustmentModel",
    "apply_regional_adjustment",
    "crossfit_regional_adjustment",
    "fit_regional_adjustment_model",
]
