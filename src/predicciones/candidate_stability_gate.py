from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .strategy import BetPolicy

STABILITY_GATE_NONE = "none"
STABILITY_GATE_HIERARCHICAL = "hierarchical_oof_gate"
STABILITY_GATE_SOFT_PARENT = "hierarchical_soft_parent_gate"
STABILITY_GATE_OUTCOME_PARENT = "hierarchical_outcome_parent_gate"
STABILITY_GATE_EXPANDING_OUTCOME = "hierarchical_expanding_outcome_gate"
STABILITY_GATE_VALIDATED_OUTCOME = "validated_outcome_gate"
STABILITY_GATE_VALIDATED_OUTCOME_SOFT = "validated_outcome_soft_gate"

_ODDS_BANDS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 1.5, "<1.5"),
    (1.5, 2.0, "1.5-2"),
    (2.0, 3.0, "2-3"),
    (3.0, 6.0, "3-6"),
    (6.0, None, "6+"),
)
_LEVEL_PRIOR_STRENGTH = {
    "league_selection_odds": 24.0,
    "selection_odds": 18.0,
    "selection": 12.0,
    "odds": 12.0,
    "global": 8.0,
}
_LEVEL_SUPPORT_THRESHOLD = {
    "league_selection_odds": 18,
    "selection_odds": 18,
    "selection": 18,
    "odds": 18,
    "global": 1,
}
_SPECIFIC_HARD_LEVELS = {"league_selection_odds", "selection_odds"}
_EXPANDING_GATE_MODES = {
    STABILITY_GATE_EXPANDING_OUTCOME,
    STABILITY_GATE_VALIDATED_OUTCOME,
    STABILITY_GATE_VALIDATED_OUTCOME_SOFT,
}
_SUPPORTED_GATE_MODES = {
    STABILITY_GATE_HIERARCHICAL,
    STABILITY_GATE_SOFT_PARENT,
    STABILITY_GATE_OUTCOME_PARENT,
    STABILITY_GATE_EXPANDING_OUTCOME,
    STABILITY_GATE_VALIDATED_OUTCOME,
    STABILITY_GATE_VALIDATED_OUTCOME_SOFT,
}


def _soft_levels_for_mode(mode: str) -> set[str]:
    if mode == STABILITY_GATE_SOFT_PARENT:
        return {"global", "selection", "odds"}
    if mode in {STABILITY_GATE_OUTCOME_PARENT, STABILITY_GATE_EXPANDING_OUTCOME}:
        return {"global", "odds"}
    return set()


def _hard_levels_for_mode(mode: str) -> set[str]:
    if mode == STABILITY_GATE_SOFT_PARENT:
        return set(_SPECIFIC_HARD_LEVELS)
    if mode in {STABILITY_GATE_OUTCOME_PARENT, STABILITY_GATE_EXPANDING_OUTCOME}:
        return {"selection", *_SPECIFIC_HARD_LEVELS}
    return {"global", "selection", "odds", *_SPECIFIC_HARD_LEVELS}


@dataclass(frozen=True)
class StabilityGateModel:
    mode: str
    regions: dict[str, dict[str, Any]]
    global_key: str
    probability_source: str


def _float_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _string_series(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str)


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


def _region_keys(frame: pd.DataFrame) -> pd.DataFrame:
    league = _string_series(frame, "league_code", "unknown")
    selection = _string_series(frame, "selection", "unknown")
    odds_band = _odds_band(_float_series(frame, "quoted_odds"))
    return pd.DataFrame(
        {
            "league_selection_odds": league + "|" + selection + "|" + odds_band,
            "selection_odds": selection + "|" + odds_band,
            "selection": selection,
            "odds": odds_band,
            "global": "global",
        },
        index=frame.index,
    )


def _scope_mask(frame: pd.DataFrame, policy: BetPolicy) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if policy.allowed_leagues:
        mask &= _string_series(frame, "league_code").isin(policy.allowed_leagues)
    if policy.allowed_outcomes:
        mask &= _string_series(frame, "selection").isin(policy.allowed_outcomes)
    odds = _float_series(frame, "quoted_odds")
    upper = odds.lt(float(policy.max_odds)) if str(policy.scope_name).startswith("mid_odds") else odds.le(float(policy.max_odds))
    return mask & odds.ge(float(policy.min_odds)) & upper


def base_policy_candidate_mask(frame: pd.DataFrame, policy: BetPolicy) -> pd.Series:
    return (
        _string_series(frame, "quote_status").eq("eligible")
        & _scope_mask(frame, policy)
        & _float_series(frame, "policy_edge").ge(float(policy.edge_threshold))
        & _float_series(frame, "policy_ev").ge(float(policy.ev_threshold))
    )


def _unit_profit(frame: pd.DataFrame) -> pd.Series:
    odds = _float_series(frame, "quoted_odds").fillna(0.0)
    if "won" in frame.columns:
        won = _float_series(frame, "won").fillna(0.0)
    else:
        won = _string_series(frame, "selection").eq(_string_series(frame, "actual_outcome")).astype(float)
    return np.where(won.gt(0.0), odds - 1.0, -1.0)


def _fold_positive_ratio(frame: pd.DataFrame) -> tuple[int, float]:
    if frame.empty or "retro_fold_id" not in frame.columns:
        return 0, 0.0
    fold_values: list[float] = []
    folds = pd.to_numeric(frame["retro_fold_id"], errors="coerce")
    for fold_id in sorted(folds.dropna().unique().tolist()):
        fold_frame = frame[folds.eq(fold_id)].copy()
        if fold_frame.empty:
            continue
        fold_values.append(float(_unit_profit(fold_frame).mean()))
    if not fold_values:
        return 0, 0.0
    return len(fold_values), float(np.mean(np.asarray(fold_values, dtype=float) > 0.0))


def _validation_unit_series(frame: pd.DataFrame) -> pd.Series:
    output = pd.Series("unit:all", index=frame.index, dtype="object")
    if "retro_fold_id" in frame.columns:
        folds = pd.to_numeric(frame["retro_fold_id"], errors="coerce")
        fold_mask = folds.notna()
        output.loc[fold_mask] = "fold:" + folds.loc[fold_mask].map(lambda value: f"{float(value):g}").astype(str)
    for column in ("pre_holdout_window", "policy_window", "window_id", "window_name"):
        if column not in frame.columns:
            continue
        values = _string_series(frame, column, "")
        window_mask = values.ne("") & values.ne("nan") & output.eq("unit:all")
        output.loc[window_mask] = "pre:" + values.loc[window_mask].astype(str)
    if "retro_segment" in frame.columns:
        segments = _string_series(frame, "retro_segment", "")
        segment_mask = segments.ne("") & segments.ne("nan") & output.eq("unit:all")
        output.loc[segment_mask] = "segment:" + segments.loc[segment_mask].astype(str)
    return output


def _validation_positive_ratio(frame: pd.DataFrame) -> tuple[int, float]:
    if frame.empty:
        return 0, 0.0
    unit_values: list[float] = []
    units = _validation_unit_series(frame)
    for _, unit_frame in frame.groupby(units, observed=True, sort=False):
        if unit_frame.empty:
            continue
        unit_values.append(float(np.asarray(_unit_profit(unit_frame), dtype=float).mean()))
    if not unit_values:
        return 0, 0.0
    return len(unit_values), float(np.mean(np.asarray(unit_values, dtype=float) > 0.0))


def _raw_region_payload(frame: pd.DataFrame) -> dict[str, float]:
    rows = int(len(frame))
    wins = float(_float_series(frame, "won").fillna(_string_series(frame, "selection").eq(_string_series(frame, "actual_outcome")).astype(float)).sum())
    losses = float(rows - wins)
    odds = _float_series(frame, "quoted_odds").replace([np.inf, -np.inf], np.nan)
    expected_wins = float(_float_series(frame, "policy_prob").clip(lower=0.0, upper=1.0).sum())
    fold_count, positive_fold_ratio = _fold_positive_ratio(frame)
    flat_roi = float(_unit_profit(frame).mean()) if rows else 0.0
    avg_odds = float(odds.mean()) if odds.notna().any() else 0.0
    observed_expected_ratio = float(wins / expected_wins) if expected_wins > 0.0 else 1.0
    return {
        "rows": rows,
        "fold_count": int(fold_count),
        "positive_fold_ratio": positive_fold_ratio,
        "flat_roi": flat_roi,
        "wins": wins,
        "losses": losses,
        "avg_odds": avg_odds,
        "expected_wins": expected_wins,
        "observed_expected_ratio": observed_expected_ratio,
    }


def _posterior_payload(payload: dict[str, float], parent_win_rate: float, prior_strength: float) -> dict[str, float]:
    wins = float(payload.get("wins", 0.0))
    losses = float(payload.get("losses", 0.0))
    alpha = wins + (float(parent_win_rate) * prior_strength)
    beta = losses + ((1.0 - float(parent_win_rate)) * prior_strength)
    denom = alpha + beta
    posterior_mean = float(alpha / denom) if denom > 0 else float(parent_win_rate)
    posterior_std = float(np.sqrt((alpha * beta) / ((denom**2) * (denom + 1.0)))) if denom > 0 else 0.0
    posterior_lower_win_rate = float(max(0.0, posterior_mean - (0.67 * posterior_std)))
    avg_odds = float(payload.get("avg_odds", 0.0))
    posterior_lower_ev = float((posterior_lower_win_rate * avg_odds) - 1.0) if avg_odds > 0.0 else -1.0
    return {
        "alpha": alpha,
        "beta": beta,
        "posterior_win_rate": posterior_mean,
        "posterior_lower_win_rate": posterior_lower_win_rate,
        "posterior_lower_ev": posterior_lower_ev,
        "prior_strength": float(prior_strength),
        "parent_win_rate": float(parent_win_rate),
    }


def _build_region_stats(training_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if training_rows.empty:
        return {}
    working = training_rows.copy()
    keys = _region_keys(working)
    raw_by_level: dict[str, dict[str, dict[str, float]]] = {}
    for level in ("global", "odds", "selection", "selection_odds", "league_selection_odds"):
        level_payloads: dict[str, dict[str, float]] = {}
        for key, group in working.groupby(keys[level], observed=True, sort=False):
            level_payloads[str(key)] = _raw_region_payload(group)
        raw_by_level[level] = level_payloads
    global_payload = raw_by_level.get("global", {}).get("global", _raw_region_payload(working))
    global_parent = float(global_payload["wins"] / max(global_payload["rows"], 1))
    regions: dict[str, dict[str, Any]] = {}
    for level in ("global", "odds", "selection", "selection_odds", "league_selection_odds"):
        for key, payload in raw_by_level.get(level, {}).items():
            if level == "global":
                parent_win_rate = global_parent
                parent_key = "global"
            elif level in {"odds", "selection"}:
                parent_key = "global"
                parent_win_rate = regions.get("global", {}).get("posterior_win_rate", global_parent)
            elif level == "selection_odds":
                parent_key = str(key).split("|", maxsplit=1)[0]
                parent_win_rate = regions.get(parent_key, regions.get("global", {})).get("posterior_win_rate", global_parent)
            else:
                parts = str(key).split("|", maxsplit=1)
                parent_key = parts[1] if len(parts) == 2 else "global"
                parent_win_rate = regions.get(parent_key, regions.get("global", {})).get("posterior_win_rate", global_parent)
            prior_strength = float(_LEVEL_PRIOR_STRENGTH[level])
            posterior = _posterior_payload(payload, float(parent_win_rate), prior_strength)
            regions[str(key)] = {
                "key": str(key),
                "level": level,
                "parent_key": parent_key,
                **payload,
                **posterior,
            }
    return regions


def _build_validated_outcome_stats(training_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if training_rows.empty:
        return {
            "__validated_outcomes__": {
                "mode": STABILITY_GATE_VALIDATED_OUTCOME,
                "stable_outcomes": [],
                "outcome_count": 0,
                "training_rows": 0,
            }
        }
    regions: dict[str, dict[str, Any]] = {}
    stable_outcomes: list[str] = []
    for selection, group in training_rows.groupby(_string_series(training_rows, "selection", "unknown"), observed=True, sort=True):
        payload = _raw_region_payload(group)
        unit_count, positive_unit_ratio = _validation_positive_ratio(group)
        payload["unit_count"] = int(unit_count)
        payload["positive_unit_ratio"] = float(positive_unit_ratio)
        payload["positive_fold_ratio"] = float(positive_unit_ratio)
        is_stable = (
            int(payload.get("rows", 0)) >= 12
            and float(payload.get("positive_unit_ratio", 0.0)) >= 0.5
            and float(payload.get("flat_roi", 0.0)) > 0.0
        )
        selection_key = str(selection)
        if is_stable:
            stable_outcomes.append(selection_key)
        regions[f"selection::{selection_key}"] = {
            "key": f"selection::{selection_key}",
            "level": "validated_outcome",
            "selection": selection_key,
            "stable": bool(is_stable),
            **payload,
        }
    regions["__validated_outcomes__"] = {
        "mode": STABILITY_GATE_VALIDATED_OUTCOME,
        "stable_outcomes": sorted(stable_outcomes),
        "outcome_count": int(len(regions)),
        "training_rows": int(len(training_rows)),
        "minimum_rows": 12,
        "minimum_positive_unit_ratio": 0.5,
        "minimum_flat_roi": 0.0,
    }
    return regions


def fit_stability_gate_model(
    training_rows: pd.DataFrame,
    policy: BetPolicy,
    *,
    probability_source: str,
    mode: str = STABILITY_GATE_HIERARCHICAL,
) -> StabilityGateModel:
    eligible = training_rows[base_policy_candidate_mask(training_rows, policy)].copy()
    gate_mode = str(mode) if str(mode) in _SUPPORTED_GATE_MODES else STABILITY_GATE_HIERARCHICAL
    if gate_mode in {STABILITY_GATE_VALIDATED_OUTCOME, STABILITY_GATE_VALIDATED_OUTCOME_SOFT}:
        regions = _build_validated_outcome_stats(eligible)
    else:
        regions = _build_region_stats(eligible)
    return StabilityGateModel(
        mode=gate_mode,
        regions=regions,
        global_key="global",
        probability_source=str(probability_source),
    )


def _apply_validated_outcome_gate(candidate_rows: pd.DataFrame, model: StabilityGateModel) -> pd.DataFrame:
    output = candidate_rows.copy()
    gate_mode = str(model.mode)
    soft_mode = gate_mode == STABILITY_GATE_VALIDATED_OUTCOME_SOFT
    metadata = model.regions.get("__validated_outcomes__", {})
    stable_outcomes = {str(value) for value in metadata.get("stable_outcomes", [])}
    selections = _string_series(output, "selection", "unknown")
    no_stable_outcomes = len(stable_outcomes) == 0
    effective_keys = ["selection::" + str(selection) for selection in selections.tolist()]
    row_payloads = [model.regions.get(key, {}) for key in effective_keys]
    region_rows = [int(payload.get("rows", 0)) for payload in row_payloads]
    positive_ratios = [
        float(payload.get("positive_unit_ratio", payload.get("positive_fold_ratio", 0.0))) for payload in row_payloads
    ]
    flat_rois = [float(payload.get("flat_roi", 0.0)) for payload in row_payloads]
    observed_expected_ratios = [
        float(np.clip(payload.get("observed_expected_ratio", 1.0), 0.55, 1.05)) for payload in row_payloads
    ]
    stable_values = [bool(payload.get("stable", False)) for payload in row_payloads]
    if soft_mode:
        pass_array = np.ones(len(output), dtype=bool)
        multipliers: list[float] = []
        reason_values: list[str] = []
        for payload, stable, rows, positive_ratio, flat_roi, observed_expected_ratio in zip(
            row_payloads,
            stable_values,
            region_rows,
            positive_ratios,
            flat_rois,
            observed_expected_ratios,
            strict=False,
        ):
            if not payload:
                multipliers.append(0.95)
                reason_values.append("validated_outcome_no_stats_soft_penalty")
                continue
            if stable:
                multipliers.append(1.0)
                reason_values.append("validated_outcome_passed")
                continue
            multiplier = float(np.clip(observed_expected_ratio, 0.70, 1.0))
            if rows < 12:
                multiplier = min(multiplier, 0.95)
            if positive_ratio < 0.5:
                multiplier = min(multiplier, 0.88)
            if flat_roi < 0.0:
                multiplier = min(multiplier, 0.90)
            multipliers.append(multiplier)
            reason_values.append("validated_outcome_soft_penalty")
    else:
        pass_array = selections.isin(stable_outcomes).to_numpy(dtype=bool) if not no_stable_outcomes else np.ones(len(output), dtype=bool)
        reason_values = np.where(
            pass_array,
            "validated_outcome_passed" if not no_stable_outcomes else "no_validated_outcome_available",
            "validated_outcome_unstable",
        ).tolist()
        multipliers = [1.0 for _ in range(len(output))]
    policy_prob = _float_series(output, "policy_prob").fillna(0.0).clip(lower=0.0, upper=1.0)
    top_ask = _float_series(output, "top_ask", default=np.nan)
    quoted_odds = _float_series(output, "quoted_odds", default=np.nan)
    if top_ask.isna().all():
        top_ask = (1.0 / quoted_odds.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    region_multiplier = pd.Series(multipliers, index=output.index, dtype=float)
    gated_prob = np.where(pass_array, (policy_prob * region_multiplier).clip(lower=0.001, upper=0.999).to_numpy(dtype=float), 0.001)
    gated_edge = np.where(pass_array, gated_prob - top_ask.fillna(1.0).to_numpy(dtype=float), -1.0)
    gated_ev = np.where(pass_array, (gated_prob * quoted_odds.fillna(0.0).to_numpy(dtype=float)) - 1.0, -1.0)
    output["stability_gate_mode"] = gate_mode
    output["region_key"] = selections
    output["effective_region_key"] = effective_keys
    output["effective_region_level"] = "validated_outcome"
    output["candidate_gate_pass"] = pass_array
    output["candidate_gate_reason"] = reason_values
    output["stability_region_rows"] = region_rows
    output["stability_posterior_lower_ev"] = np.nan
    output["stability_positive_fold_ratio"] = positive_ratios
    output["stability_flat_roi"] = flat_rois
    output["stability_region_multiplier"] = region_multiplier
    output["stability_confidence_multiplier"] = 1.0
    output["pre_gate_policy_prob"] = policy_prob
    output["pre_gate_policy_edge"] = _float_series(output, "policy_edge")
    output["pre_gate_policy_ev"] = _float_series(output, "policy_ev")
    output["gated_prob"] = gated_prob
    output["gated_edge"] = gated_edge
    output["gated_ev"] = gated_ev
    return output


def _effective_region(row_keys: pd.Series, regions: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any] | None, int]:
    max_rows = 0
    for level in ("league_selection_odds", "selection_odds", "selection", "odds", "global"):
        key = str(row_keys[level])
        payload = regions.get(key)
        if payload is None:
            continue
        rows = int(payload.get("rows", 0))
        max_rows = max(max_rows, rows)
        if rows >= int(_LEVEL_SUPPORT_THRESHOLD[level]):
            return key, payload, max_rows
    payload = regions.get("global")
    if payload is not None:
        max_rows = max(max_rows, int(payload.get("rows", 0)))
        return "global", payload, max_rows
    return "missing", None, max_rows


def apply_stability_gate(candidate_rows: pd.DataFrame, model: StabilityGateModel | None) -> pd.DataFrame:
    output = candidate_rows.copy()
    if model is None or not model.regions:
        output["stability_gate_mode"] = STABILITY_GATE_NONE
        output["candidate_gate_pass"] = True
        output["candidate_gate_reason"] = "gate_unavailable"
        return output
    gate_mode = str(model.mode) if str(model.mode) in _SUPPORTED_GATE_MODES else STABILITY_GATE_HIERARCHICAL
    if gate_mode in {STABILITY_GATE_VALIDATED_OUTCOME, STABILITY_GATE_VALIDATED_OUTCOME_SOFT}:
        return _apply_validated_outcome_gate(output, model)
    soft_levels = _soft_levels_for_mode(gate_mode)
    hard_levels = _hard_levels_for_mode(gate_mode)
    keys = _region_keys(output)
    pass_values: list[bool] = []
    reason_values: list[str] = []
    effective_keys: list[str] = []
    effective_levels: list[str] = []
    region_keys: list[str] = []
    multipliers: list[float] = []
    region_rows: list[int] = []
    lower_evs: list[float] = []
    positive_ratios: list[float] = []
    flat_rois: list[float] = []
    for index, row_keys in keys.iterrows():
        region_key = str(row_keys["league_selection_odds"])
        effective_key, payload, max_rows = _effective_region(row_keys, model.regions)
        region_keys.append(region_key)
        effective_keys.append(effective_key)
        if payload is None:
            pass_values.append(True)
            reason_values.append("no_region_stats")
            multipliers.append(1.0)
            region_rows.append(0)
            lower_evs.append(0.0)
            positive_ratios.append(1.0)
            flat_rois.append(0.0)
            effective_levels.append("missing")
            continue
        rows = int(payload.get("rows", 0))
        level = str(payload.get("level", "global"))
        confidence = float(_float_series(output.loc[[index]], "selection_confidence_score", default=np.nan).iloc[0])
        if not np.isfinite(confidence):
            confidence = float(_float_series(output.loc[[index]], "policy_confidence_score", default=0.5).iloc[0])
        posterior_lower_ev = float(payload.get("posterior_lower_ev", 0.0))
        positive_fold_ratio = float(payload.get("positive_fold_ratio", 0.0))
        flat_roi = float(payload.get("flat_roi", 0.0))
        pass_gate = True
        reason = "passed"
        hard_block_allowed = level in hard_levels
        block_reason: str | None = None
        if posterior_lower_ev < -0.03:
            block_reason = "posterior_lower_ev_below_floor"
        elif positive_fold_ratio < 0.5:
            block_reason = "positive_fold_ratio_below_floor"
        elif rows >= 24 and flat_roi < -0.15:
            block_reason = "flat_roi_below_floor"
        if block_reason and hard_block_allowed:
            pass_gate = False
            reason = block_reason
        elif block_reason:
            reason = "parent_soft_adjusted"
        elif max_rows < 12 and confidence < 0.70:
            pass_gate = False
            reason = "low_support_blocked"
        pass_values.append(pass_gate)
        reason_values.append(reason)
        observed_expected_ratio = float(payload.get("observed_expected_ratio", 1.0))
        if level in soft_levels:
            soft_multiplier = 1.0 - (0.50 * max(0.0, 1.0 - observed_expected_ratio))
            multipliers.append(float(np.clip(soft_multiplier, 0.85, 1.05)))
        else:
            multipliers.append(float(np.clip(observed_expected_ratio, 0.55, 1.10)))
        region_rows.append(rows)
        lower_evs.append(posterior_lower_ev)
        positive_ratios.append(positive_fold_ratio)
        flat_rois.append(flat_roi)
        effective_levels.append(level)
    policy_prob = _float_series(output, "policy_prob").fillna(0.0).clip(lower=0.0, upper=1.0)
    confidence_v2 = _float_series(output, "confidence_score_v2", default=np.nan).combine_first(
        _float_series(output, "selection_confidence_score", default=np.nan)
    ).combine_first(_float_series(output, "policy_confidence_score", default=0.5)).fillna(0.5).clip(lower=0.0, upper=1.0)
    region_multiplier = pd.Series(multipliers, index=output.index, dtype=float)
    confidence_multiplier = 0.85 + (0.15 * confidence_v2)
    adjusted = (policy_prob * region_multiplier * confidence_multiplier).clip(lower=0.001, upper=0.999)
    conservative_prob = np.minimum(policy_prob.to_numpy(dtype=float), adjusted.to_numpy(dtype=float))
    pass_array = np.asarray(pass_values, dtype=bool)
    top_ask = _float_series(output, "top_ask", default=np.nan)
    quoted_odds = _float_series(output, "quoted_odds", default=np.nan)
    if top_ask.isna().all():
        top_ask = (1.0 / quoted_odds.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    gated_prob = np.where(pass_array, conservative_prob, 0.001)
    gated_edge = np.where(pass_array, gated_prob - top_ask.fillna(1.0).to_numpy(dtype=float), -1.0)
    gated_ev = np.where(pass_array, (gated_prob * quoted_odds.fillna(0.0).to_numpy(dtype=float)) - 1.0, -1.0)
    output["stability_gate_mode"] = gate_mode
    output["region_key"] = region_keys
    output["effective_region_key"] = effective_keys
    output["effective_region_level"] = effective_levels
    output["candidate_gate_pass"] = pass_array
    output["candidate_gate_reason"] = reason_values
    output["stability_region_rows"] = region_rows
    output["stability_posterior_lower_ev"] = lower_evs
    output["stability_positive_fold_ratio"] = positive_ratios
    output["stability_flat_roi"] = flat_rois
    output["stability_region_multiplier"] = region_multiplier
    output["stability_confidence_multiplier"] = confidence_multiplier
    output["pre_gate_policy_prob"] = policy_prob
    output["pre_gate_policy_edge"] = _float_series(output, "policy_edge")
    output["pre_gate_policy_ev"] = _float_series(output, "policy_ev")
    output["gated_prob"] = gated_prob
    output["gated_edge"] = gated_edge
    output["gated_ev"] = gated_ev
    return output


def crossfit_stability_gate(
    oof_candidates: pd.DataFrame,
    dev_candidates: pd.DataFrame,
    holdout_candidates: pd.DataFrame,
    *,
    policy: BetPolicy,
    probability_source: str,
    fold_column: str = "retro_fold_id",
    mode: str = STABILITY_GATE_HIERARCHICAL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    gate_mode = str(mode) if str(mode) in _SUPPORTED_GATE_MODES else STABILITY_GATE_HIERARCHICAL
    if oof_candidates.empty:
        diagnostics = {"mode": STABILITY_GATE_NONE, "regions": {}, "crossfit_folds": []}
        return oof_candidates.copy(), dev_candidates.copy(), holdout_candidates.copy(), diagnostics, pd.DataFrame()
    oof_parts: list[pd.DataFrame] = []
    crossfit_rows: list[dict[str, Any]] = []
    fold_series = pd.to_numeric(oof_candidates.get(fold_column), errors="coerce")
    folds = sorted(fold_series.dropna().unique().tolist())
    for fold in folds:
        train = oof_candidates[fold_series.lt(fold)].copy()
        test = oof_candidates[fold_series.eq(fold)].copy()
        model = fit_stability_gate_model(train, policy, probability_source=probability_source, mode=gate_mode)
        gated = apply_stability_gate(test, model)
        oof_parts.append(gated)
        crossfit_rows.append(
            {
                "fold": int(fold) if float(fold).is_integer() else float(fold),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "region_count": int(len(model.regions)),
            }
        )
    full_model = fit_stability_gate_model(oof_candidates, policy, probability_source=probability_source, mode=gate_mode)
    gated_oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else apply_stability_gate(oof_candidates, full_model)
    gated_dev = apply_stability_gate(dev_candidates, full_model)
    holdout_model = full_model
    holdout_training_rows = int(len(oof_candidates))
    holdout_training_segments = ["oof"]
    if gate_mode in _EXPANDING_GATE_MODES:
        holdout_training = pd.concat([oof_candidates, dev_candidates], ignore_index=True, sort=False)
        holdout_model = fit_stability_gate_model(holdout_training, policy, probability_source=probability_source, mode=gate_mode)
        holdout_training_rows = int(len(holdout_training))
        holdout_training_segments = ["oof", "pre_holdout"]
    gated_holdout = apply_stability_gate(holdout_candidates, holdout_model)
    audit = pd.concat(
        [
            gated_oof.assign(stability_gate_split="oof"),
            gated_dev.assign(stability_gate_split="selection_dev"),
            gated_holdout.assign(stability_gate_split="locked_holdout"),
        ],
        ignore_index=True,
        sort=False,
    )
    diagnostics = {
        "mode": gate_mode,
        "regions": full_model.regions,
        "holdout_regions": holdout_model.regions,
        "crossfit_folds": crossfit_rows,
        "crossfit_training_direction": "past_folds_only",
        "pre_holdout_training_segments": ["oof"],
        "holdout_training_segments": holdout_training_segments,
        "holdout_training_rows": holdout_training_rows,
        "blocked_by_split": _blocked_by_split(audit, policy),
    }
    return gated_oof, gated_dev, gated_holdout, diagnostics, audit


def _blocked_by_split(audit: pd.DataFrame, policy: BetPolicy) -> dict[str, dict[str, Any]]:
    if audit.empty:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for split, group in audit.groupby("stability_gate_split", observed=True, sort=False):
        pass_mask = group["candidate_gate_pass"].fillna(True).astype(bool)
        reason_counts = group.loc[~pass_mask, "candidate_gate_reason"].astype(str).value_counts().to_dict()
        base_mask = base_policy_candidate_mask(group, policy)
        blocked_base = int((base_mask & ~pass_mask).sum())
        base_total = int(base_mask.sum())
        output[str(split)] = {
            "candidates": int(len(group)),
            "blocked_candidates": int((~pass_mask).sum()),
            "blocked_candidate_share": float((~pass_mask).mean()) if len(group) else 0.0,
            "base_policy_candidates": base_total,
            "blocked_base_policy_candidates": blocked_base,
            "blocked_base_policy_pick_share": float(blocked_base / base_total) if base_total > 0 else 0.0,
            "reasons": {str(key): int(value) for key, value in reason_counts.items()},
        }
    return output


def stability_gate_audit_columns(audit: pd.DataFrame) -> pd.DataFrame:
    desired = [
        "stability_gate_split",
        "stability_gate_mode",
        "match_id",
        "selection",
        "quoted_odds",
        "region_key",
        "effective_region_key",
        "effective_region_level",
        "candidate_gate_pass",
        "candidate_gate_reason",
        "policy_prob",
        "gated_prob",
        "policy_edge",
        "gated_edge",
        "policy_ev",
        "gated_ev",
    ]
    output = audit.copy()
    for column in desired:
        if column not in output.columns:
            output[column] = np.nan
    return output[desired].copy()


__all__ = [
    "STABILITY_GATE_NONE",
    "STABILITY_GATE_HIERARCHICAL",
    "STABILITY_GATE_SOFT_PARENT",
    "STABILITY_GATE_OUTCOME_PARENT",
    "STABILITY_GATE_EXPANDING_OUTCOME",
    "STABILITY_GATE_VALIDATED_OUTCOME",
    "STABILITY_GATE_VALIDATED_OUTCOME_SOFT",
    "StabilityGateModel",
    "apply_stability_gate",
    "base_policy_candidate_mask",
    "crossfit_stability_gate",
    "fit_stability_gate_model",
    "stability_gate_audit_columns",
]
