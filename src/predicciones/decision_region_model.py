from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DECISION_SCORER_HEURISTIC = "heuristic"
DECISION_SCORER_LOGIT = "dr_logit"
DECISION_SCORER_HGB = "dr_hgb"
DECISION_SCORER_LOGIT_PROB = "dr_logit_prob"
DECISION_SCORER_HGB_PROB = "dr_hgb_prob"
DECISION_SCORER_RELIABILITY_PROB = "dr_reliability_prob"

TRAINING_SCOPE_ELIGIBLE = "eligible"
TRAINING_SCOPE_ARGMAX = "argmax_all_bets"

DECISION_SCORER_NAMES = (
    DECISION_SCORER_HEURISTIC,
    DECISION_SCORER_LOGIT,
    DECISION_SCORER_HGB,
    DECISION_SCORER_LOGIT_PROB,
    DECISION_SCORER_HGB_PROB,
    DECISION_SCORER_RELIABILITY_PROB,
)
DECISION_SCORER_FEATURES_NUMERIC = (
    "selection_prob",
    "selection_edge",
    "selection_ev",
    "selection_odds",
    "selection_confidence_score",
    "season_progress_pct",
    "home_long_sample_ratio_overall",
    "away_long_sample_ratio_overall",
    "home_long_sample_ratio_side",
    "away_long_sample_ratio_side",
    "home_goal_volatility_20",
    "away_goal_volatility_20",
    "home_goals_against_volatility_20",
    "away_goals_against_volatility_20",
    "home_opponent_strength_volatility_20",
    "away_opponent_strength_volatility_20",
    "home_schedule_irregularity_20",
    "away_schedule_irregularity_20",
    "confidence_floor_continuous",
)
DECISION_SCORER_FEATURES_CATEGORICAL = (
    "league_code",
    "selection",
    "season_phase",
)


def _float_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _resolve_float_series(frame: pd.DataFrame, columns: tuple[str, ...], default: float = np.nan) -> pd.Series:
    resolved = pd.Series(default, index=frame.index, dtype=float)
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        resolved = resolved.where(resolved.notna(), values)
    return resolved


def _season_phase_labels(frame: pd.DataFrame) -> pd.Series:
    if "season_phase" in frame.columns:
        phase = frame["season_phase"].fillna("unknown").astype(str)
        return phase.where(phase.ne(""), "unknown")

    early = _float_series(frame, "season_phase_early", default=0.0).fillna(0.0)
    mid = _float_series(frame, "season_phase_mid", default=0.0).fillna(0.0)
    late = _float_series(frame, "season_phase_late", default=0.0).fillna(0.0)
    labels = pd.Series("unknown", index=frame.index, dtype=object)
    labels = labels.where(~early.gt(0.5), "early")
    labels = labels.where(~mid.gt(0.5), "mid")
    labels = labels.where(~late.gt(0.5), "late")
    return labels.astype(str)


def build_candidate_viability_frame(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_rows.copy()
    features = pd.DataFrame(index=frame.index)
    features["selection_prob"] = _resolve_float_series(frame, ("selection_prob", "policy_prob", "model_prob_raw"))
    features["selection_edge"] = _resolve_float_series(frame, ("selection_edge", "policy_edge", "edge_raw"))
    features["selection_ev"] = _resolve_float_series(frame, ("selection_ev", "policy_ev", "ev_raw"))
    features["selection_odds"] = _resolve_float_series(frame, ("selection_odds", "quoted_odds"))
    features["selection_confidence_score"] = _resolve_float_series(
        frame,
        ("selection_confidence_score", "policy_confidence_score", "confidence_score_v2", "confidence_score"),
        default=0.5,
    ).fillna(0.5)

    for column in DECISION_SCORER_FEATURES_NUMERIC[5:]:
        features[column] = _float_series(frame, column)

    features["league_code"] = frame.get("league_code", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    features["selection"] = frame.get("selection", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    features["season_phase"] = _season_phase_labels(frame)

    actual = frame.get("actual_outcome", pd.Series("", index=frame.index)).fillna("").astype(str)
    selected = features["selection"].fillna("").astype(str)
    has_actual = actual.ne("")
    features["bet_wins"] = np.where(has_actual, actual.eq(selected).astype(float), np.nan)
    return features


def _eligible_training_rows(scored_candidates: pd.DataFrame) -> pd.DataFrame:
    working = scored_candidates.copy()
    actual = working.get("actual_outcome", pd.Series("", index=working.index)).fillna("").astype(str)
    return working[working["quote_status"].astype(str).eq("eligible") & actual.ne("")].copy()


def _argmax_training_rows(scored_candidates: pd.DataFrame) -> pd.DataFrame:
    eligible = _eligible_training_rows(scored_candidates)
    if eligible.empty:
        return eligible
    ordered = eligible.sort_values(
        ["match_id", "policy_prob", "policy_ev", "quoted_odds"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    return ordered.groupby("match_id", observed=True).head(1).copy()


def training_subset(scored_candidates: pd.DataFrame, scope: str) -> pd.DataFrame:
    scope_key = str(scope or TRAINING_SCOPE_ELIGIBLE)
    if scope_key == TRAINING_SCOPE_ARGMAX:
        return _argmax_training_rows(scored_candidates)
    return _eligible_training_rows(scored_candidates)


def _dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _base_scorer_name(scorer_name: str) -> str:
    scorer_key = str(scorer_name)
    if scorer_key == DECISION_SCORER_LOGIT_PROB:
        return DECISION_SCORER_LOGIT
    if scorer_key == DECISION_SCORER_HGB_PROB:
        return DECISION_SCORER_HGB
    return scorer_key


def _adjusts_policy_probability(scorer_name: str) -> bool:
    return str(scorer_name) in {DECISION_SCORER_LOGIT_PROB, DECISION_SCORER_HGB_PROB, DECISION_SCORER_RELIABILITY_PROB}


def _band_labels(values: pd.Series, bins: list[float], labels: list[str], default: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    banded = pd.cut(numeric, bins=bins, labels=labels, include_lowest=True, right=False)
    return banded.astype("object").where(banded.notna(), default).astype(str)


def _reliability_feature_frame(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    viability = build_candidate_viability_frame(candidate_rows)
    features = pd.DataFrame(index=viability.index)
    features["selection_prob"] = pd.to_numeric(viability["selection_prob"], errors="coerce").fillna(0.5).clip(lower=0.001, upper=0.999)
    features["selection_odds"] = pd.to_numeric(viability["selection_odds"], errors="coerce")
    features["selection"] = viability["selection"].fillna("unknown").astype(str)
    features["league_code"] = viability["league_code"].fillna("unknown").astype(str)
    features["prob_band"] = _band_labels(
        features["selection_prob"],
        [0.0, 0.35, 0.45, 0.55, 0.65, 0.75, 1.01],
        ["p00_35", "p35_45", "p45_55", "p55_65", "p65_75", "p75_100"],
        "p_unknown",
    )
    features["odds_band"] = _band_labels(
        features["selection_odds"],
        [0.0, 1.5, 2.0, 3.0, 6.0, np.inf],
        ["lt_1_5", "1_5_2", "2_3", "3_6", "6_plus"],
        "odds_unknown",
    )
    features["bet_wins"] = pd.to_numeric(viability["bet_wins"], errors="coerce")
    return features


def _group_key_frame(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series("__global__", index=frame.index, dtype=object)
    return frame[columns].fillna("unknown").astype(str).agg("||".join, axis=1)


def _fit_reliability_model(training_rows: pd.DataFrame, scorer_name: str) -> "DecisionRegionModel | None":
    features = _reliability_feature_frame(training_rows)
    target = pd.to_numeric(features["bet_wins"], errors="coerce")
    valid = target.notna()
    if not bool(valid.any()):
        return None
    training = features.loc[valid].copy()
    global_rate = float(target.loc[valid].mean())
    global_rate = float(np.clip(global_rate, 0.05, 0.95))
    level_specs = [
        ("selection_odds_prob", ["selection", "odds_band", "prob_band"], 30, 36.0),
        ("selection_prob", ["selection", "prob_band"], 24, 30.0),
        ("odds_prob", ["odds_band", "prob_band"], 24, 24.0),
        ("selection", ["selection"], 18, 18.0),
        ("odds_band", ["odds_band"], 18, 14.0),
    ]
    levels: list[dict[str, Any]] = []
    for level_name, columns, min_rows, prior_strength in level_specs:
        keys = _group_key_frame(training, columns)
        groups: dict[str, dict[str, float]] = {}
        for key, group in training.groupby(keys, observed=True):
            rows = int(len(group))
            wins = float(pd.to_numeric(group["bet_wins"], errors="coerce").fillna(0.0).sum())
            mean_prob = float(pd.to_numeric(group["selection_prob"], errors="coerce").mean())
            posterior = (wins + (global_rate * prior_strength)) / (rows + prior_strength)
            groups[str(key)] = {
                "rows": rows,
                "wins": wins,
                "mean_prob": mean_prob,
                "posterior_win_rate": float(np.clip(posterior, 0.001, 0.999)),
            }
        levels.append(
            {
                "name": level_name,
                "columns": columns,
                "min_rows": int(min_rows),
                "prior_strength": float(prior_strength),
                "groups": groups,
            }
        )
    return DecisionRegionModel(
        scorer_name=str(scorer_name),
        pipeline=None,
        numeric_columns=[],
        categorical_columns=[],
        reliability_tables={"global_rate": global_rate, "levels": levels},
    )


def _predict_reliability_probabilities(frame: pd.DataFrame, reliability_tables: dict[str, Any]) -> np.ndarray:
    features = _reliability_feature_frame(frame)
    predictions = pd.Series(float(reliability_tables.get("global_rate", 0.5)), index=features.index, dtype=float)
    unresolved = pd.Series(True, index=features.index, dtype=bool)
    for level in reliability_tables.get("levels", []):
        if not bool(unresolved.any()):
            break
        columns = list(level.get("columns", []))
        groups = dict(level.get("groups", {}))
        min_rows = int(level.get("min_rows", 1))
        keys = _group_key_frame(features, columns)
        for index, key in keys[unresolved].items():
            stats = groups.get(str(key))
            if not stats or int(stats.get("rows", 0)) < min_rows:
                continue
            predictions.loc[index] = float(stats.get("posterior_win_rate", predictions.loc[index]))
            unresolved.loc[index] = False
    return predictions.clip(lower=0.001, upper=0.999).to_numpy(dtype=float)


def _attach_decision_region_outputs(frame: pd.DataFrame, scorer_name: str) -> pd.DataFrame:
    output = frame.copy()
    model_prob = pd.to_numeric(output["decision_region_model_prob"], errors="coerce").fillna(0.5).clip(lower=0.001, upper=0.999)
    confidence = _resolve_float_series(
        output,
        ("selection_confidence_score", "policy_confidence_score", "confidence_score_v2", "confidence_score"),
        default=0.5,
    ).fillna(0.5)
    selection_ev = _resolve_float_series(output, ("selection_ev", "policy_ev", "ev_raw"), default=0.0).fillna(0.0)
    output["decision_region_model_score"] = model_prob * selection_ev * confidence

    if not _adjusts_policy_probability(scorer_name):
        return output

    base_prob = _resolve_float_series(output, ("policy_prob", "selection_prob", "model_prob_raw"), default=0.5).fillna(0.5)
    conservative_prob = pd.Series(
        np.minimum(base_prob.clip(lower=0.001, upper=0.999).to_numpy(dtype=float), model_prob.to_numpy(dtype=float)),
        index=output.index,
        dtype=float,
    )
    quoted_odds = _resolve_float_series(output, ("quoted_odds", "selection_odds"), default=np.nan)
    top_ask = _resolve_float_series(output, ("top_ask",), default=np.nan)
    if top_ask.isna().all():
        top_ask = (1.0 / quoted_odds.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    base_edge = _resolve_float_series(output, ("policy_edge", "selection_edge", "edge_raw"), default=np.nan)
    base_ev = _resolve_float_series(output, ("policy_ev", "selection_ev", "ev_raw"), default=np.nan)
    adjusted_edge = (conservative_prob - top_ask).where(top_ask.notna(), base_edge)
    adjusted_ev = ((conservative_prob * quoted_odds) - 1.0).where(quoted_odds.notna() & quoted_odds.gt(0.0), base_ev)

    output["decision_adjusted_prob"] = conservative_prob.clip(lower=0.001, upper=0.999)
    output["decision_adjusted_edge"] = adjusted_edge
    output["decision_adjusted_ev"] = adjusted_ev
    output["decision_probability_adjustment"] = "conservative_model_prob"
    output["decision_probability_adjustment_delta"] = output["decision_adjusted_prob"] - base_prob
    return output


@dataclass
class DecisionRegionModel:
    scorer_name: str
    pipeline: Pipeline | None
    numeric_columns: list[str]
    categorical_columns: list[str]
    reliability_tables: dict[str, Any] | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.reliability_tables is not None:
            return _predict_reliability_probabilities(frame, self.reliability_tables)
        if self.pipeline is None:
            return np.full(len(frame), 0.5, dtype=float)
        probabilities = self.pipeline.predict_proba(frame)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            return np.full(len(frame), 0.5, dtype=float)
        return probabilities[:, 1]


def fit_decision_region_model(training_rows: pd.DataFrame, scorer_name: str) -> DecisionRegionModel | None:
    viability = build_candidate_viability_frame(training_rows)
    target = pd.to_numeric(viability["bet_wins"], errors="coerce").dropna()
    if target.empty:
        return None
    if str(scorer_name) == DECISION_SCORER_RELIABILITY_PROB:
        return _fit_reliability_model(training_rows, scorer_name)
    if target.nunique() < 2:
        return None

    numeric_columns = [
        column
        for column in DECISION_SCORER_FEATURES_NUMERIC
        if column in viability.columns and pd.to_numeric(viability[column], errors="coerce").notna().any()
    ]
    categorical_columns = [
        column
        for column in DECISION_SCORER_FEATURES_CATEGORICAL
        if column in viability.columns and viability[column].notna().any()
    ]
    features = viability.loc[target.index, [*numeric_columns, *categorical_columns]].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _dense_one_hot_encoder())]), categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    if _base_scorer_name(scorer_name) == DECISION_SCORER_HGB:
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=150,
            min_samples_leaf=10,
            random_state=42,
        )
    else:
        estimator = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )

    pipeline = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    pipeline.fit(features, target.to_numpy(dtype=int))
    return DecisionRegionModel(
        scorer_name=str(scorer_name),
        pipeline=pipeline,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def score_candidates(scored_candidates: pd.DataFrame, model: DecisionRegionModel | None, scorer_name: str | None = None) -> pd.DataFrame:
    output = scored_candidates.copy()
    scorer_key = str(scorer_name or (model.scorer_name if model is not None else DECISION_SCORER_HEURISTIC))
    if model is None or output.empty:
        output["decision_region_model_prob"] = output.get("policy_prob", pd.Series(0.5, index=output.index)).fillna(0.5).astype(float)
    else:
        if model.reliability_tables is not None:
            output["decision_region_model_prob"] = model.predict_proba(output)
        else:
            viability = build_candidate_viability_frame(output)
            design = viability[[*model.numeric_columns, *model.categorical_columns]].copy()
            output["decision_region_model_prob"] = model.predict_proba(design)
    return _attach_decision_region_outputs(output, scorer_key)


def crossfit_decision_region_scores(
    oof_candidates: pd.DataFrame,
    dev_candidates: pd.DataFrame,
    holdout_candidates: pd.DataFrame,
    *,
    scorer_name: str,
    training_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    oof_scored = oof_candidates.copy()
    dev_scored = dev_candidates.copy()
    holdout_scored = holdout_candidates.copy()
    diagnostics: dict[str, Any] = {
        "decision_scorer": str(scorer_name),
        "decision_training_scope": str(training_scope),
        "crossfit_folds": [],
    }

    if str(scorer_name) == DECISION_SCORER_HEURISTIC:
        for frame in (oof_scored, dev_scored, holdout_scored):
            frame["decision_region_scorer"] = str(scorer_name)
            frame["decision_region_training_scope"] = str(training_scope)
        diagnostics["model_fitted"] = False
        diagnostics["reason"] = "heuristic_passthrough"
        return oof_scored, dev_scored, holdout_scored, diagnostics

    fold_series = pd.to_numeric(oof_scored.get("retro_fold_id"), errors="coerce")
    fold_ids = sorted(fold_series.dropna().astype(int).unique().tolist())
    oof_probs = pd.Series(np.nan, index=oof_scored.index, dtype=float)
    for fold_id in fold_ids:
        train_subset = training_subset(oof_scored.loc[fold_series.lt(fold_id)].copy(), training_scope)
        test_mask = fold_series.eq(fold_id)
        model = fit_decision_region_model(train_subset, scorer_name)
        scored = score_candidates(oof_scored.loc[test_mask].copy(), model, scorer_name=scorer_name)
        oof_probs.loc[test_mask] = pd.to_numeric(scored["decision_region_model_prob"], errors="coerce")
        diagnostics["crossfit_folds"].append(
            {
                "fold_id": int(fold_id),
                "train_rows": int(len(train_subset)),
                "test_rows": int(test_mask.sum()),
                "model_fitted": bool(model is not None),
            }
        )

    full_training = training_subset(oof_scored, training_scope)
    full_model = fit_decision_region_model(full_training, scorer_name)
    dev_scored = score_candidates(dev_scored, full_model, scorer_name=scorer_name)
    holdout_scored = score_candidates(holdout_scored, full_model, scorer_name=scorer_name)
    oof_scored["decision_region_model_prob"] = oof_probs.fillna(_resolve_float_series(oof_scored, ("policy_prob",), default=0.5)).astype(float)
    oof_scored = _attach_decision_region_outputs(oof_scored, scorer_name)
    for frame in (oof_scored, dev_scored, holdout_scored):
        frame["decision_region_scorer"] = str(scorer_name)
        frame["decision_region_training_scope"] = str(training_scope)
    diagnostics["model_fitted"] = bool(full_model is not None)
    diagnostics["full_training_rows"] = int(len(full_training))
    diagnostics["crossfit_training_direction"] = "past_folds_only"
    diagnostics["probability_adjustment_mode"] = (
        "conservative_model_prob" if _adjusts_policy_probability(scorer_name) else "rank_score_only"
    )
    diagnostics["oof_prob_mean"] = float(pd.to_numeric(oof_scored["decision_region_model_prob"], errors="coerce").mean())
    diagnostics["dev_prob_mean"] = float(pd.to_numeric(dev_scored["decision_region_model_prob"], errors="coerce").mean()) if not dev_scored.empty else 0.0
    diagnostics["holdout_prob_mean"] = float(pd.to_numeric(holdout_scored["decision_region_model_prob"], errors="coerce").mean()) if not holdout_scored.empty else 0.0
    if _adjusts_policy_probability(scorer_name):
        diagnostics["oof_adjusted_prob_mean"] = float(pd.to_numeric(oof_scored.get("decision_adjusted_prob"), errors="coerce").mean())
        diagnostics["oof_adjusted_delta_mean"] = float(pd.to_numeric(oof_scored.get("decision_probability_adjustment_delta"), errors="coerce").mean())
    return oof_scored, dev_scored, holdout_scored, diagnostics
