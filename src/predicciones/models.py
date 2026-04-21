from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .contracts import OUTCOME_ORDER
from .dataset import sanitize_model_feature_columns


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    truth = np.zeros((len(y_true), len(OUTCOME_ORDER)))
    truth[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


@dataclass
class GoalModelBundle:
    home_model: Pipeline
    away_model: Pipeline
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    dropped_feature_columns: list[str]
    dropped_feature_reasons: dict[str, str]

    def predict_lambdas(self, feature_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = feature_rows.loc[:, self.feature_columns]
        home = np.clip(self.home_model.predict(X), 0.05, None)
        away = np.clip(self.away_model.predict(X), 0.05, None)
        return home.astype(float), away.astype(float)

    def feature_sanitization_report(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "categorical_columns": list(self.categorical_columns),
            "numeric_columns": list(self.numeric_columns),
            "dropped_feature_columns": list(self.dropped_feature_columns),
            "dropped_feature_reasons": dict(self.dropped_feature_reasons),
        }


class OutcomeCalibrator:
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.models: dict[str, IsotonicRegression | None] = {}

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "OutcomeCalibrator":
        for index, outcome in enumerate(OUTCOME_ORDER):
            target = (y_true == index).astype(int)
            if target.min() == target.max():
                self.models[outcome] = None
                continue
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(probabilities[:, index], target)
            self.models[outcome] = model
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = np.zeros_like(probabilities, dtype=float)
        for index, outcome in enumerate(OUTCOME_ORDER):
            model = self.models.get(outcome)
            if model is None:
                calibrated[:, index] = probabilities[:, index]
            else:
                calibrated[:, index] = model.predict(probabilities[:, index])
        calibrated = np.clip(calibrated, self.epsilon, None)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return calibrated


def build_goal_model(feature_rows: pd.DataFrame) -> GoalModelBundle:
    sanitization = sanitize_model_feature_columns(feature_rows)
    feature_columns = sanitization.feature_columns
    categorical_columns = sanitization.categorical_columns
    numeric_columns = sanitization.numeric_columns
    if not feature_columns:
        raise ValueError("No usable features remain after sanitizing empty columns.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if categorical_columns:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_transformer, categorical_columns))
    if numeric_columns:
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("numeric", numeric_transformer, numeric_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    home_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=350,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    )
    away_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=350,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=43,
                ),
            ),
        ]
    )
    return GoalModelBundle(
        home_model=home_pipeline,
        away_model=away_pipeline,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        dropped_feature_columns=sanitization.dropped_feature_columns,
        dropped_feature_reasons=sanitization.dropped_feature_reasons,
    )


def fit_goal_model(
    model: GoalModelBundle,
    feature_rows: pd.DataFrame,
    home_goals: pd.Series,
    away_goals: pd.Series,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> GoalModelBundle:
    X = feature_rows[model.feature_columns]
    fit_kwargs: dict[str, np.ndarray] = {}
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        fit_kwargs["model__sample_weight"] = weights
    model.home_model.fit(X, home_goals, **fit_kwargs)
    model.away_model.fit(X, away_goals, **fit_kwargs)
    return model


def _dixon_coles_tau(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - (lambda_home * lambda_away * rho)
    if home_goals == 0 and away_goals == 1:
        return 1.0 + (lambda_home * rho)
    if home_goals == 1 and away_goals == 0:
        return 1.0 + (lambda_away * rho)
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def score_matrix_from_lambdas(lambda_home: float, lambda_away: float, rho: float, max_goals: int) -> np.ndarray:
    home_support = np.arange(max_goals + 1)
    away_support = np.arange(max_goals + 1)
    home_probs = poisson.pmf(home_support, lambda_home)
    away_probs = poisson.pmf(away_support, lambda_away)
    home_probs[-1] += max(0.0, 1.0 - home_probs.sum())
    away_probs[-1] += max(0.0, 1.0 - away_probs.sum())
    matrix = np.outer(home_probs, away_probs)

    for home_goals in (0, 1):
        for away_goals in (0, 1):
            matrix[home_goals, away_goals] *= max(
                _dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho),
                1e-6,
            )

    matrix /= matrix.sum()
    return matrix


def outcome_probabilities_from_lambdas(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
    max_goals: int,
) -> np.ndarray:
    probabilities = np.zeros((len(lambda_home), len(OUTCOME_ORDER)), dtype=float)
    for index, (home_rate, away_rate) in enumerate(zip(lambda_home, lambda_away)):
        matrix = score_matrix_from_lambdas(float(home_rate), float(away_rate), rho=rho, max_goals=max_goals)
        probabilities[index, 0] = np.triu(matrix, 1).sum()
        probabilities[index, 1] = np.trace(matrix)
        probabilities[index, 2] = np.tril(matrix, -1).sum()
    return probabilities


def fit_dixon_coles_rho(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    observed_home_goals: np.ndarray,
    observed_away_goals: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    def objective(rho: float) -> float:
        tau = np.array(
            [
                max(_dixon_coles_tau(int(h), int(a), float(lh), float(la), rho), 1e-6)
                for h, a, lh, la in zip(observed_home_goals, observed_away_goals, lambda_home, lambda_away)
            ]
        )
        log_prob = (
            poisson.logpmf(observed_home_goals, lambda_home)
            + poisson.logpmf(observed_away_goals, lambda_away)
            + np.log(tau)
        )
        return float(-np.mean(log_prob))

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x if result.success else 0.0)


def estimate_feature_importance(
    model: GoalModelBundle,
    feature_rows: pd.DataFrame,
    home_goals: pd.Series,
    away_goals: pd.Series,
    max_rows: int = 600,
) -> pd.DataFrame:
    if feature_rows.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    sample = feature_rows.sample(n=min(len(feature_rows), max_rows), random_state=42) if len(feature_rows) > max_rows else feature_rows
    home_target = home_goals.loc[sample.index]
    away_target = away_goals.loc[sample.index]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`",
            category=UserWarning,
        )
        home_result = permutation_importance(
            model.home_model,
            sample[model.feature_columns],
            home_target,
            n_repeats=5,
            random_state=42,
            scoring="neg_mean_poisson_deviance",
        )
        away_result = permutation_importance(
            model.away_model,
            sample[model.feature_columns],
            away_target,
            n_repeats=5,
            random_state=43,
            scoring="neg_mean_poisson_deviance",
        )

    importance = pd.DataFrame(
        {
            "feature": model.feature_columns,
            "importance": (np.abs(home_result.importances_mean) + np.abs(away_result.importances_mean)) / 2.0,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def predicted_outcomes(probabilities: np.ndarray) -> np.ndarray:
    return probabilities.argmax(axis=1)
