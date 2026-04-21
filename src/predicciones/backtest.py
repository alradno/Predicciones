from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from .config import BacktestConfig
from .contracts import BacktestResult, FoldArtifacts, FoldWindow, OUTCOME_ORDER, RunContext
from .dataset import model_feature_columns
from .evaluation import ExecutionSummary, PolicySummary, SignalSummary
from .models import (
    OutcomeCalibrator,
    build_goal_model,
    fit_dixon_coles_rho,
    fit_goal_model,
    multiclass_brier_score,
    outcome_probabilities_from_lambdas,
    predicted_outcomes,
)
from .reporting import build_backtest_truth_summary
from .strategy import BetPolicy, add_edge_columns, optimize_policy, select_bets, summarize_bets


def _next_available_date(unique_dates: list[pd.Timestamp], threshold: pd.Timestamp) -> pd.Timestamp | None:
    for value in unique_dates:
        if value >= threshold:
            return value
    return None


def rolling_origin_windows(dataset: pd.DataFrame, config: BacktestConfig) -> list[FoldWindow]:
    ordered = dataset.sort_values(["Date", "match_id"]).reset_index(drop=True)
    unique_dates = sorted(pd.to_datetime(ordered["Date"]).dt.normalize().unique().tolist())
    min_date = pd.Timestamp(unique_dates[0])
    current_start = _next_available_date(unique_dates, min_date + pd.Timedelta(days=config.min_train_days))
    fold_id = 1
    windows: list[FoldWindow] = []

    while current_start is not None:
        train_mask = ordered["Date"] < current_start
        if int(train_mask.sum()) < config.min_train_matches:
            current_start = _next_available_date(unique_dates, current_start + pd.Timedelta(days=1))
            continue

        test_end_exclusive = current_start + pd.Timedelta(days=config.test_window_days)
        test_mask = (ordered["Date"] >= current_start) & (ordered["Date"] < test_end_exclusive)
        if int(test_mask.sum()) == 0:
            current_start = _next_available_date(unique_dates, current_start + pd.Timedelta(days=1))
            continue

        train_dates = ordered.loc[train_mask, "Date"]
        test_dates = ordered.loc[test_mask, "Date"]
        windows.append(
            FoldWindow(
                fold_id=fold_id,
                train_start=pd.Timestamp(train_dates.min()),
                train_end=pd.Timestamp(train_dates.max()),
                test_start=pd.Timestamp(test_dates.min()),
                test_end=pd.Timestamp(test_dates.max()),
                train_rows=int(train_mask.sum()),
                test_rows=int(test_mask.sum()),
            )
        )

        fold_id += 1
        current_start = _next_available_date(unique_dates, test_end_exclusive)

    return windows


def _temporal_subsets(train_rows: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = train_rows.sort_values(["Date", "match_id"]).reset_index(drop=True)
    total_rows = len(ordered)
    min_holdout = max(10, total_rows // 12)
    policy_rows = min(config.policy_matches, max(min_holdout, total_rows // 6))
    calibration_rows = min(config.calibration_matches, max(min_holdout, total_rows // 6))
    min_subtrain = min(config.min_subtrain_matches, max(30, total_rows - (2 * min_holdout)))

    while (total_rows - policy_rows - calibration_rows) < min_subtrain:
        if policy_rows > min_holdout:
            policy_rows -= 1
            continue
        if calibration_rows > min_holdout:
            calibration_rows -= 1
            continue
        break

    calibration_anchor = ordered.iloc[-(policy_rows + calibration_rows)]["Date"]
    policy_anchor = ordered.iloc[-policy_rows]["Date"]
    subtrain = ordered[ordered["Date"] < calibration_anchor].copy()
    calibration = ordered[(ordered["Date"] >= calibration_anchor) & (ordered["Date"] < policy_anchor)].copy()
    policy = ordered[ordered["Date"] >= policy_anchor].copy()

    if subtrain.empty or calibration.empty or policy.empty:
        raise ValueError("No hay suficientes datos para separar subtrain, calibracion y policy sin leakage.")

    return subtrain, calibration, policy


def _favorite_outcomes(probabilities: np.ndarray) -> np.ndarray:
    return np.take(OUTCOME_ORDER, probabilities.argmax(axis=1))


def _favorite_bets(frame: pd.DataFrame, selection_column: str, name: str) -> pd.DataFrame:
    bets = frame[
        ["match_id", "Date", "league_code", "league_name", "season", "HomeTeam", "AwayTeam", "actual_outcome", selection_column]
    ].copy()
    bets = bets.rename(columns={selection_column: "selection"})
    bets["selection_odds"] = [
        bets_row[f"odds_{bets_row['selection']}"]
        for _, bets_row in frame.assign(selection=frame[selection_column]).iterrows()
    ]
    bets["won"] = (bets["selection"] == bets["actual_outcome"]).astype(int)
    bets["flat_stake"] = 1.0
    bets["flat_profit"] = np.where(bets["won"].eq(1), bets["selection_odds"] - 1.0, -1.0)
    bets["strategy_name"] = name
    return bets


def _probability_metrics(actual: np.ndarray, probabilities: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1, 2])),
        "brier_score": multiclass_brier_score(actual, probabilities),
    }


def run_backtest(dataset: pd.DataFrame, config: BacktestConfig, run: RunContext) -> BacktestResult:
    feature_columns = model_feature_columns(dataset)
    windows = rolling_origin_windows(dataset, config)
    fold_results: list[FoldArtifacts] = []
    all_predictions: list[pd.DataFrame] = []
    all_bets: list[pd.DataFrame] = []

    for window in windows:
        train = dataset[dataset["Date"] < window.test_start].copy()
        test = dataset[(dataset["Date"] >= window.test_start) & (dataset["Date"] <= window.test_end)].copy()
        subtrain, calibration, policy_rows = _temporal_subsets(train, config)

        model = build_goal_model(subtrain[feature_columns])
        fit_goal_model(model, subtrain[feature_columns], subtrain["home_goals"], subtrain["away_goals"])

        calibration_home_lambda, calibration_away_lambda = model.predict_lambdas(calibration[feature_columns])
        rho = fit_dixon_coles_rho(
            calibration_home_lambda,
            calibration_away_lambda,
            calibration["home_goals"].to_numpy(),
            calibration["away_goals"].to_numpy(),
            config.dixon_coles_bounds,
        )
        calibration_raw = outcome_probabilities_from_lambdas(
            calibration_home_lambda,
            calibration_away_lambda,
            rho=rho,
            max_goals=config.max_poisson_goals,
        )
        calibrator = OutcomeCalibrator(epsilon=config.calibrator_epsilon).fit(calibration_raw, calibration["target"].to_numpy())

        policy_home_lambda, policy_away_lambda = model.predict_lambdas(policy_rows[feature_columns])
        policy_raw = outcome_probabilities_from_lambdas(
            policy_home_lambda,
            policy_away_lambda,
            rho=rho,
            max_goals=config.max_poisson_goals,
        )
        policy_calibrated = calibrator.transform(policy_raw)
        policy_frame = _build_prediction_frame(policy_rows, policy_home_lambda, policy_away_lambda, policy_raw, policy_calibrated)
        policy_frame = add_edge_columns(policy_frame)
        tuned_policy = optimize_policy(
            policy_frame,
            edge_thresholds=config.policy_search.edge_thresholds,
            ev_thresholds=config.policy_search.ev_thresholds,
            min_odds_options=config.policy_search.min_odds_options,
            max_odds_options=config.policy_search.max_odds_options,
            min_bets=config.policy_search.min_bets,
            kelly_fraction=config.policy_search.max_kelly_fraction,
        )

        final_model = build_goal_model(train[feature_columns])
        fit_goal_model(final_model, train[feature_columns], train["home_goals"], train["away_goals"])
        test_home_lambda, test_away_lambda = final_model.predict_lambdas(test[feature_columns])
        test_raw = outcome_probabilities_from_lambdas(
            test_home_lambda,
            test_away_lambda,
            rho=rho,
            max_goals=config.max_poisson_goals,
        )
        test_calibrated = calibrator.transform(test_raw)
        prediction_frame = _build_prediction_frame(test, test_home_lambda, test_away_lambda, test_raw, test_calibrated)
        prediction_frame["fold_id"] = window.fold_id
        prediction_frame["rho"] = rho
        prediction_frame["baseline_prediction"] = _favorite_outcomes(
            prediction_frame[[f"market_prob_{outcome}" for outcome in OUTCOME_ORDER]].to_numpy()
        )
        prediction_frame["raw_prediction"] = _favorite_outcomes(test_raw)
        prediction_frame["calibrated_prediction"] = _favorite_outcomes(test_calibrated)
        prediction_frame = add_edge_columns(prediction_frame)

        strategy_bets = select_bets(prediction_frame, tuned_policy)
        strategy_bets["strategy_name"] = "edge_policy"
        strategy_bets["fold_id"] = window.fold_id
        baseline_bets = _favorite_bets(prediction_frame, "baseline_prediction", "baseline_favorite")
        baseline_bets["fold_id"] = window.fold_id
        raw_bets = _favorite_bets(prediction_frame, "raw_prediction", "goal_model_favorite")
        raw_bets["fold_id"] = window.fold_id

        fold_metrics = {
            "baseline": {
                **_probability_metrics(
                    test["target"].to_numpy(),
                    prediction_frame[[f"market_prob_{outcome}" for outcome in OUTCOME_ORDER]].to_numpy(),
                    np.array([OUTCOME_ORDER.index(value) for value in prediction_frame["baseline_prediction"]]),
                ),
                **summarize_bets(baseline_bets),
            },
            "model_raw": {
                **_probability_metrics(test["target"].to_numpy(), test_raw, predicted_outcomes(test_raw)),
                **summarize_bets(raw_bets),
            },
            "model_calibrated": _probability_metrics(test["target"].to_numpy(), test_calibrated, predicted_outcomes(test_calibrated)),
            "strategy": summarize_bets(strategy_bets),
            "feature_sanitization": {
                "subtrain_model": model.feature_sanitization_report(),
                "full_train_model": final_model.feature_sanitization_report(),
            },
        }

        fold_results.append(
            FoldArtifacts(
                window=window,
                predictions=prediction_frame,
                bets=pd.concat([baseline_bets, raw_bets, strategy_bets], ignore_index=True, sort=False),
                metrics=fold_metrics,
                policy=tuned_policy.to_dict(),
            )
        )
        all_predictions.append(prediction_frame)
        all_bets.append(pd.concat([baseline_bets, raw_bets, strategy_bets], ignore_index=True, sort=False))

    predictions = pd.concat(all_predictions, ignore_index=True)
    bets = pd.concat(all_bets, ignore_index=True, sort=False)

    baseline_probs = predictions[[f"market_prob_{outcome}" for outcome in OUTCOME_ORDER]].to_numpy()
    raw_probs = predictions[[f"prob_{outcome}_raw" for outcome in OUTCOME_ORDER]].to_numpy()
    calibrated_probs = predictions[[f"prob_{outcome}_calibrated" for outcome in OUTCOME_ORDER]].to_numpy()
    actual = predictions["actual_target"].to_numpy()

    summary = {
        "folds": len(fold_results),
        "rows_scored": int(len(predictions)),
        "baseline": {
            **_probability_metrics(actual, baseline_probs, np.array([OUTCOME_ORDER.index(value) for value in predictions["baseline_prediction"]])),
            **summarize_bets(bets[bets["strategy_name"] == "baseline_favorite"]),
        },
        "model_raw": {
            **_probability_metrics(actual, raw_probs, predicted_outcomes(raw_probs)),
            **summarize_bets(bets[bets["strategy_name"] == "goal_model_favorite"]),
        },
        "model_calibrated": _probability_metrics(actual, calibrated_probs, predicted_outcomes(calibrated_probs)),
        "strategy": {
            **summarize_bets(bets[bets["strategy_name"] == "edge_policy"]),
            "kelly": summarize_bets(bets[bets["strategy_name"] == "edge_policy"], profit_column="kelly_profit"),
        },
        "fold_metrics": [
            {
                "window": asdict(result.window),
                "policy": result.policy,
                "metrics": result.metrics,
            }
            for result in fold_results
        ],
        "feature_sanitization": {
            "by_fold": [
                {
                    "fold_id": result.window.fold_id,
                    **result.metrics.get("feature_sanitization", {}),
                }
                for result in fold_results
            ],
            "latest_full_train_model": (
                fold_results[-1].metrics.get("feature_sanitization", {}).get("full_train_model", {}) if fold_results else {}
            ),
        },
    }
    summary["honest_diagnostics"] = build_backtest_truth_summary(summary)
    summary["signal_summary"] = SignalSummary(
        selected_probability_source=summary["honest_diagnostics"].get("selected_probability_source"),
        baseline_metrics=summary["baseline"],
        raw_metrics=summary["model_raw"],
        calibrated_metrics=summary["model_calibrated"],
        feature_sanitization=summary.get("feature_sanitization", {}),
        verdict=summary["honest_diagnostics"].get("overall_verdict"),
        extra={
            "rows_scored": int(len(predictions)),
            "folds": len(fold_results),
        },
    ).to_dict()
    summary["policy_summary"] = PolicySummary(
        selected_policy=fold_results[-1].policy if fold_results else {},
        candidate_policies=[result.policy for result in fold_results],
        holdout_metrics=summary["strategy"],
        objective={
            "name": "roi_drawdown",
            "mode": "flat_stake_backtest",
        },
        verdict=summary["honest_diagnostics"].get("overall_verdict"),
    ).to_dict()
    summary["execution_summary"] = ExecutionSummary(
        coverage={
            "mode": "flat_stake_backtest",
            "executed_bets": int(summary["strategy"].get("bets", 0)),
            "kelly_bets": int(summary["strategy"].get("kelly", {}).get("bets", 0)),
        },
        fills={
            "strategy": summary["strategy"],
            "baseline": summary["baseline"],
            "raw": summary["model_raw"],
        },
        decisions={
            "status": "simulated_flat_stake_only",
        },
        verdict="not_applicable",
    ).to_dict()

    return BacktestResult(
        run=run,
        predictions=predictions,
        bets=bets,
        fold_results=fold_results,
        summary=summary,
        artifacts={},
    )


def _build_prediction_frame(
    rows: pd.DataFrame,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
) -> pd.DataFrame:
    frame = rows[
        [
            "match_id",
            "Date",
            "league_code",
            "league_name",
            "season",
            "HomeTeam",
            "AwayTeam",
            "outcome",
            "target",
            "home_goals",
            "away_goals",
            "odds_home",
            "odds_draw",
            "odds_away",
            "market_prob_home",
            "market_prob_draw",
            "market_prob_away",
        ]
    ].copy()
    frame["actual_outcome"] = frame["outcome"]
    frame["actual_target"] = frame["target"].astype(int)
    frame["expected_goals_home"] = lambda_home
    frame["expected_goals_away"] = lambda_away
    for index, outcome in enumerate(OUTCOME_ORDER):
        frame[f"prob_{outcome}_raw"] = raw_probabilities[:, index]
        frame[f"prob_{outcome}_calibrated"] = calibrated_probabilities[:, index]
    return frame
