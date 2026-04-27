from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .backtest import _build_prediction_frame, _temporal_subsets, rolling_origin_windows
from .config import Settings
from .contracts import NetBacktestResult, NicheTrainingResult, OUTCOME_ORDER, RunContext, ShadowRunResult
from .football.dataset import build_fixture_feature_rows, model_feature_columns
from .execution_quality import (
    build_clv_rows,
    fit_fill_probability_priors,
    summarize_clv,
    summarize_fill_adjusted_ev,
    summarize_sizing,
)
from .ingestion import build_market_odds, canonicalize_matches
from .models import (
    OutcomeCalibrator,
    build_goal_model,
    fit_dixon_coles_rho,
    fit_goal_model,
    outcome_probabilities_from_lambdas,
)
from .reporting import _plot_bank_curve, _plot_calibration, _save_json, create_run_context
from .strategy import BetPolicy
from .research_candidates import (
    _evaluate_policy,
    apply_probability_source_strategy,
    build_candidate_rows,
    choose_probability_source,
    optimize_research_policy,
    select_candidate_bets,
    simulate_execution,
)
from .research_niches import _segment_filter_from_row, discover_niches
from .research_snapshots import _default_kickoff, _prepare_snapshot_frame, build_market_snapshots, load_snapshot_file
from .research_summary import _build_research_summary, load_research_run
from .selection_scoring import (
    INNER_ROLE_DISCOVERY,
    INNER_ROLE_OUTER_HOLDOUT,
    INNER_ROLE_POLICY_TUNE,
    INNER_ROLE_SELECTION_VALIDATE,
    assign_inner_validation_roles,
    conservative_score_breakdown,
)


def _policy_from_payload(payload: dict[str, object], settings: Settings) -> BetPolicy:
    return BetPolicy(
        edge_threshold=float(payload.get("edge_threshold", settings.research.provisional_edge_threshold)),
        ev_threshold=float(payload.get("ev_threshold", settings.research.provisional_ev_threshold)),
        min_odds=float(payload.get("min_odds", min(settings.backtest.policy_search.min_odds_options))),
        max_odds=float(payload.get("max_odds", max(settings.backtest.policy_search.max_odds_options))),
        kelly_fraction=float(payload.get("kelly_fraction", settings.backtest.policy_search.max_kelly_fraction)),
        family=str(payload.get("family", "edge_ev_threshold")),
        top_quantile=float(payload.get("top_quantile", 0.10)),
        allowed_leagues=tuple(payload.get("allowed_leagues", [])),
        allowed_outcomes=tuple(payload.get("allowed_outcomes", [])),
        scope_name=str(payload.get("scope_name", "global_all")),
    )


def _default_policy(settings: Settings) -> BetPolicy:
    return BetPolicy(
        edge_threshold=settings.research.provisional_edge_threshold,
        ev_threshold=settings.research.provisional_ev_threshold,
        min_odds=min(settings.backtest.policy_search.min_odds_options),
        max_odds=max(settings.backtest.policy_search.max_odds_options),
        kelly_fraction=settings.backtest.policy_search.max_kelly_fraction,
    )


def _selection_candidate_rows(candidate_rows: pd.DataFrame, role: str) -> pd.DataFrame:
    if candidate_rows.empty:
        return candidate_rows
    return candidate_rows[candidate_rows["inner_role"].astype(str).eq(role)].copy()


def _evaluate_policy_combinations(
    candidate_rows: pd.DataFrame,
    *,
    policy_ranking: pd.DataFrame,
    niches: pd.DataFrame,
    probability_source: str,
    settings: Settings,
) -> tuple[BetPolicy, dict[str, object] | None, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    candidate_policy_rows = policy_ranking.head(int(settings.research.candidate_top_policies)).copy() if not policy_ranking.empty else pd.DataFrame()
    if candidate_policy_rows.empty:
        candidate_policy_rows = pd.DataFrame(
            [{"policy_payload": _default_policy(settings).to_dict(), "shrunken_roi": 0.0, "score": 0.0}]
        )
    niche_rows = niches.head(int(settings.research.candidate_top_niches)).copy() if not niches.empty else pd.DataFrame()
    niche_payloads: list[dict[str, object] | None] = [None]
    for niche in niche_rows.to_dict(orient="records"):
        niche_payloads.append(niche)

    for policy_row in candidate_policy_rows.to_dict(orient="records"):
        policy = _policy_from_payload(dict(policy_row.get("policy_payload", {})), settings)
        policy_reference_roi = float(policy_row.get("shrunken_roi", policy_row.get("raw_roi", 0.0)) or 0.0)
        for niche_row in niche_payloads:
            segment_filter = _segment_filter_from_row(pd.Series(niche_row)) if niche_row is not None else None
            _, execution_rows, metrics = _evaluate_policy(
                candidate_rows=candidate_rows,
                policy=policy,
                probability_source=probability_source,
                settings=settings,
                allow_proxy=settings.execution.allow_closing_proxy_for_research,
                segment_filter=segment_filter,
            )
            if metrics["executed"] <= 0:
                continue
            fold_roi = execution_rows.loc[execution_rows["execution_status"].astype(str).eq("executed")].groupby("fold_id", observed=True).apply(
                lambda group: float(group["net_profit"].sum() / group["accepted_stake"].sum()) if float(group["accepted_stake"].sum()) > 0 else 0.0
            )
            niche_reference_roi = 0.0 if niche_row is None else float(niche_row.get("shrunken_roi", niche_row.get("raw_roi", 0.0)) or 0.0)
            references = [policy_reference_roi]
            if niche_row is not None:
                references.append(niche_reference_roi)
            reference_roi = float(np.mean(references)) if references else 0.0
            breakdown = conservative_score_breakdown(
                metrics,
                fold_roi=fold_roi,
                reference_roi=reference_roi,
                positive_fold_target=settings.research.positive_fold_ratio,
                prior_bets=settings.research.min_segment_bets,
                drawdown_weight=settings.research.drawdown_weight,
                generalization_gap_weight=settings.research.generalization_gap_weight,
                positive_penalty_weight=settings.research.positive_penalty_weight,
            )
            rows.append(
                {
                    "policy_payload": policy.to_dict(),
                    "filters_json": "" if niche_row is None else str(niche_row.get("filters_json", "")),
                    "dimensions": "global" if niche_row is None else str(niche_row.get("dimensions", "")),
                    "executed": int(metrics["executed"]),
                    "roi": float(metrics["roi"]),
                    "profit": float(metrics["profit"]),
                    "stake": float(metrics["stake"]),
                    **breakdown,
                }
            )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return _default_policy(settings), None, ranking
    ranking = ranking.sort_values(["score", "generalization_gap", "executed"], ascending=[False, True, False]).reset_index(drop=True)
    champion = ranking.iloc[0].to_dict()
    champion_policy = _policy_from_payload(dict(champion["policy_payload"]), settings)
    champion_niche = None
    if str(champion.get("filters_json", "")).strip():
        champion_niche = {"filters_json": str(champion["filters_json"]), "dimensions": str(champion.get("dimensions", ""))}
    return champion_policy, champion_niche, ranking


def run_net_backtest(dataset: pd.DataFrame, snapshots: pd.DataFrame, settings: Settings, run: RunContext) -> NetBacktestResult:
    feature_columns = model_feature_columns(dataset)
    windows = rolling_origin_windows(dataset, settings.backtest)
    if len(windows) <= settings.research.holdout_windows:
        raise ValueError("No hay suficientes folds para separar train y holdout en el backtest neto.")

    holdout_fold_ids = {window.fold_id for window in windows[-settings.research.holdout_windows :]}
    prediction_frames: list[pd.DataFrame] = []

    for window in windows:
        train = dataset[dataset["Date"] < window.test_start].copy()
        test = dataset[(dataset["Date"] >= window.test_start) & (dataset["Date"] <= window.test_end)].copy()
        subtrain, calibration, _ = _temporal_subsets(train, settings.backtest)

        model = build_goal_model(subtrain[feature_columns])
        fit_goal_model(model, subtrain[feature_columns], subtrain["home_goals"], subtrain["away_goals"])

        calibration_home_lambda, calibration_away_lambda = model.predict_lambdas(calibration[feature_columns])
        rho = fit_dixon_coles_rho(
            calibration_home_lambda,
            calibration_away_lambda,
            calibration["home_goals"].to_numpy(),
            calibration["away_goals"].to_numpy(),
            settings.backtest.dixon_coles_bounds,
        )
        calibration_raw = outcome_probabilities_from_lambdas(
            calibration_home_lambda,
            calibration_away_lambda,
            rho=rho,
            max_goals=settings.backtest.max_poisson_goals,
        )
        calibrator = OutcomeCalibrator(epsilon=settings.backtest.calibrator_epsilon).fit(
            calibration_raw,
            calibration["target"].to_numpy(),
        )

        final_model = build_goal_model(train[feature_columns])
        fit_goal_model(final_model, train[feature_columns], train["home_goals"], train["away_goals"])
        test_home_lambda, test_away_lambda = final_model.predict_lambdas(test[feature_columns])
        test_raw = outcome_probabilities_from_lambdas(
            test_home_lambda,
            test_away_lambda,
            rho=rho,
            max_goals=settings.backtest.max_poisson_goals,
        )
        test_calibrated = calibrator.transform(test_raw)
        prediction_frame = _build_prediction_frame(test, test_home_lambda, test_away_lambda, test_raw, test_calibrated)
        prediction_frame["fold_id"] = window.fold_id
        prediction_frame["rho"] = rho
        prediction_frame["fold_segment"] = np.where(prediction_frame["fold_id"].isin(holdout_fold_ids), "holdout", "train")
        prediction_frame["baseline_prediction"] = np.take(
            OUTCOME_ORDER,
            prediction_frame[[f"market_prob_{outcome}" for outcome in OUTCOME_ORDER]].to_numpy().argmax(axis=1),
        )
        prediction_frame["raw_prediction"] = np.take(OUTCOME_ORDER, test_raw.argmax(axis=1))
        prediction_frame["calibrated_prediction"] = np.take(OUTCOME_ORDER, test_calibrated.argmax(axis=1))
        prediction_frame["kickoff_time"] = _default_kickoff(prediction_frame["Date"], settings.snapshot.default_kickoff_hour)
        prediction_frames.append(prediction_frame)

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["Date", "match_id"]).reset_index(drop=True)
    predictions, inner_fold_assignments = assign_inner_validation_roles(predictions)
    candidate_rows = build_candidate_rows(predictions, snapshots, settings)
    selected_probability_source, probability_decision = choose_probability_source(predictions, candidate_rows, settings)
    strategy_probability_source = str(probability_decision.get("decision", selected_probability_source))
    predictions, candidate_rows = apply_probability_source_strategy(predictions, candidate_rows, probability_decision, settings)

    discovery_rows = _selection_candidate_rows(candidate_rows, INNER_ROLE_DISCOVERY)
    policy_tune_rows = _selection_candidate_rows(candidate_rows, INNER_ROLE_POLICY_TUNE)
    selection_validate_rows = _selection_candidate_rows(candidate_rows, INNER_ROLE_SELECTION_VALIDATE)
    holdout_rows = candidate_rows[candidate_rows["inner_role"].astype(str).eq(INNER_ROLE_OUTER_HOLDOUT)].copy()
    provisional_policy = _default_policy(settings)
    _, discovery_execution, _ = _evaluate_policy(
        discovery_rows,
        policy=provisional_policy,
        probability_source=strategy_probability_source,
        settings=settings,
        allow_proxy=settings.execution.allow_closing_proxy_for_research,
    )
    niches = discover_niches(discovery_execution, settings, top_n=settings.research.candidate_top_niches)
    tuned_policy, policy_ranking = optimize_research_policy(
        policy_tune_rows,
        probability_source=strategy_probability_source,
        settings=settings,
        return_ranking=True,
        top_k=settings.research.candidate_top_policies,
    )
    champion_policy, champion_niche_row, combination_ranking = _evaluate_policy_combinations(
        selection_validate_rows,
        policy_ranking=policy_ranking,
        niches=niches,
        probability_source=strategy_probability_source,
        settings=settings,
    )
    if champion_policy is None:
        champion_policy = tuned_policy
    top_niche_filter = json.loads(str(champion_niche_row["filters_json"])) if champion_niche_row is not None else None

    _, training_execution, _ = _evaluate_policy(
        candidate_rows[candidate_rows["fold_segment"].astype(str) == "train"],
        policy=champion_policy,
        probability_source=strategy_probability_source,
        settings=settings,
        allow_proxy=settings.execution.allow_closing_proxy_for_research,
    )
    _, holdout_execution, _ = _evaluate_policy(
        holdout_rows,
        policy=champion_policy,
        probability_source=strategy_probability_source,
        settings=settings,
        allow_proxy=settings.execution.allow_closing_proxy_for_research,
    )
    _, training_niche_execution, _ = _evaluate_policy(
        candidate_rows[candidate_rows["fold_segment"].astype(str) == "train"],
        policy=champion_policy,
        probability_source=strategy_probability_source,
        settings=settings,
        allow_proxy=settings.execution.allow_closing_proxy_for_research,
        segment_filter=top_niche_filter,
    )
    _, holdout_niche_execution, _ = _evaluate_policy(
        holdout_rows,
        policy=champion_policy,
        probability_source=strategy_probability_source,
        settings=settings,
        allow_proxy=settings.execution.allow_closing_proxy_for_research,
        segment_filter=top_niche_filter,
    )

    execution_rows = pd.concat([training_execution, holdout_execution], ignore_index=True, sort=False)
    if not execution_rows.empty:
        matched_niche_ids = set(training_niche_execution["match_id"].tolist() + holdout_niche_execution["match_id"].tolist())
        execution_rows["top_niche_match"] = execution_rows["match_id"].isin(matched_niche_ids)
    net_bet_rows = execution_rows[execution_rows["execution_status"] == "executed"].copy().reset_index(drop=True)
    clv_rows = build_clv_rows(
        execution_rows,
        executed_odds_column="executed_odds",
        closing_reference_odds_column="closing_reference_odds",
        closing_reference_prob_column="closing_reference_prob",
        source_column="clv_source",
    )
    fill_model_summary = fit_fill_probability_priors(
        execution_rows.assign(fill_rate=pd.to_numeric(execution_rows.get("fill_rate", 0.0), errors="coerce").fillna(0.0))
        if not execution_rows.empty
        else execution_rows
    )
    execution_quality = {
        "clv_summary": summarize_clv(clv_rows, total_rows=int(len(execution_rows))),
        "fill_model_summary": fill_model_summary,
        "fill_adjusted_ev_summary": summarize_fill_adjusted_ev(execution_rows),
        "sizing_summary": summarize_sizing(execution_rows),
    }

    summary = _build_research_summary(
        predictions=predictions,
        training_execution=training_execution,
        holdout_execution=holdout_execution,
        training_niche_execution=training_niche_execution,
        holdout_niche_execution=holdout_niche_execution,
        probability_decision=probability_decision,
        selected_probability_source=strategy_probability_source,
        selected_policy=champion_policy,
        niches=niches,
        settings=settings,
        inner_fold_assignments=inner_fold_assignments,
        policy_ranking=policy_ranking,
        combination_ranking=combination_ranking,
        execution_quality=execution_quality,
        top_niche_filter=top_niche_filter,
    )
    artifacts = _save_net_backtest_artifacts(
        run=run,
        predictions=predictions,
        candidate_rows=candidate_rows,
        execution_rows=execution_rows,
        net_bet_rows=net_bet_rows,
        niches=niches,
        inner_fold_assignments=inner_fold_assignments,
        policy_ranking=policy_ranking,
        combination_ranking=combination_ranking,
        clv_rows=clv_rows,
        execution_quality=execution_quality,
        summary=summary,
    )
    return NetBacktestResult(
        run=run,
        predictions=predictions,
        candidate_rows=candidate_rows,
        execution_rows=execution_rows,
        net_bet_rows=net_bet_rows,
        niches=niches,
        summary=summary,
        artifacts=artifacts,
    )


def _save_net_backtest_artifacts(
    run: RunContext,
    predictions: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    execution_rows: pd.DataFrame,
    net_bet_rows: pd.DataFrame,
    niches: pd.DataFrame,
    inner_fold_assignments: pd.DataFrame,
    policy_ranking: pd.DataFrame,
    combination_ranking: pd.DataFrame,
    clv_rows: pd.DataFrame,
    execution_quality: dict[str, object],
    summary: dict[str, object],
) -> dict[str, Path]:
    run_dir = run.run_dir
    predictions_path = run_dir / "prediction_rows.csv"
    candidate_path = run_dir / "candidate_rows.csv"
    execution_path = run_dir / "execution_rows.csv"
    net_bet_path = run_dir / "net_bet_rows.csv"
    niches_path = run_dir / "niches.csv"
    inner_fold_assignments_path = run_dir / "inner_fold_assignments.csv"
    regional_source_path = run_dir / "regional_source_diagnostics.csv"
    policy_ranking_path = run_dir / "policy_ranking.csv"
    combination_ranking_path = run_dir / "niche_ranking.csv"
    clv_rows_path = run_dir / "clv_rows.csv"
    execution_quality_path = run_dir / "execution_quality.json"
    summary_path = run_dir / "summary.json"
    promotion_path = run_dir / "promotion_report.json"
    bank_training_path = run_dir / "bank_curve_training.png"
    bank_holdout_path = run_dir / "bank_curve_holdout.png"
    calibration_path = run_dir / "calibration_curves.png"

    predictions.to_csv(predictions_path, index=False)
    candidate_rows.to_csv(candidate_path, index=False)
    execution_rows.to_csv(execution_path, index=False)
    net_bet_rows.to_csv(net_bet_path, index=False)
    niches.to_csv(niches_path, index=False)
    inner_fold_assignments.to_csv(inner_fold_assignments_path, index=False)
    pd.DataFrame(dict(summary.get("signal_summary", {})).get("regional_diagnostics", {}).get("rows", [])).to_csv(
        regional_source_path,
        index=False,
    )
    policy_ranking.to_csv(policy_ranking_path, index=False)
    combination_ranking.to_csv(combination_ranking_path, index=False)
    clv_rows.to_csv(clv_rows_path, index=False)
    _save_json(execution_quality_path, execution_quality)
    _save_json(summary_path, summary)
    _save_json(promotion_path, summary["promotion_report"])
    _plot_calibration(predictions, calibration_path)
    _plot_bank_curve(execution_rows[execution_rows["fold_segment"] == "train"].rename(columns={"net_profit": "flat_profit"}), bank_training_path)
    _plot_bank_curve(execution_rows[execution_rows["fold_segment"] == "holdout"].rename(columns={"net_profit": "flat_profit"}), bank_holdout_path)

    return {
        "predictions": predictions_path,
        "candidate_rows": candidate_path,
        "execution_rows": execution_path,
        "net_bet_rows": net_bet_path,
        "niches": niches_path,
        "inner_fold_assignments": inner_fold_assignments_path,
        "regional_source_diagnostics": regional_source_path,
        "policy_ranking": policy_ranking_path,
        "niche_ranking": combination_ranking_path,
        "clv_rows": clv_rows_path,
        "execution_quality": execution_quality_path,
        "summary": summary_path,
        "promotion_report": promotion_path,
        "calibration": calibration_path,
        "bank_training": bank_training_path,
        "bank_holdout": bank_holdout_path,
    }


def train_niche_bundle(
    settings: Settings,
    bundle_matches: pd.DataFrame,
    feature_rows: pd.DataFrame,
    research_run_dir: Path,
) -> NicheTrainingResult:
    summary, predictions, _ = load_research_run(research_run_dir)
    feature_columns = model_feature_columns(feature_rows)
    final_model = build_goal_model(feature_rows[feature_columns])
    fit_goal_model(final_model, feature_rows[feature_columns], feature_rows["home_goals"], feature_rows["away_goals"])
    lambda_home, lambda_away = final_model.predict_lambdas(feature_rows[feature_columns])
    rho = fit_dixon_coles_rho(
        lambda_home,
        lambda_away,
        feature_rows["home_goals"].to_numpy(),
        feature_rows["away_goals"].to_numpy(),
        settings.backtest.dixon_coles_bounds,
    )

    probability_source = str(summary.get("selected_probability_source", "raw"))
    probability_decision = dict(summary.get("probability_decision", {}))
    calibrator = None
    if probability_source == "calibrated" or bool(probability_decision.get("requires_calibrator")):
        calibrator = OutcomeCalibrator(epsilon=settings.backtest.calibrator_epsilon).fit(
            predictions[[f"prob_{outcome}_raw" for outcome in OUTCOME_ORDER]].to_numpy(),
            predictions["actual_target"].to_numpy(),
        )

    run = create_run_context(settings.paths.models_dir, "train_niche")
    model_path = run.run_dir / "model_bundle.joblib"
    summary_path = run.run_dir / "summary.json"
    payload = {
        "model": final_model,
        "calibrator": calibrator,
        "rho": rho,
        "policy": summary.get("selected_policy", {}),
        "probability_source": probability_source,
        "probability_decision": probability_decision,
        "niche_filter": summary.get("top_niche"),
        "feature_columns": feature_columns,
        "history_matches": bundle_matches,
        "rolling_window": settings.backtest.rolling_window,
        "max_poisson_goals": settings.backtest.max_poisson_goals,
        "execution_assumptions": summary.get("execution_assumptions", {}),
        "research_run_dir": str(research_run_dir),
        "promotion_report": summary.get("promotion_report", {}),
    }
    joblib.dump(payload, model_path)
    training_summary = {
        "rows": int(len(feature_rows)),
        "probability_source": probability_source,
        "probability_decision": probability_decision,
        "policy": summary.get("selected_policy", {}),
        "niche_filter": summary.get("top_niche"),
        "rho": rho,
        "research_run_dir": str(research_run_dir),
        "promotion_report": summary.get("promotion_report", {}),
    }
    _save_json(summary_path, training_summary)
    return NicheTrainingResult(run=run, model_path=model_path, summary_path=summary_path, summary=training_summary)


def shadow_run_from_bundle(
    settings: Settings,
    fixtures_path: Path,
    model_path: Path,
    snapshot_files: list[Path] | None = None,
) -> ShadowRunResult:
    payload = joblib.load(model_path)
    fixtures = canonicalize_matches(pd.read_csv(fixtures_path), require_results=False)
    feature_rows = build_fixture_feature_rows(
        history_matches=payload["history_matches"],
        fixtures=fixtures,
        rolling_window=int(payload["rolling_window"]),
    )
    market = build_market_odds(fixtures)
    dataset = feature_rows.merge(
        market,
        on=["match_id", "Date", "league_code", "season", "HomeTeam", "AwayTeam"],
        how="left",
    )
    dataset["league_name"] = dataset.get("league_name", dataset["league_code"])
    dataset["outcome"] = ""
    dataset["target"] = 0
    dataset["home_goals"] = np.nan
    dataset["away_goals"] = np.nan
    snapshots = build_market_snapshots(market, settings=settings, source_name="input_quote", source_type="user_quote")
    if snapshot_files:
        reference = fixtures.copy()
        if "league_name" not in reference.columns:
            reference["league_name"] = reference["league_code"]
        extra = [load_snapshot_file(path, reference, settings) for path in snapshot_files]
        snapshots = _prepare_snapshot_frame(pd.concat([snapshots, *extra], ignore_index=True, sort=False), settings)

    model = payload["model"]
    rho = float(payload["rho"])
    feature_columns = payload["feature_columns"]
    lambda_home, lambda_away = model.predict_lambdas(dataset[feature_columns])
    raw_probabilities = outcome_probabilities_from_lambdas(
        lambda_home,
        lambda_away,
        rho=rho,
        max_goals=int(payload["max_poisson_goals"]),
    )
    calibrator = payload.get("calibrator")
    calibrated_probabilities = raw_probabilities.copy() if calibrator is None else calibrator.transform(raw_probabilities)

    predictions = _build_prediction_frame(dataset, lambda_home, lambda_away, raw_probabilities, calibrated_probabilities)
    predictions["actual_outcome"] = pd.NA
    predictions["actual_target"] = np.nan
    predictions["baseline_prediction"] = np.take(
        OUTCOME_ORDER,
        predictions[[f"market_prob_{outcome}" for outcome in OUTCOME_ORDER]].to_numpy().argmax(axis=1),
    )
    predictions["raw_prediction"] = np.take(OUTCOME_ORDER, raw_probabilities.argmax(axis=1))
    predictions["calibrated_prediction"] = np.take(OUTCOME_ORDER, calibrated_probabilities.argmax(axis=1))
    predictions["fold_segment"] = "shadow"
    predictions["kickoff_time"] = _default_kickoff(predictions["Date"], settings.snapshot.default_kickoff_hour)

    candidate_rows = build_candidate_rows(predictions, snapshots, settings)
    probability_decision = dict(payload.get("probability_decision", {}))
    if probability_decision:
        predictions, candidate_rows = apply_probability_source_strategy(predictions, candidate_rows, probability_decision, settings)
    policy = payload.get("policy", {})
    selected = select_candidate_bets(
        candidate_rows=candidate_rows,
        policy=BetPolicy(
            edge_threshold=float(policy.get("edge_threshold", settings.research.provisional_edge_threshold)),
            ev_threshold=float(policy.get("ev_threshold", settings.research.provisional_ev_threshold)),
            min_odds=float(policy.get("min_odds", min(settings.backtest.policy_search.min_odds_options))),
            max_odds=float(policy.get("max_odds", max(settings.backtest.policy_search.max_odds_options))),
            kelly_fraction=float(policy.get("kelly_fraction", settings.backtest.policy_search.max_kelly_fraction)),
        ),
        probability_source=str(payload.get("probability_source", "raw")),
        segment_filter=payload.get("niche_filter"),
    )
    planned_bets = simulate_execution(
        selected_bets=selected,
        execution=settings.execution,
        flat_stake=settings.research.flat_stake,
        allow_proxy=True,
    )
    if planned_bets.empty:
        planned_bets = selected.copy()
        planned_bets["execution_status"] = "no_bet"
        planned_bets["executed_odds"] = np.nan
        planned_bets["accepted_stake"] = 0.0
        planned_bets["gross_profit"] = np.nan
        planned_bets["net_profit"] = np.nan
        planned_bets["won"] = np.nan
    else:
        planned_bets["execution_status"] = np.where(planned_bets["execution_status"] == "executed", "planned", planned_bets["execution_status"])
        planned_bets["net_profit"] = np.nan
        planned_bets["gross_profit"] = np.nan
        planned_bets["won"] = np.nan

    run = create_run_context(settings.paths.runs_dir, "shadow")
    predictions_path = run.run_dir / "predictions.csv"
    candidates_path = run.run_dir / "candidate_rows.csv"
    planned_path = run.run_dir / "planned_bets.csv"
    summary_path = run.run_dir / "summary.json"
    predictions.to_csv(predictions_path, index=False)
    candidate_rows.to_csv(candidates_path, index=False)
    planned_bets.to_csv(planned_path, index=False)
    _save_json(
        summary_path,
        {
            "fixtures": int(len(predictions)),
            "planned_bets": int(len(planned_bets)),
            "probability_source": payload.get("probability_source", "raw"),
            "niche_filter": payload.get("niche_filter"),
            "execution_assumptions": payload.get("execution_assumptions", {}),
        },
    )
    return ShadowRunResult(
        run=run,
        predictions=predictions,
        candidate_rows=candidate_rows,
        planned_bets=planned_bets,
        artifacts={
            "predictions": predictions_path,
            "candidate_rows": candidates_path,
            "planned_bets": planned_path,
            "summary": summary_path,
        },
    )
