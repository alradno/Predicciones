from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss

from .config import Settings
from .contracts import OUTCOME_ORDER, OUTCOME_TO_TARGET
from .models import (
    GoalModelBundle,
    OutcomeCalibrator,
    build_goal_model,
    fit_dixon_coles_rho,
    fit_goal_model,
    multiclass_brier_score,
    outcome_probabilities_from_lambdas,
    score_matrix_from_lambdas,
)
from .reporting import create_run_context


MODEL_NAME = "football_sim_poisson_v1"
TARGET_COLUMNS = {"home_goals", "away_goals", "outcome", "total_goals", "btts"}
SPLIT_ORDER = ("train", "dev", "locked_holdout")
MARKET_REFERENCE_PREFIXES = ("odds_", "market_prob_")
MARKET_REFERENCE_COLUMNS = {"market_reference_present"}
METADATA_COLUMNS = {
    "match_id",
    "match_date",
    "match_start_time",
    "as_of_time",
    "known_before_match",
    "league_code",
    "season",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
    "split",
    "feature_role",
    "feature_family_set",
}
MIN_READINESS_ROWS = {"train": 500, "dev": 100, "locked_holdout": 100}


@dataclass(frozen=True)
class FootballSimTrainingResult:
    summary: dict[str, Any]
    artifacts: dict[str, Path]


def train_football_simulator(
    settings: Settings,
    dataset_path: Path | str | None = None,
    model_name: str = MODEL_NAME,
    exclude_market_reference: bool = True,
) -> FootballSimTrainingResult:
    if model_name != MODEL_NAME:
        raise ValueError(f"Modelo no soportado: {model_name}. Usa {MODEL_NAME}.")

    resolved_dataset_path = Path(dataset_path) if dataset_path else settings.paths.outputs_dir / "sim_data" / "simulation_training_dataset.csv"
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(
            "No encuentro simulation_training_dataset.csv. Ejecuta antes: "
            "collect-sim-data, normalize-sim-data, build-sim-features y export-sim-training-dataset."
        )

    dataset = pd.read_csv(resolved_dataset_path)
    _validate_training_dataset(dataset, exclude_market_reference=exclude_market_reference)
    manifest = _load_training_manifest(resolved_dataset_path)
    feature_columns, feature_manifest = _resolve_feature_columns(
        dataset=dataset,
        manifest=manifest,
        exclude_market_reference=exclude_market_reference,
    )
    if not feature_columns:
        raise ValueError("No quedan features entrenables despues de aplicar el contrato leakage-free.")

    prepared = _prepare_training_frame(dataset)
    train = prepared[prepared["split"] == "train"].copy()
    if train.empty:
        raise ValueError("El dataset no contiene filas train para entrenar football_sim_poisson_v1.")

    model = build_goal_model(train[feature_columns])
    fit_goal_model(model, train[feature_columns], train["home_goals"], train["away_goals"])
    train_home_lambda, train_away_lambda = model.predict_lambdas(train[feature_columns])
    rho = fit_dixon_coles_rho(
        train_home_lambda,
        train_away_lambda,
        train["home_goals"].to_numpy(),
        train["away_goals"].to_numpy(),
        settings.backtest.dixon_coles_bounds,
    )

    train_raw = outcome_probabilities_from_lambdas(
        train_home_lambda,
        train_away_lambda,
        rho=rho,
        max_goals=settings.backtest.max_poisson_goals,
    )
    calibrator = OutcomeCalibrator(epsilon=settings.backtest.calibrator_epsilon).fit(
        train_raw,
        train["target"].to_numpy(),
    )

    league_baseline = _fit_league_mean_baseline(train)
    predictions, metrics_by_split, calibration_report = _evaluate_splits(
        frame=prepared,
        model=model,
        feature_columns=feature_columns,
        rho=rho,
        calibrator=calibrator,
        league_baseline=league_baseline,
        max_goals=settings.backtest.max_poisson_goals,
    )
    selected_probability_source = _select_probability_source(metrics_by_split)
    readiness = _readiness_status(
        frame=prepared,
        metrics_by_split=metrics_by_split,
        selected_probability_source=selected_probability_source,
        leakage_violations=_leakage_violations(dataset, exclude_market_reference=exclude_market_reference),
    )

    model_root = settings.paths.outputs_dir / "sim_models"
    run = create_run_context(model_root, model_name)
    artifacts = _write_artifacts(
        run_dir=run.run_dir,
        model=model,
        calibrator=calibrator,
        rho=rho,
        selected_probability_source=selected_probability_source,
        dataset_path=resolved_dataset_path,
        predictions=predictions,
        metrics_by_split=metrics_by_split,
        calibration_report=calibration_report,
        feature_manifest=feature_manifest,
        readiness=readiness,
        settings=settings,
    )
    latest_pointer = settings.paths.outputs_dir / "latest_football_sim_model.txt"
    latest_pointer.write_text(str(artifacts["model_bundle"]), encoding="utf-8")
    artifacts["latest_football_sim_model"] = latest_pointer

    summary = {
        "model_name": model_name,
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "dataset_path": str(resolved_dataset_path),
        "rows": int(len(prepared)),
        "split_counts": {split: int((prepared["split"] == split).sum()) for split in SPLIT_ORDER},
        "feature_count": len(feature_columns),
        "rho": float(rho),
        "selected_probability_source": selected_probability_source,
        "readiness_status": readiness["readiness_status"],
        "readiness_blockers": readiness["readiness_blockers"],
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        "picks_emitidos": 0,
    }
    return FootballSimTrainingResult(summary=summary, artifacts=artifacts)


def _validate_training_dataset(dataset: pd.DataFrame, exclude_market_reference: bool) -> None:
    required = {"home_goals", "away_goals", "outcome", "split"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset de simulacion incompleto; faltan columnas: {missing}")
    if exclude_market_reference:
        market_columns = _market_reference_columns(dataset)
        if market_columns:
            raise ValueError(
                "El dataset contiene columnas market_reference y exclude_market_reference=true: "
                + ", ".join(market_columns)
            )


def _load_training_manifest(dataset_path: Path) -> dict[str, Any]:
    manifest_path = dataset_path.with_name("simulation_training_manifest.json")
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_feature_columns(
    dataset: pd.DataFrame,
    manifest: dict[str, Any],
    exclude_market_reference: bool,
) -> tuple[list[str], dict[str, Any]]:
    declared = list(manifest.get("feature_columns") or [])
    if not declared:
        declared = [
            column
            for column in dataset.columns
            if column not in TARGET_COLUMNS and column not in METADATA_COLUMNS
        ]

    dropped: dict[str, str] = {}
    feature_columns: list[str] = []
    for column in declared:
        if column not in dataset.columns:
            dropped[column] = "missing_from_dataset"
            continue
        if column in TARGET_COLUMNS or column in METADATA_COLUMNS:
            dropped[column] = "target_or_metadata"
            continue
        if exclude_market_reference and _is_market_reference_column(column):
            dropped[column] = "market_reference_excluded"
            continue
        if column != "league_code" and not pd.api.types.is_numeric_dtype(dataset[column]):
            dropped[column] = "non_numeric_not_supported_in_v1"
            continue
        feature_columns.append(column)

    return feature_columns, {
        "declared_feature_columns": declared,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "dropped_feature_columns": sorted(dropped),
        "dropped_feature_reasons": dropped,
        "train_allowed_source": "simulation_training_manifest.feature_columns",
        "market_reference_training_enabled": False,
        "policy_reoptimized": False,
        "global_roi_actionable": False,
    }


def _prepare_training_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
    frame = frame.dropna(subset=["home_goals", "away_goals", "outcome", "split"]).copy()
    frame["outcome"] = frame["outcome"].astype(str).str.lower()
    frame = frame[frame["outcome"].isin(OUTCOME_TO_TARGET)].copy()
    frame["target"] = frame["outcome"].map(OUTCOME_TO_TARGET).astype(int)
    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
        frame = frame.sort_values(["match_date", "match_id" if "match_id" in frame.columns else "split"]).reset_index(drop=True)
    return frame


def _fit_league_mean_baseline(train: pd.DataFrame) -> dict[str, Any]:
    global_home = float(max(train["home_goals"].mean(), 0.05))
    global_away = float(max(train["away_goals"].mean(), 0.05))
    by_league: dict[str, dict[str, float]] = {}
    if "league_code" in train.columns:
        grouped = train.groupby("league_code", dropna=False)
        for league, rows in grouped:
            by_league[str(league)] = {
                "home_lambda": float(max(rows["home_goals"].mean(), 0.05)),
                "away_lambda": float(max(rows["away_goals"].mean(), 0.05)),
            }
    return {
        "global": {"home_lambda": global_home, "away_lambda": global_away},
        "by_league": by_league,
    }


def _baseline_lambdas(frame: pd.DataFrame, baseline: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    global_home = baseline["global"]["home_lambda"]
    global_away = baseline["global"]["away_lambda"]
    home: list[float] = []
    away: list[float] = []
    for row in frame.itertuples(index=False):
        league = str(getattr(row, "league_code", ""))
        league_values = baseline.get("by_league", {}).get(league)
        home.append(float(league_values["home_lambda"] if league_values else global_home))
        away.append(float(league_values["away_lambda"] if league_values else global_away))
    return np.asarray(home, dtype=float), np.asarray(away, dtype=float)


def _evaluate_splits(
    frame: pd.DataFrame,
    model: GoalModelBundle,
    feature_columns: list[str],
    rho: float,
    calibrator: OutcomeCalibrator,
    league_baseline: dict[str, Any],
    max_goals: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    prediction_frames: list[pd.DataFrame] = []
    metrics_by_split: dict[str, Any] = {}
    calibration_report: dict[str, Any] = {}

    for split in SPLIT_ORDER:
        split_frame = frame[frame["split"] == split].copy()
        if split_frame.empty:
            metrics_by_split[split] = {"rows": 0}
            calibration_report[split] = []
            continue

        home_lambda, away_lambda = model.predict_lambdas(split_frame[feature_columns])
        raw_probs = outcome_probabilities_from_lambdas(home_lambda, away_lambda, rho=rho, max_goals=max_goals)
        calibrated_probs = calibrator.transform(raw_probs)
        baseline_home, baseline_away = _baseline_lambdas(split_frame, league_baseline)
        baseline_probs = outcome_probabilities_from_lambdas(
            baseline_home,
            baseline_away,
            rho=0.0,
            max_goals=max_goals,
        )
        internal_elo_probs = _internal_elo_probabilities(split_frame)
        split_predictions = _prediction_frame(
            split_frame,
            home_lambda,
            away_lambda,
            raw_probs,
            calibrated_probs,
            baseline_probs,
            internal_elo_probs,
            rho=rho,
            max_goals=max_goals,
        )
        prediction_frames.append(split_predictions)
        metrics_by_split[split] = {
            "rows": int(len(split_frame)),
            "raw": _metrics(split_frame, home_lambda, away_lambda, raw_probs),
            "calibrated": _metrics(split_frame, home_lambda, away_lambda, calibrated_probs),
            "league_mean_poisson_baseline": _metrics(split_frame, baseline_home, baseline_away, baseline_probs),
        }
        if internal_elo_probs is not None:
            metrics_by_split[split]["internal_elo_baseline"] = _outcome_only_metrics(split_frame, internal_elo_probs)
        calibration_report[split] = _calibration_deciles(split_frame["target"].to_numpy(), raw_probs)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return predictions, metrics_by_split, calibration_report


def _prediction_frame(
    frame: pd.DataFrame,
    home_lambda: np.ndarray,
    away_lambda: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    baseline_probs: np.ndarray,
    internal_elo_probs: np.ndarray | None,
    rho: float,
    max_goals: int,
) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "match_id",
            "match_date",
            "league_code",
            "season",
            "home_team_name",
            "away_team_name",
            "split",
            "outcome",
            "home_goals",
            "away_goals",
        )
        if column in frame.columns
    ]
    output = frame[columns].copy()
    output["target"] = frame["target"].to_numpy()
    output["lambda_home"] = home_lambda
    output["lambda_away"] = away_lambda
    for index, outcome in enumerate(OUTCOME_ORDER):
        output[f"prob_{outcome}_raw"] = raw_probs[:, index]
        output[f"prob_{outcome}_calibrated"] = calibrated_probs[:, index]
        output[f"prob_{outcome}_league_mean_baseline"] = baseline_probs[:, index]
        if internal_elo_probs is not None:
            output[f"prob_{outcome}_internal_elo_baseline"] = internal_elo_probs[:, index]
    output["predicted_outcome_raw"] = [OUTCOME_ORDER[index] for index in np.argmax(raw_probs, axis=1)]
    output["predicted_outcome_calibrated"] = [OUTCOME_ORDER[index] for index in np.argmax(calibrated_probs, axis=1)]

    market_probs = [_scoreline_market_probabilities(float(h), float(a), rho=rho, max_goals=max_goals) for h, a in zip(home_lambda, away_lambda)]
    for key in ("over_2_5", "under_2_5", "btts_yes", "btts_no"):
        output[f"prob_{key}_raw"] = [payload[key] for payload in market_probs]
    return output


def _metrics(frame: pd.DataFrame, home_lambda: np.ndarray, away_lambda: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    output = _outcome_only_metrics(frame, probabilities)
    actual_home = frame["home_goals"].to_numpy(dtype=float)
    actual_away = frame["away_goals"].to_numpy(dtype=float)
    output.update(
        {
            "home_goal_mae": float(np.mean(np.abs(actual_home - home_lambda))),
            "away_goal_mae": float(np.mean(np.abs(actual_away - away_lambda))),
            "total_goal_mae": float(np.mean(np.abs((actual_home + actual_away) - (home_lambda + away_lambda)))),
            "goal_poisson_nll": _goal_poisson_nll(actual_home, actual_away, home_lambda, away_lambda),
        }
    )
    return output


def _outcome_only_metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    target = frame["target"].to_numpy(dtype=int)
    predicted = np.argmax(probabilities, axis=1)
    actual_outcomes = frame["outcome"].astype(str).to_numpy()
    predicted_no_draw = np.where(probabilities[:, OUTCOME_ORDER.index("home")] >= probabilities[:, OUTCOME_ORDER.index("away")], "home", "away")
    non_draw_mask = actual_outcomes != "draw"
    winner_accuracy = (
        float(np.mean(predicted_no_draw[non_draw_mask] == actual_outcomes[non_draw_mask]))
        if np.any(non_draw_mask)
        else None
    )
    return {
        "outcome_log_loss": _safe_log_loss(target, probabilities),
        "outcome_brier": multiclass_brier_score(target, probabilities),
        "accuracy_1x2": float(accuracy_score(target, predicted)),
        "winner_accuracy_no_draw": winner_accuracy,
        "actual_draw_rate": float(np.mean(actual_outcomes == "draw")),
        "predicted_draw_rate": float(np.mean(predicted == OUTCOME_ORDER.index("draw"))),
        "average_draw_probability": float(np.mean(probabilities[:, OUTCOME_ORDER.index("draw")])),
    }


def _safe_log_loss(target: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(target) == 0:
        return None
    return float(log_loss(target, np.clip(probabilities, 1e-9, 1.0), labels=[0, 1, 2]))


def _goal_poisson_nll(actual_home: np.ndarray, actual_away: np.ndarray, home_lambda: np.ndarray, away_lambda: np.ndarray) -> float:
    home_rate = np.clip(home_lambda, 1e-6, None)
    away_rate = np.clip(away_lambda, 1e-6, None)
    return float(-np.mean(poisson.logpmf(actual_home, home_rate) + poisson.logpmf(actual_away, away_rate)))


def _internal_elo_probabilities(frame: pd.DataFrame) -> np.ndarray | None:
    if "internal_elo_diff" not in frame.columns:
        return None
    diff = pd.to_numeric(frame["internal_elo_diff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    home_strength = 1.0 / (1.0 + np.power(10.0, -diff / 400.0))
    draw_prob = np.full(len(frame), 0.26, dtype=float)
    probabilities = np.zeros((len(frame), len(OUTCOME_ORDER)), dtype=float)
    probabilities[:, OUTCOME_ORDER.index("home")] = (1.0 - draw_prob) * home_strength
    probabilities[:, OUTCOME_ORDER.index("away")] = (1.0 - draw_prob) * (1.0 - home_strength)
    probabilities[:, OUTCOME_ORDER.index("draw")] = draw_prob
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _scoreline_market_probabilities(lambda_home: float, lambda_away: float, rho: float, max_goals: int) -> dict[str, float]:
    matrix = score_matrix_from_lambdas(lambda_home, lambda_away, rho=rho, max_goals=max_goals)
    home_goals = np.arange(max_goals + 1)[:, None]
    away_goals = np.arange(max_goals + 1)[None, :]
    total = home_goals + away_goals
    over_2_5 = float(matrix[total > 2.5].sum())
    btts_yes = float(matrix[(home_goals > 0) & (away_goals > 0)].sum())
    return {
        "over_2_5": over_2_5,
        "under_2_5": float(1.0 - over_2_5),
        "btts_yes": btts_yes,
        "btts_no": float(1.0 - btts_yes),
    }


def _calibration_deciles(target: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    if len(target) == 0:
        return []
    confidence = probabilities.max(axis=1)
    predicted = np.argmax(probabilities, axis=1)
    correct = (predicted == target).astype(float)
    frame = pd.DataFrame({"confidence": confidence, "correct": correct})
    bins = min(10, len(frame))
    if bins <= 1:
        frame["decile"] = 0
    else:
        frame["decile"] = pd.qcut(frame["confidence"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for decile, group in frame.groupby("decile", dropna=False):
        rows.append(
            {
                "decile": int(decile) if pd.notna(decile) else -1,
                "rows": int(len(group)),
                "avg_confidence": float(group["confidence"].mean()),
                "accuracy": float(group["correct"].mean()),
            }
        )
    return rows


def _select_probability_source(metrics_by_split: dict[str, Any]) -> str:
    dev = metrics_by_split.get("dev", {})
    raw = dev.get("raw", {}).get("outcome_log_loss")
    calibrated = dev.get("calibrated", {}).get("outcome_log_loss")
    if raw is None or calibrated is None:
        return "raw"
    return "calibrated" if calibrated <= raw * 0.99 else "raw"


def _readiness_status(
    frame: pd.DataFrame,
    metrics_by_split: dict[str, Any],
    selected_probability_source: str,
    leakage_violations: list[str],
) -> dict[str, Any]:
    split_counts = {split: int((frame["split"] == split).sum()) for split in SPLIT_ORDER}
    blockers: list[str] = []
    for split, minimum in MIN_READINESS_ROWS.items():
        if split_counts.get(split, 0) < minimum:
            blockers.append(f"{split}_sample_below_{minimum}")
    blockers.extend(leakage_violations)

    dev_model = metrics_by_split.get("dev", {}).get(selected_probability_source, {}).get("outcome_log_loss")
    dev_baseline = metrics_by_split.get("dev", {}).get("league_mean_poisson_baseline", {}).get("outcome_log_loss")
    holdout_model = metrics_by_split.get("locked_holdout", {}).get(selected_probability_source, {}).get("outcome_log_loss")
    holdout_baseline = metrics_by_split.get("locked_holdout", {}).get("league_mean_poisson_baseline", {}).get("outcome_log_loss")

    dev_beats = dev_model is not None and dev_baseline is not None and dev_model < dev_baseline
    holdout_not_worse = holdout_model is not None and holdout_baseline is not None and holdout_model <= holdout_baseline
    if not dev_beats:
        blockers.append("dev_does_not_beat_league_mean_baseline")
    if not holdout_not_worse:
        blockers.append("locked_holdout_worse_than_league_mean_baseline")

    if not blockers:
        status = "predictive_ready"
    elif not dev_beats and not holdout_not_worse:
        status = "rejected"
    else:
        status = "research_only"
    return {
        "readiness_status": status,
        "readiness_blockers": blockers,
        "split_counts": split_counts,
        "selected_probability_source": selected_probability_source,
        "dev_beats_baseline": bool(dev_beats),
        "locked_holdout_not_worse_than_baseline": bool(holdout_not_worse),
        "min_readiness_rows": dict(MIN_READINESS_ROWS),
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        "picks_emitidos": 0,
    }


def _leakage_violations(dataset: pd.DataFrame, exclude_market_reference: bool) -> list[str]:
    violations: list[str] = []
    if "known_before_match" in dataset.columns:
        known = pd.to_numeric(dataset["known_before_match"], errors="coerce").fillna(0)
        if not bool((known == 1).all()):
            violations.append("known_before_match_violation")
    if exclude_market_reference and _market_reference_columns(dataset):
        violations.append("market_reference_columns_present")
    return violations


def _write_artifacts(
    run_dir: Path,
    model: GoalModelBundle,
    calibrator: OutcomeCalibrator,
    rho: float,
    selected_probability_source: str,
    dataset_path: Path,
    predictions: pd.DataFrame,
    metrics_by_split: dict[str, Any],
    calibration_report: dict[str, Any],
    feature_manifest: dict[str, Any],
    readiness: dict[str, Any],
    settings: Settings,
) -> dict[str, Path]:
    model_path = run_dir / "model_bundle.joblib"
    predictions_path = run_dir / "simulation_predictions.csv"
    model_report_path = run_dir / "simulation_model_report.json"
    metrics_path = run_dir / "simulation_metrics_by_split.json"
    calibration_path = run_dir / "simulation_calibration_report.json"
    feature_manifest_path = run_dir / "simulation_feature_manifest.json"
    readiness_path = run_dir / "simulation_readiness_report.json"

    predictions.to_csv(predictions_path, index=False)
    _write_json(metrics_path, metrics_by_split)
    _write_json(calibration_path, calibration_report)
    _write_json(feature_manifest_path, feature_manifest)
    _write_json(readiness_path, readiness)
    model_report = {
        "model_name": MODEL_NAME,
        "dataset_path": str(dataset_path),
        "rho": float(rho),
        "selected_probability_source": selected_probability_source,
        "readiness_status": readiness["readiness_status"],
        "readiness_blockers": readiness["readiness_blockers"],
        "artifacts": {
            "model_bundle": str(model_path),
            "simulation_predictions": str(predictions_path),
            "simulation_metrics_by_split": str(metrics_path),
            "simulation_calibration_report": str(calibration_path),
            "simulation_feature_manifest": str(feature_manifest_path),
            "simulation_readiness_report": str(readiness_path),
        },
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        "picks_emitidos": 0,
    }
    _write_json(model_report_path, model_report)
    joblib.dump(
        {
            "model_name": MODEL_NAME,
            "model": model,
            "calibrator": calibrator,
            "rho": rho,
            "selected_probability_source": selected_probability_source,
            "feature_columns": feature_manifest["feature_columns"],
            "max_poisson_goals": settings.backtest.max_poisson_goals,
            "policy_reoptimized": False,
            "global_roi_actionable": False,
            "picks_emitidos": 0,
        },
        model_path,
    )
    return {
        "model_bundle": model_path,
        "simulation_predictions": predictions_path,
        "simulation_model_report": model_report_path,
        "simulation_metrics_by_split": metrics_path,
        "simulation_calibration_report": calibration_path,
        "simulation_feature_manifest": feature_manifest_path,
        "simulation_readiness_report": readiness_path,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _market_reference_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if _is_market_reference_column(column)]


def _is_market_reference_column(column: str) -> bool:
    return column in MARKET_REFERENCE_COLUMNS or column.startswith(MARKET_REFERENCE_PREFIXES)
