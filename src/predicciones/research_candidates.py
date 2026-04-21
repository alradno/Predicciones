from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .config import ExecutionConfig, Settings
from .contracts import OUTCOME_ORDER
from .execution_quality import attach_fill_adjusted_ev
from .selection_scoring import (
    DEFAULT_REGIONAL_SOURCE_FIELDS,
    INNER_ROLE_DISCOVERY,
    INNER_ROLE_POLICY_TUNE,
    conservative_score_breakdown,
)
from .strategy import BetPolicy, score_policy_metrics
from .research_snapshots import _categorize_time_bucket, _edge_band, _odds_band, _season_phase


def _probability_array(frame: pd.DataFrame, probability_source: str) -> np.ndarray:
    return frame[[f"prob_{outcome}_{probability_source}" for outcome in OUTCOME_ORDER]].to_numpy()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _multiclass_accuracy(actual: np.ndarray, probabilities: np.ndarray) -> float:
    actual_values = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(actual_values)
    if not mask.any() or probabilities.size == 0:
        return 0.0

    labels = actual_values[mask].astype(int)
    probs = probabilities[mask]
    if labels.shape[0] != probs.shape[0]:
        limit = min(len(labels), len(probs))
        labels = labels[:limit]
        probs = probs[:limit]
    if labels.size == 0 or probs.size == 0:
        return 0.0
    predicted = np.argmax(probs, axis=1)
    return float(np.mean(predicted == labels))


def _multiclass_brier_score(actual: np.ndarray, probabilities: np.ndarray) -> float:
    actual_values = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(actual_values)
    if not mask.any() or probabilities.size == 0:
        return float("inf")

    labels = actual_values[mask].astype(int)
    probs = probabilities[mask]
    if labels.shape[0] != probs.shape[0]:
        limit = min(len(labels), len(probs))
        labels = labels[:limit]
        probs = probs[:limit]
    if labels.size == 0 or probs.size == 0:
        return float("inf")

    one_hot = np.zeros_like(probs, dtype=float)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _probability_quality_report(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if actual.size == 0 or probabilities.size == 0:
        return {
            "accuracy": 0.0,
            "log_loss": float("inf"),
            "brier_score": float("inf"),
            "n_rows": 0.0,
        }

    return {
        "accuracy": _multiclass_accuracy(actual, probabilities),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1, 2])),
        "brier_score": _multiclass_brier_score(actual, probabilities),
        "n_rows": float(len(actual)),
    }


def _quality_key(report: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(report.get("log_loss", float("inf"))),
        float(report.get("brier_score", float("inf"))),
        -float(report.get("accuracy", 0.0)),
    )


@dataclass(frozen=True)
class ProbabilitySourceDecision:
    decision: str
    criterion: str
    rationale: str
    global_best_source: str
    selected_source_strategy: str
    raw: dict[str, Any]
    calibrated: dict[str, Any]
    regional_source_map: dict[str, dict[str, str]]
    regional_diagnostics: dict[str, Any]
    requires_calibrator: bool
    comparison: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "criterion": self.criterion,
            "rationale": self.rationale,
            "global_best_source": self.global_best_source,
            "selected_source_strategy": self.selected_source_strategy,
            "raw": self.raw,
            "calibrated": self.calibrated,
            "regional_source_map": self.regional_source_map,
            "regional_diagnostics": self.regional_diagnostics,
            "requires_calibrator": self.requires_calibrator,
            "comparison": self.comparison,
        }


def _source_metrics_report(predictions: pd.DataFrame, probability_source: str) -> dict[str, Any]:
    actual = predictions["actual_target"].to_numpy()
    probabilities = _probability_array(predictions, probability_source)
    quality = _probability_quality_report(actual, probabilities)
    return {
        "accuracy": quality["accuracy"],
        "log_loss": quality["log_loss"],
        "brier_score": quality["brier_score"],
        "quality": quality,
        "n_rows": int(len(predictions)),
    }


def _match_level_region_frame(predictions: pd.DataFrame, candidate_rows: pd.DataFrame) -> pd.DataFrame:
    working = predictions.copy()
    if "match_id" not in working.columns:
        working = working.reset_index(drop=True).copy()
        working["match_id"] = np.arange(len(working))
    working = working.drop_duplicates(subset=["match_id"]).copy()
    if candidate_rows.empty or "match_id" not in candidate_rows.columns:
        return working

    rows: list[dict[str, Any]] = []
    for match_id, group in candidate_rows.groupby("match_id", observed=True, sort=False):
        record: dict[str, Any] = {"match_id": match_id}
        record["league_code"] = str(group["league_code"].iloc[0]) if "league_code" in group.columns else ""
        record["time_bucket"] = str(group["time_bucket"].iloc[0]) if "time_bucket" in group.columns else ""
        if "market_prob" in group.columns:
            favorite_row = group.sort_values(["market_prob", "quoted_odds"], ascending=[False, True]).iloc[0]
        else:
            favorite_row = group.iloc[0]
        record["odds_band"] = str(favorite_row.get("odds_band", ""))
        if "edge_calibrated" in group.columns:
            edge_row = group.assign(_edge_value=pd.to_numeric(group["edge_calibrated"], errors="coerce").fillna(-np.inf))
            edge_row = edge_row.sort_values("_edge_value", ascending=False).iloc[0]
            record["edge_band_calibrated"] = str(edge_row.get("edge_band_calibrated", ""))
        else:
            record["edge_band_calibrated"] = ""
        rows.append(record)
    region_rows = pd.DataFrame(rows)
    return working.merge(region_rows, on="match_id", how="left")


def _regional_source_map(
    region_rows: pd.DataFrame,
    settings: Settings,
    global_best_source: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any], bool]:
    regional_fields = tuple(getattr(settings.research, "regional_source_fields", DEFAULT_REGIONAL_SOURCE_FIELDS))
    diagnostics: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, str]] = {field: {} for field in regional_fields}
    requires_calibrator = global_best_source == "calibrated"
    policy_rows = region_rows[region_rows.get("inner_role", pd.Series("", index=region_rows.index)).astype(str).eq(INNER_ROLE_POLICY_TUNE)].copy()
    if policy_rows.empty:
        return mapping, {"fields": list(regional_fields), "rows": diagnostics}, requires_calibrator

    for field in regional_fields:
        if field not in policy_rows.columns:
            continue
        for region_value, group in policy_rows.groupby(field, dropna=False, observed=True):
            region_label = "" if pd.isna(region_value) else str(region_value)
            raw_report = _source_metrics_report(group, "raw")
            calibrated_report = _source_metrics_report(group, "calibrated")
            log_loss_delta = float(raw_report["log_loss"] - calibrated_report["log_loss"])
            brier_delta = float(raw_report["brier_score"] - calibrated_report["brier_score"])
            qualifies = (
                int(len(group)) >= int(settings.research.regional_source_min_rows)
                and log_loss_delta >= float(settings.research.regional_source_min_log_loss_delta)
                and (
                    not bool(settings.research.regional_source_require_non_worse_brier)
                    or brier_delta >= 0.0
                )
            )
            chosen = "calibrated" if qualifies else global_best_source
            if region_label:
                mapping.setdefault(field, {})[region_label] = chosen
            diagnostics.append(
                {
                    "field": field,
                    "region": region_label,
                    "rows": int(len(group)),
                    "raw_log_loss": float(raw_report["log_loss"]),
                    "calibrated_log_loss": float(calibrated_report["log_loss"]),
                    "log_loss_delta": log_loss_delta,
                    "raw_brier_score": float(raw_report["brier_score"]),
                    "calibrated_brier_score": float(calibrated_report["brier_score"]),
                    "brier_score_delta": brier_delta,
                    "chosen_source": chosen,
                    "qualifies_regional_override": bool(qualifies),
                }
            )
            requires_calibrator = requires_calibrator or chosen == "calibrated"
    return mapping, {"fields": list(regional_fields), "rows": diagnostics}, requires_calibrator


def _row_effective_source(
    row: pd.Series,
    *,
    global_best_source: str,
    regional_source_map: dict[str, dict[str, str]],
    regional_fields: tuple[str, ...],
) -> tuple[str, float, dict[str, int]]:
    calibrated_votes = 0
    raw_votes = 0
    applicable = 0
    for field in regional_fields:
        value = row.get(field, "")
        region = "" if pd.isna(value) else str(value)
        if not region:
            continue
        field_map = regional_source_map.get(field, {})
        if region not in field_map:
            continue
        applicable += 1
        if str(field_map[region]) == "calibrated":
            calibrated_votes += 1
        else:
            raw_votes += 1
    if applicable <= 0:
        return global_best_source, 0.5, {"applicable": 0, "calibrated_votes": 0, "raw_votes": 0}
    if calibrated_votes > raw_votes:
        source = "calibrated"
        confidence = float(calibrated_votes / applicable)
    elif raw_votes > calibrated_votes:
        source = "raw"
        confidence = float(raw_votes / applicable)
    else:
        source = global_best_source
        tied_votes = calibrated_votes if source == "calibrated" else raw_votes
        confidence = float(tied_votes / applicable) if applicable > 0 else 0.5
    return source, confidence if confidence > 0 else 0.5, {
        "applicable": applicable,
        "calibrated_votes": calibrated_votes,
        "raw_votes": raw_votes,
    }


def apply_probability_source_strategy(
    predictions: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    decision_payload: dict[str, Any],
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = str(decision_payload.get("selected_source_strategy", settings.research.regional_source_strategy))
    global_best_source = str(decision_payload.get("global_best_source", decision_payload.get("decision", "raw")))
    regional_source_map = {
        str(field): {str(key): str(value) for key, value in mapping.items()}
        for field, mapping in dict(decision_payload.get("regional_source_map", {})).items()
    }
    regional_fields = tuple(getattr(settings.research, "regional_source_fields", DEFAULT_REGIONAL_SOURCE_FIELDS))

    enriched_predictions = predictions.copy()
    enriched_candidates = candidate_rows.copy()
    match_regions = _match_level_region_frame(enriched_predictions, enriched_candidates)
    source_rows: list[dict[str, Any]] = []
    for row in match_regions.itertuples(index=False):
        source, confidence, votes = _row_effective_source(
            pd.Series(row._asdict()),
            global_best_source=global_best_source,
            regional_source_map=regional_source_map,
            regional_fields=regional_fields,
        )
        payload = {"match_id": row.match_id, "regional_probability_source": source, "regional_calibration_confidence": confidence}
        payload.update({f"regional_vote_{key}": int(value) for key, value in votes.items()})
        source_rows.append(payload)
    source_frame = pd.DataFrame(source_rows)
    if not source_frame.empty:
        enriched_predictions = enriched_predictions.merge(source_frame, on="match_id", how="left")
        enriched_candidates = enriched_candidates.merge(source_frame, on="match_id", how="left")
    else:
        enriched_predictions["regional_probability_source"] = global_best_source
        enriched_predictions["regional_calibration_confidence"] = 0.5
        enriched_candidates["regional_probability_source"] = global_best_source
        enriched_candidates["regional_calibration_confidence"] = 0.5

    if strategy != "regional_best_with_global_fallback":
        enriched_predictions["regional_probability_source"] = global_best_source
        enriched_candidates["regional_probability_source"] = global_best_source
        enriched_predictions["regional_calibration_confidence"] = 0.5
        enriched_candidates["regional_calibration_confidence"] = 0.5

    for outcome in OUTCOME_ORDER:
        enriched_predictions[f"prob_{outcome}_effective"] = np.where(
            enriched_predictions["regional_probability_source"].astype(str).eq("calibrated"),
            enriched_predictions[f"prob_{outcome}_calibrated"],
            enriched_predictions[f"prob_{outcome}_raw"],
        )
    enriched_predictions["effective_prediction"] = np.take(
        OUTCOME_ORDER,
        enriched_predictions[[f"prob_{outcome}_effective" for outcome in OUTCOME_ORDER]].to_numpy().argmax(axis=1),
    )
    enriched_candidates["model_prob_effective"] = np.where(
        enriched_candidates["regional_probability_source"].astype(str).eq("calibrated"),
        enriched_candidates["model_prob_calibrated"],
        enriched_candidates["model_prob_raw"],
    )
    enriched_candidates["edge_effective"] = np.where(
        enriched_candidates["regional_probability_source"].astype(str).eq("calibrated"),
        enriched_candidates["edge_calibrated"],
        enriched_candidates["edge_raw"],
    )
    enriched_candidates["ev_effective"] = np.where(
        enriched_candidates["regional_probability_source"].astype(str).eq("calibrated"),
        enriched_candidates["ev_calibrated"],
        enriched_candidates["ev_raw"],
    )
    enriched_candidates = attach_fill_adjusted_ev(
        enriched_candidates,
        probability_column="model_prob_effective",
        edge_column="edge_effective",
        default_notional=1.0,
    )
    return enriched_predictions, enriched_candidates


def _best_snapshot_for_outcome(
    snapshots: pd.DataFrame,
    outcome: str,
    decision_time: pd.Timestamp,
    config: ExecutionConfig,
    allow_proxy: bool,
) -> tuple[str, pd.Series | None]:
    if snapshots.empty:
        return "no_match_snapshots", None

    working = snapshots.copy()
    working["quote_age_minutes"] = (decision_time - working["snapshot_time"]).dt.total_seconds().div(60.0)
    eligible = working[
        working["snapshot_time"].le(decision_time)
        & working["quote_age_minutes"].ge(0.0)
        & working["quote_age_minutes"].le(config.max_quote_age_minutes)
        & working[f"odds_{outcome}"].notna()
        & working["liquidity"].ge(config.min_liquidity)
    ].copy()
    if not allow_proxy:
        eligible = eligible[eligible["source_type"] != "closing_proxy"].copy()

    if eligible.empty:
        return "no_eligible_quote", None

    best_index = eligible[f"odds_{outcome}"].astype(float).idxmax()
    return "eligible", eligible.loc[best_index]


def build_candidate_rows(predictions: pd.DataFrame, snapshots: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    snapshot_lookup = {
        int(match_id): group.reset_index(drop=True)
        for match_id, group in snapshots.groupby("match_id", observed=True)
    }
    rows: list[dict[str, Any]] = []

    for row in predictions.itertuples(index=False):
        kickoff_time = getattr(row, "kickoff_time", pd.NaT)
        if pd.isna(kickoff_time):
            kickoff_time = pd.Timestamp(row.Date).normalize() + pd.to_timedelta(settings.snapshot.default_kickoff_hour, unit="h")
        decision_time = pd.Timestamp(kickoff_time) - pd.to_timedelta(settings.execution.decision_minutes_before_kickoff, unit="m")
        snapshot_group = snapshot_lookup.get(int(row.match_id), pd.DataFrame())
        allow_proxy = settings.execution.allow_closing_proxy_for_research

        for outcome in OUTCOME_ORDER:
            status, snapshot = _best_snapshot_for_outcome(
                snapshot_group,
                outcome=outcome,
                decision_time=decision_time,
                config=settings.execution,
                allow_proxy=allow_proxy,
            )
            quoted_odds = float(snapshot[f"odds_{outcome}"]) if snapshot is not None else np.nan
            if snapshot is not None and f"market_prob_{outcome}" in snapshot.index:
                market_prob = float(snapshot[f"market_prob_{outcome}"])
            elif snapshot is not None:
                implied_total = 0.0
                implied_probs: dict[str, float] = {}
                for item in OUTCOME_ORDER:
                    odds_value = float(snapshot.get(f"odds_{item}", np.nan))
                    implied_probs[item] = 0.0 if pd.isna(odds_value) or odds_value <= 0 else (1.0 / odds_value)
                    implied_total += implied_probs[item]
                market_prob = (implied_probs[outcome] / implied_total) if implied_total > 0 else float(
                    getattr(row, f"market_prob_{outcome}", np.nan)
                )
            else:
                market_prob = float(getattr(row, f"market_prob_{outcome}", np.nan))
            snapshot_time = pd.Timestamp(snapshot["snapshot_time"]) if snapshot is not None else pd.NaT
            quote_age_minutes = float((decision_time - snapshot_time).total_seconds() / 60.0) if snapshot is not None else np.nan
            time_to_kickoff_minutes = (
                float(snapshot["time_to_kickoff_minutes"])
                if snapshot is not None and "time_to_kickoff_minutes" in snapshot.index
                else (
                    float((pd.Timestamp(kickoff_time) - snapshot_time).total_seconds() / 60.0)
                    if snapshot is not None
                    else np.nan
                )
            )
            time_bucket = (
                str(snapshot["time_bucket"])
                if snapshot is not None and "time_bucket" in snapshot.index
                else _categorize_time_bucket(time_to_kickoff_minutes, settings.snapshot.time_bucket_edges)
            )

            rows.append(
                {
                    "match_id": row.match_id,
                    "fold_id": getattr(row, "fold_id", np.nan),
                    "fold_segment": getattr(row, "fold_segment", "unknown"),
                    "inner_role": getattr(row, "inner_role", ""),
                    "Date": row.Date,
                    "kickoff_time": kickoff_time,
                    "decision_time": decision_time,
                    "league_code": row.league_code,
                    "league_name": row.league_name,
                    "season": row.season,
                    "HomeTeam": row.HomeTeam,
                    "AwayTeam": row.AwayTeam,
                    "selection": outcome,
                    "actual_outcome": getattr(row, "actual_outcome", pd.NA),
                    "actual_target": getattr(row, "actual_target", np.nan),
                    "baseline_prediction": getattr(row, "baseline_prediction", ""),
                    "raw_prediction": getattr(row, "raw_prediction", ""),
                    "calibrated_prediction": getattr(row, "calibrated_prediction", ""),
                    "expected_goals_home": getattr(row, "expected_goals_home", np.nan),
                    "expected_goals_away": getattr(row, "expected_goals_away", np.nan),
                    "model_prob_raw": float(getattr(row, f"prob_{outcome}_raw")),
                    "model_prob_calibrated": float(getattr(row, f"prob_{outcome}_calibrated")),
                    "quoted_odds": quoted_odds,
                    "market_prob": market_prob,
                    "source_name": str(snapshot["source_name"]) if snapshot is not None else "",
                    "source_type": str(snapshot["source_type"]) if snapshot is not None else "",
                    "snapshot_time": snapshot_time,
                    "quote_age_minutes": quote_age_minutes,
                    "time_to_kickoff_minutes": time_to_kickoff_minutes,
                    "minutes_to_kickoff": time_to_kickoff_minutes,
                    "time_bucket": time_bucket,
                    "liquidity": float(snapshot["liquidity"]) if snapshot is not None else np.nan,
                    "available_ask_size": float(snapshot["liquidity"]) if snapshot is not None else np.nan,
                    "movement_regime": str(snapshot.get(f"movement_regime_{outcome}", "unknown")) if snapshot is not None else "unknown",
                    "movement_delta": float(snapshot.get(f"movement_delta_{outcome}", np.nan)) if snapshot is not None else np.nan,
                    "quote_status": status,
                    "month": int(pd.Timestamp(row.Date).month),
                    "season_phase": _season_phase(pd.Timestamp(row.Date)),
                }
            )

    candidate_rows = pd.DataFrame(rows)
    if candidate_rows.empty:
        return candidate_rows

    candidate_rows["edge_raw"] = candidate_rows["model_prob_raw"] - candidate_rows["market_prob"]
    candidate_rows["edge_calibrated"] = candidate_rows["model_prob_calibrated"] - candidate_rows["market_prob"]
    candidate_rows["ev_raw"] = (candidate_rows["model_prob_raw"] * candidate_rows["quoted_odds"]) - 1.0
    candidate_rows["ev_calibrated"] = (candidate_rows["model_prob_calibrated"] * candidate_rows["quoted_odds"]) - 1.0
    candidate_rows["odds_band"] = candidate_rows["quoted_odds"].map(_odds_band)
    candidate_rows["edge_band_raw"] = candidate_rows["edge_raw"].map(_edge_band)
    candidate_rows["edge_band_calibrated"] = candidate_rows["edge_calibrated"].map(_edge_band)
    return candidate_rows


def _apply_segment_filter(frame: pd.DataFrame, segment_filter: dict[str, Any] | None) -> pd.DataFrame:
    if not segment_filter:
        return frame
    working = frame
    for key, value in segment_filter.items():
        working = working[working[key] == value]
    return working


def select_candidate_bets(
    candidate_rows: pd.DataFrame,
    policy: BetPolicy,
    probability_source: str,
    segment_filter: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if candidate_rows.empty:
        return pd.DataFrame()

    edge_column = f"edge_{probability_source}"
    ev_column = f"ev_{probability_source}"
    prob_column = f"model_prob_{probability_source}"
    working = _apply_segment_filter(candidate_rows, segment_filter).copy()
    working = working[
        working["quote_status"].eq("eligible")
        & working["quoted_odds"].between(policy.min_odds, policy.max_odds, inclusive="both")
        & working[edge_column].ge(policy.edge_threshold)
        & working[ev_column].ge(policy.ev_threshold)
    ].copy()
    if working.empty:
        return working

    working = working.sort_values(["match_id", ev_column, edge_column, "quoted_odds"], ascending=[True, False, False, False])
    selected = working.groupby("match_id", observed=True).head(1).copy()
    selected["selection_prob"] = selected[prob_column]
    selected["selection_edge"] = selected[edge_column]
    selected["selection_ev"] = selected[ev_column]
    selected["requested_stake"] = 1.0
    selected["probability_source"] = probability_source
    selected["policy_edge_threshold"] = policy.edge_threshold
    selected["policy_ev_threshold"] = policy.ev_threshold
    selected["policy_min_odds"] = policy.min_odds
    selected["policy_max_odds"] = policy.max_odds
    return selected.reset_index(drop=True)


def simulate_execution(
    selected_bets: pd.DataFrame,
    execution: ExecutionConfig,
    flat_stake: float,
    allow_proxy: bool,
) -> pd.DataFrame:
    if selected_bets.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for row in selected_bets.itertuples(index=False):
        requested_stake = min(flat_stake, float(getattr(row, "requested_stake", flat_stake)))
        if pd.isna(row.quoted_odds):
            status = "rejected_no_quote"
            accepted_stake = 0.0
            executed_odds = np.nan
        elif str(row.quote_status) != "eligible":
            status = f"rejected_{row.quote_status}"
            accepted_stake = 0.0
            executed_odds = np.nan
        elif str(row.source_type) == "closing_proxy" and not allow_proxy:
            status = "rejected_proxy_source"
            accepted_stake = 0.0
            executed_odds = np.nan
        else:
            status = "executed"
            accepted_stake = min(requested_stake, execution.max_stake)
            executed_odds = float(row.quoted_odds) * (1.0 - execution.slippage_rate)
        fill_rate = _safe_ratio(accepted_stake, requested_stake) if requested_stake > 0 else 0.0

        actual_value = getattr(row, "actual_outcome", pd.NA)
        has_actual = pd.notna(actual_value) and str(actual_value).strip() != ""
        won = int(has_actual and str(actual_value) == str(row.selection)) if status == "executed" else np.nan
        if status == "executed" and has_actual:
            gross_profit = accepted_stake * ((executed_odds - 1.0) if won == 1 else -1.0)
            net_profit = accepted_stake * (((executed_odds - 1.0) * (1.0 - execution.commission_rate)) if won == 1 else -1.0)
        else:
            gross_profit = np.nan
            net_profit = np.nan

        payload = row._asdict()
        payload.update(
            {
                "requested_stake": requested_stake,
                "accepted_stake": accepted_stake,
                "executed_odds": executed_odds,
                "execution_status": status,
                "fill_rate": fill_rate,
                "partial_fill": bool(fill_rate < 0.999) if status == "executed" else False,
                "available_ask_size": getattr(row, "available_ask_size", np.nan),
                "minutes_to_kickoff": getattr(row, "minutes_to_kickoff", np.nan),
                "book_age_seconds": (
                    float(getattr(row, "quote_age_minutes", np.nan)) * 60.0
                    if pd.notna(getattr(row, "quote_age_minutes", np.nan))
                    else np.nan
                ),
                "price_provenance": "proxy" if str(getattr(row, "source_type", "")) == "closing_proxy" else "exact",
                "closing_reference_odds": np.nan,
                "closing_reference_prob": np.nan,
                "clv_source": "",
                "won": won,
                "gross_profit": gross_profit,
                "net_profit": net_profit,
            }
        )
        rows.append(payload)

    return pd.DataFrame(rows)


def summarize_execution(rows: pd.DataFrame, profit_column: str = "net_profit") -> dict[str, float]:
    if rows.empty:
        return {
            "bets": 0,
            "wins": 0,
            "profit": 0.0,
            "roi": 0.0,
            "yield": 0.0,
            "max_drawdown": 0.0,
            "executed": 0,
            "rejected": 0,
            "stake": 0.0,
        }

    executed = rows[rows["execution_status"] == "executed"].copy()
    if executed.empty:
        return {
            "bets": int(len(rows)),
            "wins": 0,
            "profit": 0.0,
            "roi": 0.0,
            "yield": 0.0,
            "max_drawdown": 0.0,
            "executed": 0,
            "rejected": int(len(rows)),
            "stake": 0.0,
        }

    profits = executed[profit_column].dropna().astype(float)
    cumulative = profits.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    stake = float(executed["accepted_stake"].sum())
    return {
        "bets": int(len(rows)),
        "wins": int(executed.get("won", pd.Series(0, index=executed.index)).fillna(0).eq(1).sum()),
        "profit": float(profits.sum()),
        "roi": _safe_ratio(float(profits.sum()), stake),
        "yield": float(profits.mean()) if not profits.empty else 0.0,
        "max_drawdown": float(drawdown.max()) if not drawdown.empty else 0.0,
        "executed": int(len(executed)),
        "rejected": int(len(rows) - len(executed)),
        "stake": stake,
    }


def _roi_by_fold(rows: pd.DataFrame) -> pd.Series:
    executed = rows[rows["execution_status"] == "executed"].copy()
    if executed.empty or "fold_id" not in executed.columns:
        return pd.Series(dtype=float)
    grouped = executed.groupby("fold_id", observed=True).agg(profit=("net_profit", "sum"), stake=("accepted_stake", "sum"))
    return grouped["profit"] / grouped["stake"].replace(0.0, np.nan)


def _evaluate_policy(
    candidate_rows: pd.DataFrame,
    policy: BetPolicy,
    probability_source: str,
    settings: Settings,
    allow_proxy: bool,
    segment_filter: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    selected = select_candidate_bets(
        candidate_rows=candidate_rows,
        policy=policy,
        probability_source=probability_source,
        segment_filter=segment_filter,
    )
    execution_rows = simulate_execution(
        selected_bets=selected,
        execution=settings.execution,
        flat_stake=settings.research.flat_stake,
        allow_proxy=allow_proxy,
    )
    return selected, execution_rows, summarize_execution(execution_rows)


def choose_probability_source(
    predictions: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    settings: Settings,
) -> tuple[str, dict[str, Any]]:
    """Choose the probability source with a global OOF leader plus regional calibrated overrides."""

    train_predictions = predictions[predictions["fold_segment"].astype(str) == "train"].copy()
    if train_predictions.empty:
        raise ValueError("No hay filas de train para elegir la fuente probabilistica.")
    if "inner_role" in train_predictions.columns:
        discovery_predictions = train_predictions[train_predictions["inner_role"].astype(str).eq(INNER_ROLE_DISCOVERY)].copy()
    else:
        discovery_predictions = train_predictions.copy()
    if discovery_predictions.empty:
        discovery_predictions = train_predictions.copy()

    raw_report = _source_metrics_report(discovery_predictions, "raw")
    calibrated_report = _source_metrics_report(discovery_predictions, "calibrated")
    provisional_policy = BetPolicy(
        edge_threshold=settings.research.provisional_edge_threshold,
        ev_threshold=settings.research.provisional_ev_threshold,
        min_odds=min(settings.backtest.policy_search.min_odds_options),
        max_odds=max(settings.backtest.policy_search.max_odds_options),
        kelly_fraction=settings.backtest.policy_search.max_kelly_fraction,
    )
    train_candidate_rows = candidate_rows[candidate_rows["fold_segment"].astype(str).eq("train")].copy()
    _, raw_execution, raw_metrics = _evaluate_policy(
        train_candidate_rows,
        policy=provisional_policy,
        probability_source="raw",
        settings=settings,
        allow_proxy=settings.execution.allow_closing_proxy_for_research,
    )
    _, calibrated_execution, calibrated_metrics = _evaluate_policy(
        train_candidate_rows,
        policy=provisional_policy,
        probability_source="calibrated",
        settings=settings,
        allow_proxy=settings.execution.allow_closing_proxy_for_research,
    )
    raw_report["net_strategy"] = raw_metrics
    raw_report["selected_bets"] = int(len(raw_execution))
    calibrated_report["net_strategy"] = calibrated_metrics
    calibrated_report["selected_bets"] = int(len(calibrated_execution))
    global_best_source = "calibrated" if _quality_key(calibrated_report["quality"]) < _quality_key(raw_report["quality"]) else "raw"
    selected_report = calibrated_report if global_best_source == "calibrated" else raw_report
    other_report = raw_report if global_best_source == "calibrated" else calibrated_report
    region_rows = _match_level_region_frame(train_predictions, candidate_rows[candidate_rows["fold_segment"].astype(str).eq("train")].copy())
    regional_source_map, regional_diagnostics, requires_calibrator = _regional_source_map(
        region_rows=region_rows,
        settings=settings,
        global_best_source=global_best_source,
    )
    rationale = (
        "La fuente global se elige por calidad OOF en discovery "
        "(log_loss, luego brier_score y por ultimo accuracy). "
        "Los overrides regionales solo permiten calibrated en policy_tune cuando cumplen minimo de filas y mejora estable."
    )

    payload = ProbabilitySourceDecision(
        decision="effective",
        criterion="oof_probability_quality",
        rationale=rationale,
        global_best_source=global_best_source,
        selected_source_strategy=str(settings.research.regional_source_strategy),
        raw=raw_report,
        calibrated=calibrated_report,
        regional_source_map=regional_source_map,
        regional_diagnostics=regional_diagnostics,
        requires_calibrator=requires_calibrator,
        comparison={
            "log_loss_delta": float(raw_report["log_loss"] - calibrated_report["log_loss"]),
            "brier_score_delta": float(raw_report["brier_score"] - calibrated_report["brier_score"]),
            "accuracy_delta": float(calibrated_report["accuracy"] - raw_report["accuracy"]),
            "selected_quality_log_loss": float(selected_report["log_loss"]),
            "selected_quality_brier_score": float(selected_report["brier_score"]),
            "selected_quality_accuracy": float(selected_report["accuracy"]),
            "other_quality_log_loss": float(other_report["log_loss"]),
            "other_quality_brier_score": float(other_report["brier_score"]),
            "other_quality_accuracy": float(other_report["accuracy"]),
        },
    )
    return global_best_source, payload.to_dict()


def optimize_research_policy(
    candidate_rows: pd.DataFrame,
    probability_source: str,
    settings: Settings,
    *,
    return_ranking: bool = False,
    top_k: int | None = None,
) -> BetPolicy | tuple[BetPolicy, pd.DataFrame]:
    """Optimize the betting policy using economic outcomes only."""

    best_policy = BetPolicy(
        edge_threshold=settings.research.provisional_edge_threshold,
        ev_threshold=settings.research.provisional_ev_threshold,
        min_odds=min(settings.backtest.policy_search.min_odds_options),
        max_odds=max(settings.backtest.policy_search.max_odds_options),
        kelly_fraction=settings.backtest.policy_search.max_kelly_fraction,
    )
    best_score = float("-inf")
    min_bets = max(settings.backtest.policy_search.min_bets, settings.research.min_segment_bets)
    ranking_rows: list[dict[str, Any]] = []

    for edge, ev, min_odds, max_odds in product(
        settings.backtest.policy_search.edge_thresholds,
        settings.backtest.policy_search.ev_thresholds,
        settings.backtest.policy_search.min_odds_options,
        settings.backtest.policy_search.max_odds_options,
    ):
        if min_odds >= max_odds:
            continue
        candidate = BetPolicy(
            edge_threshold=float(edge),
            ev_threshold=float(ev),
            min_odds=float(min_odds),
            max_odds=float(max_odds),
            kelly_fraction=settings.backtest.policy_search.max_kelly_fraction,
        )
        _, execution_rows, metrics = _evaluate_policy(
            candidate_rows=candidate_rows,
            policy=candidate,
            probability_source=probability_source,
            settings=settings,
            allow_proxy=settings.execution.allow_closing_proxy_for_research,
        )
        if metrics["executed"] < min_bets:
            continue

        fold_roi = _roi_by_fold(execution_rows)
        breakdown = conservative_score_breakdown(
            metrics,
            fold_roi=fold_roi,
            positive_fold_target=settings.research.positive_fold_ratio,
            prior_bets=min_bets,
            drawdown_weight=settings.research.drawdown_weight,
            generalization_gap_weight=settings.research.generalization_gap_weight,
            positive_penalty_weight=settings.research.positive_penalty_weight,
        )
        score = float(breakdown["score"])
        ranking_rows.append(
            {
                "edge_threshold": float(edge),
                "ev_threshold": float(ev),
                "min_odds": float(min_odds),
                "max_odds": float(max_odds),
                "executed": int(metrics["executed"]),
                "profit": float(metrics["profit"]),
                "roi": float(metrics["roi"]),
                "stake": float(metrics["stake"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                **breakdown,
                "policy_payload": candidate.to_dict(),
            }
        )

        if score > best_score:
            best_policy = candidate
            best_score = score

    ranking = pd.DataFrame(ranking_rows)
    if not ranking.empty:
        ranking = ranking.sort_values(["score", "generalization_gap", "executed"], ascending=[False, True, False]).reset_index(drop=True)
        if top_k is not None:
            ranking = ranking.head(int(top_k)).reset_index(drop=True)
    if return_ranking:
        return best_policy, ranking
    return best_policy
