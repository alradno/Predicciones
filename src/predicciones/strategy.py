from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import uuid
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import OUTCOME_ORDER
from .execution_quality import attach_fill_adjusted_ev
from .selection_scoring import conservative_score_breakdown


@dataclass(frozen=True)
class BetPolicy:
    edge_threshold: float
    ev_threshold: float
    min_odds: float
    max_odds: float
    kelly_fraction: float = 0.25
    family: str = "edge_ev_threshold"
    top_quantile: float = 0.10
    allowed_leagues: tuple[str, ...] = ()
    allowed_outcomes: tuple[str, ...] = ()
    scope_name: str = "global_all"

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["allowed_leagues"] = list(self.allowed_leagues)
        payload["allowed_outcomes"] = list(self.allowed_outcomes)
        return payload


def _float_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _clamp01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def _resolve_probability_source(frame: pd.DataFrame, probability_source: str) -> str:
    requested = str(probability_source or "calibrated")
    requested_prob = f"model_prob_{requested}"
    requested_edge = f"edge_{requested}"
    requested_ev = f"ev_{requested}"
    if requested_prob in frame.columns and requested_edge in frame.columns and requested_ev in frame.columns:
        return requested
    if all(f"prob_{outcome}_{requested}" in frame.columns for outcome in OUTCOME_ORDER):
        return requested

    fallback = "raw" if requested != "raw" else "calibrated"
    fallback_prob = f"model_prob_{fallback}"
    fallback_edge = f"edge_{fallback}"
    fallback_ev = f"ev_{fallback}"
    if fallback_prob in frame.columns and fallback_edge in frame.columns and fallback_ev in frame.columns:
        return fallback
    if all(f"prob_{outcome}_{fallback}" in frame.columns for outcome in OUTCOME_ORDER):
        return fallback

    if requested_prob in frame.columns:
        return requested
    if fallback_prob in frame.columns:
        return fallback
    if all(f"prob_{outcome}_{requested}" in frame.columns for outcome in OUTCOME_ORDER):
        return requested
    if all(f"prob_{outcome}_{fallback}" in frame.columns for outcome in OUTCOME_ORDER):
        return fallback
    raise KeyError(f"No model probability columns found for requested source '{probability_source}'.")


def _match_probability_margin(frame: pd.DataFrame, probability_column: str) -> pd.Series:
    if frame.empty or "match_id" not in frame.columns or probability_column not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)

    margins = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("match_id", observed=True, sort=False):
        probs = pd.to_numeric(group[probability_column], errors="coerce").to_numpy(dtype=float)
        if probs.size == 0:
            continue
        for position, index in enumerate(group.index):
            prob = float(probs[position]) if np.isfinite(probs[position]) else np.nan
            if np.isnan(prob):
                margins.at[index] = np.nan
                continue
            others = np.delete(probs, position)
            others = others[np.isfinite(others)]
            runner_up = float(np.max(others)) if others.size else 0.0
            margins.at[index] = prob - runner_up
    return margins.fillna(0.0)


def _match_score_margin(frame: pd.DataFrame, score_column: str) -> pd.Series:
    if frame.empty or "match_id" not in frame.columns or score_column not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)

    margins = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("match_id", observed=True, sort=False):
        scores = pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=float)
        if scores.size == 0:
            continue
        for position, index in enumerate(group.index):
            score = float(scores[position]) if np.isfinite(scores[position]) else np.nan
            if np.isnan(score):
                margins.at[index] = np.nan
                continue
            others = np.delete(scores, position)
            others = others[np.isfinite(others)]
            runner_up = float(np.max(others)) if others.size else 0.0
            margins.at[index] = score - runner_up
    return margins.fillna(0.0)


def _confidence_from_signals(
    frame: pd.DataFrame,
    *,
    score_column: str,
    probability_source: str,
) -> pd.DataFrame:
    output = frame.copy()
    raw_prob = _float_series(output, "model_prob_raw", default=np.nan)
    calibrated_prob = _float_series(output, "model_prob_calibrated", default=np.nan)
    source_prob = pd.to_numeric(output[score_column], errors="coerce")
    source_gap = (raw_prob - calibrated_prob).abs().fillna(0.0)

    output["policy_score_margin"] = _clamp01(_match_score_margin(output, score_column).fillna(0.0) / 0.25)
    output["policy_probability_margin_signal"] = _clamp01(output["policy_probability_margin"].fillna(0.0) / 0.20)
    output["policy_value_buffer_signal"] = _clamp01(
        (output["policy_value_buffer"].fillna(0.0) / 0.05).clip(lower=-3.0, upper=3.0).apply(lambda value: 1.0 / (1.0 + np.exp(-float(value))))
    )
    output["policy_source_agreement"] = _clamp01(1.0 - (source_gap / 0.15).clip(lower=0.0, upper=1.0))
    output["policy_confidence_score"] = _clamp01(
        0.30 * output["policy_probability_margin_signal"].fillna(0.0)
        + 0.25 * output["policy_score_margin"].fillna(0.0)
        + 0.20 * output["policy_source_consensus"].fillna(0.5)
        + 0.10 * output["policy_quote_freshness"].fillna(0.5)
        + 0.10 * output["policy_source_agreement"].fillna(0.5)
        + 0.05 * output["policy_value_buffer_signal"].fillna(0.5)
    )
    if probability_source == "raw":
        alt_prob = calibrated_prob
    else:
        alt_prob = raw_prob
    output["policy_confidence_divergence_to_v1"] = (source_prob - alt_prob).abs().fillna(source_gap)
    return output


def _price_provenance(frame: pd.DataFrame) -> pd.Series:
    if "price_provenance" in frame.columns:
        return frame["price_provenance"].fillna("unknown").astype(str)
    source_type = frame.get("source_type", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    return source_type.map(
        {
            "user_quote": "exact",
            "user_quote_snapshot": "exact",
            "polymarket_checkpoint": "exact",
            "closing_proxy": "proxy",
            "bookmaker_proxy": "proxy",
            "resolution_only": "resolution_only",
        }
    ).fillna("unknown")


def _decision_score(frame: pd.DataFrame) -> pd.Series:
    return (
        0.30 * frame["policy_value_buffer"].fillna(0.0)
        + 0.20 * frame["policy_fill_adjusted_ev"].fillna(0.0)
        + 0.20 * frame["policy_ev"].fillna(0.0)
        + 0.10 * frame["policy_edge"].fillna(0.0)
        + 0.10 * frame["policy_probability_margin"].fillna(0.0)
        + 0.05 * frame["policy_source_consensus"].fillna(0.5)
        + 0.05 * frame["policy_quote_freshness"].fillna(0.5)
    )


def _uncertainty_multiplier(frame: pd.DataFrame) -> pd.Series:
    confidence_score = _clamp01(
        0.25 * frame["policy_source_consensus"].fillna(0.5)
        + 0.20 * frame["policy_probability_margin_signal"].fillna(0.0)
        + 0.15 * frame["policy_quote_freshness"].fillna(0.5)
        + 0.20 * frame.get("policy_regional_calibration_confidence", pd.Series(0.5, index=frame.index)).fillna(0.5)
        + 0.20 * frame["expected_fill_probability"].fillna(0.5)
    )
    return 0.35 + (0.65 * confidence_score)


def _base_kelly_fraction(frame: pd.DataFrame) -> pd.Series:
    odds = pd.to_numeric(frame.get("quoted_odds", np.nan), errors="coerce")
    probability = pd.to_numeric(frame.get("policy_prob", np.nan), errors="coerce")
    raw_kelly = ((probability * odds) - 1.0) / (odds - 1.0).replace(0.0, np.nan)
    raw_kelly = raw_kelly.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return raw_kelly.clip(lower=0.0)


def _scalar_confidence(value: float) -> float:
    if not np.isfinite(value):
        return 0.5
    return float(np.clip(value, 0.0, 1.0))


def _row_selection_confidence(row: pd.Series | pd.Series, outcome: str, probability_source: str) -> dict[str, float]:
    source = str(probability_source or "calibrated")
    candidate_probs: list[float] = []
    for candidate_outcome in OUTCOME_ORDER:
        prob = float(getattr(row, f"prob_{candidate_outcome}_{source}", np.nan))
        if np.isfinite(prob):
            candidate_probs.append(prob)
    selected_prob = float(getattr(row, f"prob_{outcome}_{source}", np.nan))
    runner_up = max((prob for prob in candidate_probs if not np.isclose(prob, selected_prob, atol=1e-12)), default=0.0)
    probability_margin = max(selected_prob - runner_up, 0.0)

    raw_prob = float(getattr(row, f"prob_{outcome}_raw", np.nan))
    calibrated_prob = float(getattr(row, f"prob_{outcome}_calibrated", np.nan))
    if np.isfinite(raw_prob) and np.isfinite(calibrated_prob):
        source_agreement = 1.0 - min(abs(raw_prob - calibrated_prob), 1.0)
        divergence = abs(selected_prob - (calibrated_prob if source == "raw" else raw_prob))
    else:
        source_agreement = 0.5
        divergence = 0.0

    quote_age = float(getattr(row, "quote_age_minutes", np.nan))
    quote_freshness = 1.0 - np.clip((quote_age if np.isfinite(quote_age) else 60.0) / 60.0, 0.0, 1.0)

    odds = float(getattr(row, f"odds_{outcome}", np.nan))
    top_ask = float(getattr(row, "top_ask", np.nan))
    if not np.isfinite(top_ask) or top_ask <= 0.0:
        top_ask = (1.0 / odds) if np.isfinite(odds) and odds > 0.0 else np.nan
    value_buffer = selected_prob - top_ask if np.isfinite(top_ask) else selected_prob - float(getattr(row, f"market_prob_{outcome}", np.nan))
    value_signal = 1.0 / (1.0 + np.exp(-np.clip(value_buffer / 0.05, -3.0, 3.0))) if np.isfinite(value_buffer) else 0.5

    score_signal = _scalar_confidence(probability_margin / 0.20)
    confidence_score = _scalar_confidence(
        0.40 * score_signal
        + 0.25 * source_agreement
        + 0.20 * quote_freshness
        + 0.15 * value_signal
    )
    return {
        "selection_probability_margin": probability_margin,
        "selection_source_consensus": source_agreement,
        "selection_quote_freshness": quote_freshness,
        "selection_value_buffer": value_buffer if np.isfinite(value_buffer) else np.nan,
        "selection_confidence_score": confidence_score,
        "selection_confidence_divergence_to_v1": divergence,
        "confidence_score": confidence_score,
        "confidence_divergence_to_v1": divergence,
        "policy_confidence_score": confidence_score,
        "policy_confidence_divergence_to_v1": divergence,
    }


def add_edge_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    return add_edge_columns_for_source(predictions, probability_source="calibrated")


def add_edge_columns_for_source(predictions: pd.DataFrame, probability_source: str) -> pd.DataFrame:
    frame = predictions.copy()
    for outcome in OUTCOME_ORDER:
        frame[f"edge_{outcome}"] = frame[f"prob_{outcome}_{probability_source}"] - frame[f"market_prob_{outcome}"]
        frame[f"ev_{outcome}"] = frame[f"prob_{outcome}_{probability_source}"] * frame[f"odds_{outcome}"] - 1.0
    return frame


def select_bets(predictions: pd.DataFrame, policy: BetPolicy) -> pd.DataFrame:
    return select_bets_for_source(predictions, policy=policy, probability_source="calibrated")


def select_bets_for_source(predictions: pd.DataFrame, policy: BetPolicy, probability_source: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int | pd.Timestamp]] = []
    resolved_probability_source = _resolve_probability_source(predictions, probability_source)

    for row in predictions.itertuples(index=False):
        candidates: list[tuple[str, float, float, float]] = []
        for outcome in OUTCOME_ORDER:
            odds = float(getattr(row, f"odds_{outcome}", np.nan))
            edge = float(getattr(row, f"edge_{outcome}", np.nan))
            ev = float(getattr(row, f"ev_{outcome}", np.nan))
            if np.isnan(odds) or np.isnan(edge) or np.isnan(ev):
                continue
            if odds < policy.min_odds or odds > policy.max_odds:
                continue
            if edge < policy.edge_threshold or ev < policy.ev_threshold:
                continue
            candidates.append((outcome, edge, ev, odds))

        if not candidates:
            continue

        outcome, edge, ev, odds = max(candidates, key=lambda item: (item[2], item[1], -item[3]))
        prob = float(getattr(row, f"prob_{outcome}_{probability_source}"))
        actual_value = getattr(row, "actual_outcome", pd.NA)
        has_actual = pd.notna(actual_value) and str(actual_value).strip() != ""
        actual = str(actual_value) if has_actual else ""
        flat_profit = (odds - 1.0) if has_actual and actual == outcome else (-1.0 if has_actual else np.nan)
        confidence_payload = _row_selection_confidence(row, outcome, probability_source)
        base_kelly = max(((prob * odds) - 1.0) / max(odds - 1.0, 1e-9), 0.0)
        confidence_score = _scalar_confidence(float(confidence_payload.get("selection_confidence_score", 0.5)))
        uncertainty_multiplier = 0.35 + (0.65 * confidence_score)
        kelly_fraction = min(base_kelly * uncertainty_multiplier, policy.kelly_fraction)

        rows.append(
            {
                "match_id": row.match_id,
                "Date": row.Date,
                "league_code": row.league_code,
                "league_name": row.league_name,
                "season": row.season,
                "HomeTeam": row.HomeTeam,
                "AwayTeam": row.AwayTeam,
                "selection": outcome,
                "selection_prob": prob,
                "selection_odds": odds,
                "selection_edge": edge,
                "selection_ev": ev,
                **confidence_payload,
                "decision_score": float(getattr(row, "policy_decision_score", getattr(row, "policy_rank_score", np.nan))),
                "policy_rank_score": float(getattr(row, "policy_rank_score", np.nan)),
                "probability_source": resolved_probability_source,
                "actual_outcome": actual,
                "won": int(actual == outcome) if has_actual else np.nan,
                "flat_stake": 1.0,
                "flat_profit": flat_profit,
                "base_kelly_fraction": base_kelly,
                "uncertainty_multiplier": uncertainty_multiplier,
                "requested_stake": kelly_fraction,
                "kelly_fraction": kelly_fraction,
                "kelly_profit": flat_profit * kelly_fraction if has_actual else np.nan,
            }
        )

    return pd.DataFrame(rows)


def summarize_bets(bets: pd.DataFrame, profit_column: str = "flat_profit") -> dict[str, float]:
    if bets.empty:
        return {
            "bets": 0,
            "wins": 0,
            "profit": 0.0,
            "roi": 0.0,
            "yield": 0.0,
            "max_drawdown": 0.0,
        }

    profits = bets[profit_column].dropna().astype(float)
    if profits.empty:
        return {
            "bets": int(len(bets)),
            "wins": int(bets["won"].fillna(0).eq(1).sum()),
            "profit": 0.0,
            "roi": 0.0,
            "yield": 0.0,
            "max_drawdown": 0.0,
        }
    cumulative = profits.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    stake = float(len(bets)) if profit_column == "flat_profit" else float(bets["kelly_fraction"].sum())
    stake = stake if stake > 0 else 1.0

    return {
        "bets": int(len(bets)),
        "wins": int((bets["won"] == 1).sum()),
        "profit": float(profits.sum()),
        "roi": float(profits.sum() / stake),
        "yield": float(profits.mean()),
        "max_drawdown": float(drawdown.max()),
    }


def score_policy_metrics(
    metrics: dict[str, float],
    fold_roi: pd.Series | None = None,
    positive_fold_target: float = 0.0,
    drawdown_weight: float = 0.25,
    stability_bonus: float = 0.0,
    reference_roi: float | None = None,
    generalization_gap_weight: float = 0.10,
    positive_penalty_weight: float = 0.10,
) -> float:
    """Score a policy candidate using economic outcomes only."""
    breakdown = conservative_score_breakdown(
        metrics,
        fold_roi=fold_roi,
        reference_roi=reference_roi,
        positive_fold_target=positive_fold_target,
        drawdown_weight=drawdown_weight,
        generalization_gap_weight=generalization_gap_weight,
        positive_penalty_weight=positive_penalty_weight,
    )
    return float(breakdown["score"] + stability_bonus)


def _policy_stability_bonus(bets: pd.DataFrame) -> float:
    if bets.empty:
        return 0.0
    components = []
    for column, weight in (
        ("selection_probability_margin", 0.05),
        ("selection_source_consensus", 0.02),
        ("selection_quote_freshness", 0.02),
        ("selection_confidence_score", 0.04),
    ):
        if column not in bets.columns:
            continue
        values = pd.to_numeric(bets[column], errors="coerce").dropna()
        if values.empty:
            continue
        components.append(float(values.median()) * weight)
    return float(sum(components))


def optimize_policy(
    predictions: pd.DataFrame,
    edge_thresholds: Iterable[float],
    ev_thresholds: Iterable[float],
    min_odds_options: Iterable[float],
    max_odds_options: Iterable[float],
    min_bets: int,
    kelly_fraction: float,
    probability_source: str = "calibrated",
) -> BetPolicy:
    """Optimize a betting policy using only economic criteria."""

    best_policy = BetPolicy(edge_threshold=0.03, ev_threshold=0.02, min_odds=1.2, max_odds=10.0, kelly_fraction=kelly_fraction)
    best_score = float("-inf")

    for edge, ev, min_odds, max_odds in product(edge_thresholds, ev_thresholds, min_odds_options, max_odds_options):
        if min_odds >= max_odds:
            continue
        candidate = BetPolicy(
            edge_threshold=float(edge),
            ev_threshold=float(ev),
            min_odds=float(min_odds),
            max_odds=float(max_odds),
            kelly_fraction=kelly_fraction,
        )
        bets = select_bets_for_source(predictions, candidate, probability_source=probability_source)
        if len(bets) < min_bets:
            continue

        metrics = summarize_bets(bets)
        score = score_policy_metrics(
            metrics,
            drawdown_weight=0.25,
            stability_bonus=_policy_stability_bonus(bets),
        )
        if score > best_score:
            best_policy = candidate
            best_score = score

    return best_policy


def candidate_score_columns(candidate_rows: pd.DataFrame, probability_source: str, slippage_cushion: float) -> pd.DataFrame:
    frame = candidate_rows.copy()
    resolved_probability_source = _resolve_probability_source(frame, probability_source)
    frame.attrs["resolved_probability_source"] = resolved_probability_source
    prob_column = f"model_prob_{resolved_probability_source}"
    edge_column = f"edge_{resolved_probability_source}"
    ev_column = f"ev_{resolved_probability_source}"
    if "top_ask" not in frame.columns:
        quoted = frame.get("quoted_odds")
        quoted_series = pd.to_numeric(quoted, errors="coerce") if quoted is not None else pd.Series(np.nan, index=frame.index, dtype=float)
        fallback_ask = _float_series(frame, "market_prob_home", default=np.nan)
        frame["top_ask"] = np.where(quoted_series.notna() & quoted_series.gt(0.0), 1.0 / quoted_series, fallback_ask)
    if "gated_prob" in frame.columns:
        base_prob = pd.to_numeric(frame[prob_column], errors="coerce")
        gated_prob = pd.to_numeric(frame["gated_prob"], errors="coerce").combine_first(base_prob)
        frame["policy_prob"] = np.minimum(base_prob.fillna(gated_prob).to_numpy(dtype=float), gated_prob.fillna(base_prob).to_numpy(dtype=float))
    elif "decision_adjusted_prob" in frame.columns:
        base_prob = pd.to_numeric(frame[prob_column], errors="coerce")
        decision_prob = pd.to_numeric(frame["decision_adjusted_prob"], errors="coerce").combine_first(base_prob)
        frame["policy_prob"] = np.minimum(base_prob.fillna(decision_prob).to_numpy(dtype=float), decision_prob.fillna(base_prob).to_numpy(dtype=float))
    elif "regional_adjusted_prob" in frame.columns:
        frame["policy_prob"] = pd.to_numeric(frame["regional_adjusted_prob"], errors="coerce").combine_first(
            pd.to_numeric(frame[prob_column], errors="coerce")
        )
    else:
        frame["policy_prob"] = frame[prob_column]
    if "gated_edge" in frame.columns:
        base_edge = pd.to_numeric(frame[edge_column], errors="coerce")
        gated_edge = pd.to_numeric(frame["gated_edge"], errors="coerce").combine_first(base_edge)
        frame["policy_edge"] = np.minimum(base_edge.fillna(gated_edge).to_numpy(dtype=float), gated_edge.fillna(base_edge).to_numpy(dtype=float))
    elif "decision_adjusted_edge" in frame.columns:
        base_edge = pd.to_numeric(frame[edge_column], errors="coerce")
        decision_edge = pd.to_numeric(frame["decision_adjusted_edge"], errors="coerce").combine_first(base_edge)
        frame["policy_edge"] = np.minimum(base_edge.fillna(decision_edge).to_numpy(dtype=float), decision_edge.fillna(base_edge).to_numpy(dtype=float))
    elif "regional_adjusted_edge" in frame.columns:
        frame["policy_edge"] = pd.to_numeric(frame["regional_adjusted_edge"], errors="coerce").combine_first(
            pd.to_numeric(frame[edge_column], errors="coerce")
        )
    else:
        frame["policy_edge"] = frame[edge_column]
    if "gated_ev" in frame.columns:
        base_ev = pd.to_numeric(frame[ev_column], errors="coerce")
        gated_ev = pd.to_numeric(frame["gated_ev"], errors="coerce").combine_first(base_ev)
        frame["policy_ev"] = np.minimum(base_ev.fillna(gated_ev).to_numpy(dtype=float), gated_ev.fillna(base_ev).to_numpy(dtype=float))
    elif "decision_adjusted_ev" in frame.columns:
        base_ev = pd.to_numeric(frame[ev_column], errors="coerce")
        decision_ev = pd.to_numeric(frame["decision_adjusted_ev"], errors="coerce").combine_first(base_ev)
        frame["policy_ev"] = np.minimum(base_ev.fillna(decision_ev).to_numpy(dtype=float), decision_ev.fillna(base_ev).to_numpy(dtype=float))
    elif "regional_adjusted_ev" in frame.columns:
        frame["policy_ev"] = pd.to_numeric(frame["regional_adjusted_ev"], errors="coerce").combine_first(
            pd.to_numeric(frame[ev_column], errors="coerce")
        )
    else:
        frame["policy_ev"] = frame[ev_column]
    frame["slippage_cushion"] = float(slippage_cushion)
    frame["policy_value_buffer"] = frame["policy_prob"] - frame["top_ask"] - float(slippage_cushion)
    if "gated_prob" in frame.columns or "decision_adjusted_prob" in frame.columns or "regional_adjusted_prob" in frame.columns:
        temp_prob_column = "__policy_prob_temp__"
        frame[temp_prob_column] = frame["policy_prob"]
        frame["policy_probability_margin"] = _match_probability_margin(frame, temp_prob_column)
        frame = frame.drop(columns=[temp_prob_column])
    else:
        frame["policy_probability_margin"] = _match_probability_margin(frame, prob_column)
    raw_prob = _float_series(frame, "model_prob_raw", default=np.nan)
    calibrated_prob = _float_series(frame, "model_prob_calibrated", default=np.nan)
    consensus = 1.0 - (raw_prob - calibrated_prob).abs()
    frame["policy_source_consensus"] = _clamp01(consensus.fillna(0.5))
    frame["price_provenance"] = _price_provenance(frame)
    quote_age = _float_series(frame, "quote_age_minutes", default=np.nan)
    frame["policy_quote_age_minutes"] = quote_age
    freshness = 1.0 - (quote_age.fillna(60.0).clip(lower=0.0, upper=60.0) / 60.0)
    frame["policy_quote_freshness"] = _clamp01(freshness.fillna(0.5))
    frame = _confidence_from_signals(frame, score_column=prob_column, probability_source=probability_source)
    if "confidence_score_v2" in frame.columns:
        frame["policy_signal_confidence_score"] = frame["policy_confidence_score"]
        frame["policy_confidence_score"] = pd.to_numeric(frame["confidence_score_v2"], errors="coerce").fillna(
            pd.to_numeric(frame["policy_confidence_score"], errors="coerce").fillna(0.5)
        ).clip(lower=0.0, upper=1.0)
    if "regional_calibration_confidence" in frame.columns:
        frame["policy_regional_calibration_confidence"] = pd.to_numeric(
            frame["regional_calibration_confidence"], errors="coerce"
        ).fillna(0.5).clip(lower=0.0, upper=1.0)
    else:
        frame["policy_regional_calibration_confidence"] = 0.5
    frame = attach_fill_adjusted_ev(
        frame,
        probability_column="policy_prob",
        edge_column="policy_edge",
        default_notional=1.0,
    )
    frame["policy_fill_adjusted_ev"] = frame["fill_adjusted_ev"]
    frame["policy_expected_fill_probability"] = frame["expected_fill_probability"]
    frame["policy_decision_score"] = _decision_score(frame)
    frame["policy_rank_score"] = frame["policy_decision_score"]
    if "decision_region_model_prob" in frame.columns:
        frame["policy_decision_model_prob"] = pd.to_numeric(frame["decision_region_model_prob"], errors="coerce").fillna(
            pd.to_numeric(frame["policy_prob"], errors="coerce").fillna(0.5)
        )
    if "decision_region_model_score" in frame.columns:
        scorer_score = pd.to_numeric(frame["decision_region_model_score"], errors="coerce")
        frame["policy_decision_score"] = scorer_score.combine_first(frame["policy_decision_score"])
        frame["policy_rank_score"] = frame["policy_decision_score"]
    frame["policy_decision_scorer"] = frame.get(
        "decision_region_scorer",
        pd.Series("heuristic", index=frame.index, dtype=object),
    ).fillna("heuristic").astype(str)
    frame["policy_decision_training_scope"] = frame.get(
        "decision_region_training_scope",
        pd.Series("", index=frame.index, dtype=object),
    ).fillna("").astype(str)
    frame["policy_uncertainty_multiplier"] = _uncertainty_multiplier(frame)
    frame["policy_base_kelly"] = _base_kelly_fraction(frame)
    return frame


def _apply_policy_scope(candidate_rows: pd.DataFrame, policy: BetPolicy) -> pd.DataFrame:
    working = candidate_rows.copy()
    if policy.allowed_leagues:
        working = working[working["league_code"].astype(str).isin(policy.allowed_leagues)].copy()
    if policy.allowed_outcomes:
        working = working[working["selection"].astype(str).isin(policy.allowed_outcomes)].copy()
    return working


def _odds_in_scope(odds: pd.Series, policy: BetPolicy) -> pd.Series:
    lower = odds.ge(float(policy.min_odds))
    if str(policy.scope_name).startswith("mid_odds"):
        upper = odds.lt(float(policy.max_odds))
    else:
        upper = odds.le(float(policy.max_odds))
    return lower & upper


def select_candidate_rows(
    candidate_rows: pd.DataFrame,
    policy: BetPolicy,
    probability_source: str,
    slippage_cushion: float = 0.0,
) -> pd.DataFrame:
    if candidate_rows.empty:
        return pd.DataFrame()

    working = candidate_score_columns(candidate_rows, probability_source=probability_source, slippage_cushion=slippage_cushion)
    working = _apply_policy_scope(working, policy)
    working = working[
        working["quote_status"].astype(str).eq("eligible")
        & _odds_in_scope(working["quoted_odds"], policy)
    ].copy()
    if working.empty:
        return working

    family = str(policy.family or "edge_ev_threshold")
    if family == "edge_ev_threshold":
        working = working[
            working["policy_edge"].ge(policy.edge_threshold)
            & working["policy_ev"].ge(policy.ev_threshold)
        ].copy()
        if working.empty:
            return working
        working = working.sort_values(
            [
                "match_id",
                "policy_decision_score",
                "policy_confidence_score",
                "policy_score_margin",
                "policy_ev",
                "policy_edge",
                "policy_probability_margin",
                "policy_source_consensus",
                "quoted_odds",
                "policy_quote_age_minutes",
            ],
            ascending=[True, False, False, False, False, False, False, False, True, True],
        )
        selected = working.groupby("match_id", observed=True).head(1).copy()
    elif family == "ranked_one_pick":
        working = working[working["policy_rank_score"].gt(0.0)].copy()
        if working.empty:
            return working
        working = working.sort_values(
            [
                "match_id",
                "policy_decision_score",
                "policy_confidence_score",
                "policy_score_margin",
                "policy_probability_margin",
                "policy_prob",
                "policy_edge",
                "quoted_odds",
                "policy_quote_age_minutes",
            ],
            ascending=[True, False, False, False, False, False, False, True, True],
        )
        selected = working.groupby("match_id", observed=True).head(1).copy()
    elif family == "top_quantile":
        working = working[
            working["policy_edge"].ge(policy.edge_threshold)
            & working["policy_ev"].ge(policy.ev_threshold)
        ].copy()
        if working.empty:
            return working
        working = working.sort_values(
            [
                "match_id",
                "policy_decision_score",
                "policy_confidence_score",
                "policy_score_margin",
                "policy_probability_margin",
                "policy_ev",
                "policy_edge",
                "quoted_odds",
                "policy_quote_age_minutes",
            ],
            ascending=[True, False, False, False, False, False, False, True, True],
        )
        working = working.groupby("match_id", observed=True).head(1).copy()
        decision_time = pd.to_datetime(working["decision_time"], utc=True, errors="coerce")
        iso = decision_time.dt.isocalendar()
        week_key = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
        working["policy_week"] = week_key
        rows: list[pd.DataFrame] = []
        quantile = float(np.clip(policy.top_quantile, 0.01, 1.0))
        for _, group in working.groupby("policy_week", observed=True):
            keep = max(1, int(np.ceil(len(group) * quantile)))
            rows.append(
                group.sort_values(
                    [
                        "policy_decision_score",
                        "policy_confidence_score",
                        "policy_score_margin",
                        "policy_probability_margin",
                        "policy_ev",
                        "policy_edge",
                        "policy_quote_age_minutes",
                        "quoted_odds",
                    ],
                    ascending=[False, False, False, False, False, False, True, True],
                ).head(keep)
            )
        selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=working.columns)
    else:
        raise ValueError(f"Familia de politica no soportada: {family}")

    if selected.empty:
        return selected
    selected = selected.copy()
    selected["selection_prob"] = selected["policy_prob"]
    selected["selection_edge"] = selected["policy_edge"]
    selected["selection_ev"] = selected["policy_ev"]
    selected["selection_probability_margin"] = selected["policy_probability_margin"]
    selected["selection_source_consensus"] = selected["policy_source_consensus"]
    selected["selection_quote_freshness"] = selected["policy_quote_freshness"]
    selected["selection_value_buffer"] = selected["policy_value_buffer"]
    selected["selection_confidence_score"] = selected["policy_confidence_score"]
    selected["selection_confidence_divergence_to_v1"] = selected["policy_confidence_divergence_to_v1"]
    selected["confidence_score"] = selected["policy_confidence_score"]
    selected["confidence_divergence_to_v1"] = selected["policy_confidence_divergence_to_v1"]
    selected["policy_confidence_score"] = selected["policy_confidence_score"]
    selected["policy_confidence_divergence_to_v1"] = selected["policy_confidence_divergence_to_v1"]
    selected["expected_fill_probability"] = selected["policy_expected_fill_probability"]
    selected["fill_adjusted_ev"] = selected["policy_fill_adjusted_ev"]
    selected["decision_score"] = selected["policy_decision_score"]
    selected["regional_calibration_confidence"] = selected["policy_regional_calibration_confidence"]
    selected["uncertainty_multiplier"] = selected["policy_uncertainty_multiplier"]
    selected["base_kelly_fraction"] = selected["policy_base_kelly"]
    selected["requested_stake"] = np.minimum(
        float(policy.kelly_fraction),
        selected["policy_base_kelly"].fillna(0.0).clip(lower=0.0) * selected["policy_uncertainty_multiplier"].fillna(0.35),
    )
    selected["policy_score_margin"] = selected["policy_score_margin"]
    selected["policy_family"] = family
    selected["policy_top_quantile"] = float(policy.top_quantile)
    selected["policy_allowed_leagues"] = ",".join(policy.allowed_leagues)
    selected["policy_allowed_outcomes"] = ",".join(policy.allowed_outcomes)
    selected["policy_scope_name"] = str(policy.scope_name)
    selected["policy_decision_scorer"] = selected["policy_decision_scorer"]
    selected["policy_decision_training_scope"] = selected["policy_decision_training_scope"]
    selected["probability_source"] = str(working.attrs.get("resolved_probability_source", probability_source))
    if "decision_id" not in selected.columns:
        selected["decision_id"] = [str(uuid.uuid4()) for _ in range(len(selected))]
    return selected.reset_index(drop=True)
