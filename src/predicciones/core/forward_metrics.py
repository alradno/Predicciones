from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class ForwardDecisionResult:
    lane_id: str
    event_id: str
    selection_id: str
    ev: float | None
    odds: float | None
    stake: float
    profit_units: float | None
    settled: bool
    decision_implied_probability: float | None = None
    closing_implied_probability: float | None = None
    decision_decimal_odds: float | None = None
    closing_decimal_odds: float | None = None


def _finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def calculate_clv_probability(
    decision_implied_probability: float | None,
    closing_implied_probability: float | None,
) -> float | None:
    decision = _finite_float(decision_implied_probability)
    closing = _finite_float(closing_implied_probability)
    if decision is None or closing is None:
        return None
    return float(closing - decision)


def calculate_clv_decimal_odds(
    decision_decimal_odds: float | None,
    closing_decimal_odds: float | None,
) -> float | None:
    decision = _finite_float(decision_decimal_odds)
    closing = _finite_float(closing_decimal_odds)
    if decision is None or closing is None:
        return None
    return float(decision - closing)


def calculate_max_drawdown_units(profits: list[float]) -> float | None:
    if not profits:
        return None

    cumulative_profit = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for profit in profits:
        cumulative_profit += float(profit)
        peak = max(peak, cumulative_profit)
        max_drawdown = max(max_drawdown, peak - cumulative_profit)
    return float(max_drawdown)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = percentile * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return float(lower_value + ((upper_value - lower_value) * weight))


def bootstrap_roi_lower_bound(
    returns: list[float],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> float | None:
    if len(returns) < 40:
        return None
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    sample_size = len(returns)
    bootstrap_rois: list[float] = []
    for _ in range(n_bootstrap):
        total_return = 0.0
        for _ in range(sample_size):
            total_return += float(rng.choice(returns))
        bootstrap_rois.append(total_return / sample_size)

    lower_tail = 1.0 - confidence
    return _percentile(bootstrap_rois, lower_tail)


def _ev_bucket(ev: float | None) -> str:
    value = _finite_float(ev)
    if value is None:
        return "unknown"
    if value < 0.0:
        return "ev < 0"
    if value < 0.02:
        return "0 <= ev < 0.02"
    if value < 0.05:
        return "0.02 <= ev < 0.05"
    return "ev >= 0.05"


def _odds_bucket(odds: float | None) -> str:
    value = _finite_float(odds)
    if value is None:
        return "unknown"
    if value < 1.2:
        return "odds < 1.2"
    if value < 1.5:
        return "1.2 <= odds < 1.5"
    if value < 2.0:
        return "1.5 <= odds < 2.0"
    if value < 3.0:
        return "2.0 <= odds < 3.0"
    if value < 6.0:
        return "3.0 <= odds < 6.0"
    return "odds >= 6.0"


def _settled_returns(decisions: list[ForwardDecisionResult]) -> list[float]:
    returns: list[float] = []
    for decision in decisions:
        profit = _finite_float(decision.profit_units)
        if decision.settled and profit is not None:
            returns.append(profit)
    return returns


def _bucket_report(
    decisions: list[ForwardDecisionResult],
    *,
    bucket_names: list[str],
    bucket_for_decision,
) -> list[dict]:
    rows: list[dict] = []
    for bucket_name in bucket_names:
        bucket_decisions = [decision for decision in decisions if bucket_for_decision(decision) == bucket_name]
        returns = _settled_returns(bucket_decisions)
        profit_units = float(sum(returns)) if returns else 0.0
        rows.append(
            {
                "bucket": bucket_name,
                "decisions": len(bucket_decisions),
                "settled_decisions": len(returns),
                "profit_units": profit_units,
                "roi": float(profit_units / len(returns)) if returns else None,
            }
        )
    return rows


def build_ev_bucket_report(decisions: list[ForwardDecisionResult]) -> list[dict]:
    return _bucket_report(
        decisions,
        bucket_names=[
            "ev < 0",
            "0 <= ev < 0.02",
            "0.02 <= ev < 0.05",
            "ev >= 0.05",
            "unknown",
        ],
        bucket_for_decision=lambda decision: _ev_bucket(decision.ev),
    )


def build_odds_bucket_report(decisions: list[ForwardDecisionResult]) -> list[dict]:
    return _bucket_report(
        decisions,
        bucket_names=[
            "odds < 1.2",
            "1.2 <= odds < 1.5",
            "1.5 <= odds < 2.0",
            "2.0 <= odds < 3.0",
            "3.0 <= odds < 6.0",
            "odds >= 6.0",
            "unknown",
        ],
        bucket_for_decision=lambda decision: _odds_bucket(decision.odds),
    )


def build_forward_metrics_report(decisions: list[ForwardDecisionResult]) -> dict:
    returns = _settled_returns(decisions)
    profit_units = float(sum(returns)) if returns else None
    clv_probabilities = [
        value
        for value in (
            calculate_clv_probability(
                decision.decision_implied_probability,
                decision.closing_implied_probability,
            )
            for decision in decisions
        )
        if value is not None
    ]
    clv_decimal_odds = [
        value
        for value in (
            calculate_clv_decimal_odds(
                decision.decision_decimal_odds,
                decision.closing_decimal_odds,
            )
            for decision in decisions
        )
        if value is not None
    ]

    return {
        "settled_decisions": len(returns),
        "flat_stake_roi": float(profit_units / len(returns)) if returns else None,
        "flat_stake_profit_units": profit_units,
        "max_drawdown_units": calculate_max_drawdown_units(returns),
        "avg_clv_probability": _average(clv_probabilities),
        "median_clv_probability": float(median(clv_probabilities)) if clv_probabilities else None,
        "avg_clv_decimal_odds": _average(clv_decimal_odds),
        "median_clv_decimal_odds": float(median(clv_decimal_odds)) if clv_decimal_odds else None,
        "roi_lower_bound_95": bootstrap_roi_lower_bound(returns, confidence=0.95),
        "ev_bucket_report": build_ev_bucket_report(decisions),
        "odds_bucket_report": build_odds_bucket_report(decisions),
    }
