from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FILL_PRIORS = {
    "exact": 0.78,
    "proxy": 0.55,
    "resolution_only": 0.10,
    "user_quote": 0.92,
    "closing_proxy": 0.45,
    "unknown": 0.55,
}


def _clamp01(values: pd.Series | float) -> pd.Series | float:
    if isinstance(values, pd.Series):
        return values.clip(lower=0.0, upper=1.0)
    return float(np.clip(values, 0.0, 1.0))


def _float_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _text_series(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=str)
    return frame[column].fillna(default).astype(str)


def _bucket_label(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(str).replace({"nan": ""})


def fit_fill_probability_priors(fill_rows: pd.DataFrame, *, min_rows: int = 20) -> dict[str, Any]:
    if fill_rows.empty or "fill_rate" not in fill_rows.columns:
        return {
            "mode": "default",
            "rows": 0,
            "overall_fill_rate": 0.0,
            "feature_priors": {},
        }
    working = fill_rows.copy()
    working["observed_fill_rate"] = pd.to_numeric(working["fill_rate"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    feature_priors: dict[str, dict[str, float]] = {}
    feature_counts: dict[str, dict[str, int]] = {}
    bucketed_features = {
        "price_provenance": _text_series(working, "price_provenance", "unknown"),
        "selection": _text_series(working, "selection", ""),
        "league_code": _text_series(working, "league_code", ""),
        "notional_bucket": _bucket_label(
            _float_series(working, "notional", default=0.0),
            [0.0, 10.0, 25.0, 50.0, 100.0, np.inf],
            ["<=10", "10-25", "25-50", "50-100", ">100"],
        ),
        "top_ask_bucket": _bucket_label(
            _float_series(working, "top_ask", default=np.nan),
            [0.0, 0.20, 0.40, 0.60, 0.80, 1.0],
            ["<=0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", ">0.80"],
        ),
        "age_bucket": _bucket_label(
            _float_series(working, "book_age_seconds", default=np.nan),
            [-np.inf, 5.0, 15.0, 60.0, np.inf],
            ["<=5s", "5-15s", "15-60s", ">60s"],
        ),
        "minutes_bucket": _bucket_label(
            _float_series(working, "minutes_to_kickoff", default=np.nan),
            [-np.inf, 15.0, 45.0, 120.0, np.inf],
            ["<=15m", "15-45m", "45-120m", ">120m"],
        ),
    }
    for feature, values in bucketed_features.items():
        grouped = (
            pd.DataFrame({"bucket": values, "fill_rate": working["observed_fill_rate"]})
            .groupby("bucket", observed=True)
            .agg(rows=("fill_rate", "size"), mean_fill_rate=("fill_rate", "mean"))
            .reset_index()
        )
        if grouped.empty:
            continue
        feature_priors[feature] = {}
        feature_counts[feature] = {}
        for row in grouped.itertuples(index=False):
            if int(row.rows) < min_rows:
                continue
            bucket = str(row.bucket)
            if not bucket:
                continue
            feature_priors[feature][bucket] = float(row.mean_fill_rate)
            feature_counts[feature][bucket] = int(row.rows)
    return {
        "mode": "observed",
        "rows": int(len(working)),
        "overall_fill_rate": float(working["observed_fill_rate"].mean()),
        "feature_priors": feature_priors,
        "feature_counts": feature_counts,
    }


def _prior_lookup(priors: dict[str, Any] | None, feature: str, bucket: str) -> float | None:
    if not priors:
        return None
    feature_priors = priors.get("feature_priors", {})
    if feature not in feature_priors:
        return None
    value = feature_priors[feature].get(bucket)
    if value is None:
        return None
    return float(value)


def estimate_fill_probability(
    frame: pd.DataFrame,
    *,
    priors: dict[str, Any] | None = None,
    default_notional: float = 1.0,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    provenance = _text_series(frame, "price_provenance", "")
    source_type = _text_series(frame, "source_type", "")
    quote_age_minutes = _float_series(frame, "quote_age_minutes", default=np.nan)
    book_age_seconds = _float_series(frame, "book_age_seconds", default=np.nan)
    top_ask = _float_series(frame, "top_ask", default=np.nan)
    available_ask_size = _float_series(frame, "available_ask_size", default=np.nan)
    notional = _float_series(frame, "notional", default=default_notional).fillna(default_notional)
    minutes_to_kickoff = _float_series(frame, "minutes_to_kickoff", default=np.nan)

    source_prior = provenance.map(DEFAULT_FILL_PRIORS).fillna(
        source_type.map(DEFAULT_FILL_PRIORS).fillna(DEFAULT_FILL_PRIORS["unknown"])
    )
    age_seconds = book_age_seconds.fillna(quote_age_minutes * 60.0)
    freshness = 1.0 - np.clip(age_seconds.fillna(300.0) / 300.0, 0.0, 1.0)
    kickoff_alignment = 1.0 - np.clip((minutes_to_kickoff.fillna(45.0) - 45.0).abs() / 180.0, 0.0, 1.0)
    ask_depth_ratio = np.clip(available_ask_size.fillna(notional) / notional.replace(0.0, np.nan), 0.0, 1.0).fillna(0.5)
    price_balance = 1.0 - np.clip((top_ask.fillna(0.5) - 0.5).abs() / 0.5, 0.0, 1.0)

    values = (
        0.40 * source_prior
        + 0.20 * freshness
        + 0.15 * kickoff_alignment
        + 0.15 * ask_depth_ratio
        + 0.10 * price_balance
    )

    if priors:
        prior_components: list[pd.Series] = []
        lookup_rows = {
            "price_provenance": provenance,
            "selection": _text_series(frame, "selection", ""),
            "league_code": _text_series(frame, "league_code", ""),
            "notional_bucket": _bucket_label(
                notional,
                [0.0, 10.0, 25.0, 50.0, 100.0, np.inf],
                ["<=10", "10-25", "25-50", "50-100", ">100"],
            ),
            "top_ask_bucket": _bucket_label(
                top_ask,
                [0.0, 0.20, 0.40, 0.60, 0.80, 1.0],
                ["<=0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", ">0.80"],
            ),
            "age_bucket": _bucket_label(
                age_seconds,
                [-np.inf, 5.0, 15.0, 60.0, np.inf],
                ["<=5s", "5-15s", "15-60s", ">60s"],
            ),
            "minutes_bucket": _bucket_label(
                minutes_to_kickoff,
                [-np.inf, 15.0, 45.0, 120.0, np.inf],
                ["<=15m", "15-45m", "45-120m", ">120m"],
            ),
        }
        for feature, buckets in lookup_rows.items():
            prior_components.append(
                buckets.map(lambda bucket: _prior_lookup(priors, feature, str(bucket))).astype(float)
            )
        prior_frame = pd.concat(prior_components, axis=1)
        prior_mean = prior_frame.mean(axis=1, skipna=True).fillna(float(priors.get("overall_fill_rate", 0.0)))
        values = (0.55 * values) + (0.45 * prior_mean)

    return _clamp01(values.fillna(DEFAULT_FILL_PRIORS["unknown"]))


def attach_fill_adjusted_ev(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    edge_column: str | None = None,
    priors: dict[str, Any] | None = None,
    default_notional: float = 1.0,
) -> pd.DataFrame:
    if frame.empty:
        output = frame.copy()
        output["expected_fill_probability"] = pd.Series(dtype=float)
        output["expected_edge_after_costs"] = pd.Series(dtype=float)
        output["fill_adjusted_ev"] = pd.Series(dtype=float)
        return output
    output = frame.copy()
    top_ask = _float_series(output, "top_ask", default=np.nan)
    fee_rate = _float_series(output, "fee_rate", default=0.0).fillna(0.0)
    slippage_cost = _float_series(output, "slippage_cushion", default=0.0).fillna(0.0)
    if "policy_value_buffer" in output.columns:
        edge_after_costs = _float_series(output, "policy_value_buffer", default=np.nan)
    else:
        probability = _float_series(output, probability_column, default=np.nan)
        base_edge = _float_series(output, edge_column, default=np.nan) if edge_column else probability - top_ask
        fee_cost = fee_rate * top_ask * (1.0 - top_ask)
        edge_after_costs = base_edge - fee_cost - slippage_cost
    output["expected_fill_probability"] = estimate_fill_probability(output, priors=priors, default_notional=default_notional)
    output["expected_edge_after_costs"] = edge_after_costs.fillna(0.0)
    output["fill_adjusted_ev"] = output["expected_fill_probability"] * output["expected_edge_after_costs"]
    return output


def build_clv_rows(
    frame: pd.DataFrame,
    *,
    executed_odds_column: str,
    closing_reference_odds_column: str | None = None,
    executed_prob_column: str | None = None,
    closing_reference_prob_column: str | None = None,
    source_column: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "clv_odds",
                "clv_prob",
                "closing_reference_odds",
                "closing_reference_prob",
                "executed_odds",
                "executed_implied_prob",
                "clv_source",
            ]
        )
    output = frame.copy()
    executed_odds = _float_series(output, executed_odds_column, default=np.nan)
    if executed_prob_column:
        executed_prob = _float_series(output, executed_prob_column, default=np.nan)
    else:
        executed_prob = np.where(executed_odds > 0.0, 1.0 / executed_odds, np.nan)
        executed_prob = pd.Series(executed_prob, index=output.index, dtype=float)
    closing_odds = _float_series(output, closing_reference_odds_column, default=np.nan) if closing_reference_odds_column else pd.Series(np.nan, index=output.index, dtype=float)
    if closing_reference_prob_column:
        closing_prob = _float_series(output, closing_reference_prob_column, default=np.nan)
    else:
        closing_prob = np.where(closing_odds > 0.0, 1.0 / closing_odds, np.nan)
        closing_prob = pd.Series(closing_prob, index=output.index, dtype=float)
    output["executed_odds"] = executed_odds
    output["executed_implied_prob"] = executed_prob
    output["closing_reference_odds"] = closing_odds
    output["closing_reference_prob"] = closing_prob
    output["clv_odds"] = executed_odds - closing_odds
    output["clv_prob"] = closing_prob - executed_prob
    output["clv_source"] = _text_series(output, source_column, "missing") if source_column else "missing"
    mask = output["closing_reference_prob"].notna() & output["executed_implied_prob"].notna()
    return output[mask].copy()


def summarize_clv(clv_rows: pd.DataFrame, *, total_rows: int = 0) -> dict[str, Any]:
    total = int(total_rows or len(clv_rows))
    if clv_rows.empty:
        return {
            "rows": 0,
            "coverage": 0.0 if total <= 0 else 0.0,
            "mean_clv": 0.0,
            "mean_clv_odds": 0.0,
            "median_clv": 0.0,
            "positive_rate": 0.0,
            "source_counts": {},
        }
    return {
        "rows": int(len(clv_rows)),
        "coverage": float(len(clv_rows) / total) if total > 0 else 0.0,
        "mean_clv": float(pd.to_numeric(clv_rows["clv_prob"], errors="coerce").mean()),
        "mean_clv_odds": float(pd.to_numeric(clv_rows["clv_odds"], errors="coerce").mean()),
        "median_clv": float(pd.to_numeric(clv_rows["clv_prob"], errors="coerce").median()),
        "positive_rate": float(pd.to_numeric(clv_rows["clv_prob"], errors="coerce").ge(0.0).mean()),
        "source_counts": clv_rows["clv_source"].value_counts().to_dict() if "clv_source" in clv_rows.columns else {},
    }


def summarize_fill_adjusted_ev(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "fill_adjusted_ev" not in frame.columns:
        return {"mean": 0.0, "median": 0.0, "positive_rate": 0.0}
    values = pd.to_numeric(frame["fill_adjusted_ev"], errors="coerce").dropna()
    if values.empty:
        return {"mean": 0.0, "median": 0.0, "positive_rate": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "positive_rate": float(values.ge(0.0).mean()),
    }


def summarize_sizing(frame: pd.DataFrame, *, requested_column: str = "requested_stake") -> dict[str, Any]:
    if frame.empty or requested_column not in frame.columns:
        return {"mean_requested_stake": 0.0, "median_requested_stake": 0.0, "positive_rate": 0.0}
    values = pd.to_numeric(frame[requested_column], errors="coerce").dropna()
    if values.empty:
        return {"mean_requested_stake": 0.0, "median_requested_stake": 0.0, "positive_rate": 0.0}
    return {
        "mean_requested_stake": float(values.mean()),
        "median_requested_stake": float(values.median()),
        "positive_rate": float(values.gt(0.0).mean()),
    }
