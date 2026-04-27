from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Settings
from ..core.edge_hypothesis import EdgeHypothesisInputs, classify_edge_hypothesis
from ..core.forward_metrics import ForwardDecisionResult, build_forward_metrics_report
from ..core.promotion_state import PromotionInputs, classify_promotion_status
from ..reporting import create_run_context
from .runtime import default_multi_market_db_path, get_market_lane_spec, init_multi_market_db


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def _settlement_wins(selected: str, contract_outcome: str, winning: str) -> bool:
    selected_key = _normalized(selected)
    contract_key = _normalized(contract_outcome)
    winning_key = _normalized(winning)
    return bool(winning_key and winning_key in {selected_key, contract_key})


def _book_age_seconds(decision_time: Any, book_timestamp: Any) -> float | None:
    decision = pd.to_datetime(decision_time, utc=True, errors="coerce")
    book = pd.to_datetime(book_timestamp, utc=True, errors="coerce")
    if pd.isna(decision) or pd.isna(book):
        return None
    return float((decision - book).total_seconds())


def _rolling_roi_min(returns: list[float], window: int = 50) -> float | None:
    if len(returns) < 2:
        return None
    effective_window = min(window, len(returns))
    rois = [
        float(sum(returns[index : index + effective_window]) / effective_window)
        for index in range(0, len(returns) - effective_window + 1)
    ]
    return min(rois) if rois else None


def _load_forward_rows(connection, lane_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT *
        FROM mm_lane_forward_ledger
        WHERE lane_id = ?
          AND decision_status IN ('valid_forward_sample', 'settled')
        ORDER BY decision_time, created_at, decision_id
        """,
        connection,
        params=(lane_id,),
    )


def _load_settlements(connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT market_id, event_slug, winning_outcome, resolved_at, status
        FROM mm_raw_settlements
        WHERE COALESCE(winning_outcome, '') <> ''
        """,
        connection,
    )


def evaluate_forward_lane(
    settings: Settings,
    lane_id: str,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    spec = get_market_lane_spec(lane_id)
    database_path = Path(db_path) if db_path else default_multi_market_db_path(settings)
    connection = init_multi_market_db(database_path)
    try:
        rows = _load_forward_rows(connection, spec.lane_id)
        settlements = _load_settlements(connection)
    finally:
        connection.close()

    settlement_by_market = {
        str(row.market_id): row
        for row in settlements.itertuples(index=False)
        if str(row.market_id or "")
    }
    fresh_flags: list[bool] = []
    decisions: list[ForwardDecisionResult] = []
    returns: list[float] = []

    for row in rows.itertuples(index=False):
        top_ask = _safe_float(getattr(row, "top_ask", None))
        quoted_odds = _safe_float(getattr(row, "quoted_odds", None))
        if quoted_odds is None and top_ask is not None and top_ask > 0.0:
            quoted_odds = float(1.0 / top_ask)

        book_age = _book_age_seconds(getattr(row, "decision_time", None), getattr(row, "book_timestamp", None))
        if top_ask is not None:
            fresh_flags.append(
                bool(book_age is not None and 0.0 <= book_age <= settings.polymarket.decision_book_freshness_seconds)
            )

        settlement = settlement_by_market.get(str(getattr(row, "market_id", "")))
        settled = settlement is not None
        won = False
        profit_units: float | None = None
        if settled:
            won = _settlement_wins(
                selected=str(getattr(row, "selected_outcome", "")),
                contract_outcome=str(getattr(row, "contract_outcome", "")),
                winning=str(getattr(settlement, "winning_outcome", "")),
            )
            profit_units = float((quoted_odds or 1.0) - 1.0) if won else -1.0
            returns.append(profit_units)

        closing_top_ask = _safe_float(getattr(row, "closing_top_ask", None))
        decisions.append(
            ForwardDecisionResult(
                lane_id=spec.lane_id,
                event_id=str(getattr(row, "event_slug", "")),
                selection_id=str(getattr(row, "selected_outcome", "")),
                ev=_safe_float(getattr(row, "ev", None)),
                odds=quoted_odds,
                stake=1.0,
                profit_units=profit_units,
                settled=settled,
                decision_implied_probability=top_ask,
                closing_implied_probability=closing_top_ask,
                decision_decimal_odds=quoted_odds,
                closing_decimal_odds=(1.0 / closing_top_ask) if closing_top_ask and closing_top_ask > 0.0 else None,
            )
        )

    metrics = build_forward_metrics_report(decisions)
    fresh_book_rate = float(sum(1 for flag in fresh_flags if flag) / len(fresh_flags)) if fresh_flags else 0.0
    observed_roi = metrics.get("flat_stake_roi")
    avg_clv = metrics.get("avg_clv_probability")
    rolling_roi_min = _rolling_roi_min(returns)
    promotion_decision = classify_promotion_status(
        PromotionInputs(
            valid_forward_decisions=int(len(rows)),
            settled_decisions=int(metrics["settled_decisions"]),
            fresh_book_rate=fresh_book_rate,
            roi_lower_bound_95=metrics.get("roi_lower_bound_95"),
            avg_clv=avg_clv,
            max_drawdown_units=metrics.get("max_drawdown_units"),
        )
    )
    edge_decision = classify_edge_hypothesis(
        EdgeHypothesisInputs(
            lane_id=spec.lane_id,
            valid_forward_decisions=int(len(rows)),
            settled_decisions=int(metrics["settled_decisions"]),
            fresh_book_rate=fresh_book_rate,
            observed_roi=observed_roi,
            roi_lower_bound_95=metrics.get("roi_lower_bound_95"),
            avg_clv=avg_clv,
            rolling_roi_min=rolling_roi_min,
            promotion_status=promotion_decision.promotion_status,
        )
    )

    run = create_run_context(settings.paths.runs_dir, f"{spec.lane_id}_forward_evaluation")
    lane_dir = settings.paths.outputs_dir / "lanes" / spec.lane_id
    lane_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "lane_id": spec.lane_id,
        "benchmark_id": spec.benchmark_id,
        "database_path": str(database_path),
        "valid_forward_decisions": int(len(rows)),
        "settled_decisions": int(metrics["settled_decisions"]),
        "fresh_book_rate": fresh_book_rate,
        "forward_metrics": metrics,
        "rolling_roi_min": rolling_roi_min,
        "promotion_status": promotion_decision.promotion_status.value,
        "promotion_blockers": list(promotion_decision.promotion_blockers),
        "paper_stake_allowed": promotion_decision.paper_stake_allowed,
        "capital_stake_allowed": promotion_decision.capital_stake_allowed,
        "roi_45_hypothesis": edge_decision.to_dict(),
        "policy_reoptimized": False,
        "locked_holdout_used_for_training": False,
    }
    artifacts = {
        "run_forward_evaluation_report": run.run_dir / "forward_evaluation_report.json",
        "lane_forward_evaluation_report": lane_dir / "forward_evaluation_report.json",
    }
    for path in artifacts.values():
        path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    sample_path = lane_dir / "sample_report.json"
    if sample_path.exists():
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        payload["roi_45_hypothesis"] = edge_decision.to_dict()
        payload["forward_metrics"] = metrics
        payload["promotion_status"] = promotion_decision.promotion_status.value
        payload["promotion_blockers"] = list(promotion_decision.promotion_blockers)
        sample_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return summary, artifacts
