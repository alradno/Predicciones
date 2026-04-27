from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtest import _build_prediction_frame
from ..config import Settings
from ..contracts import PolymarketShadowResult
from ..core.promotion_state import (
    ForwardSampleInputs,
    ForwardSampleThresholds,
    SampleStatus,
    classify_forward_sample,
)
from ..execution_quality import (
    attach_fill_adjusted_ev,
    build_clv_rows,
    fit_fill_probability_priors,
    summarize_clv,
    summarize_fill_adjusted_ev,
    summarize_sizing,
)
from ..data_sources import PolymarketClobClient, PolymarketGammaClient
from ..football.dataset import build_fixture_feature_rows
from ..ingestion import build_market_odds, canonicalize_matches
from ..models import outcome_probabilities_from_lambdas
from ..markets.common import (
    BUNDLE_STATUS_PROVISIONAL,
    GROUP_ROLE_ORDER,
    PRICE_PROVENANCE_EXACT,
    SOURCE_MODE_FORWARD,
    SOURCE_MODE_RETRO,
    VALIDATION_STAGE_SHADOW,
    build_polymarket_lifecycle_summary,
    format_polymarket_lifecycle_label,
    _BOOK_AGE_BUCKET_ORDER,
    _book_age_bucket,
    _clean_json,
    _iso_timestamp,
    _skip_reason_blocker,
    _utcnow,
)
from ..markets.storage import (
    _catalog_lookup,
    _checkpoint_lookup,
    _group_lookup,
    _resolution_lookup,
    _upsert_rows,
    default_polymarket_db_path,
    init_polymarket_db,
)
from ..markets.ingest import collect_polymarket, discover_market_catalog
from ..markets.sim import (
    _decision_from_prediction,
    _derive_fixtures_from_groups,
    _load_policy_bundle,
    _match_fixture_to_group,
    _policy_from_raw,
    _settle_fill_row,
)
from ..reporting import _save_json, create_run_context


FORWARD_SAMPLE_TARGET_VALID_DECISIONS = 100
FORWARD_SAMPLE_TARGET_SETTLED_DECISIONS = 40
FORWARD_SAMPLE_TARGET_FRESH_BOOK_RATE = 0.80


def _drawdown_from_profit(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    bank = values.cumsum()
    peak = bank.cummax()
    return float((bank - peak).min())


def _plot_shadow_bank_curve(fill_rows: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    if fill_rows.empty:
        plt.title("Polymarket Shadow Bank Curve")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return
    for notional, group in fill_rows.groupby("notional", observed=True):
        settled = group[group["status"] == "settled"].sort_values("created_at")
        if settled.empty:
            continue
        bank = settled["net_profit"].cumsum()
        plt.plot(range(1, len(bank) + 1), bank, label=f"{int(notional)} USDC")
    plt.title("Polymarket Shadow Bank Curve")
    plt.xlabel("Bets")
    plt.ylabel("Net Profit (USDC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _table_columns(connection, table_name: str) -> set[str]:
    try:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    except Exception:  # pragma: no cover - defensive against schema races
        return set()
    return {str(row[1]) for row in rows if len(row) > 1}


def _prune_rows_to_table(connection, table_name: str, rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    table_columns = _table_columns(connection, table_name)
    if not table_columns:
        return rows
    keep_columns = [column for column in rows.columns if column in table_columns]
    return rows.loc[:, keep_columns].copy()


def _text_column(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=object)
    values = frame[column].fillna(default).astype(str)
    return values.replace({"nan": default, "None": default, "NaT": default})


def _number_column(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _has_text(values: pd.Series) -> pd.Series:
    return ~values.astype(str).str.strip().isin({"", "nan", "None", "NaT"})


def _forward_sample_flags(
    decisions: pd.DataFrame,
    decision_book_freshness_seconds: float = 15.0,
) -> pd.DataFrame:
    frame = decisions.copy()
    required_columns = [
        "decision_id",
        "run_id",
        "match_id",
        "group_key",
        "league_code",
        "kickoff_time",
        "decision_time",
        "created_at",
        "mapping_status",
        "selection",
        "skip_reason",
        "price_provenance",
        "validation_stage",
        "model_prob",
        "top_ask",
        "expected_edge",
        "expected_ev",
        "available_ask_size",
        "minutes_to_kickoff",
        "book_age_seconds",
        "snapshot_time",
    ]
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = np.nan if column in {"model_prob", "top_ask", "expected_edge", "expected_ev", "available_ask_size", "minutes_to_kickoff", "book_age_seconds"} else ""

    selection = _text_column(frame, "selection")
    skip_reason = _text_column(frame, "skip_reason")
    mapping_status = _text_column(frame, "mapping_status")
    group_key = _text_column(frame, "group_key")
    snapshot_time = _text_column(frame, "snapshot_time")
    price_provenance = _text_column(frame, "price_provenance")
    book_age_seconds = _number_column(frame, "book_age_seconds")
    model_prob = _number_column(frame, "model_prob")
    top_ask = _number_column(frame, "top_ask")

    policy_selected = selection.str.strip().ne("") & skip_reason.str.strip().eq("")
    coverage_statuses = {"missing_book", "stale_book", "no_ask_ladder"}
    coverage_reasons = {"no_book_checkpoint", "stale_book", "no_ask_ladder"}
    complete_mapping = (
        mapping_status.str.strip().eq("complete")
        | (
            _has_text(group_key)
            & (
                mapping_status.str.strip().isin(coverage_statuses)
                | skip_reason.str.strip().isin(coverage_reasons)
            )
        )
    )
    has_book_snapshot = _has_text(snapshot_time)
    fresh_book = book_age_seconds.notna() & book_age_seconds.le(float(decision_book_freshness_seconds))
    finite_price = model_prob.between(0.0, 1.0, inclusive="neither") & top_ask.between(0.0, 1.0, inclusive="neither")
    exact_price = price_provenance.str.strip().eq(PRICE_PROVENANCE_EXACT) | (
        price_provenance.str.strip().eq("") & has_book_snapshot
    )
    valid_forward_sample = policy_selected & complete_mapping & has_book_snapshot & fresh_book & finite_price & exact_price

    blocker = pd.Series("other", index=frame.index, dtype=object)
    blocker.loc[valid_forward_sample] = ""
    blocker.loc[~complete_mapping] = "mapping"
    blocker.loc[complete_mapping & ~policy_selected & skip_reason.str.strip().eq("policy_rejected")] = "policy_not_in_zone"
    blocker.loc[complete_mapping & ~policy_selected & selection.str.strip().eq("") & skip_reason.str.strip().eq("")] = "policy_not_in_zone"
    blocker.loc[complete_mapping & skip_reason.str.strip().eq("no_book_checkpoint")] = "coverage_missing_book"
    blocker.loc[complete_mapping & skip_reason.str.strip().eq("no_ask_ladder")] = "coverage_missing_book"
    blocker.loc[complete_mapping & skip_reason.str.strip().eq("stale_book")] = "coverage_stale_book"
    blocker.loc[
        complete_mapping
        & (~has_book_snapshot)
        & ~valid_forward_sample
        & ~skip_reason.str.strip().eq("stale_book")
    ] = "coverage_missing_book"
    blocker.loc[complete_mapping & has_book_snapshot & ~fresh_book & ~valid_forward_sample] = "coverage_stale_book"
    blocker.loc[complete_mapping & policy_selected & has_book_snapshot & fresh_book & ~finite_price] = "invalid_price"
    blocker.loc[complete_mapping & policy_selected & has_book_snapshot & fresh_book & finite_price & ~exact_price] = "non_exact_price"

    frame["policy_selected"] = policy_selected.astype(int)
    frame["complete_mapping"] = complete_mapping.astype(int)
    frame["has_book_snapshot"] = has_book_snapshot.astype(int)
    frame["fresh_book"] = fresh_book.astype(int)
    frame["finite_price"] = finite_price.astype(int)
    frame["exact_price"] = exact_price.astype(int)
    frame["valid_forward_sample"] = valid_forward_sample.astype(int)
    frame["forward_sample_blocker"] = blocker
    frame["decision_month"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce").dt.tz_localize(None).dt.to_period("M").astype(str)
    frame.loc[frame["decision_month"].eq("NaT"), "decision_month"] = ""
    return frame


def _blocker_summary(flagged: pd.DataFrame) -> list[dict[str, Any]]:
    if flagged.empty or "forward_sample_blocker" not in flagged.columns:
        return []
    blocked = flagged[flagged["forward_sample_blocker"].astype(str).str.strip().ne("")]
    if blocked.empty:
        return []
    rows = (
        blocked.groupby("forward_sample_blocker", observed=True)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "forward_sample_blocker"], ascending=[False, True])
    )
    return rows.to_dict(orient="records")


def _summarize_forward_frame(
    decisions: pd.DataFrame,
    fills: pd.DataFrame,
    decision_book_freshness_seconds: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    flagged = _forward_sample_flags(decisions, decision_book_freshness_seconds)
    valid = flagged[flagged["valid_forward_sample"].astype(int).eq(1)].copy()

    if fills.empty:
        fill_rows = fills.copy()
    else:
        fill_rows = fills.copy()
        if "decision_id" not in fill_rows.columns:
            fill_rows["decision_id"] = ""
    valid_decision_ids = set(valid["decision_id"].astype(str).tolist()) if "decision_id" in valid.columns else set()
    settled_fills = fill_rows[
        _text_column(fill_rows, "status").str.strip().eq("settled")
        & _text_column(fill_rows, "decision_id").isin(valid_decision_ids)
    ].copy() if not fill_rows.empty else pd.DataFrame()

    total_decisions = int(len(flagged))
    policy_selected_decisions = int(flagged["policy_selected"].sum()) if not flagged.empty else 0
    valid_forward_decisions = int(len(valid))
    settled_unique_decisions = int(settled_fills["decision_id"].astype(str).nunique()) if not settled_fills.empty else 0
    total_cost = float(pd.to_numeric(settled_fills.get("cost_basis", pd.Series(dtype=float)), errors="coerce").sum()) if not settled_fills.empty else 0.0
    net_pnl = float(pd.to_numeric(settled_fills.get("net_profit", pd.Series(dtype=float)), errors="coerce").sum()) if not settled_fills.empty else 0.0
    fresh_book_rate = float(flagged["fresh_book"].mean()) if not flagged.empty else 0.0
    selected_fresh_book_rate = (
        float(flagged.loc[flagged["policy_selected"].astype(int).eq(1), "fresh_book"].mean())
        if policy_selected_decisions > 0
        else 0.0
    )
    valid_rate = float(valid_forward_decisions / total_decisions) if total_decisions > 0 else 0.0
    settlement_rate = float(settled_unique_decisions / valid_forward_decisions) if valid_forward_decisions > 0 else 0.0

    blockers = _blocker_summary(flagged)
    coverage_blocker_count = int(
        sum(int(item["count"]) for item in blockers if str(item["forward_sample_blocker"]).startswith("coverage_"))
    )

    classifier_fresh_book_rate = fresh_book_rate if total_decisions > 0 else 1.0
    sample_decision = classify_forward_sample(
        ForwardSampleInputs(
            valid_forward_decisions=valid_forward_decisions,
            settled_decisions=settled_unique_decisions,
            fresh_book_rate=classifier_fresh_book_rate,
            selected_candidates=policy_selected_decisions,
        ),
        ForwardSampleThresholds(
            min_valid_forward_decisions=FORWARD_SAMPLE_TARGET_VALID_DECISIONS,
            min_settled_decisions=FORWARD_SAMPLE_TARGET_SETTLED_DECISIONS,
            min_fresh_book_rate=FORWARD_SAMPLE_TARGET_FRESH_BOOK_RATE,
        ),
    )
    sample_status = sample_decision.sample_status.value
    if total_decisions <= 0:
        next_action = "Run predicciones lane run-shadow with the frozen policy to start accumulating forward decisions."
    elif sample_decision.sample_status == SampleStatus.sample_ready:
        next_action = "Forward sample is ready for decision-region analysis under the frozen policy."
    elif sample_decision.sample_status == SampleStatus.settlement_pending:
        next_action = "Keep the policy frozen and wait for more selected forward decisions to settle."
    elif sample_decision.sample_status == SampleStatus.coverage_blocked or coverage_blocker_count > max(policy_selected_decisions, valid_forward_decisions):
        next_action = "Improve forward book capture coverage before drawing ROI conclusions."
    else:
        next_action = "Keep collecting shadow decisions with the frozen policy until sample targets are reached."
    roi_display_mode = (
        "hidden_until_sample_ready"
        if sample_decision.sample_status != SampleStatus.sample_ready
        else ("actionable" if sample_decision.actionable_roi else "diagnostic_only")
    )

    by_league = []
    if not valid.empty:
        by_league = (
            valid.groupby("league_code", observed=True)
            .agg(valid_forward_decisions=("decision_id", "nunique"))
            .reset_index()
            .sort_values("valid_forward_decisions", ascending=False)
            .to_dict(orient="records")
        )
    by_selection = []
    if not valid.empty:
        by_selection = (
            valid.groupby("selection", observed=True)
            .agg(valid_forward_decisions=("decision_id", "nunique"))
            .reset_index()
            .sort_values("valid_forward_decisions", ascending=False)
            .to_dict(orient="records")
        )
    by_month = []
    if not valid.empty:
        by_month = (
            valid.groupby("decision_month", observed=True)
            .agg(valid_forward_decisions=("decision_id", "nunique"))
            .reset_index()
            .sort_values("decision_month")
            .to_dict(orient="records")
        )

    summary = {
        "sample_status": sample_status,
        "sample_blockers": list(sample_decision.sample_blockers),
        "next_action": next_action,
        "target_valid_forward_decisions": FORWARD_SAMPLE_TARGET_VALID_DECISIONS,
        "target_settled_decisions": FORWARD_SAMPLE_TARGET_SETTLED_DECISIONS,
        "target_fresh_book_rate": FORWARD_SAMPLE_TARGET_FRESH_BOOK_RATE,
        "total_decisions": total_decisions,
        "policy_selected_decisions": policy_selected_decisions,
        "valid_forward_decisions": valid_forward_decisions,
        "valid_forward_decision_rate": valid_rate,
        "open_valid_decisions": int(max(valid_forward_decisions - settled_unique_decisions, 0)),
        "settled_decisions": settled_unique_decisions,
        "settled_unique_decisions": settled_unique_decisions,
        "settled_fill_rows": int(len(settled_fills)),
        "settlement_rate": settlement_rate,
        "fresh_book_rate": fresh_book_rate,
        "selected_fresh_book_rate": selected_fresh_book_rate,
        "net_pnl": net_pnl,
        "total_cost_basis": total_cost,
        "net_roi": float(net_pnl / total_cost) if total_cost > 0 else 0.0,
        "actionable_roi": bool(sample_decision.actionable_roi),
        "can_reopen_decision_region_analysis": bool(sample_decision.can_reopen_decision_region_analysis),
        "roi_display_mode": roi_display_mode,
        "capital_promotion_allowed": False,
        "diagnostic_only": {
            "raw_roi": float(net_pnl / total_cost) if total_cost > 0 else 0.0,
            "settled_roi": float(net_pnl / total_cost) if total_cost > 0 else 0.0,
            "net_pnl": net_pnl,
            "total_cost_basis": total_cost,
            "reason": "ROI is not actionable until sample_ready"
            if sample_decision.sample_status != SampleStatus.sample_ready
            else "sample_ready allows decision-region analysis only; it is not capital promotion",
        },
        "blockers": blockers,
        "by_league": by_league,
        "by_selection": by_selection,
        "by_month": by_month,
    }
    return summary, flagged


def _build_forward_sample_report(
    current_decisions: pd.DataFrame,
    current_fills: pd.DataFrame,
    cumulative_decisions: pd.DataFrame,
    cumulative_fills: pd.DataFrame,
    decision_book_freshness_seconds: float = 15.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    current_summary, _ = _summarize_forward_frame(current_decisions, current_fills, decision_book_freshness_seconds)
    cumulative_summary, cumulative_ledger = _summarize_forward_frame(
        cumulative_decisions,
        cumulative_fills,
        decision_book_freshness_seconds,
    )
    return {
        "sample_status": cumulative_summary["sample_status"],
        "sample_blockers": cumulative_summary.get("sample_blockers", []),
        "valid_forward_decisions": cumulative_summary.get("valid_forward_decisions", 0),
        "settled_decisions": cumulative_summary.get("settled_decisions", cumulative_summary.get("settled_unique_decisions", 0)),
        "fresh_book_rate": cumulative_summary.get("fresh_book_rate", 0.0),
        "actionable_roi": bool(cumulative_summary.get("actionable_roi", False)),
        "can_reopen_decision_region_analysis": bool(cumulative_summary.get("can_reopen_decision_region_analysis", False)),
        "roi_display_mode": cumulative_summary.get("roi_display_mode", "hidden_until_sample_ready"),
        "capital_promotion_allowed": False,
        "diagnostic_only": cumulative_summary.get("diagnostic_only", {}),
        "next_action": cumulative_summary["next_action"],
        "policy_reoptimized": False,
        "t45m_policy_touched": False,
        "decision_book_freshness_seconds": float(decision_book_freshness_seconds),
        "current": current_summary,
        "cumulative": cumulative_summary,
    }, cumulative_ledger


def _resolve_default_policy_bundle_path(settings: Settings, explicit_path: Path | str | None = None) -> tuple[Path | None, str]:
    if explicit_path:
        path = Path(explicit_path)
        if path.is_dir():
            path = path / "policy_bundle.json"
        return path, "explicit"
    pointer = settings.paths.outputs_dir / "latest_polymarket_policy.txt"
    if not pointer.exists():
        return None, "model_bundle_fallback"
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        return None, "model_bundle_fallback"
    path = Path(value)
    if path.is_dir():
        path = path / "policy_bundle.json"
    if not path.exists():
        return None, "model_bundle_fallback_missing_latest_policy"
    return path, "latest_polymarket_policy"


def _build_forward_sample_manifest(
    *,
    run_id: str,
    db_path: Path,
    model_path: Path,
    policy_bundle_path: Path | None,
    policy_bundle_resolution: str,
    payload: dict[str, Any],
    probability_source: str,
    policy_source_mode: str,
    policy_payload: dict[str, Any],
    policy_coverage_status: str,
    policy_bundle_status: str,
    policy_minimum_quality: str,
    settings: Settings,
    forward_sample_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": _iso_timestamp(_utcnow()),
        "run_id": str(run_id),
        "database_path": str(db_path),
        "model_path": str(model_path),
        "policy_bundle_path": str(policy_bundle_path) if policy_bundle_path else "",
        "policy_bundle_resolution": policy_bundle_resolution,
        "policy_source_mode": policy_source_mode,
        "policy_reoptimized": False,
        "t45m_policy_touched": False,
        "live_money_used": False,
        "market_type": "1X2",
        "supported_leagues": list(settings.polymarket.supported_leagues),
        "model_variant": str(payload.get("model_variant", payload.get("variant_name", ""))),
        "probability_source": probability_source,
        "decision_offset_minutes": int(settings.polymarket.decision_offset_minutes),
        "decision_book_freshness_seconds": float(settings.polymarket.decision_book_freshness_seconds),
        "book_freshness_seconds_legacy": float(settings.polymarket.book_freshness_seconds),
        "policy": policy_payload,
        "policy_coverage_status": policy_coverage_status,
        "policy_bundle_status": policy_bundle_status,
        "policy_minimum_quality_tier": policy_minimum_quality,
        "forward_sample_status": forward_sample_report.get("sample_status", ""),
        "forward_sample_targets": {
            "valid_forward_decisions": FORWARD_SAMPLE_TARGET_VALID_DECISIONS,
            "settled_unique_decisions": FORWARD_SAMPLE_TARGET_SETTLED_DECISIONS,
            "fresh_book_rate": FORWARD_SAMPLE_TARGET_FRESH_BOOK_RATE,
        },
    }


def _band_label(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype(str).replace({"nan": ""})


def _summarize_decision_region_slice(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []
    rows = []
    for value, group in frame.groupby(column, observed=True):
        cost = float(pd.to_numeric(group["cost_basis"], errors="coerce").sum())
        pnl = float(pd.to_numeric(group["net_profit"], errors="coerce").sum())
        rows.append(
            {
                column: str(value),
                "valid_forward_decisions": int(group["decision_id"].astype(str).nunique()),
                "settled_unique_decisions": int(group.loc[group["settled"].astype(bool), "decision_id"].astype(str).nunique()),
                "settled_fill_rows": int(pd.to_numeric(group["settled_fill_rows"], errors="coerce").sum()),
                "net_pnl": pnl,
                "total_cost_basis": cost,
                "net_roi": float(pnl / cost) if cost > 0 else 0.0,
                "avg_edge": float(pd.to_numeric(group["expected_edge"], errors="coerce").mean()) if "expected_edge" in group.columns else 0.0,
                "avg_ev": float(pd.to_numeric(group["expected_ev"], errors="coerce").mean()) if "expected_ev" in group.columns else 0.0,
            }
        )
    return rows


def _build_forward_decision_region_report(
    forward_sample_report: dict[str, Any],
    forward_sample_ledger: pd.DataFrame,
    cumulative_fills: pd.DataFrame,
) -> dict[str, Any]:
    if forward_sample_report.get("sample_status") != "sample_ready":
        return {
            "status": "not_generated_sample_not_ready",
            "sample_status": forward_sample_report.get("sample_status", ""),
            "required_status": "sample_ready",
            "policy_reoptimized": False,
            "t45m_policy_touched": False,
        }
    ledger = forward_sample_ledger.copy()
    if ledger.empty:
        return {"status": "not_generated_empty_ledger", "policy_reoptimized": False, "t45m_policy_touched": False}
    ledger = ledger[ledger["valid_forward_sample"].astype(int).eq(1)].copy()
    if ledger.empty:
        return {"status": "not_generated_no_valid_forward_sample", "policy_reoptimized": False, "t45m_policy_touched": False}
    fills = cumulative_fills.copy()
    if fills.empty:
        fills = pd.DataFrame(columns=["decision_id", "status", "cost_basis", "net_profit", "fill_adjusted_ev"])
    settled = fills[_text_column(fills, "status").str.strip().eq("settled")].copy()
    fill_summary = (
        settled.groupby("decision_id", observed=True)
        .agg(
            settled_fill_rows=("decision_id", "count"),
            cost_basis=("cost_basis", "sum"),
            net_profit=("net_profit", "sum"),
            fill_adjusted_ev=("fill_adjusted_ev", "mean"),
        )
        .reset_index()
    ) if not settled.empty else pd.DataFrame(columns=["decision_id", "settled_fill_rows", "cost_basis", "net_profit", "fill_adjusted_ev"])
    frame = ledger.merge(fill_summary, on="decision_id", how="left")
    frame["settled_fill_rows"] = pd.to_numeric(frame.get("settled_fill_rows", 0), errors="coerce").fillna(0)
    frame["cost_basis"] = pd.to_numeric(frame.get("cost_basis", 0.0), errors="coerce").fillna(0.0)
    frame["net_profit"] = pd.to_numeric(frame.get("net_profit", 0.0), errors="coerce").fillna(0.0)
    frame["settled"] = frame["settled_fill_rows"].gt(0)
    frame["quoted_odds"] = np.where(pd.to_numeric(frame["top_ask"], errors="coerce").gt(0.0), 1.0 / pd.to_numeric(frame["top_ask"], errors="coerce"), np.nan)
    frame["odds_band"] = _band_label(frame["quoted_odds"], [0.0, 1.5, 2.0, 3.0, 6.0, np.inf], ["<1.5", "1.5-2", "2-3", "3-6", "6+"])
    frame["edge_band"] = _band_label(frame["expected_edge"], [-np.inf, 0.0, 0.03, 0.06, 0.10, np.inf], ["<=0", "0-3pp", "3-6pp", "6-10pp", "10pp+"])
    frame["ev_band"] = _band_label(frame["expected_ev"], [-np.inf, 0.0, 0.05, 0.10, 0.20, np.inf], ["<=0", "0-5pp", "5-10pp", "10-20pp", "20pp+"])
    confidence_column = ""
    for candidate in ("selection_confidence_score", "expected_fill_probability", "model_prob"):
        if candidate in frame.columns and pd.to_numeric(frame[candidate], errors="coerce").notna().any():
            confidence_column = candidate
            break
    if confidence_column:
        ranks = pd.to_numeric(frame[confidence_column], errors="coerce").rank(method="first")
        frame["confidence_decile"] = pd.qcut(ranks, q=min(10, max(1, int(ranks.notna().sum()))), duplicates="drop").astype(str)
    else:
        frame["confidence_decile"] = ""

    cost = float(frame["cost_basis"].sum())
    pnl = float(frame["net_profit"].sum())
    return {
        "status": "generated",
        "sample_status": "sample_ready",
        "policy_reoptimized": False,
        "t45m_policy_touched": False,
        "valid_forward_decisions": int(frame["decision_id"].astype(str).nunique()),
        "settled_unique_decisions": int(frame.loc[frame["settled"], "decision_id"].astype(str).nunique()),
        "net_pnl": pnl,
        "total_cost_basis": cost,
        "net_roi": float(pnl / cost) if cost > 0 else 0.0,
        "confidence_column": confidence_column,
        "by_outcome": _summarize_decision_region_slice(frame, "selection"),
        "by_league": _summarize_decision_region_slice(frame, "league_code"),
        "by_odds_band": _summarize_decision_region_slice(frame, "odds_band"),
        "by_edge_band": _summarize_decision_region_slice(frame, "edge_band"),
        "by_ev_band": _summarize_decision_region_slice(frame, "ev_band"),
        "by_confidence_decile": _summarize_decision_region_slice(frame, "confidence_decile"),
        "by_month": _summarize_decision_region_slice(frame, "decision_month"),
    }


def _summarize_shadow(
    decisions: pd.DataFrame,
    fills: pd.DataFrame,
    mappings: pd.DataFrame,
    decision_window_minutes: float = 45.0,
    decision_book_freshness_seconds: float = 15.0,
    book_freshness_seconds: float | None = None,
    decision_capture_lead_seconds: float = 15.0,
    decision_capture_retry_attempts: int = 3,
    decision_capture_retry_delay_seconds: float = 0.75,
    bundle_status: str = BUNDLE_STATUS_PROVISIONAL,
) -> dict[str, Any]:
    if book_freshness_seconds is not None:
        decision_book_freshness_seconds = float(book_freshness_seconds)

    settled = fills[fills["status"] == "settled"].copy() if not fills.empty else pd.DataFrame()
    total_cost = float(settled["cost_basis"].sum()) if not settled.empty else 0.0
    total_profit = float(settled["net_profit"].sum()) if not settled.empty else 0.0
    capacity_curve = (
        settled.groupby("notional", observed=True)
        .agg(
            bets=("fill_id", "count"),
            pnl=("net_profit", "sum"),
            total_cost=("cost_basis", "sum"),
            fill_rate=("fill_rate", "mean"),
            partial_fill_rate=("partial_fill", "mean"),
        )
        .reset_index()
    ) if not settled.empty else pd.DataFrame(columns=["notional", "bets", "pnl", "total_cost", "fill_rate", "partial_fill_rate"])
    if not capacity_curve.empty:
        capacity_curve["roi"] = np.where(capacity_curve["total_cost"] > 0, capacity_curve["pnl"] / capacity_curve["total_cost"], 0.0)

    roi_by_league = (
        settled.groupby("league_code", observed=True)
        .agg(bets=("fill_id", "count"), pnl=("net_profit", "sum"), total_cost=("cost_basis", "sum"))
        .reset_index()
    ) if not settled.empty and "league_code" in settled.columns else pd.DataFrame(columns=["league_code", "bets", "pnl", "total_cost"])
    if not roi_by_league.empty:
        roi_by_league["roi"] = np.where(roi_by_league["total_cost"] > 0, roi_by_league["pnl"] / roi_by_league["total_cost"], 0.0)

    if not settled.empty and "kickoff_time" in settled.columns:
        settled["window"] = pd.to_datetime(settled["kickoff_time"], utc=True).dt.tz_localize(None).dt.to_period("M").astype(str)
        roi_by_window = (
            settled.groupby("window", observed=True)
            .agg(bets=("fill_id", "count"), pnl=("net_profit", "sum"), total_cost=("cost_basis", "sum"))
            .reset_index()
        )
        roi_by_window["roi"] = np.where(roi_by_window["total_cost"] > 0, roi_by_window["pnl"] / roi_by_window["total_cost"], 0.0)
    else:
        roi_by_window = pd.DataFrame(columns=["window", "bets", "pnl", "total_cost", "roi"])

    skip_counts = (
        decisions[decisions["skip_reason"].astype(str) != ""]
        .groupby("skip_reason", observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    ) if not decisions.empty else pd.DataFrame(columns=["skip_reason", "count"])
    skip_reason_totals = {str(item["skip_reason"]): int(item["count"]) for item in skip_counts.to_dict(orient="records")}

    blocked_decisions = decisions[decisions["skip_reason"].astype(str) != ""].copy() if not decisions.empty else pd.DataFrame()
    blocker_breakdown: list[dict[str, Any]] = []
    blocker_counts: Counter[str] = Counter()
    if not blocked_decisions.empty:
        blocked_decisions["blocker"] = blocked_decisions["skip_reason"].map(_skip_reason_blocker)
        blocker_counts.update(blocked_decisions["blocker"].astype(str).tolist())
        blocker_order = {"mapping": 0, "coverage": 1, "policy": 2, "other": 3, "none": 4}
        for blocker, _count in sorted(blocker_counts.items(), key=lambda item: (-item[1], blocker_order.get(item[0], 99), item[0])):
            subset = blocked_decisions[blocked_decisions["blocker"].astype(str) == blocker]
            reason_counts = (
                subset.groupby("skip_reason", observed=True)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            blocker_breakdown.append(
                {
                    "blocker": blocker,
                    "count": int(len(subset)),
                    "reasons": reason_counts.to_dict(orient="records"),
                }
            )

    snapshot_present = pd.Series(dtype=bool)
    book_age_seconds = pd.Series(dtype=float)
    if not decisions.empty:
        snapshot_present = decisions["snapshot_time"].map(
            lambda value: str(value).strip() not in {"", "NaT", "nan", "None"}
        )
        book_age_seconds = pd.to_numeric(decisions["book_age_seconds"], errors="coerce")
    fresh_mask = book_age_seconds.notna() & book_age_seconds.le(float(decision_book_freshness_seconds))
    stale_mask = book_age_seconds.notna() & book_age_seconds.gt(float(decision_book_freshness_seconds))
    age_bucket_counts = (
        book_age_seconds.dropna().map(_book_age_bucket).value_counts().reset_index()
        if not book_age_seconds.dropna().empty
        else pd.DataFrame(columns=["index", "count"])
    )
    if not age_bucket_counts.empty:
        age_bucket_counts.columns = ["bucket", "count"]
        age_bucket_counts["_order"] = age_bucket_counts["bucket"].map(lambda bucket: _BOOK_AGE_BUCKET_ORDER.get(str(bucket), 99))
        age_bucket_counts = age_bucket_counts.sort_values(["_order", "bucket"]).drop(columns=["_order"]).reset_index(drop=True)
        total_age_rows = float(age_bucket_counts["count"].sum())
        age_bucket_counts["share"] = np.where(total_age_rows > 0, age_bucket_counts["count"] / total_age_rows, 0.0)

    decision_count = int(len(decisions))
    selected_count = int(decisions["selection"].astype(str).ne("").sum()) if not decisions.empty else 0
    covered_count = int(snapshot_present.sum()) if not decisions.empty else 0
    complete_mapping_count = int(decisions["mapping_status"].astype(str).eq("complete").sum()) if not decisions.empty else 0
    price_provenance_counts = (
        decisions["price_provenance"].astype(str).replace({"nan": "", "None": ""}).value_counts().to_dict()
        if not decisions.empty and "price_provenance" in decisions.columns
        else {}
    )
    validation_stage_counts = (
        decisions["validation_stage"].astype(str).replace({"nan": "", "None": ""}).value_counts().to_dict()
        if not decisions.empty and "validation_stage" in decisions.columns
        else {}
    )
    blocker_totals = dict(blocker_counts)
    timing_blockers = {
        "no_book_checkpoint": int(skip_reason_totals.get("no_book_checkpoint", 0)),
        "stale_book": int(skip_reason_totals.get("stale_book", 0)),
        "no_ask_ladder": int(skip_reason_totals.get("no_ask_ladder", 0)),
    }
    timing_failure_count = int(sum(timing_blockers.values()))
    if decision_count <= 0:
        timing_failure_reason = "no_decisions"
    elif timing_failure_count <= 0:
        timing_failure_reason = "none"
    elif timing_blockers["no_book_checkpoint"] == decision_count and timing_blockers["stale_book"] == 0:
        timing_failure_reason = "no_book_checkpoint_everywhere"
    elif timing_blockers["stale_book"] == decision_count and timing_blockers["no_book_checkpoint"] == 0:
        timing_failure_reason = "stale_book_everywhere"
    elif covered_count == 0:
        timing_failure_reason = "no_book_snapshot_any_decision"
    elif fresh_mask.sum() == 0:
        timing_failure_reason = "books_captured_but_not_fresh_enough"
    else:
        timing_failure_reason = "mixed_timing_blockers"
    timing_failure_detail = (
        f"coverage={covered_count}/{decision_count}, "
        f"fresh={int(fresh_mask.sum()) if not decisions.empty else 0}, "
        f"stale={int(stale_mask.sum()) if not decisions.empty else 0}, "
        f"no_book_checkpoint={timing_blockers['no_book_checkpoint']}, "
        f"no_ask_ladder={timing_blockers['no_ask_ladder']}"
    )
    primary_blocker = ""
    if blocker_counts:
        blocker_order = {"mapping": 0, "coverage": 1, "policy": 2, "other": 3, "none": 4}
        primary_blocker = sorted(blocker_counts.items(), key=lambda item: (-item[1], blocker_order.get(item[0], 99), item[0]))[0][0]

    decision_coverage = {
        "decision_window_minutes": float(decision_window_minutes),
        "decision_window_label": f"T-{int(round(float(decision_window_minutes)))}m",
        "decision_window_book_freshness_seconds": float(decision_book_freshness_seconds),
        "decision_window_capture_lead_seconds": float(decision_capture_lead_seconds),
        "decision_window_capture_retry_attempts": int(decision_capture_retry_attempts),
        "decision_window_capture_retry_delay_seconds": float(decision_capture_retry_delay_seconds),
        "total_decisions": decision_count,
        "decisions_with_group_key": int(decisions["group_key"].astype(str).str.strip().ne("").sum()) if not decisions.empty else 0,
        "decisions_with_complete_mapping": complete_mapping_count,
        "decisions_with_book_snapshot": covered_count,
        "decisions_with_fresh_book": int(fresh_mask.sum()) if not decisions.empty else 0,
        "decisions_with_stale_book": int(stale_mask.sum()) if not decisions.empty else 0,
        "decisions_with_missing_book": int(book_age_seconds.isna().sum()) if not decisions.empty else 0,
        "decisions_with_selection": selected_count,
        "selection_rate": float(selected_count / decision_count) if decision_count > 0 else 0.0,
        "book_snapshot_coverage_rate": float(covered_count / decision_count) if decision_count > 0 else 0.0,
        "fresh_book_rate": float(fresh_mask.mean()) if not decisions.empty else 0.0,
        "stale_book_rate": float(stale_mask.mean()) if not decisions.empty else 0.0,
        "missing_book_rate": float(book_age_seconds.isna().mean()) if not decisions.empty else 0.0,
        "timing_blockers": timing_blockers,
        "timing_failure_count": timing_failure_count,
        "timing_failure_rate": float(timing_failure_count / decision_count) if decision_count > 0 else 0.0,
        "timing_failure_reason": timing_failure_reason,
        "timing_failure_detail": timing_failure_detail,
        "book_age_seconds": {
            "median": float(book_age_seconds.dropna().median()) if not book_age_seconds.dropna().empty else 0.0,
            "p90": float(book_age_seconds.dropna().quantile(0.9)) if not book_age_seconds.dropna().empty else 0.0,
            "max": float(book_age_seconds.dropna().max()) if not book_age_seconds.dropna().empty else 0.0,
        },
        "book_age_buckets": age_bucket_counts.to_dict(orient="records"),
    }

    fill_model_summary = fit_fill_probability_priors(fills)
    fill_quality_rows = attach_fill_adjusted_ev(
        fills.copy(),
        probability_column="model_prob",
        edge_column="expected_edge",
        priors=fill_model_summary,
        default_notional=float(fills["notional"].median()) if not fills.empty and "notional" in fills.columns else 10.0,
    )
    if not fill_quality_rows.empty:
        closing_reference_prob = pd.to_numeric(
            pd.Series(fill_quality_rows.get("closing_reference_prob", np.nan), index=fill_quality_rows.index),
            errors="coerce",
        )
        fill_quality_rows["executed_odds"] = np.where(
            pd.to_numeric(fill_quality_rows["effective_vwap"], errors="coerce").gt(0.0),
            1.0 / pd.to_numeric(fill_quality_rows["effective_vwap"], errors="coerce"),
            np.nan,
        )
        fill_quality_rows["closing_reference_odds"] = np.where(
            closing_reference_prob.gt(0.0),
            1.0 / closing_reference_prob,
            np.nan,
        )
    clv_rows = build_clv_rows(
        fill_quality_rows,
        executed_odds_column="executed_odds",
        closing_reference_odds_column="closing_reference_odds",
        executed_prob_column="effective_vwap",
        closing_reference_prob_column="closing_reference_prob",
        source_column="clv_source",
    )
    clv_summary = summarize_clv(clv_rows, total_rows=int(len(fill_quality_rows)))
    fill_adjusted_ev_summary = summarize_fill_adjusted_ev(fill_quality_rows)
    sizing_summary = summarize_sizing(fills, requested_column="notional")

    lifecycle = build_polymarket_lifecycle_summary(
        source_mode=SOURCE_MODE_FORWARD,
        bundle_status=bundle_status,
        price_provenance_counts=price_provenance_counts,
        validation_stage=VALIDATION_STAGE_SHADOW,
    )
    lifecycle["validation_stage_counts"] = validation_stage_counts

    return {
        "settled_bets": int(len(settled)),
        "all_fill_rows": int(len(fills)),
        "net_pnl": total_profit,
        "net_roi": float(total_profit / total_cost) if total_cost > 0 else 0.0,
        "total_cost_basis": total_cost,
        "fill_rate": float(fills["fill_rate"].mean()) if not fills.empty else 0.0,
        "partial_fill_rate": float(fills["partial_fill"].mean()) if not fills.empty else 0.0,
        "fee_drag": float(settled["fee_paid"].sum() / total_cost) if total_cost > 0 else 0.0,
        "slippage_vs_top_ask": float((settled["raw_vwap"] - settled["top_ask"]).mean()) if not settled.empty else 0.0,
        "effective_slippage_vs_top_ask": float((settled["effective_vwap"] - settled["top_ask"]).mean()) if not settled.empty else 0.0,
        "drawdown": _drawdown_from_profit(settled.sort_values("created_at")["net_profit"]) if not settled.empty else 0.0,
        "mapping_precision": float(mappings["mapping_status"].eq("complete").mean()) if not mappings.empty else 0.0,
        "skip_reasons": skip_counts.to_dict(orient="records"),
        "skip_reason_blockers": blocker_breakdown,
        "primary_blocker": primary_blocker,
        "primary_blocker_count": int(blocker_counts.get(primary_blocker, 0)) if primary_blocker else 0,
        "blocker_totals": blocker_totals,
        "decision_coverage": decision_coverage,
        "observed_fill_rate": float(pd.to_numeric(fills["fill_rate"], errors="coerce").mean()) if not fills.empty else 0.0,
        "clv_summary": clv_summary,
        "fill_model_summary": fill_model_summary,
        "fill_adjusted_ev_summary": fill_adjusted_ev_summary,
        "sizing_summary": sizing_summary,
        "clv_rows": clv_rows.to_dict(orient="records"),
        "bundle_status": lifecycle["bundle_status"],
        "validation_stage": lifecycle["validation_stage"],
        "price_provenance": lifecycle["price_provenance"],
        "bundle_readiness": lifecycle["bundle_readiness"],
        "price_provenance_counts": price_provenance_counts,
        "validation_stage_counts": validation_stage_counts,
        "lifecycle": lifecycle,
        "roi_by_notional": capacity_curve.to_dict(orient="records"),
        "roi_by_league": roi_by_league.to_dict(orient="records"),
        "roi_by_window": roi_by_window.to_dict(orient="records"),
    }


def _save_shadow_artifacts(
    run_dir: Path,
    decisions: pd.DataFrame,
    fills: pd.DataFrame,
    mappings: pd.DataFrame,
    summary: dict[str, Any],
    forward_sample_ledger: pd.DataFrame | None = None,
) -> dict[str, Path]:
    decision_path = run_dir / "decision_rows.csv"
    fill_path = run_dir / "fill_rows.csv"
    mapping_path = run_dir / "market_mapping.csv"
    summary_path = run_dir / "shadow_summary.json"
    capacity_path = run_dir / "capacity_curve.csv"
    skip_path = run_dir / "skip_reasons.csv"
    clv_path = run_dir / "clv_rows.csv"
    execution_quality_path = run_dir / "execution_quality.json"
    bank_curve_path = run_dir / "bank_curve.png"
    forward_sample_report_path = run_dir / "forward_sample_report.json"
    forward_sample_ledger_path = run_dir / "forward_sample_ledger.csv"
    forward_sample_blockers_path = run_dir / "forward_sample_blockers.csv"
    forward_sample_manifest_path = run_dir / "forward_sample_manifest.json"
    forward_decision_region_report_path = run_dir / "forward_decision_region_report.json"

    decisions.to_csv(decision_path, index=False)
    fills.to_csv(fill_path, index=False)
    mappings.to_csv(mapping_path, index=False)
    _save_json(summary_path, summary)
    pd.DataFrame(summary.get("roi_by_notional", [])).to_csv(capacity_path, index=False)
    pd.DataFrame(summary.get("skip_reasons", [])).to_csv(skip_path, index=False)
    pd.DataFrame(summary.get("clv_rows", [])).to_csv(clv_path, index=False)
    _save_json(
        execution_quality_path,
        {
            "clv_summary": summary.get("clv_summary", {}),
            "fill_model_summary": summary.get("fill_model_summary", {}),
            "fill_adjusted_ev_summary": summary.get("fill_adjusted_ev_summary", {}),
            "sizing_summary": summary.get("sizing_summary", {}),
        },
    )
    forward_sample = summary.get("forward_sample", {})
    _save_json(forward_sample_report_path, forward_sample)
    _save_json(forward_sample_manifest_path, summary.get("forward_sample_manifest", {}))
    _save_json(forward_decision_region_report_path, summary.get("forward_decision_region_report", {}))
    if forward_sample_ledger is None:
        forward_sample_ledger = pd.DataFrame()
    forward_sample_ledger.to_csv(forward_sample_ledger_path, index=False)
    pd.DataFrame((forward_sample.get("cumulative", {}) or {}).get("blockers", [])).to_csv(
        forward_sample_blockers_path,
        index=False,
    )
    _plot_shadow_bank_curve(fills, bank_curve_path)
    return {
        "decision_rows": decision_path,
        "fill_rows": fill_path,
        "market_mapping": mapping_path,
        "shadow_summary": summary_path,
        "capacity_curve": capacity_path,
        "skip_reasons": skip_path,
        "clv_rows": clv_path,
        "execution_quality": execution_quality_path,
        "bank_curve": bank_curve_path,
        "forward_sample_report": forward_sample_report_path,
        "forward_sample_ledger": forward_sample_ledger_path,
        "forward_sample_blockers": forward_sample_blockers_path,
        "forward_sample_manifest": forward_sample_manifest_path,
        "forward_decision_region_report": forward_decision_region_report_path,
    }


def shadow_polymarket(
    settings: Settings,
    db_path: Path | str | None = None,
    fixtures_path: Path | str | None = None,
    model_path: Path | str | None = None,
    policy_bundle_path: Path | str | None = None,
) -> PolymarketShadowResult:
    db_path = Path(db_path) if db_path else default_polymarket_db_path(settings)
    connection = init_polymarket_db(db_path)
    catalog = _catalog_lookup(connection)
    groups = _group_lookup(connection)
    checkpoints = _checkpoint_lookup(connection)
    resolutions = _resolution_lookup(connection)
    if catalog.empty or groups.empty or checkpoints.empty:
        collect_polymarket(settings=settings, db_path=db_path, stream_seconds=0)
        connection = init_polymarket_db(db_path)
        catalog = _catalog_lookup(connection)
        groups = _group_lookup(connection)
        checkpoints = _checkpoint_lookup(connection)
        resolutions = _resolution_lookup(connection)

    if model_path is None:
        latest_model = settings.paths.outputs_dir / "latest_niche_model.txt"
        if not latest_model.exists():
            raise FileNotFoundError("No encuentro un niche model. Ejecuta antes `predicciones train-niche`.")
        model_path = Path(latest_model.read_text(encoding="utf-8").strip())
    model_path = Path(model_path)
    payload = joblib.load(Path(model_path))
    policy_payload = payload.get("policy", {})
    probability_source = str(payload.get("probability_source", "raw"))
    policy_source_mode = "model_bundle"
    policy_bundle_ref = ""
    policy_coverage_status = ""
    policy_bundle_status = ""
    policy_minimum_quality = ""
    resolved_policy_bundle_path, policy_bundle_resolution = _resolve_default_policy_bundle_path(
        settings,
        explicit_path=policy_bundle_path,
    )
    if resolved_policy_bundle_path:
        bundle = _load_policy_bundle(resolved_policy_bundle_path)
        policy_payload = bundle.get("policy", policy_payload)
        probability_source = str(bundle.get("probability_source", probability_source))
        policy_source_mode = str(bundle.get("source_mode", SOURCE_MODE_RETRO))
        policy_bundle_ref = str(resolved_policy_bundle_path)
        policy_coverage_status = str(bundle.get("coverage_status", ""))
        policy_bundle_status = str(bundle.get("bundle_status", ""))
        policy_minimum_quality = str(bundle.get("minimum_quality_tier", ""))
    policy = _policy_from_raw(policy_payload, settings)

    if fixtures_path:
        fixtures_raw = pd.read_csv(fixtures_path)
        fixtures = canonicalize_matches(fixtures_raw, require_results=False)
    else:
        fixtures = canonicalize_matches(_derive_fixtures_from_groups(connection, payload["history_matches"]), require_results=False)
        if fixtures.empty:
            raise ValueError("No hay grupos completos en la base de Polymarket para construir fixtures.")

    fixtures["league_name"] = fixtures.get("league_name", fixtures["league_code"]).fillna(fixtures["league_code"])
    if "kickoff_time" not in fixtures.columns:
        fixtures["kickoff_time"] = pd.NaT
    missing_kickoff = fixtures["kickoff_time"].isna()
    fixtures.loc[missing_kickoff, "kickoff_time"] = (
        pd.to_datetime(fixtures.loc[missing_kickoff, "Date"]).dt.normalize()
        + pd.to_timedelta(settings.snapshot.default_kickoff_hour, unit="h")
    )
    fixtures["group_key"] = fixtures.apply(lambda row: _match_fixture_to_group(row, groups, settings=settings), axis=1)

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
    predictions["kickoff_time"] = fixtures["kickoff_time"].where(fixtures["kickoff_time"].notna(), predictions["Date"])
    predictions["group_key"] = fixtures["group_key"].values
    predictions["raw_prediction"] = [GROUP_ROLE_ORDER[index] for index in raw_probabilities.argmax(axis=1)]
    predictions["calibrated_prediction"] = [GROUP_ROLE_ORDER[index] for index in calibrated_probabilities.argmax(axis=1)]
    predictions["league_name"] = dataset["league_name"]

    decision_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        mapping_rows.append(
            {
                "match_id": str(row.match_id),
                "league_code": str(row.league_code),
                "league_name": str(row.league_name),
                "HomeTeam": str(row.HomeTeam),
                "AwayTeam": str(row.AwayTeam),
                "group_key": str(getattr(row, "group_key", "")),
                "mapping_status": "complete" if str(getattr(row, "group_key", "")) else "missing",
            }
        )
        decision_row, fills = _decision_from_prediction(
            pd.Series(row._asdict()),
            groups=groups,
            catalog=catalog,
            checkpoints=checkpoints,
            settings=settings,
            probability_source=probability_source,
            policy=policy,
        )
        decision_rows.append(decision_row)
        fill_rows.extend(fills)

    decisions = pd.DataFrame(decision_rows)
    fills = pd.DataFrame(fill_rows)
    mappings = pd.DataFrame(mapping_rows)

    if not fills.empty:
        fills["winning_role"] = fills["group_key"].map(
            lambda key: str(resolutions[resolutions["group_key"].astype(str) == str(key)]["winning_role"].iloc[0])
            if not resolutions[resolutions["group_key"].astype(str) == str(key)].empty
            else ""
        )
        fills = pd.DataFrame([_settle_fill_row(row, str(row["winning_role"])) for _, row in fills.iterrows()])
        fills = fills.merge(decisions[["decision_id", "league_code", "kickoff_time"]], on="decision_id", how="left")

    run = create_run_context(settings.paths.runs_dir, "shadow_polymarket")
    if not decisions.empty:
        decisions["run_id"] = run.run_id
    if not fills.empty:
        fills["run_id"] = run.run_id

    decisions_to_save = _prune_rows_to_table(connection, "pm_shadow_decisions", decisions)
    fills_to_save = _prune_rows_to_table(connection, "pm_shadow_fills", fills)
    _upsert_rows(connection, "pm_shadow_decisions", decisions_to_save.to_dict(orient="records") if not decisions_to_save.empty else [])
    _upsert_rows(connection, "pm_shadow_fills", fills_to_save.to_dict(orient="records") if not fills_to_save.empty else [])
    cumulative_decisions = pd.read_sql_query("SELECT * FROM pm_shadow_decisions", connection)
    cumulative_fills = pd.read_sql_query("SELECT * FROM pm_shadow_fills", connection)
    summary = _summarize_shadow(
        decisions,
        fills,
        mappings,
        decision_window_minutes=float(settings.polymarket.decision_offset_minutes),
        decision_book_freshness_seconds=float(settings.polymarket.decision_book_freshness_seconds),
        decision_capture_lead_seconds=float(settings.polymarket.decision_capture_lead_seconds),
        decision_capture_retry_attempts=int(settings.polymarket.decision_capture_retry_attempts),
        decision_capture_retry_delay_seconds=float(settings.polymarket.decision_capture_retry_delay_seconds),
        bundle_status=policy_bundle_status or BUNDLE_STATUS_PROVISIONAL,
    )
    summary.update(
        {
            "source_mode": SOURCE_MODE_FORWARD,
            "probability_source": probability_source,
            "policy": policy.to_dict(),
            "policy_source_mode": policy_source_mode,
            "policy_bundle_path": policy_bundle_ref,
            "policy_coverage_status": policy_coverage_status,
            "policy_bundle_status": policy_bundle_status,
            "policy_minimum_quality_tier": policy_minimum_quality,
        }
    )
    forward_sample_report, forward_sample_ledger = _build_forward_sample_report(
        decisions,
        fills,
        cumulative_decisions,
        cumulative_fills,
        decision_book_freshness_seconds=float(settings.polymarket.decision_book_freshness_seconds),
    )
    summary["forward_sample"] = forward_sample_report
    summary["forward_sample_manifest"] = _build_forward_sample_manifest(
        run_id=run.run_id,
        db_path=db_path,
        model_path=model_path,
        policy_bundle_path=resolved_policy_bundle_path,
        policy_bundle_resolution=policy_bundle_resolution,
        payload=payload,
        probability_source=probability_source,
        policy_source_mode=policy_source_mode,
        policy_payload=policy.to_dict(),
        policy_coverage_status=policy_coverage_status,
        policy_bundle_status=policy_bundle_status,
        policy_minimum_quality=policy_minimum_quality,
        settings=settings,
        forward_sample_report=forward_sample_report,
    )
    summary["forward_decision_region_report"] = _build_forward_decision_region_report(
        forward_sample_report,
        forward_sample_ledger,
        cumulative_fills,
    )
    artifacts = _save_shadow_artifacts(
        run.run_dir,
        decisions,
        fills,
        mappings,
        summary,
        forward_sample_ledger=forward_sample_ledger,
    )
    return PolymarketShadowResult(
        run=run,
        database_path=db_path,
        decisions=decisions,
        fills=fills,
        mappings=mappings,
        summary=summary,
        artifacts=artifacts,
    )


def report_polymarket(run_dir: Path | str) -> tuple[dict[str, Any], str]:
    root = Path(run_dir)
    summary = json.loads((root / "shadow_summary.json").read_text(encoding="utf-8"))
    collect_summary_path = root / "collect_summary.json"
    collect_summary = json.loads(collect_summary_path.read_text(encoding="utf-8")) if collect_summary_path.exists() else {}
    lifecycle = summary.get("lifecycle", {}) or {
        "validation_stage": summary.get("validation_stage", VALIDATION_STAGE_SHADOW),
        "price_provenance": summary.get("price_provenance", "unknown"),
        "bundle_readiness": summary.get("bundle_readiness", summary.get("bundle_status", BUNDLE_STATUS_PROVISIONAL)),
    }
    coverage = summary.get("decision_coverage", {}) or {}
    blocker_totals = summary.get("blocker_totals", {}) or {}
    blocker_label = summary.get("primary_blocker", "")
    blocker_count = int(summary.get("primary_blocker_count", 0) or 0)
    skip_reasons = summary.get("skip_reasons", []) or []
    skip_reason_text = ", ".join(f"{item['skip_reason']}={item['count']}" for item in skip_reasons[:4]) or "none"
    collect_capture = collect_summary.get("capture", {}) or {}
    collect_stream = collect_summary.get("stream", {}) or {}
    collect_capture_text = (
        f"book_attempts={collect_capture.get('book_fetch_attempts', 0)}, "
        f"book_successes={collect_capture.get('book_fetch_successes', 0)}, "
        f"book_missing={collect_capture.get('book_fetch_missing', 0)}, "
        f"book_failed={collect_capture.get('book_fetch_failed', 0)}, "
        f"book_retry_attempts={collect_capture.get('book_fetch_retry_attempts', 0)}, "
        f"book_retry_successes={collect_capture.get('book_fetch_retry_successes', 0)}"
    ) if collect_capture else "none"
    collect_stream_text = (
        f"precheck_attempts={collect_stream.get('decision_precheck_attempts', 0)}, "
        f"ready_groups={collect_stream.get('decision_precheck_ready_groups', 0)}, "
        f"retry_attempts={collect_stream.get('decision_precheck_retry_attempts', 0)}, "
        f"retry_successes={collect_stream.get('decision_precheck_retry_successes', 0)}, "
        f"stale_candidates={collect_stream.get('decision_precheck_stale_candidates', 0)}, "
        f"missing_books={collect_stream.get('decision_precheck_missing_books', 0)}, "
        f"fetch_failures={collect_stream.get('decision_precheck_fetch_failures', 0)}"
    ) if collect_stream else "none"
    coverage_text = (
        f"{coverage.get('decisions_with_book_snapshot', 0)}/{coverage.get('total_decisions', 0)} snapshots, "
        f"fresh={coverage.get('decisions_with_fresh_book', 0)}, "
        f"stale={coverage.get('decisions_with_stale_book', 0)}, "
        f"missing={coverage.get('decisions_with_missing_book', 0)}, "
        f"timing_failures={coverage.get('timing_failure_count', 0)}, "
        f"reason={coverage.get('timing_failure_reason', 'none')}"
    )
    blocker_text = blocker_label or "none"
    if blocker_label:
        blocker_text = f"{blocker_label} ({blocker_count})"
    blocker_breakdown = ", ".join(f"{name}={count}" for name, count in blocker_totals.items()) or "none"
    timing_breakdown = ", ".join(f"{name}={count}" for name, count in coverage.get("timing_blockers", {}).items()) or "none"
    timing_failure_reason = coverage.get("timing_failure_reason", "none")
    timing_failure_detail = coverage.get("timing_failure_detail", "")
    capture_profile = summary.get("decision_capture_profile", {}) or {}
    clv_summary = summary.get("clv_summary", {}) or {}
    fill_adjusted_ev_summary = summary.get("fill_adjusted_ev_summary", {}) or {}
    forward_sample = summary.get("forward_sample", {}) or {}
    forward_cumulative = forward_sample.get("cumulative", {}) or {}
    forward_status = forward_sample.get("sample_status", forward_cumulative.get("sample_status", "unknown"))
    forward_roi_display_mode = forward_sample.get("roi_display_mode", forward_cumulative.get("roi_display_mode", "hidden_until_sample_ready"))
    forward_roi_text = (
        "roi=hidden_until_sample_ready"
        if forward_roi_display_mode == "hidden_until_sample_ready"
        else f"roi={forward_cumulative.get('net_roi', 0.0):.4f}"
    )
    forward_sample_text = (
        f"valid={forward_cumulative.get('valid_forward_decisions', 0)}/"
        f"{forward_cumulative.get('target_valid_forward_decisions', FORWARD_SAMPLE_TARGET_VALID_DECISIONS)}, "
        f"settled={forward_cumulative.get('settled_unique_decisions', 0)}/"
        f"{forward_cumulative.get('target_settled_decisions', FORWARD_SAMPLE_TARGET_SETTLED_DECISIONS)}, "
        f"fresh_rate={forward_cumulative.get('fresh_book_rate', 0.0):.4f}, "
        f"{forward_roi_text}"
    )
    lines = [
        f"- Source mode: {summary.get('source_mode', SOURCE_MODE_FORWARD)}",
        f"- Lifecycle: {format_polymarket_lifecycle_label(lifecycle)}",
        f"- Validation stage: {lifecycle.get('validation_stage', summary.get('validation_stage', VALIDATION_STAGE_SHADOW))}",
        f"- Price provenance: {lifecycle.get('price_provenance', summary.get('price_provenance', 'unknown'))}",
        f"- Bundle readiness: {lifecycle.get('bundle_readiness', summary.get('bundle_readiness', summary.get('bundle_status', BUNDLE_STATUS_PROVISIONAL)))}",
        f"- Settled bets: {summary.get('settled_bets', 0)}",
        f"- Net ROI: {summary.get('net_roi', 0.0):.4f}",
        f"- Net PnL: {summary.get('net_pnl', 0.0):.4f}",
        f"- Fill rate: {summary.get('fill_rate', 0.0):.4f}",
        f"- Fill-adjusted EV mean: {fill_adjusted_ev_summary.get('mean', 0.0):.4f}",
        f"- Mean CLV: {clv_summary.get('mean_clv', 0.0):.4f}",
        f"- CLV coverage: {clv_summary.get('coverage', 0.0):.4f}",
        f"- Partial fill rate: {summary.get('partial_fill_rate', 0.0):.4f}",
        f"- Mapping precision: {summary.get('mapping_precision', 0.0):.4f}",
        f"- Primary blocker: {blocker_text}",
        f"- Blocker breakdown: {blocker_breakdown}",
        f"- Timing blockers: {timing_breakdown}",
        f"- Timing failure reason: {timing_failure_reason}",
        f"- Timing failure detail: {timing_failure_detail or 'none'}",
        f"- Decision capture profile: freshness={capture_profile.get('decision_book_freshness_seconds', 'n/a')}, lead={capture_profile.get('decision_capture_lead_seconds', 'n/a')}, retries={capture_profile.get('decision_capture_retry_attempts', 'n/a')}, retry_delay={capture_profile.get('decision_capture_retry_delay_seconds', 'n/a')}",
        f"- Capture diagnostics: {collect_capture_text}",
        f"- Decision precheck diagnostics: {collect_stream_text}",
        f"- Decision-window coverage: {coverage_text}",
        f"- Forward sample status: {forward_status}",
        f"- Forward sample ROI display mode: {forward_roi_display_mode}",
        f"- Forward sample actionable ROI: {str(forward_sample.get('actionable_roi', forward_cumulative.get('actionable_roi', False))).lower()}",
        f"- Forward sample cumulative: {forward_sample_text}",
        f"- Forward sample next action: {forward_sample.get('next_action', forward_cumulative.get('next_action', 'none'))}",
        f"- Skip reasons: {skip_reason_text}",
    ]
    summary["collect_summary"] = collect_summary
    summary["capture_diagnostics"] = collect_capture
    summary["decision_precheck_diagnostics"] = collect_stream
    return summary, "\n".join(lines)
