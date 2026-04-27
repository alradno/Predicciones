from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..reporting import create_run_context


ODDS_BANDS = ((0.0, 1.5, "<1.5"), (1.5, 2.0, "1.5-2"), (2.0, 3.0, "2-3"), (3.0, 6.0, "3-6"), (6.0, np.inf, "6+"))
TRAIN_BUCKETS = {"oof", "pre_holdout"}
REPORT_COLUMNS = [
    "match_key",
    "validation_bucket",
    "analysis_split",
    "league_code",
    "selection",
    "odds_band",
    "quoted_odds",
    "policy_prob",
    "policy_edge",
    "policy_ev",
    "confidence",
    "unit_profit",
    "won",
    "region_roi",
    "region_positive_split_ratio",
    "noise_low_confidence",
    "noise_high_odds",
    "noise_unstable_region",
    "noise_positive_ev_but_unstable",
]


@dataclass(frozen=True)
class OfflineReviewResult:
    run_dir: Path
    summary: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _maybe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _metric(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"bets": 0, "wins": 0, "hit_rate": None, "profit": 0.0, "roi": None, "avg_ev": None, "avg_edge": None, "avg_odds": None}
    profit = float(rows["unit_profit"].sum())
    bets = int(len(rows))
    wins = int(rows["won"].sum())
    return {
        "bets": bets,
        "wins": wins,
        "hit_rate": float(wins / bets) if bets else None,
        "profit": profit,
        "roi": float(profit / bets) if bets else None,
        "avg_ev": float(rows["policy_ev"].mean()) if "policy_ev" in rows else None,
        "avg_edge": float(rows["policy_edge"].mean()) if "policy_edge" in rows else None,
        "avg_odds": float(rows["quoted_odds"].mean()) if "quoted_odds" in rows else None,
    }


def _group_metrics(rows: pd.DataFrame, by: str) -> dict[str, Any]:
    if rows.empty or by not in rows:
        return {}
    result: dict[str, Any] = {}
    for key, part in rows.groupby(by, dropna=False):
        result[str(key)] = _metric(part)
    return result


def _odds_band(value: Any) -> str:
    odds = _safe_float(value, default=np.nan)
    if not math.isfinite(odds):
        return "unknown"
    for low, high, label in ODDS_BANDS:
        if odds >= low and odds < high:
            return label
    return "unknown"


def _confidence_column(df: pd.DataFrame) -> str | None:
    for name in ("confidence_score_v2", "policy_confidence_score", "policy_signal_confidence_score", "confidence_score"):
        if name in df.columns:
            return name
    return None


def _match_key(df: pd.DataFrame) -> pd.Series:
    for name in ("match_id", "group_key"):
        if name in df.columns:
            values = df[name].astype("string")
            if values.notna().any():
                return values.fillna("")
    return pd.Series(np.arange(len(df)).astype(str), index=df.index)


def _analysis_split(row: pd.Series) -> str:
    segment = str(row.get("retro_segment", "") or "")
    if segment == "discovery_train":
        fold = row.get("retro_fold_id")
        return f"oof_fold_{int(fold)}" if pd.notna(fold) else "oof"
    if segment == "selection_dev":
        window = str(row.get("pre_holdout_window", "") or "")
        return window if window and window != "nan" else "pre_holdout"
    if segment == "locked_holdout":
        return "locked_holdout"
    return segment or "unknown"


def _validation_bucket(row: pd.Series) -> str:
    segment = str(row.get("retro_segment", "") or "")
    if segment == "discovery_train":
        return "oof"
    if segment == "selection_dev":
        return "pre_holdout"
    if segment == "locked_holdout":
        return "locked_holdout"
    return segment or "unknown"


def _load_policy(policy_bundle: Path | None, retro_run_dir: Path) -> dict[str, Any]:
    bundle_path = policy_bundle
    if bundle_path is None:
        candidate = retro_run_dir / "policy_bundle.json"
        bundle_path = candidate if candidate.exists() else None
    payload = _maybe_json(bundle_path) if bundle_path else {}
    policy = dict(payload.get("policy") or {})
    return {
        "policy_bundle_path": str(bundle_path) if bundle_path else None,
        "edge_threshold": _safe_float(policy.get("edge_threshold"), 0.0),
        "ev_threshold": _safe_float(policy.get("ev_threshold"), 0.0),
        "min_odds": _safe_float(policy.get("min_odds"), 0.0),
        "max_odds": _safe_float(policy.get("max_odds"), float("inf")),
        "allowed_leagues": [str(x) for x in policy.get("allowed_leagues") or []],
        "allowed_outcomes": [str(x) for x in policy.get("allowed_outcomes") or []],
        "family": str(policy.get("family") or payload.get("policy_family") or "edge_ev_threshold"),
        "scope_name": str(policy.get("scope_name") or payload.get("scope_name") or "unknown"),
        "policy_reoptimized": False,
        "thresholds_changed": False,
        "scopes_changed": False,
    }


def _prepare_candidates(retro_run_dir: Path, policy: dict[str, Any]) -> pd.DataFrame:
    path = retro_run_dir / "retro_candidate_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df

    df = df.copy()
    df["match_key"] = _match_key(df)
    df["analysis_split"] = df.apply(_analysis_split, axis=1)
    df["validation_bucket"] = df.apply(_validation_bucket, axis=1)
    df["quoted_odds"] = pd.to_numeric(df.get("quoted_odds"), errors="coerce")
    df["top_ask"] = pd.to_numeric(df.get("top_ask"), errors="coerce")
    df["policy_prob"] = pd.to_numeric(df.get("policy_prob"), errors="coerce")
    df["policy_edge"] = pd.to_numeric(df.get("policy_edge"), errors="coerce")
    df["policy_ev"] = pd.to_numeric(df.get("policy_ev"), errors="coerce")
    df["policy_rank_score"] = pd.to_numeric(df.get("policy_rank_score"), errors="coerce")
    df["odds_band"] = df["quoted_odds"].map(_odds_band)

    confidence = _confidence_column(df)
    df["confidence"] = pd.to_numeric(df[confidence], errors="coerce") if confidence else np.nan
    df["selection"] = df["selection"].astype(str)
    df["actual_outcome"] = df["actual_outcome"].astype(str)
    df["won"] = df["selection"].eq(df["actual_outcome"]).astype(int)
    df["unit_profit"] = np.where(df["won"].eq(1), df["quoted_odds"] - 1.0, -1.0)

    mask = df.get("quote_status", pd.Series("", index=df.index)).astype(str).eq("eligible")
    mask &= df["quoted_odds"].ge(policy["min_odds"]) & df["quoted_odds"].le(policy["max_odds"])
    mask &= df["policy_edge"].ge(policy["edge_threshold"]) & df["policy_ev"].ge(policy["ev_threshold"])
    if policy["allowed_leagues"]:
        mask &= df.get("league_code", pd.Series("", index=df.index)).astype(str).isin(policy["allowed_leagues"])
    if policy["allowed_outcomes"]:
        mask &= df["selection"].isin(policy["allowed_outcomes"])
    df["base_policy_candidate"] = mask.fillna(False)
    return df


def _select_one_per_match(candidates: pd.DataFrame, score_col: str = "policy_rank_score") -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    rows = candidates.copy()
    if score_col not in rows:
        rows[score_col] = np.nan
    rows["_score"] = pd.to_numeric(rows[score_col], errors="coerce")
    rows["_tie_ev"] = pd.to_numeric(rows.get("policy_ev"), errors="coerce")
    rows["_tie_prob"] = pd.to_numeric(rows.get("policy_prob"), errors="coerce")
    rows = rows.sort_values(["match_key", "_score", "_tie_ev", "_tie_prob"], ascending=[True, False, False, False])
    return rows.drop_duplicates("match_key", keep="first").drop(columns=["_score", "_tie_ev", "_tie_prob"], errors="ignore")


def _split_summary(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "by_validation_bucket": _group_metrics(rows, "validation_bucket"),
        "by_split": _group_metrics(rows, "analysis_split"),
        "by_outcome": _group_metrics(rows, "selection"),
        "by_league": _group_metrics(rows, "league_code"),
        "by_odds_band": _group_metrics(rows, "odds_band"),
        "by_confidence_decile": _group_metrics(rows, "confidence_decile"),
    }


def _add_confidence_deciles(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    if rows.empty or rows["confidence"].notna().sum() < 2:
        rows["confidence_decile"] = "unknown"
        return rows
    ranked = rows["confidence"].rank(method="first")
    rows["confidence_decile"] = pd.qcut(ranked, q=min(10, len(rows)), labels=False, duplicates="drop")
    rows["confidence_decile"] = rows["confidence_decile"].map(lambda x: f"d{int(x) + 1}" if pd.notna(x) else "unknown")
    return rows


def _region_stability(base_picks: pd.DataFrame) -> pd.DataFrame:
    train = base_picks[base_picks["validation_bucket"].isin(TRAIN_BUCKETS)].copy()
    if train.empty:
        return pd.DataFrame(columns=["selection", "odds_band", "region_bets", "region_roi", "region_positive_split_ratio"])
    split_roi = (
        train.groupby(["selection", "odds_band", "analysis_split"], dropna=False)["unit_profit"]
        .agg(["sum", "count"])
        .reset_index()
    )
    split_roi["split_roi"] = split_roi["sum"] / split_roi["count"]
    positive = (
        split_roi.groupby(["selection", "odds_band"], dropna=False)["split_roi"]
        .agg(lambda values: float((pd.Series(values) > 0).mean()))
        .reset_index(name="region_positive_split_ratio")
    )
    region = (
        train.groupby(["selection", "odds_band"], dropna=False)
        .agg(region_bets=("unit_profit", "size"), region_profit=("unit_profit", "sum"), region_hit_rate=("won", "mean"))
        .reset_index()
    )
    region["region_roi"] = region["region_profit"] / region["region_bets"]
    return region.merge(positive, on=["selection", "odds_band"], how="left")


def _attach_region(rows: pd.DataFrame, region: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    out = rows.merge(region, on=["selection", "odds_band"], how="left")
    out["region_bets"] = out["region_bets"].fillna(0).astype(int)
    out["region_roi"] = out["region_roi"].fillna(0.0)
    out["region_positive_split_ratio"] = out["region_positive_split_ratio"].fillna(0.0)
    return out


def _status(metrics: dict[str, Any]) -> str:
    oof = (metrics.get("by_validation_bucket") or {}).get("oof") or {}
    pre = (metrics.get("by_validation_bucket") or {}).get("pre_holdout") or {}
    holdout = (metrics.get("by_validation_bucket") or {}).get("locked_holdout") or {}
    if (oof.get("roi") is None) or float(oof.get("roi") or 0.0) <= 0:
        return "rejected_oof_negative"
    if (pre.get("roi") is None) or float(pre.get("roi") or 0.0) <= 0:
        return "rejected_pre_holdout_negative"
    if int(holdout.get("bets") or 0) < 40:
        return "research_positive_but_holdout_small"
    return "forward_candidate_pending_sample"


def _write_noise_report(run_dir: Path, selected: pd.DataFrame, retro_run_dir: Path, policy: dict[str, Any]) -> dict[str, Any]:
    rows = _add_confidence_deciles(selected)
    rows["noise_low_confidence"] = rows["confidence"].fillna(0.0).lt(0.70)
    rows["noise_high_odds"] = rows["quoted_odds"].fillna(0.0).ge(3.0)
    rows["noise_unstable_region"] = rows["region_roi"].le(0.0) | rows["region_positive_split_ratio"].lt(2.0 / 3.0)
    rows["noise_positive_ev_but_unstable"] = rows["policy_ev"].gt(0.0) & (
        rows["noise_low_confidence"] | rows["noise_high_odds"] | rows["noise_unstable_region"]
    )
    csv_path = run_dir / "decision_region_noise_report.csv"
    rows[[col for col in REPORT_COLUMNS if col in rows.columns]].to_csv(csv_path, index=False)

    fill_audit = _fill_audit(retro_run_dir)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "retro_run_dir": str(retro_run_dir),
        "policy": policy,
        "policy_reoptimized": False,
        "locked_holdout_used_for_training": False,
        "selected_picks": int(len(rows)),
        "noise_positive_ev_but_unstable_share": float(rows["noise_positive_ev_but_unstable"].mean()) if len(rows) else None,
        "noise_flag_counts": {
            "low_confidence": int(rows["noise_low_confidence"].sum()),
            "high_odds": int(rows["noise_high_odds"].sum()),
            "unstable_region": int(rows["noise_unstable_region"].sum()),
            "positive_ev_but_unstable": int(rows["noise_positive_ev_but_unstable"].sum()),
        },
        "metrics": _split_summary(rows),
        "fill_audit": fill_audit,
        "csv_path": str(csv_path),
    }
    _write_json(run_dir / "decision_region_noise_report.json", payload)
    return payload


def _fill_audit(retro_run_dir: Path) -> dict[str, Any]:
    path = retro_run_dir / "retro_fill_rows.csv"
    if not path.exists():
        return {"available": False}
    fills = pd.read_csv(path)
    if fills.empty or "decision_id" not in fills:
        return {"available": True, "rows": int(len(fills)), "canonical_decisions": 0}
    fills["notional"] = pd.to_numeric(fills.get("notional"), errors="coerce")
    fills["net_profit"] = pd.to_numeric(fills.get("net_profit"), errors="coerce")
    canonical = fills.sort_values("notional").drop_duplicates("decision_id", keep="first")
    stake = float(canonical["notional"].sum())
    profit = float(canonical["net_profit"].sum())
    return {
        "available": True,
        "rows": int(len(fills)),
        "canonical_rule": "minimum_notional_per_decision",
        "canonical_decisions": int(canonical["decision_id"].nunique()),
        "canonical_stake": stake,
        "canonical_profit": profit,
        "canonical_roi": float(profit / stake) if stake else None,
    }


def _evaluate_variant(name: str, candidates: pd.DataFrame, policy: dict[str, Any], score_col: str = "policy_rank_score") -> dict[str, Any]:
    selected = _select_one_per_match(candidates, score_col=score_col)
    selected = _add_confidence_deciles(selected)
    metrics = _split_summary(selected)
    return {
        "name": name,
        "selected_picks": int(len(selected)),
        "policy_reoptimized": False,
        "thresholds_changed": False,
        "scopes_changed": False,
        "locked_holdout_used_for_training": False,
        "policy_edge_threshold": policy["edge_threshold"],
        "policy_ev_threshold": policy["ev_threshold"],
        "metrics": metrics,
        "status": _status(metrics),
    }


def _write_policy_comparison(run_dir: Path, df: pd.DataFrame, region: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    base_candidates = df[df["base_policy_candidate"]].copy()
    comparisons: list[dict[str, Any]] = []
    comparisons.append(_evaluate_variant("frozen_policy_current", base_candidates, policy))

    if "policy_uncertainty_multiplier" in df.columns:
        conservative = df.copy()
        multiplier = pd.to_numeric(conservative["policy_uncertainty_multiplier"], errors="coerce").clip(0.0, 1.0).fillna(1.0)
        conservative["policy_prob"] = conservative["policy_prob"] * multiplier
        conservative["policy_edge"] = conservative["policy_prob"] - conservative["top_ask"]
        conservative["policy_ev"] = conservative["policy_prob"] * conservative["quoted_odds"] - 1.0
        mask = conservative["base_policy_candidate"]
        mask &= conservative["policy_edge"].ge(policy["edge_threshold"]) & conservative["policy_ev"].ge(policy["ev_threshold"])
        comparisons.append(_evaluate_variant("frozen_policy_conservative_existing_multiplier", conservative[mask], policy))

    stable = _attach_region(df, region)
    stable_mask = stable["base_policy_candidate"]
    stable_mask &= stable["region_roi"].gt(0.0) & stable["region_positive_split_ratio"].ge(2.0 / 3.0)
    comparisons.append(_evaluate_variant("frozen_policy_pre_holdout_stability_exclusion", stable[stable_mask], policy))

    rows = []
    for item in comparisons:
        for bucket, metric in (item["metrics"].get("by_validation_bucket") or {}).items():
            rows.append(
                {
                    "comparison": item["name"],
                    "validation_bucket": bucket,
                    "bets": metric.get("bets"),
                    "roi": metric.get("roi"),
                    "hit_rate": metric.get("hit_rate"),
                    "status": item["status"],
                }
            )
    csv_path = run_dir / "frozen_policy_comparison_report.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "policy": policy,
        "policy_reoptimized": False,
        "thresholds_changed": False,
        "scopes_changed": False,
        "locked_holdout_used_for_training": False,
        "comparisons": comparisons,
        "csv_path": str(csv_path),
    }
    _write_json(run_dir / "frozen_policy_comparison_report.json", payload)
    return payload


def _write_reformulation_report(run_dir: Path, df: pd.DataFrame, region: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    rows = _attach_region(df[df["base_policy_candidate"]].copy(), region)
    if rows.empty:
        rows["stable_rank_score"] = []
    else:
        base = pd.to_numeric(rows.get("policy_rank_score"), errors="coerce").fillna(pd.to_numeric(rows.get("policy_ev"), errors="coerce"))
        stability = (0.50 + rows["region_positive_split_ratio"]).clip(0.25, 1.25)
        roi_factor = (1.0 + rows["region_roi"]).clip(0.25, 1.50)
        rows["stable_rank_score"] = base.fillna(0.0) * stability * roi_factor
    selected = _select_one_per_match(rows, score_col="stable_rank_score")
    selected = _add_confidence_deciles(selected)
    metrics = _split_summary(selected)
    status = _status(metrics)
    selected["reformulation_status"] = status
    csv_path = run_dir / "decision_region_reformulation_report.csv"
    keep = [
        "match_key",
        "validation_bucket",
        "analysis_split",
        "league_code",
        "selection",
        "odds_band",
        "quoted_odds",
        "policy_ev",
        "confidence",
        "region_roi",
        "region_positive_split_ratio",
        "stable_rank_score",
        "unit_profit",
        "won",
        "reformulation_status",
    ]
    selected[[col for col in keep if col in selected.columns]].to_csv(csv_path, index=False)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "policy_reoptimized": False,
        "one_pick_per_match": True,
        "locked_holdout_used_for_training": False,
        "allowed_training_buckets": sorted(TRAIN_BUCKETS),
        "status": status,
        "metrics": metrics,
        "csv_path": str(csv_path),
    }
    _write_json(run_dir / "decision_region_reformulation_report.json", payload)
    return payload


def run_offline_decision_region_review(
    retro_run_dir: Path,
    outputs_dir: Path,
    policy_bundle: Path | None = None,
) -> OfflineReviewResult:
    retro_run_dir = Path(retro_run_dir)
    outputs_dir = Path(outputs_dir)
    run = create_run_context(outputs_dir / "runs", "offline_decision_region_review")
    policy = _load_policy(policy_bundle, retro_run_dir)
    df = _prepare_candidates(retro_run_dir, policy)
    base_picks = _select_one_per_match(df[df["base_policy_candidate"]])
    region = _region_stability(base_picks)
    selected = _attach_region(base_picks, region)

    noise = _write_noise_report(run.run_dir, selected, retro_run_dir, policy)
    comparison = _write_policy_comparison(run.run_dir, df, region, policy)
    reformulation = _write_reformulation_report(run.run_dir, df, region, policy)

    summary = {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "retro_run_dir": str(retro_run_dir),
        "diagnostic_only": True,
        "policy_written": False,
        "policy_reoptimized": False,
        "t45m_policy_touched": False,
        "sqlite_written": False,
        "locked_holdout_used_for_training": False,
        "noise_report": {
            "selected_picks": noise["selected_picks"],
            "noise_positive_ev_but_unstable_share": noise["noise_positive_ev_but_unstable_share"],
            "noise_flag_counts": noise["noise_flag_counts"],
        },
        "best_frozen_policy_comparison": _best_comparison(comparison),
        "reformulation_status": reformulation["status"],
        "artifacts": {
            "decision_region_noise_report_json": str(run.run_dir / "decision_region_noise_report.json"),
            "decision_region_noise_report_csv": str(run.run_dir / "decision_region_noise_report.csv"),
            "frozen_policy_comparison_report_json": str(run.run_dir / "frozen_policy_comparison_report.json"),
            "frozen_policy_comparison_report_csv": str(run.run_dir / "frozen_policy_comparison_report.csv"),
            "decision_region_reformulation_report_json": str(run.run_dir / "decision_region_reformulation_report.json"),
            "decision_region_reformulation_report_csv": str(run.run_dir / "decision_region_reformulation_report.csv"),
        },
    }
    _write_json(run.run_dir / "summary.json", summary)
    (outputs_dir / "latest_offline_decision_region_review.txt").write_text(str(run.run_dir), encoding="utf-8")
    return OfflineReviewResult(run_dir=run.run_dir, summary=summary)


def _best_comparison(report: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for item in report.get("comparisons") or []:
        buckets = item.get("metrics", {}).get("by_validation_bucket", {})
        oof_roi = _safe_float((buckets.get("oof") or {}).get("roi"), -999.0)
        pre_roi = _safe_float((buckets.get("pre_holdout") or {}).get("roi"), -999.0)
        holdout_roi = _safe_float((buckets.get("locked_holdout") or {}).get("roi"), -999.0)
        key = (oof_roi, pre_roi, holdout_roi)
        candidate = {"name": item["name"], "status": item["status"], "oof_roi": oof_roi, "pre_holdout_roi": pre_roi, "holdout_roi": holdout_roi}
        if best is None or key > best["_key"]:
            best = {**candidate, "_key": key}
    if best is None:
        return {}
    best.pop("_key", None)
    return best


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _count(con: sqlite3.Connection, table: str, where: str = "") -> int:
    if not _table_exists(con, table):
        return 0
    query = f"SELECT COUNT(*) FROM {table}"
    if where:
        query += f" WHERE {where}"
    return int(con.execute(query).fetchone()[0] or 0)


def _scalar(con: sqlite3.Connection, query: str) -> Any:
    try:
        row = con.execute(query).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def build_forward_capture_status(db_path: Path, collect_summary_path: Path | None = None) -> dict[str, Any]:
    db_path = Path(db_path)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "db_path": str(db_path),
        "read_only": True,
        "writes_performed": False,
    }
    if not db_path.exists():
        payload.update({"capture_liveness": "db_missing", "counts": {}})
        return payload

    now = datetime.now(UTC)
    con = _connect_read_only(db_path)
    try:
        latest_checkpoint = _scalar(con, "SELECT MAX(timestamp) FROM pm_book_checkpoints")
        latest_dt = _parse_dt(latest_checkpoint)
        age = float((now - latest_dt).total_seconds()) if latest_dt else None
        counts = {
            "book_checkpoints": _count(con, "pm_book_checkpoints"),
            "book_best": _count(con, "pm_book_best"),
            "trades": _count(con, "pm_trades"),
            "shadow_decisions": _count(con, "pm_shadow_decisions"),
            "shadow_fills": _count(con, "pm_shadow_fills"),
            "market_groups": _count(con, "pm_market_groups"),
            "mapped_groups": _count(con, "pm_market_groups", "mapping_status = 'complete'"),
            "decision_checkpoints": _count(con, "pm_book_checkpoints", "event_type = 'decision_checkpoint'"),
            "periodic_checkpoints": _count(con, "pm_book_checkpoints", "event_type = 'periodic_checkpoint'"),
        }
        recent_cutoff = (now - pd.Timedelta(hours=1)).isoformat()
        counts["checkpoints_last_hour"] = _count(con, "pm_book_checkpoints", f"timestamp >= '{recent_cutoff}'")
        upcoming = _upcoming_decision_windows(con, now)
    finally:
        con.close()

    if age is None:
        liveness = "no_checkpoints"
    elif age <= 180:
        liveness = "active_recent_checkpoint"
    elif age <= 900:
        liveness = "stale_warning"
    else:
        liveness = "stale_blocked"

    payload.update(
        {
            "counts": counts,
            "latest_checkpoint": latest_checkpoint,
            "latest_checkpoint_age_seconds": age,
            "capture_liveness": liveness,
            "upcoming_t45m_windows": upcoming,
            "collect_summary": _collect_summary_status(collect_summary_path),
        }
    )
    return payload


def _upcoming_decision_windows(con: sqlite3.Connection, now: datetime) -> list[dict[str, Any]]:
    if not _table_exists(con, "pm_market_groups"):
        return []
    rows = con.execute(
        "SELECT group_key, league_code, home_team, away_team, game_start_time "
        "FROM pm_market_groups WHERE mapping_status = 'complete'"
    ).fetchall()
    upcoming: list[dict[str, Any]] = []
    for group_key, league_code, home_team, away_team, kickoff_raw in rows:
        kickoff = _parse_dt(kickoff_raw)
        if kickoff is None:
            continue
        decision = kickoff - pd.Timedelta(minutes=45)
        if decision < now - pd.Timedelta(minutes=5):
            continue
        upcoming.append(
            {
                "decision_time": decision.isoformat(),
                "seconds_until_decision": float((decision - now).total_seconds()),
                "kickoff_time": kickoff.isoformat(),
                "group_key": group_key,
                "league_code": league_code,
                "home_team": home_team,
                "away_team": away_team,
            }
        )
    return sorted(upcoming, key=lambda item: item["decision_time"])[:12]


def _collect_summary_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False}
    path = Path(path)
    if not path.exists():
        return {"available": False, "path": str(path)}
    payload = _maybe_json(path)
    stream = payload.get("stream") or {}
    return {
        "available": True,
        "path": str(path),
        "stream_seconds": payload.get("stream_seconds"),
        "checkpoint_rows_inserted": payload.get("checkpoint_rows_inserted"),
        "market_stream_errors": stream.get("market_stream_errors", 0),
        "sports_stream_errors": stream.get("sports_stream_errors", 0),
        "market_stream_last_error": stream.get("market_stream_last_error"),
        "sports_stream_last_error": stream.get("sports_stream_last_error"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline decision-region review helpers.")
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="Generate read-only reports from an existing retro run.")
    review.add_argument("--retro-run-dir", required=True)
    review.add_argument("--outputs-dir", required=True)
    review.add_argument("--policy-bundle", default="")

    monitor = sub.add_parser("monitor", help="Read-only forward capture status from SQLite.")
    monitor.add_argument("--db-path", required=True)
    monitor.add_argument("--collect-summary", default="")
    monitor.add_argument("--json", action="store_true")
    return parser


def _print_monitor(payload: dict[str, Any]) -> None:
    counts = payload.get("counts") or {}
    print(f"capture_liveness: {payload.get('capture_liveness')}")
    print(f"db_path: {payload.get('db_path')}")
    print(f"read_only: {payload.get('read_only')} writes_performed: {payload.get('writes_performed')}")
    print(
        "counts: "
        f"checkpoints={counts.get('book_checkpoints', 0)} "
        f"last_hour={counts.get('checkpoints_last_hour', 0)} "
        f"decision_ckpt={counts.get('decision_checkpoints', 0)} "
        f"best={counts.get('book_best', 0)} "
        f"trades={counts.get('trades', 0)} "
        f"decisions={counts.get('shadow_decisions', 0)} "
        f"fills={counts.get('shadow_fills', 0)} "
        f"groups={counts.get('mapped_groups', 0)}/{counts.get('market_groups', 0)}"
    )
    print(f"latest_checkpoint: {payload.get('latest_checkpoint')} age_seconds={payload.get('latest_checkpoint_age_seconds')}")
    collect = payload.get("collect_summary") or {}
    if collect.get("available"):
        print(
            "collect_summary: "
            f"{collect.get('path')} "
            f"checkpoint_rows_inserted={collect.get('checkpoint_rows_inserted')} "
            f"market_ws_errors={collect.get('market_stream_errors')} "
            f"sports_ws_errors={collect.get('sports_stream_errors')}"
        )
    upcoming = payload.get("upcoming_t45m_windows") or []
    if upcoming:
        print("upcoming_t45m:")
        for item in upcoming[:5]:
            print(
                f"- {item.get('decision_time')} in {int(item.get('seconds_until_decision', 0))}s "
                f"{item.get('league_code')} {item.get('home_team')} vs {item.get('away_team')}"
            )
    else:
        print("upcoming_t45m: none")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "review":
        result = run_offline_decision_region_review(
            retro_run_dir=Path(args.retro_run_dir),
            outputs_dir=Path(args.outputs_dir),
            policy_bundle=Path(args.policy_bundle) if args.policy_bundle else None,
        )
        print(f"offline_decision_region_review: {result.run_dir}")
        print(json.dumps(result.summary, indent=2, ensure_ascii=True, default=_json_default))
        return 0
    if args.command == "monitor":
        payload = build_forward_capture_status(
            db_path=Path(args.db_path),
            collect_summary_path=Path(args.collect_summary) if args.collect_summary else None,
        )
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default))
        else:
            _print_monitor(payload)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
