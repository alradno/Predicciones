from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import OUTCOME_ORDER


DEFAULT_ODDS_BANDS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 1.5, "<1.5"),
    (1.5, 2.0, "1.5-2.0"),
    (2.0, 3.0, "2.0-3.0"),
    (3.0, 6.0, "3.0-6.0"),
    (6.0, None, "6.0+"),
)

_DEFAULT_SELECTION_COLUMNS = (
    "selection",
    "selected_outcome",
    "bet_outcome",
    "chosen_outcome",
    "prediction",
    "outcome",
)
_DEFAULT_ODDS_COLUMNS = (
    "selection_odds",
    "executed_odds",
    "quoted_odds",
    "odds",
    "price",
)
_DEFAULT_PROFIT_COLUMNS = (
    "net_profit",
    "flat_profit",
    "gross_profit",
    "kelly_profit",
    "profit",
    "pnl",
)
_DEFAULT_STAKE_COLUMNS = (
    "accepted_stake",
    "flat_stake",
    "requested_stake",
    "cost_basis",
    "stake",
    "wager",
    "stake_amount",
)
_DEFAULT_CONFIDENCE_COLUMNS = (
    "confidence_score",
    "selection_confidence",
    "policy_confidence",
    "decision_confidence",
    "backoff_confidence",
)
_DEFAULT_FOLD_COLUMNS = ("fold_id", "policy_fold", "fold_segment")
_DEFAULT_WINDOW_COLUMNS = (
    "policy_window",
    "window_id",
    "window_name",
    "policy_week",
    "decision_window",
    "fold_window",
)


@dataclass(frozen=True)
class ResolvedDecisionRegionColumns:
    selection: str | None
    odds: str | None
    profit: str | None
    stake: str | None
    confidence: str | None


def _as_frame(selected_bets: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(selected_bets, pd.DataFrame):
        return selected_bets.copy()
    return pd.DataFrame(list(selected_bets))


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _resolve_columns(
    frame: pd.DataFrame,
    selection_column: str | None = None,
    odds_column: str | None = None,
    profit_column: str | None = None,
    stake_column: str | None = None,
    confidence_column: str | None = None,
) -> ResolvedDecisionRegionColumns:
    return ResolvedDecisionRegionColumns(
        selection=selection_column or _first_existing_column(frame, _DEFAULT_SELECTION_COLUMNS),
        odds=odds_column or _first_existing_column(frame, _DEFAULT_ODDS_COLUMNS),
        profit=profit_column or _first_existing_column(frame, _DEFAULT_PROFIT_COLUMNS),
        stake=stake_column or _first_existing_column(frame, _DEFAULT_STAKE_COLUMNS),
        confidence=confidence_column or _first_existing_column(frame, _DEFAULT_CONFIDENCE_COLUMNS),
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _extract_group_rows(frame: pd.DataFrame, group_column: str | None) -> dict[str, pd.DataFrame]:
    if group_column is None or group_column not in frame.columns:
        return {}
    working = frame.copy()
    if group_column in {"fold_id"}:
        working = working[working[group_column].notna()].copy()
    groups: dict[str, pd.DataFrame] = {}
    for key, group in working.groupby(group_column, observed=True, dropna=True):
        groups[str(key)] = group.copy()
    return groups


def _prepare_profit_and_stake(
    frame: pd.DataFrame,
    columns: ResolvedDecisionRegionColumns,
) -> pd.DataFrame:
    working = frame.copy()

    if columns.selection and columns.selection in working.columns:
        working["__selection__"] = working[columns.selection].astype(str)
    else:
        working["__selection__"] = ""

    if "won" in working.columns:
        working["__won__"] = pd.to_numeric(working["won"], errors="coerce")
    elif "actual_outcome" in working.columns and columns.selection and columns.selection in working.columns:
        working["__won__"] = (working["actual_outcome"].astype(str) == working[columns.selection].astype(str)).astype(float)
    else:
        working["__won__"] = np.nan

    if columns.odds and columns.odds in working.columns:
        working["__odds__"] = pd.to_numeric(working[columns.odds], errors="coerce")
    else:
        working["__odds__"] = np.nan

    profit_column = columns.profit if columns.profit and columns.profit in working.columns else None
    if profit_column is not None:
        working["__profit__"] = pd.to_numeric(working[profit_column], errors="coerce")
    elif working["__won__"].notna().any() and working["__odds__"].notna().any():
        working["__profit__"] = np.where(
            working["__won__"].fillna(0).eq(1),
            working["__odds__"] - 1.0,
            -1.0,
        )
    else:
        working["__profit__"] = np.nan

    stake_column = columns.stake if columns.stake and columns.stake in working.columns else None
    if stake_column is not None:
        working["__stake__"] = pd.to_numeric(working[stake_column], errors="coerce")
    elif working["__profit__"].notna().any():
        working["__stake__"] = 1.0
    else:
        working["__stake__"] = np.nan

    return working


def _summarize_frame(frame: pd.DataFrame, columns: ResolvedDecisionRegionColumns) -> dict[str, Any]:
    if frame.empty:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate": 0.0,
            "profit": 0.0,
            "roi": 0.0,
            "stake": 0.0,
            "avg_odds": None,
            "median_odds": None,
            "avg_confidence": None,
            "median_confidence": None,
            "unique_matches": 0,
        }

    working = _prepare_profit_and_stake(frame, columns)
    profit_series = pd.to_numeric(working["__profit__"], errors="coerce")
    stake_series = pd.to_numeric(working["__stake__"], errors="coerce")
    odds_series = pd.to_numeric(working["__odds__"], errors="coerce")
    won_series = pd.to_numeric(working["__won__"], errors="coerce")
    confidence_series = pd.to_numeric(working[columns.confidence], errors="coerce") if columns.confidence and columns.confidence in working.columns else pd.Series(dtype=float)

    profit = float(profit_series.dropna().sum()) if not profit_series.dropna().empty else 0.0
    stake = float(stake_series.dropna().sum()) if not stake_series.dropna().empty else float(len(working))
    wins = int(won_series.fillna(0).eq(1).sum()) if not won_series.empty else 0
    hit_rate = float(wins / len(working)) if len(working) > 0 else 0.0
    roi = _safe_ratio(profit, stake) or 0.0

    overview = {
        "bets": int(len(working)),
        "wins": wins,
        "win_rate": hit_rate,
        "profit": profit,
        "roi": roi,
        "stake": stake,
        "avg_odds": float(odds_series.dropna().mean()) if not odds_series.dropna().empty else None,
        "median_odds": float(odds_series.dropna().median()) if not odds_series.dropna().empty else None,
        "avg_confidence": float(confidence_series.dropna().mean()) if not confidence_series.dropna().empty else None,
        "median_confidence": float(confidence_series.dropna().median()) if not confidence_series.dropna().empty else None,
        "unique_matches": int(working["match_id"].nunique()) if "match_id" in working.columns else int(len(working)),
    }

    if "match_id" in working.columns:
        overview["match_coverage"] = float(working["match_id"].nunique() / len(working)) if len(working) else 0.0
    if "selection_edge" in working.columns:
        edge_series = pd.to_numeric(working["selection_edge"], errors="coerce")
        overview["avg_edge"] = float(edge_series.dropna().mean()) if not edge_series.dropna().empty else None
    if "selection_ev" in working.columns:
        ev_series = pd.to_numeric(working["selection_ev"], errors="coerce")
        overview["avg_ev"] = float(ev_series.dropna().mean()) if not ev_series.dropna().empty else None

    return overview


def _summary_by_group(
    frame: pd.DataFrame,
    group_column: str | None,
    columns: ResolvedDecisionRegionColumns,
) -> dict[str, Any]:
    grouped = _extract_group_rows(frame, group_column)
    if not grouped:
        return {}
    return {
        group_key: _summarize_frame(group_frame, columns)
        for group_key, group_frame in grouped.items()
    }


def _assign_odds_band(odds: pd.Series, odds_band_edges: Sequence[float]) -> pd.Series:
    if odds.empty:
        return pd.Series(dtype="object")
    edges = list(float(edge) for edge in odds_band_edges)
    if len(edges) < 2:
        raise ValueError("odds_band_edges must contain at least two edges")
    if not np.isinf(edges[-1]):
        edges = edges + [np.inf]
    labels: list[str] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        if np.isinf(upper):
            labels.append(f"{lower:g}+")
        elif lower <= 0 and upper <= 1.5:
            labels.append("<1.5")
        else:
            labels.append(f"{lower:g}-{upper:g}")
    return pd.cut(odds, bins=edges, labels=labels, include_lowest=True, right=False)


def _summary_by_odds_band(
    frame: pd.DataFrame,
    columns: ResolvedDecisionRegionColumns,
    odds_band_edges: Sequence[float],
) -> dict[str, Any]:
    if columns.odds is None or columns.odds not in frame.columns:
        return {}
    working = frame.copy()
    working["__odds_band__"] = _assign_odds_band(pd.to_numeric(working[columns.odds], errors="coerce"), odds_band_edges)
    working = working[working["__odds_band__"].notna()].copy()
    if working.empty:
        return {}
    summary: dict[str, Any] = {}
    for band, group in working.groupby("__odds_band__", observed=True, dropna=True):
        summary[str(band)] = _summarize_frame(group, columns)
    return summary


def _confidence_deciles(frame: pd.DataFrame, confidence_column: str) -> pd.Series:
    confidence = pd.to_numeric(frame[confidence_column], errors="coerce")
    valid = confidence.notna()
    if not valid.any():
        return pd.Series(dtype="Int64")
    ranked = confidence[valid].rank(method="first", pct=True)
    deciles = np.ceil(ranked * 10.0).astype(int).clip(1, 10)
    result = pd.Series(index=frame.index, dtype="Int64")
    result.loc[valid] = deciles.to_numpy()
    return result


def _summary_by_confidence_decile(frame: pd.DataFrame, columns: ResolvedDecisionRegionColumns) -> dict[str, Any] | None:
    if columns.confidence is None or columns.confidence not in frame.columns:
        return None
    working = frame.copy()
    working["__confidence_decile__"] = _confidence_deciles(working, columns.confidence)
    working = working[working["__confidence_decile__"].notna()].copy()
    if working.empty:
        return None
    return {
        str(int(decile)): _summarize_frame(group, columns)
        for decile, group in working.groupby("__confidence_decile__", observed=True, dropna=True)
    }


def _first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def _collapse_selected_bet_rows(selected_bets: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frame = _as_frame(selected_bets)
    if frame.empty or "decision_id" not in frame.columns:
        return frame

    working = frame.copy()
    working = working[working["decision_id"].notna()].copy()
    if working.empty:
        return frame

    if not working["decision_id"].duplicated().any():
        if "fill_count" not in working.columns:
            working["fill_count"] = 1
        return working

    sort_columns = [column for column in ("created_at", "fill_id", "fill_index", "timestamp") if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns, kind="mergesort").copy()

    additive_columns = {
        "accepted_stake",
        "amount",
        "cost_basis",
        "exposure",
        "flat_profit",
        "flat_stake",
        "gross_profit",
        "kelly_profit",
        "net_profit",
        "notional",
        "pnl",
        "profit",
        "quantity",
        "requested_stake",
        "size",
        "stake",
        "stake_amount",
        "wager",
    }
    additive_columns = {column for column in additive_columns if column in working.columns}
    for column in additive_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    grouped = working.groupby("decision_id", sort=False)
    pieces: list[pd.DataFrame | pd.Series] = []
    metadata_columns = [
        column
        for column in working.columns
        if column not in additive_columns and column not in {"decision_id", "fill_count"}
    ]
    if metadata_columns:
        pieces.append(grouped[metadata_columns].first())
    if additive_columns:
        pieces.append(grouped[sorted(additive_columns)].sum(min_count=1))
    pieces.append(grouped.size().rename("fill_count"))

    collapsed = pd.concat(pieces, axis=1).reset_index()
    ordered_columns = ["decision_id"]
    ordered_columns.extend(
        column
        for column in working.columns
        if column != "decision_id" and column in collapsed.columns and column != "fill_count"
    )
    ordered_columns.append("fill_count")
    return collapsed.loc[:, ordered_columns].copy()


def summarize_selected_bets(
    selected_bets: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    selection_column: str | None = None,
    odds_column: str | None = None,
    profit_column: str | None = None,
    stake_column: str | None = None,
    confidence_column: str | None = None,
    fold_columns: Sequence[str] = _DEFAULT_FOLD_COLUMNS,
    window_columns: Sequence[str] = _DEFAULT_WINDOW_COLUMNS,
    odds_band_edges: Sequence[float] = (0.0, 1.5, 2.0, 3.0, 6.0),
) -> dict[str, Any]:
    """
    Summarize the bets a policy actually selected.

    The helper is intentionally flexible so it can consume either a DataFrame
    or an iterable of dictionaries. It detects common column names for the
    selected outcome, odds, profit/stake, confidence and grouping dimensions.
    """

    frame = _collapse_selected_bet_rows(selected_bets)
    columns = _resolve_columns(
        frame,
        selection_column=selection_column,
        odds_column=odds_column,
        profit_column=profit_column,
        stake_column=stake_column,
        confidence_column=confidence_column,
    )

    if frame.empty:
        return {
            "overview": _summarize_frame(frame, columns),
            "by_outcome": {},
            "by_odds_band": {},
            "by_fold": {},
            "by_window": {},
            "by_confidence_decile": None,
            "column_map": {
                "selection": columns.selection,
                "odds": columns.odds,
                "profit": columns.profit,
                "stake": columns.stake,
                "confidence": columns.confidence,
            },
        }

    working = frame.copy()
    if columns.selection and columns.selection in working.columns:
        working["__selection_region__"] = working[columns.selection].astype(str)
    elif "selection" in working.columns:
        working["__selection_region__"] = working["selection"].astype(str)
    else:
        working["__selection_region__"] = ""

    outcome_summary: dict[str, Any] = {}
    if "selection" in working.columns or columns.selection is not None:
        for outcome in OUTCOME_ORDER:
            if columns.selection and columns.selection in working.columns:
                group = working[working[columns.selection].astype(str).eq(outcome)].copy()
            elif "selection" in working.columns:
                group = working[working["selection"].astype(str).eq(outcome)].copy()
            else:
                group = pd.DataFrame(columns=working.columns)
            if not group.empty:
                outcome_summary[outcome] = _summarize_frame(group, columns)

        if not outcome_summary and "__selection_region__" in working.columns:
            for outcome, group in working.groupby("__selection_region__", observed=True, dropna=True):
                outcome_summary[str(outcome)] = _summarize_frame(group, columns)

    fold_column = _first_existing_column(working, fold_columns)
    window_column = _first_existing_column(working, window_columns)

    summary: dict[str, Any] = {
        "overview": _summarize_frame(working, columns),
        "by_outcome": outcome_summary,
        "by_odds_band": _summary_by_odds_band(working, columns, odds_band_edges),
        "by_fold": _summary_by_group(working, fold_column, columns),
        "by_window": _summary_by_group(working, window_column, columns),
        "by_confidence_decile": _summary_by_confidence_decile(working, columns),
        "column_map": {
            "selection": columns.selection,
            "odds": columns.odds,
            "profit": columns.profit,
            "stake": columns.stake,
            "confidence": columns.confidence,
            "fold": fold_column,
            "window": window_column,
        },
        "odds_band_edges": list(odds_band_edges),
    }
    return summary


def build_decision_region_diagnostics(
    selected_bets: pd.DataFrame | Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Alias for summarize_selected_bets, kept as a clearer public entry point.
    """

    return summarize_selected_bets(selected_bets, **kwargs)


def format_decision_region_summary(summary: dict[str, Any]) -> str:
    overview = summary.get("overview", {})
    by_outcome = summary.get("by_outcome", {})
    by_odds_band = summary.get("by_odds_band", {})
    confidence = summary.get("by_confidence_decile")
    lines = [
        f"- bets: {overview.get('bets', 0)}",
        f"- win_rate: {overview.get('win_rate', 0.0):.4f}",
        f"- roi: {overview.get('roi', 0.0):.4f}",
    ]
    if by_outcome:
        lines.append(f"- outcomes: {', '.join(sorted(by_outcome.keys()))}")
    if by_odds_band:
        lines.append(f"- odds bands: {', '.join(sorted(by_odds_band.keys()))}")
    if confidence:
        lines.append(f"- confidence deciles: {', '.join(sorted(confidence.keys(), key=lambda x: int(x)))}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_ODDS_BANDS",
    "build_decision_region_diagnostics",
    "format_decision_region_summary",
    "summarize_selected_bets",
]
