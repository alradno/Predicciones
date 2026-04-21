from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import METADATA_COLUMNS, OUTCOME_TO_TARGET

ROLLING_METRICS = (
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "yellows",
    "reds",
    "points",
    "goal_diff",
    "xg_proxy",
)

LEAGUE_SEASON_MATCH_COUNTS = {
    "E0": 38.0,
    "SP1": 38.0,
    "D1": 34.0,
    "I1": 38.0,
    "F1": 34.0,
    "N1": 34.0,
    "P1": 34.0,
    "MEX": 34.0,
    "USA": 34.0,
}

CONFIDENCE_HISTORY_MIN = 20
CONFIDENCE_LONG_OVERALL_TARGET = 40
CONFIDENCE_LONG_SIDE_TARGET = 20


def _mean_from_history(history: Iterable[dict[str, float]], field_name: str) -> float:
    values = [float(item[field_name]) for item in history if pd.notna(item.get(field_name))]
    return float(np.mean(values)) if values else np.nan


def _sum_from_history(history: Iterable[dict[str, float]], field_name: str, limit: int | None = None) -> float:
    values = [float(item[field_name]) for item in history if pd.notna(item.get(field_name))]
    if limit is not None:
        values = values[-int(limit) :]
    return float(np.sum(values)) if values else np.nan


def _exp_mean_from_history(
    history: Iterable[dict[str, float]],
    field_name: str,
    alpha: float = 0.35,
    limit: int | None = None,
) -> float:
    values = [float(item[field_name]) for item in history if pd.notna(item.get(field_name))]
    if limit is not None:
        values = values[-int(limit) :]
    if not values:
        return np.nan
    weights = np.array([(1.0 - alpha) ** offset for offset in range(len(values) - 1, -1, -1)], dtype=float)
    weights /= weights.sum()
    return float(np.dot(np.array(values, dtype=float), weights))


def _count_recent_matches(history: Iterable[dict[str, float]], current_date: pd.Timestamp, horizon_days: int) -> float:
    total = 0
    for item in history:
        match_date = pd.Timestamp(item.get("date")) if item.get("date") is not None else pd.NaT
        if pd.isna(match_date):
            continue
        delta_days = float((pd.Timestamp(current_date) - match_date).days)
        if 0.0 <= delta_days <= float(horizon_days):
            total += 1
    return float(total)


def _safe_ratio(numerator: float, denominator: float, fallback: float = np.nan) -> float:
    if denominator == 0:
        return fallback
    return numerator / denominator


def _coverage_ratio(sample_size: float, target_size: float) -> float:
    if pd.isna(sample_size) or pd.isna(target_size) or float(target_size) <= 0.0:
        return np.nan
    return float(np.clip(float(sample_size) / float(target_size), 0.0, 1.0))


def _shrink_metric(metric: float, sample_size: float, prior_mean: float, prior_strength: float) -> float:
    if pd.isna(metric):
        return np.nan
    n = max(float(sample_size), 0.0)
    weight = n / (n + float(prior_strength))
    return float((weight * metric) + ((1.0 - weight) * float(prior_mean)))


def _volatility_from_history(history: Iterable[dict[str, float]], field_name: str, limit: int) -> float:
    values = [float(item[field_name]) for item in history if pd.notna(item.get(field_name))]
    values = values[-int(limit) :]
    if len(values) < 2:
        return np.nan
    return float(np.std(np.array(values, dtype=float), ddof=0))


def _schedule_irregularity_from_history(history: Iterable[dict[str, float]], limit: int) -> float:
    dates = [pd.Timestamp(item["date"]) for item in history if pd.notna(item.get("date"))]
    dates = dates[-int(limit) :]
    if len(dates) < 3:
        return np.nan
    gaps = [
        float((dates[index] - dates[index - 1]).days)
        for index in range(1, len(dates))
    ]
    if len(gaps) < 2:
        return np.nan
    return float(np.std(np.asarray(gaps, dtype=float), ddof=0))


def _sample_adequacy_flag(sample_size: float, minimum: float) -> float:
    return float(float(sample_size) >= float(minimum))


def _pair_mean_or_nan(left: float, right: float) -> float:
    values = [float(value) for value in (left, right) if pd.notna(value)]
    return float(np.mean(values)) if values else np.nan


@dataclass
class TeamState:
    rolling_window: int
    overall_history: deque[dict[str, float]] = field(init=False)
    side_history: dict[str, deque[dict[str, float]]] = field(init=False)
    long_overall_history: deque[dict[str, float]] = field(init=False)
    long_side_history: dict[str, deque[dict[str, float]]] = field(init=False)
    opponent_overall_history: deque[dict[str, float]] = field(init=False)
    opponent_side_history: dict[str, deque[dict[str, float]]] = field(init=False)
    elo: float = 1500.0
    last_date: pd.Timestamp | None = None
    matches_played: int = 0

    def __post_init__(self) -> None:
        # Keep short windows for recent form, but let the longer buffers carry
        # a meaningfully deeper history so confidence signals do not saturate.
        long_overall_window = max(self.rolling_window, CONFIDENCE_LONG_OVERALL_TARGET)
        long_side_window = max(self.rolling_window, CONFIDENCE_LONG_SIDE_TARGET)
        self.overall_history = deque(maxlen=self.rolling_window)
        self.side_history = {
            "home": deque(maxlen=self.rolling_window),
            "away": deque(maxlen=self.rolling_window),
        }
        self.long_overall_history = deque(maxlen=long_overall_window)
        self.long_side_history = {
            "home": deque(maxlen=long_side_window),
            "away": deque(maxlen=long_side_window),
        }
        self.opponent_overall_history = deque(maxlen=long_overall_window)
        self.opponent_side_history = {
            "home": deque(maxlen=long_side_window),
            "away": deque(maxlen=long_side_window),
        }


@dataclass
class LeagueState:
    home_matches: int = 0
    home_wins: int = 0
    draws: int = 0
    goal_diff_total: float = 0.0
    team_match_samples: int = 0
    goals_for_total: float = 0.0
    goals_against_total: float = 0.0


class HistoricalFeatureBuilder:
    def __init__(self, rolling_window: int = 8, k_factor: float = 24.0) -> None:
        self.rolling_window = rolling_window
        self.k_factor = k_factor

    def transform(self, matches: pd.DataFrame, update_states: bool = True) -> pd.DataFrame:
        teams: dict[tuple[str, str], TeamState] = defaultdict(lambda: TeamState(self.rolling_window))
        h2h_history: dict[tuple[str, str, str], deque[dict[str, float | str]]] = defaultdict(
            lambda: deque(maxlen=self.rolling_window)
        )
        leagues: dict[str, LeagueState] = defaultdict(LeagueState)
        rows: list[dict[str, float | str | int | pd.Timestamp]] = []

        ordered = matches.sort_values(["Date", "league_code", "HomeTeam", "AwayTeam", "match_id"]).reset_index(drop=True)
        for row in ordered.itertuples(index=False):
            league_key = str(row.league_code)
            home_key = (league_key, str(row.HomeTeam))
            away_key = (league_key, str(row.AwayTeam))
            home_state = teams[home_key]
            away_state = teams[away_key]
            league_state = leagues[league_key]

            expected_home = 1.0 / (1.0 + 10 ** ((away_state.elo - home_state.elo) / 400))
            expected_away = 1.0 - expected_home
            features = {
                "match_id": int(row.match_id),
                "Date": row.Date,
                "league_code": row.league_code,
                "league_name": row.league_name,
                "season": row.season,
                "HomeTeam": row.HomeTeam,
                "AwayTeam": row.AwayTeam,
                "day_of_week": int(pd.Timestamp(row.Date).dayofweek),
                "month": int(pd.Timestamp(row.Date).month),
                "is_weekend": int(pd.Timestamp(row.Date).dayofweek in {5, 6}),
                "home_matches_played": float(home_state.matches_played),
                "away_matches_played": float(away_state.matches_played),
                "home_matches_played_pre": float(home_state.matches_played),
                "away_matches_played_pre": float(away_state.matches_played),
                "elo_home": float(home_state.elo),
                "elo_away": float(away_state.elo),
                "elo_diff": float(home_state.elo - away_state.elo),
                "elo_expected_home": float(expected_home),
                "elo_expected_away": float(expected_away),
                "league_home_win_rate": _safe_ratio(league_state.home_wins, league_state.home_matches, 0.45),
                "league_draw_rate": _safe_ratio(league_state.draws, league_state.home_matches, 0.25),
                "league_home_goal_diff": _safe_ratio(league_state.goal_diff_total, league_state.home_matches, 0.25),
            }

            features.update(self._team_snapshot(home_state, league_state, prefix="home", side="home", current_date=row.Date))
            features.update(self._team_snapshot(away_state, league_state, prefix="away", side="away", current_date=row.Date))
            features.update(self._structural_match_features(features, row.Date))
            features.update(self._head_to_head_snapshot(h2h_history, row))
            rows.append(features)

            if update_states and pd.notna(getattr(row, "FTHG", np.nan)) and pd.notna(getattr(row, "FTAG", np.nan)) and pd.notna(
                getattr(row, "outcome", None)
            ):
                self._update_team_states(home_state, away_state, league_state, h2h_history, row, expected_home)

        feature_rows = pd.DataFrame(rows)
        return feature_rows.sort_values(["Date", "league_code", "HomeTeam", "AwayTeam", "match_id"]).reset_index(drop=True)

    def _team_snapshot(
        self,
        state: TeamState,
        league_state: LeagueState,
        prefix: str,
        side: str,
        current_date: pd.Timestamp,
    ) -> dict[str, float]:
        overall = state.overall_history
        side_history = state.side_history[side]
        long_overall = state.long_overall_history
        long_side_history = state.long_side_history[side]
        opponent_overall = state.opponent_overall_history
        opponent_side_history = state.opponent_side_history[side]
        rest_days = 14.0 if state.last_date is None else float((pd.Timestamp(current_date) - pd.Timestamp(state.last_date)).days)
        rest_days = float(np.clip(rest_days, 0.0, 30.0))
        league_goal_mean = _safe_ratio(league_state.goals_for_total, league_state.team_match_samples, 1.35)
        league_conceded_mean = _safe_ratio(league_state.goals_against_total, league_state.team_match_samples, 1.35)
        overall_coverage_ratio = _coverage_ratio(float(len(long_overall)), float(CONFIDENCE_LONG_OVERALL_TARGET))
        side_coverage_ratio = _coverage_ratio(float(len(long_side_history)), float(CONFIDENCE_LONG_SIDE_TARGET))

        snapshot: dict[str, float] = {
            f"{prefix}_rest_days": rest_days,
            f"{prefix}_matches_last_7d": _count_recent_matches(overall, current_date, 7),
            f"{prefix}_matches_last_14d": _count_recent_matches(overall, current_date, 14),
            f"{prefix}_matches_last_21d": _count_recent_matches(overall, current_date, 21),
            f"{prefix}_overall_sample_size": float(len(overall)),
            f"{prefix}_side_sample_size": float(len(side_history)),
            f"{prefix}_long_overall_sample_size": float(len(long_overall)),
            f"{prefix}_long_side_sample_size": float(len(long_side_history)),
            f"{prefix}_points_last_5": _sum_from_history(long_overall, "points", limit=5),
            f"{prefix}_points_last_10": _sum_from_history(long_overall, "points", limit=10),
            f"{prefix}_sample_coverage_ratio_overall": overall_coverage_ratio,
            f"{prefix}_sample_coverage_ratio_side": side_coverage_ratio,
            f"{prefix}_long_sample_ratio_overall": overall_coverage_ratio,
            f"{prefix}_long_sample_ratio_side": side_coverage_ratio,
            f"{prefix}_sample_shortage_ratio_overall": (
                float(1.0 - overall_coverage_ratio) if pd.notna(overall_coverage_ratio) else np.nan
            ),
            f"{prefix}_sample_shortage_ratio_side": (
                float(1.0 - side_coverage_ratio) if pd.notna(side_coverage_ratio) else np.nan
            ),
            f"{prefix}_low_confidence_overall_continuous": (
                float(1.0 - overall_coverage_ratio) if pd.notna(overall_coverage_ratio) else np.nan
            ),
            f"{prefix}_low_confidence_side_continuous": (
                float(1.0 - side_coverage_ratio) if pd.notna(side_coverage_ratio) else np.nan
            ),
            f"{prefix}_low_confidence_match_score": float(
                max(
                    1.0 - overall_coverage_ratio if pd.notna(overall_coverage_ratio) else 0.0,
                    1.0 - side_coverage_ratio if pd.notna(side_coverage_ratio) else 0.0,
                )
            ),
            f"{prefix}_schedule_irregularity_20": _schedule_irregularity_from_history(long_overall, 20),
        }
        points_last_5_count = float(min(len(long_overall), 5))
        points_last_10_count = float(min(len(long_overall), 10))
        snapshot[f"{prefix}_points_per_match_last_5"] = _safe_ratio(
            snapshot[f"{prefix}_points_last_5"],
            points_last_5_count,
            np.nan,
        )
        snapshot[f"{prefix}_points_per_match_last_10"] = _safe_ratio(
            snapshot[f"{prefix}_points_last_10"],
            points_last_10_count,
            np.nan,
        )
        for metric in ROLLING_METRICS:
            snapshot[f"{prefix}_overall_{metric}"] = _mean_from_history(overall, metric)
            snapshot[f"{prefix}_{side}_{metric}"] = _mean_from_history(side_history, metric)
        snapshot[f"{prefix}_exp_attack_overall"] = _exp_mean_from_history(overall, "goals_for")
        snapshot[f"{prefix}_exp_defense_overall"] = _exp_mean_from_history(overall, "goals_against")
        snapshot[f"{prefix}_exp_attack_{side}"] = _exp_mean_from_history(side_history, "goals_for")
        snapshot[f"{prefix}_exp_defense_{side}"] = _exp_mean_from_history(side_history, "goals_against")
        snapshot[f"{prefix}_exp_attack_overall_shrunk"] = _shrink_metric(
            snapshot[f"{prefix}_exp_attack_overall"],
            snapshot[f"{prefix}_overall_sample_size"],
            league_goal_mean,
            prior_strength=8.0,
        )
        snapshot[f"{prefix}_exp_defense_overall_shrunk"] = _shrink_metric(
            snapshot[f"{prefix}_exp_defense_overall"],
            snapshot[f"{prefix}_overall_sample_size"],
            league_conceded_mean,
            prior_strength=8.0,
        )
        snapshot[f"{prefix}_exp_attack_{side}_shrunk"] = _shrink_metric(
            snapshot[f"{prefix}_exp_attack_{side}"],
            snapshot[f"{prefix}_side_sample_size"],
            snapshot[f"{prefix}_exp_attack_overall_shrunk"],
            prior_strength=5.0,
        )
        snapshot[f"{prefix}_exp_defense_{side}_shrunk"] = _shrink_metric(
            snapshot[f"{prefix}_exp_defense_{side}"],
            snapshot[f"{prefix}_side_sample_size"],
            snapshot[f"{prefix}_exp_defense_overall_shrunk"],
            prior_strength=5.0,
        )
        snapshot[f"{prefix}_league_relative_attack"] = (
            snapshot[f"{prefix}_exp_attack_overall"] - league_goal_mean
            if pd.notna(snapshot[f"{prefix}_exp_attack_overall"])
            else np.nan
        )
        snapshot[f"{prefix}_league_relative_defense"] = (
            league_conceded_mean - snapshot[f"{prefix}_exp_defense_overall"]
            if pd.notna(snapshot[f"{prefix}_exp_defense_overall"])
            else np.nan
        )
        snapshot[f"{prefix}_league_relative_attack_shrunk"] = (
            snapshot[f"{prefix}_exp_attack_overall_shrunk"] - league_goal_mean
            if pd.notna(snapshot[f"{prefix}_exp_attack_overall_shrunk"])
            else np.nan
        )
        snapshot[f"{prefix}_league_relative_defense_shrunk"] = (
            league_conceded_mean - snapshot[f"{prefix}_exp_defense_overall_shrunk"]
            if pd.notna(snapshot[f"{prefix}_exp_defense_overall_shrunk"])
            else np.nan
        )
        overall_ratio = _safe_ratio(snapshot[f"{prefix}_overall_sample_size"], snapshot[f"{prefix}_overall_sample_size"] + 8.0, 0.0)
        side_ratio = _safe_ratio(snapshot[f"{prefix}_side_sample_size"], snapshot[f"{prefix}_side_sample_size"] + 5.0, 0.0)
        snapshot[f"{prefix}_shrinkage_ratio_overall"] = float(overall_ratio)
        snapshot[f"{prefix}_shrinkage_ratio_side"] = float(side_ratio)
        snapshot[f"{prefix}_sample_adequacy_overall_5"] = _sample_adequacy_flag(snapshot[f"{prefix}_long_overall_sample_size"], 5.0)
        snapshot[f"{prefix}_sample_adequacy_overall_10"] = _sample_adequacy_flag(snapshot[f"{prefix}_long_overall_sample_size"], 10.0)
        snapshot[f"{prefix}_sample_adequacy_overall_15"] = _sample_adequacy_flag(snapshot[f"{prefix}_long_overall_sample_size"], 15.0)
        snapshot[f"{prefix}_sample_adequacy_overall_20"] = _sample_adequacy_flag(snapshot[f"{prefix}_long_overall_sample_size"], 20.0)
        snapshot[f"{prefix}_sample_adequacy_side_5"] = _sample_adequacy_flag(snapshot[f"{prefix}_long_side_sample_size"], 5.0)
        snapshot[f"{prefix}_sample_adequacy_side_10"] = _sample_adequacy_flag(snapshot[f"{prefix}_long_side_sample_size"], 10.0)
        snapshot[f"{prefix}_sample_adequacy_side_15"] = _sample_adequacy_flag(snapshot[f"{prefix}_long_side_sample_size"], 15.0)
        snapshot[f"{prefix}_sample_adequacy_side_20"] = _sample_adequacy_flag(snapshot[f"{prefix}_long_side_sample_size"], 20.0)
        snapshot[f"{prefix}_low_confidence_overall"] = float(snapshot[f"{prefix}_sample_adequacy_overall_5"] == 0.0)
        snapshot[f"{prefix}_low_confidence_side"] = float(snapshot[f"{prefix}_sample_adequacy_side_5"] == 0.0)
        for horizon in (5, 10, 20):
            snapshot[f"{prefix}_goals_for_volatility_{horizon}"] = _volatility_from_history(long_overall, "goals_for", horizon)
            snapshot[f"{prefix}_goals_against_volatility_{horizon}"] = _volatility_from_history(long_overall, "goals_against", horizon)
            snapshot[f"{prefix}_opponent_attack_overall_{horizon}"] = _exp_mean_from_history(
                opponent_overall,
                "opponent_attack_overall_shrunk",
                limit=horizon,
            )
            snapshot[f"{prefix}_opponent_defense_overall_{horizon}"] = _exp_mean_from_history(
                opponent_overall,
                "opponent_defense_overall_shrunk",
                limit=horizon,
            )
            snapshot[f"{prefix}_opponent_attack_side_{horizon}"] = _exp_mean_from_history(
                opponent_side_history,
                "opponent_attack_side_shrunk",
                limit=horizon,
            )
            snapshot[f"{prefix}_opponent_defense_side_{horizon}"] = _exp_mean_from_history(
                opponent_side_history,
                "opponent_defense_side_shrunk",
                limit=horizon,
            )
            snapshot[f"{prefix}_opponent_strength_overall_{horizon}"] = _pair_mean_or_nan(
                snapshot[f"{prefix}_opponent_attack_overall_{horizon}"],
                snapshot[f"{prefix}_opponent_defense_overall_{horizon}"],
            )
            snapshot[f"{prefix}_opponent_strength_side_{horizon}"] = _pair_mean_or_nan(
                snapshot[f"{prefix}_opponent_attack_side_{horizon}"],
                snapshot[f"{prefix}_opponent_defense_side_{horizon}"],
            )
            snapshot[f"{prefix}_opponent_strength_volatility_{horizon}"] = _volatility_from_history(
                opponent_overall,
                "opponent_strength_overall",
                horizon,
            )
        snapshot[f"{prefix}_goal_volatility_20"] = snapshot[f"{prefix}_goals_for_volatility_20"]
        return snapshot

    @staticmethod
    def _structural_match_features(features: dict[str, float | str | int | pd.Timestamp], current_date: pd.Timestamp) -> dict[str, float]:
        league_code = str(features.get("league_code", ""))
        season_month = int(pd.Timestamp(current_date).month)
        season_phase_early = float(season_month in {8, 9, 10})
        season_phase_mid = float(season_month in {11, 12, 1, 2})
        season_phase_late = float(season_month in {3, 4, 5, 6})
        scheduled_matches = float(LEAGUE_SEASON_MATCH_COUNTS.get(league_code, 38.0))
        matches_played_values = [
            float(value)
            for value in (
                features.get("home_matches_played_pre", np.nan),
                features.get("away_matches_played_pre", np.nan),
            )
            if pd.notna(value)
        ]
        average_matches_played = float(np.mean(matches_played_values)) if matches_played_values else np.nan
        season_progress_pct = (
            float(np.clip(average_matches_played / scheduled_matches, 0.0, 1.0))
            if pd.notna(average_matches_played) and scheduled_matches > 0.0
            else np.nan
        )
        matches_remaining_estimate = (
            float(max(scheduled_matches - average_matches_played, 0.0))
            if pd.notna(average_matches_played)
            else np.nan
        )
        confidence_components = [
            float(features.get("home_shrinkage_ratio_overall", np.nan)),
            float(features.get("away_shrinkage_ratio_overall", np.nan)),
            float(features.get("home_shrinkage_ratio_side", np.nan)),
            float(features.get("away_shrinkage_ratio_side", np.nan)),
        ]
        adequacy_components = [
            float(features.get("home_long_sample_ratio_overall", np.nan)),
            float(features.get("away_long_sample_ratio_overall", np.nan)),
            float(features.get("home_long_sample_ratio_side", np.nan)),
            float(features.get("away_long_sample_ratio_side", np.nan)),
        ]
        stability_components = [
            float(features.get("home_goal_volatility_20", np.nan)),
            float(features.get("away_goal_volatility_20", np.nan)),
            float(features.get("home_goals_against_volatility_20", np.nan)),
            float(features.get("away_goals_against_volatility_20", np.nan)),
            float(features.get("home_opponent_strength_volatility_20", np.nan)),
            float(features.get("away_opponent_strength_volatility_20", np.nan)),
        ]
        shrinkage_conf = float(np.nanmean(confidence_components)) if any(pd.notna(v) for v in confidence_components) else np.nan
        adequacy_conf = float(np.nanmean(adequacy_components)) if any(pd.notna(v) for v in adequacy_components) else np.nan
        stability_penalty = (
            float(np.clip(np.nanmean(stability_components) / 1.5, 0.0, 1.0))
            if any(pd.notna(v) for v in stability_components)
            else np.nan
        )
        stability_conf = float(1.0 - stability_penalty) if pd.notna(stability_penalty) else np.nan
        season_conf = (
            float(np.clip(season_progress_pct, 0.25, 1.0))
            if pd.notna(season_progress_pct)
            else np.nan
        )
        confidence_score_v2 = (
            float(
                np.clip(
                    0.30 * shrinkage_conf
                    + 0.30 * adequacy_conf
                    + 0.25 * stability_conf
                    + 0.15 * season_conf,
                    0.0,
                    1.0,
                )
            )
            if all(pd.notna(v) for v in (shrinkage_conf, adequacy_conf, stability_conf, season_conf))
            else np.nan
        )
        home_attack = float(features.get("home_exp_attack_overall", np.nan))
        away_attack = float(features.get("away_exp_attack_overall", np.nan))
        home_defense = float(features.get("home_exp_defense_overall", np.nan))
        away_defense = float(features.get("away_exp_defense_overall", np.nan))
        home_side_attack = float(features.get("home_exp_attack_home", np.nan))
        away_side_attack = float(features.get("away_exp_attack_away", np.nan))
        home_side_defense = float(features.get("home_exp_defense_home", np.nan))
        away_side_defense = float(features.get("away_exp_defense_away", np.nan))
        home_attack_shrunk = float(features.get("home_exp_attack_overall_shrunk", np.nan))
        away_attack_shrunk = float(features.get("away_exp_attack_overall_shrunk", np.nan))
        home_defense_shrunk = float(features.get("home_exp_defense_overall_shrunk", np.nan))
        away_defense_shrunk = float(features.get("away_exp_defense_overall_shrunk", np.nan))
        home_side_attack_shrunk = float(features.get("home_exp_attack_home_shrunk", np.nan))
        away_side_attack_shrunk = float(features.get("away_exp_attack_away_shrunk", np.nan))
        home_side_defense_shrunk = float(features.get("home_exp_defense_home_shrunk", np.nan))
        away_side_defense_shrunk = float(features.get("away_exp_defense_away_shrunk", np.nan))
        home_schedule_strength_5 = float(features.get("home_opponent_strength_overall_5", np.nan))
        away_schedule_strength_5 = float(features.get("away_opponent_strength_overall_5", np.nan))
        home_schedule_strength_10 = float(features.get("home_opponent_strength_overall_10", np.nan))
        away_schedule_strength_10 = float(features.get("away_opponent_strength_overall_10", np.nan))
        home_schedule_strength_20 = float(features.get("home_opponent_strength_overall_20", np.nan))
        away_schedule_strength_20 = float(features.get("away_opponent_strength_overall_20", np.nan))
        home_rest_days = float(features.get("home_rest_days", np.nan))
        away_rest_days = float(features.get("away_rest_days", np.nan))
        home_short_turnaround_3d = float(home_rest_days <= 3.0) if pd.notna(home_rest_days) else np.nan
        away_short_turnaround_3d = float(away_rest_days <= 3.0) if pd.notna(away_rest_days) else np.nan
        home_short_turnaround_5d = float(home_rest_days <= 5.0) if pd.notna(home_rest_days) else np.nan
        away_short_turnaround_5d = float(away_rest_days <= 5.0) if pd.notna(away_rest_days) else np.nan
        return {
            "season_progress_pct": season_progress_pct,
            "matches_remaining_estimate": matches_remaining_estimate,
            "points_form_diff_5": (
                float(features.get("home_points_last_5", np.nan)) - float(features.get("away_points_last_5", np.nan))
                if pd.notna(float(features.get("home_points_last_5", np.nan)))
                and pd.notna(float(features.get("away_points_last_5", np.nan)))
                else np.nan
            ),
            "points_form_diff_10": (
                float(features.get("home_points_last_10", np.nan)) - float(features.get("away_points_last_10", np.nan))
                if pd.notna(float(features.get("home_points_last_10", np.nan)))
                and pd.notna(float(features.get("away_points_last_10", np.nan)))
                else np.nan
            ),
            "home_attack_minus_away_defense": home_attack - away_defense if pd.notna(home_attack) and pd.notna(away_defense) else np.nan,
            "away_attack_minus_home_defense": away_attack - home_defense if pd.notna(away_attack) and pd.notna(home_defense) else np.nan,
            "home_home_attack_minus_away_away_defense": (
                home_side_attack - away_side_defense
                if pd.notna(home_side_attack) and pd.notna(away_side_defense)
                else np.nan
            ),
            "away_away_attack_minus_home_home_defense": (
                away_side_attack - home_side_defense
                if pd.notna(away_side_attack) and pd.notna(home_side_defense)
                else np.nan
            ),
            "attack_balance_diff": home_attack - away_attack if pd.notna(home_attack) and pd.notna(away_attack) else np.nan,
            "defense_balance_diff": away_defense - home_defense if pd.notna(home_defense) and pd.notna(away_defense) else np.nan,
            "fixture_congestion_diff_7d": float(features.get("home_matches_last_7d", np.nan)) - float(features.get("away_matches_last_7d", np.nan)),
            "fixture_congestion_diff_14d": float(features.get("home_matches_last_14d", np.nan)) - float(features.get("away_matches_last_14d", np.nan)),
            "fixture_congestion_diff_21d": float(features.get("home_matches_last_21d", np.nan)) - float(features.get("away_matches_last_21d", np.nan)),
            "league_relative_attack_diff": float(features.get("home_league_relative_attack", np.nan)) - float(features.get("away_league_relative_attack", np.nan)),
            "league_relative_defense_diff": float(features.get("home_league_relative_defense", np.nan)) - float(features.get("away_league_relative_defense", np.nan)),
            "home_attack_minus_away_defense_shrunk": (
                home_attack_shrunk - away_defense_shrunk
                if pd.notna(home_attack_shrunk) and pd.notna(away_defense_shrunk)
                else np.nan
            ),
            "away_attack_minus_home_defense_shrunk": (
                away_attack_shrunk - home_defense_shrunk
                if pd.notna(away_attack_shrunk) and pd.notna(home_defense_shrunk)
                else np.nan
            ),
            "home_home_attack_minus_away_away_defense_shrunk": (
                home_side_attack_shrunk - away_side_defense_shrunk
                if pd.notna(home_side_attack_shrunk) and pd.notna(away_side_defense_shrunk)
                else np.nan
            ),
            "away_away_attack_minus_home_home_defense_shrunk": (
                away_side_attack_shrunk - home_side_defense_shrunk
                if pd.notna(away_side_attack_shrunk) and pd.notna(home_side_defense_shrunk)
                else np.nan
            ),
            "attack_balance_diff_shrunk": (
                home_attack_shrunk - away_attack_shrunk
                if pd.notna(home_attack_shrunk) and pd.notna(away_attack_shrunk)
                else np.nan
            ),
            "defense_balance_diff_shrunk": (
                away_defense_shrunk - home_defense_shrunk
                if pd.notna(home_defense_shrunk) and pd.notna(away_defense_shrunk)
                else np.nan
            ),
            "league_relative_attack_diff_shrunk": (
                float(features.get("home_league_relative_attack_shrunk", np.nan))
                - float(features.get("away_league_relative_attack_shrunk", np.nan))
            ),
            "league_relative_defense_diff_shrunk": (
                float(features.get("home_league_relative_defense_shrunk", np.nan))
                - float(features.get("away_league_relative_defense_shrunk", np.nan))
            ),
            "opponent_strength_diff_5": (
                home_schedule_strength_5 - away_schedule_strength_5
                if pd.notna(home_schedule_strength_5) and pd.notna(away_schedule_strength_5)
                else np.nan
            ),
            "opponent_strength_diff_10": (
                home_schedule_strength_10 - away_schedule_strength_10
                if pd.notna(home_schedule_strength_10) and pd.notna(away_schedule_strength_10)
                else np.nan
            ),
            "opponent_strength_diff_20": (
                home_schedule_strength_20 - away_schedule_strength_20
                if pd.notna(home_schedule_strength_20) and pd.notna(away_schedule_strength_20)
                else np.nan
            ),
            "fixture_congestion_weighted_diff_7d_5": (
                float(features.get("home_matches_last_7d", np.nan)) * home_schedule_strength_5
                - float(features.get("away_matches_last_7d", np.nan)) * away_schedule_strength_5
                if pd.notna(home_schedule_strength_5) and pd.notna(away_schedule_strength_5)
                else np.nan
            ),
            "fixture_congestion_weighted_diff_14d_5": (
                float(features.get("home_matches_last_14d", np.nan)) * home_schedule_strength_5
                - float(features.get("away_matches_last_14d", np.nan)) * away_schedule_strength_5
                if pd.notna(home_schedule_strength_5) and pd.notna(away_schedule_strength_5)
                else np.nan
            ),
            "fixture_congestion_weighted_diff_21d_5": (
                float(features.get("home_matches_last_21d", np.nan)) * home_schedule_strength_5
                - float(features.get("away_matches_last_21d", np.nan)) * away_schedule_strength_5
                if pd.notna(home_schedule_strength_5) and pd.notna(away_schedule_strength_5)
                else np.nan
            ),
            "fixture_congestion_weighted_diff_7d_10": (
                float(features.get("home_matches_last_7d", np.nan)) * home_schedule_strength_10
                - float(features.get("away_matches_last_7d", np.nan)) * away_schedule_strength_10
                if pd.notna(home_schedule_strength_10) and pd.notna(away_schedule_strength_10)
                else np.nan
            ),
            "fixture_congestion_weighted_diff_14d_10": (
                float(features.get("home_matches_last_14d", np.nan)) * home_schedule_strength_10
                - float(features.get("away_matches_last_14d", np.nan)) * away_schedule_strength_10
                if pd.notna(home_schedule_strength_10) and pd.notna(away_schedule_strength_10)
                else np.nan
            ),
            "fixture_congestion_weighted_diff_21d_10": (
                float(features.get("home_matches_last_21d", np.nan)) * home_schedule_strength_10
                - float(features.get("away_matches_last_21d", np.nan)) * away_schedule_strength_10
                if pd.notna(home_schedule_strength_10) and pd.notna(away_schedule_strength_10)
                else np.nan
            ),
            "fixture_congestion_weighted_diff_7d_20": (
                float(features.get("home_matches_last_7d", np.nan)) * home_schedule_strength_20
                - float(features.get("away_matches_last_7d", np.nan)) * away_schedule_strength_20
                if pd.notna(home_schedule_strength_20) and pd.notna(away_schedule_strength_20)
                else np.nan
            ),
            "fixture_congestion_weighted_diff_14d_20": (
                float(features.get("home_matches_last_14d", np.nan)) * home_schedule_strength_20
                - float(features.get("away_matches_last_14d", np.nan)) * away_schedule_strength_20
                if pd.notna(home_schedule_strength_20) and pd.notna(away_schedule_strength_20)
                else np.nan
            ),
            "fixture_congestion_weighted_diff_21d_20": (
                float(features.get("home_matches_last_21d", np.nan)) * home_schedule_strength_20
                - float(features.get("away_matches_last_21d", np.nan)) * away_schedule_strength_20
                if pd.notna(home_schedule_strength_20) and pd.notna(away_schedule_strength_20)
                else np.nan
            ),
            "rest_vs_schedule_diff_5": (
                (home_rest_days - away_rest_days) - (home_schedule_strength_5 - away_schedule_strength_5)
                if pd.notna(home_rest_days)
                and pd.notna(away_rest_days)
                and pd.notna(home_schedule_strength_5)
                and pd.notna(away_schedule_strength_5)
                else np.nan
            ),
            "rest_vs_schedule_diff_10": (
                (home_rest_days - away_rest_days) - (home_schedule_strength_10 - away_schedule_strength_10)
                if pd.notna(home_rest_days)
                and pd.notna(away_rest_days)
                and pd.notna(home_schedule_strength_10)
                and pd.notna(away_schedule_strength_10)
                else np.nan
            ),
            "rest_vs_schedule_diff_20": (
                (home_rest_days - away_rest_days) - (home_schedule_strength_20 - away_schedule_strength_20)
                if pd.notna(home_rest_days)
                and pd.notna(away_rest_days)
                and pd.notna(home_schedule_strength_20)
                and pd.notna(away_schedule_strength_20)
                else np.nan
            ),
            "turnaround_diff_3d": (
                home_short_turnaround_3d - away_short_turnaround_3d
                if pd.notna(home_short_turnaround_3d) and pd.notna(away_short_turnaround_3d)
                else np.nan
            ),
            "turnaround_diff_5d": (
                home_short_turnaround_5d - away_short_turnaround_5d
                if pd.notna(home_short_turnaround_5d) and pd.notna(away_short_turnaround_5d)
                else np.nan
            ),
            "overall_confidence_balance": (
                float(features.get("home_shrinkage_ratio_overall", np.nan))
                - float(features.get("away_shrinkage_ratio_overall", np.nan))
            ),
            "side_confidence_balance": (
                float(features.get("home_shrinkage_ratio_side", np.nan))
                - float(features.get("away_shrinkage_ratio_side", np.nan))
            ),
            "low_confidence_match_flag": float(
                max(
                    float(features.get("home_low_confidence_overall", 0.0)),
                    float(features.get("away_low_confidence_overall", 0.0)),
                    float(features.get("home_low_confidence_side", 0.0)),
                    float(features.get("away_low_confidence_side", 0.0)),
                )
            ),
            "rest_advantage_x_fixture_congestion_diff_7d": (
                float(features.get("rest_advantage", np.nan)) * float(features.get("fixture_congestion_diff_7d", np.nan))
                if pd.notna(float(features.get("rest_advantage", np.nan)))
                and pd.notna(float(features.get("fixture_congestion_diff_7d", np.nan)))
                else np.nan
            ),
            "season_phase_early_x_league_relative_attack_diff_shrunk": season_phase_early
            * float(features.get("league_relative_attack_diff_shrunk", np.nan)),
            "season_phase_mid_x_league_relative_attack_diff_shrunk": season_phase_mid
            * float(features.get("league_relative_attack_diff_shrunk", np.nan)),
            "season_phase_late_x_league_relative_attack_diff_shrunk": season_phase_late
            * float(features.get("league_relative_attack_diff_shrunk", np.nan)),
            "season_phase_early_x_league_relative_defense_diff_shrunk": season_phase_early
            * float(features.get("league_relative_defense_diff_shrunk", np.nan)),
            "season_phase_mid_x_league_relative_defense_diff_shrunk": season_phase_mid
            * float(features.get("league_relative_defense_diff_shrunk", np.nan)),
            "season_phase_late_x_league_relative_defense_diff_shrunk": season_phase_late
            * float(features.get("league_relative_defense_diff_shrunk", np.nan)),
            "matchup_balance_interaction_shrunk": (
                float(features.get("home_attack_minus_away_defense_shrunk", np.nan))
                * float(features.get("away_attack_minus_home_defense_shrunk", np.nan))
                if pd.notna(float(features.get("home_attack_minus_away_defense_shrunk", np.nan)))
                and pd.notna(float(features.get("away_attack_minus_home_defense_shrunk", np.nan)))
                else np.nan
            ),
            "season_phase_early": season_phase_early,
            "season_phase_mid": season_phase_mid,
            "season_phase_late": season_phase_late,
            "confidence_score_v2": confidence_score_v2,
            "confidence_floor_continuous": (
                float(
                    np.nanmin(
                        [
                            float(features.get("home_long_sample_ratio_overall", np.nan)),
                            float(features.get("away_long_sample_ratio_overall", np.nan)),
                            float(features.get("home_long_sample_ratio_side", np.nan)),
                            float(features.get("away_long_sample_ratio_side", np.nan)),
                        ]
                    )
                )
                if any(
                    pd.notna(value)
                    for value in (
                        features.get("home_long_sample_ratio_overall", np.nan),
                        features.get("away_long_sample_ratio_overall", np.nan),
                        features.get("home_long_sample_ratio_side", np.nan),
                        features.get("away_long_sample_ratio_side", np.nan),
                    )
                )
                else np.nan
            ),
            "rest_advantage_x_season_progress_pct": (
                float(features.get("rest_advantage", np.nan)) * season_progress_pct
                if pd.notna(float(features.get("rest_advantage", np.nan))) and pd.notna(season_progress_pct)
                else np.nan
            ),
            "points_form_diff_5_x_season_progress_pct": (
                float(features.get("points_form_diff_5", np.nan)) * season_progress_pct
                if pd.notna(float(features.get("points_form_diff_5", np.nan))) and pd.notna(season_progress_pct)
                else np.nan
            ),
            "points_form_diff_10_x_season_progress_pct": (
                float(features.get("points_form_diff_10", np.nan)) * season_progress_pct
                if pd.notna(float(features.get("points_form_diff_10", np.nan))) and pd.notna(season_progress_pct)
                else np.nan
            ),
            "home_attack_minus_away_defense_shrunk_x_confidence_score_v2": (
                float(features.get("home_attack_minus_away_defense_shrunk", np.nan)) * confidence_score_v2
                if pd.notna(float(features.get("home_attack_minus_away_defense_shrunk", np.nan))) and pd.notna(confidence_score_v2)
                else np.nan
            ),
            "away_attack_minus_home_defense_shrunk_x_confidence_score_v2": (
                float(features.get("away_attack_minus_home_defense_shrunk", np.nan)) * confidence_score_v2
                if pd.notna(float(features.get("away_attack_minus_home_defense_shrunk", np.nan))) and pd.notna(confidence_score_v2)
                else np.nan
            ),
        }

    def _head_to_head_snapshot(
        self,
        h2h_history: dict[tuple[str, str, str], deque[dict[str, float | str]]],
        row: tuple,
    ) -> dict[str, float]:
        key = (str(row.league_code), *sorted([str(row.HomeTeam), str(row.AwayTeam)]))
        records = list(h2h_history[key])
        if not records:
            return {
                "h2h_home_points_avg": np.nan,
                "h2h_home_goal_diff_avg": np.nan,
                "h2h_draw_rate": np.nan,
            }

        points_samples: list[float] = []
        goal_diff_samples: list[float] = []
        draw_samples: list[float] = []
        for record in records:
            if record["team_a"] == row.HomeTeam:
                points_samples.append(float(record["points_a"]))
                goal_diff_samples.append(float(record["goals_a"]) - float(record["goals_b"]))
            else:
                points_samples.append(float(record["points_b"]))
                goal_diff_samples.append(float(record["goals_b"]) - float(record["goals_a"]))
            draw_samples.append(float(record["is_draw"]))

        return {
            "h2h_home_points_avg": float(np.mean(points_samples)),
            "h2h_home_goal_diff_avg": float(np.mean(goal_diff_samples)),
            "h2h_draw_rate": float(np.mean(draw_samples)),
        }

    def _update_team_states(
        self,
        home_state: TeamState,
        away_state: TeamState,
        league_state: LeagueState,
        h2h_history: dict[tuple[str, str, str], deque[dict[str, float | str]]],
        row: tuple,
        expected_home: float,
    ) -> None:
        home_points = 3.0 if row.outcome == "home" else 1.0 if row.outcome == "draw" else 0.0
        away_points = 3.0 if row.outcome == "away" else 1.0 if row.outcome == "draw" else 0.0
        home_stats = self._match_stats(row, side="home", points=home_points)
        away_stats = self._match_stats(row, side="away", points=away_points)
        home_opponent_context = self._opponent_context(away_state, league_state, opponent_side="away", match_date=row.Date)
        away_opponent_context = self._opponent_context(home_state, league_state, opponent_side="home", match_date=row.Date)

        home_state.overall_history.append(home_stats)
        home_state.side_history["home"].append(home_stats)
        home_state.long_overall_history.append(home_stats)
        home_state.long_side_history["home"].append(home_stats)
        home_state.opponent_overall_history.append(home_opponent_context)
        home_state.opponent_side_history["home"].append(home_opponent_context)
        away_state.overall_history.append(away_stats)
        away_state.side_history["away"].append(away_stats)
        away_state.long_overall_history.append(away_stats)
        away_state.long_side_history["away"].append(away_stats)
        away_state.opponent_overall_history.append(away_opponent_context)
        away_state.opponent_side_history["away"].append(away_opponent_context)
        home_state.last_date = pd.Timestamp(row.Date)
        away_state.last_date = pd.Timestamp(row.Date)
        home_state.matches_played += 1
        away_state.matches_played += 1

        goal_diff = float(row.FTHG) - float(row.FTAG)
        actual_home = 1.0 if goal_diff > 0 else 0.0 if goal_diff < 0 else 0.5
        margin_multiplier = float(np.log1p(abs(goal_diff) + 1.0))
        change = self.k_factor * margin_multiplier * (actual_home - expected_home)
        home_state.elo += change
        away_state.elo -= change

        league_state.home_matches += 1
        league_state.home_wins += int(row.outcome == "home")
        league_state.draws += int(row.outcome == "draw")
        league_state.goal_diff_total += goal_diff
        league_state.team_match_samples += 2
        league_state.goals_for_total += float(row.FTHG) + float(row.FTAG)
        league_state.goals_against_total += float(row.FTHG) + float(row.FTAG)

        key = (str(row.league_code), *sorted([str(row.HomeTeam), str(row.AwayTeam)]))
        h2h_history[key].append(
            {
                "team_a": row.HomeTeam,
                "team_b": row.AwayTeam,
                "goals_a": float(row.FTHG),
                "goals_b": float(row.FTAG),
                "points_a": home_points,
                "points_b": away_points,
                "is_draw": float(row.outcome == "draw"),
            }
        )

    @staticmethod
    def _match_stats(row: tuple, side: str, points: float) -> dict[str, float]:
        goals_for = float(row.FTHG if side == "home" else row.FTAG)
        goals_against = float(row.FTAG if side == "home" else row.FTHG)
        shots_for = float(getattr(row, "HS", np.nan) if side == "home" else getattr(row, "AS", np.nan))
        shots_against = float(getattr(row, "AS", np.nan) if side == "home" else getattr(row, "HS", np.nan))
        shots_on_target_for = float(getattr(row, "HST", np.nan) if side == "home" else getattr(row, "AST", np.nan))
        shots_on_target_against = float(getattr(row, "AST", np.nan) if side == "home" else getattr(row, "HST", np.nan))
        corners_for = float(getattr(row, "HC", np.nan) if side == "home" else getattr(row, "AC", np.nan))
        corners_against = float(getattr(row, "AC", np.nan) if side == "home" else getattr(row, "HC", np.nan))
        yellows = float(getattr(row, "HY", np.nan) if side == "home" else getattr(row, "AY", np.nan))
        reds = float(getattr(row, "HR", np.nan) if side == "home" else getattr(row, "AR", np.nan))
        xg_proxy = (
            0.08 * np.nan_to_num(shots_for, nan=0.0)
            + 0.18 * np.nan_to_num(shots_on_target_for, nan=0.0)
            + 0.03 * np.nan_to_num(corners_for, nan=0.0)
            - 0.05 * np.nan_to_num(reds, nan=0.0)
        )
        goal_diff = goals_for - goals_against
        return {
            "date": pd.Timestamp(row.Date),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "shots_for": shots_for,
            "shots_against": shots_against,
            "shots_on_target_for": shots_on_target_for,
            "shots_on_target_against": shots_on_target_against,
            "corners_for": corners_for,
            "corners_against": corners_against,
            "yellows": yellows,
            "reds": reds,
            "points": float(points),
            "goal_diff": goal_diff,
            "xg_proxy": float(xg_proxy),
        }

    @staticmethod
    def _opponent_context(
        opponent_state: TeamState,
        league_state: LeagueState,
        opponent_side: str,
        match_date: pd.Timestamp,
    ) -> dict[str, float]:
        league_goal_mean = _safe_ratio(league_state.goals_for_total, league_state.team_match_samples, 1.35)
        league_conceded_mean = _safe_ratio(league_state.goals_against_total, league_state.team_match_samples, 1.35)
        overall_sample = float(len(opponent_state.long_overall_history))
        side_history = opponent_state.long_side_history[opponent_side]
        side_sample = float(len(side_history))
        attack_overall = _exp_mean_from_history(opponent_state.long_overall_history, "goals_for")
        defense_overall = _exp_mean_from_history(opponent_state.long_overall_history, "goals_against")
        attack_side = _exp_mean_from_history(side_history, "goals_for")
        defense_side = _exp_mean_from_history(side_history, "goals_against")
        attack_overall_shrunk = _shrink_metric(attack_overall, overall_sample, league_goal_mean, prior_strength=8.0)
        defense_overall_shrunk = _shrink_metric(defense_overall, overall_sample, league_conceded_mean, prior_strength=8.0)
        attack_side_shrunk = _shrink_metric(attack_side, side_sample, attack_overall_shrunk, prior_strength=5.0)
        defense_side_shrunk = _shrink_metric(defense_side, side_sample, defense_overall_shrunk, prior_strength=5.0)
        return {
            "date": pd.Timestamp(match_date),
            "opponent_attack_overall_shrunk": attack_overall_shrunk,
            "opponent_defense_overall_shrunk": defense_overall_shrunk,
            "opponent_attack_side_shrunk": attack_side_shrunk,
            "opponent_defense_side_shrunk": defense_side_shrunk,
            "opponent_strength_overall": _pair_mean_or_nan(attack_overall_shrunk, defense_overall_shrunk),
        }


def build_feature_rows(matches: pd.DataFrame, rolling_window: int = 8) -> pd.DataFrame:
    builder = HistoricalFeatureBuilder(rolling_window=rolling_window)
    features = builder.transform(matches, update_states=True)
    enriched = matches[list(METADATA_COLUMNS) + ["FTHG", "FTAG", "outcome"]].merge(features, on=list(METADATA_COLUMNS), how="left")
    enriched["target"] = enriched["outcome"].map(OUTCOME_TO_TARGET)
    enriched["home_goals"] = enriched["FTHG"].astype(float)
    enriched["away_goals"] = enriched["FTAG"].astype(float)
    enriched["rest_advantage"] = enriched["home_rest_days"] - enriched["away_rest_days"]
    return enriched


def build_fixture_feature_rows(history_matches: pd.DataFrame, fixtures: pd.DataFrame, rolling_window: int = 8) -> pd.DataFrame:
    history = history_matches.copy()
    history["is_fixture"] = 0
    future = fixtures.copy()
    future["is_fixture"] = 1
    if "match_id" not in future.columns:
        future["match_id"] = np.arange(len(history), len(history) + len(future), dtype=int)

    combined = (
        pd.concat([history, future], ignore_index=True, sort=False)
        .sort_values(["Date", "league_code", "HomeTeam", "AwayTeam", "match_id"])
        .reset_index(drop=True)
    )
    builder = HistoricalFeatureBuilder(rolling_window=rolling_window)
    features = builder.transform(combined, update_states=True)
    fixture_features = combined.loc[combined["is_fixture"].eq(1), list(METADATA_COLUMNS)].merge(
        features, on=list(METADATA_COLUMNS), how="left"
    )
    fixture_features["rest_advantage"] = fixture_features["home_rest_days"] - fixture_features["away_rest_days"]
    return fixture_features.reset_index(drop=True)


def model_feature_columns(feature_rows: pd.DataFrame) -> list[str]:
    excluded = {
        "match_id",
        "Date",
        "kickoff_time",
        "league_name",
        "season",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "outcome",
        "target",
        "home_goals",
        "away_goals",
    }
    return [column for column in feature_rows.columns if column not in excluded]


@dataclass(frozen=True)
class FeatureColumnSanitization:
    feature_columns: list[str]
    dropped_feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    dropped_feature_reasons: dict[str, str]


def sanitize_model_feature_columns(
    feature_rows: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> FeatureColumnSanitization:
    candidate_columns = list(feature_columns) if feature_columns is not None else model_feature_columns(feature_rows)
    active_columns: list[str] = []
    dropped_columns: list[str] = []
    dropped_reasons: dict[str, str] = {}

    for column in candidate_columns:
        if column not in feature_rows.columns:
            dropped_columns.append(column)
            dropped_reasons[column] = "missing_from_frame"
            continue

        observed = feature_rows[column].replace([np.inf, -np.inf], np.nan).notna().sum()
        if observed == 0:
            dropped_columns.append(column)
            dropped_reasons[column] = "all_missing"
            continue

        active_columns.append(column)

    categorical_columns = [column for column in active_columns if column == "league_code"]
    numeric_columns = [column for column in active_columns if column not in categorical_columns]
    return FeatureColumnSanitization(
        feature_columns=active_columns,
        dropped_feature_columns=dropped_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        dropped_feature_reasons=dropped_reasons,
    )


VARIANT_FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "v1": ("baseline",),
    "v2": ("baseline", "recent_form", "congestion"),
    "v3": ("baseline", "recent_form", "congestion", "matchup_raw"),
    "v2r": ("baseline", "shrinkage", "congestion"),
    "v3r": ("baseline", "shrinkage", "congestion", "matchup_shrunk", "season_phase"),
    "v4": ("baseline", "shrinkage", "congestion", "rival_context", "uncertainty"),
    "v5": (
        "baseline",
        "shrinkage",
        "congestion",
        "rival_context",
        "uncertainty",
        "matchup_shrunk",
        "season_phase",
        "season_state",
        "interactions",
    ),
    "v6": (
        "baseline",
        "shrinkage",
        "congestion",
        "rival_context",
        "uncertainty",
        "confidence_long",
        "matchup_shrunk",
        "season_phase",
        "season_state",
        "interactions",
    ),
    "v4b": ("baseline", "shrinkage", "congestion", "rival_context", "uncertainty", "confidence_backoff"),
    "v5b": (
        "baseline",
        "shrinkage",
        "congestion",
        "rival_context",
        "uncertainty",
        "matchup_shrunk",
        "season_phase",
        "season_state",
        "interactions",
        "confidence_backoff",
    ),
    "v6b": (
        "baseline",
        "shrinkage",
        "congestion",
        "rival_context",
        "uncertainty",
        "confidence_long",
        "matchup_shrunk",
        "season_phase",
        "season_state",
        "interactions",
        "confidence_backoff",
    ),
    "v7": (
        "baseline",
        "shrinkage",
        "congestion",
        "rival_context",
        "uncertainty",
        "confidence_long",
        "matchup_shrunk",
        "season_phase",
        "season_state",
        "interactions",
        "regime_interactions",
    ),
    "v7b": (
        "baseline",
        "shrinkage",
        "congestion",
        "rival_context",
        "uncertainty",
        "confidence_long",
        "matchup_shrunk",
        "season_phase",
        "season_state",
        "interactions",
        "regime_interactions",
        "confidence_backoff",
    ),
}


def variant_feature_families(variant: str) -> list[str]:
    return list(VARIANT_FEATURE_FAMILIES.get(str(variant).lower(), VARIANT_FEATURE_FAMILIES["v1"]))


def feature_family_columns(feature_rows: pd.DataFrame) -> dict[str, list[str]]:
    columns = model_feature_columns(feature_rows)
    family_sets: dict[str, set[str]] = {
        "recent_form": {
            "home_exp_attack_overall",
            "away_exp_attack_overall",
            "home_exp_defense_overall",
            "away_exp_defense_overall",
            "home_exp_attack_home",
            "away_exp_attack_away",
            "home_exp_defense_home",
            "away_exp_defense_away",
            "home_league_relative_attack",
            "home_league_relative_defense",
            "away_league_relative_attack",
            "away_league_relative_defense",
        },
        "congestion": {
            "home_matches_last_7d",
            "home_matches_last_14d",
            "home_matches_last_21d",
            "away_matches_last_7d",
            "away_matches_last_14d",
            "away_matches_last_21d",
            "rest_advantage",
            "fixture_congestion_diff_7d",
            "fixture_congestion_diff_14d",
            "fixture_congestion_diff_21d",
        },
        "shrinkage": {
            "home_exp_attack_overall_shrunk",
            "away_exp_attack_overall_shrunk",
            "home_exp_defense_overall_shrunk",
            "away_exp_defense_overall_shrunk",
            "home_exp_attack_home_shrunk",
            "away_exp_attack_away_shrunk",
            "home_exp_defense_home_shrunk",
            "away_exp_defense_away_shrunk",
            "home_league_relative_attack_shrunk",
            "home_league_relative_defense_shrunk",
            "away_league_relative_attack_shrunk",
            "away_league_relative_defense_shrunk",
        },
        "matchup_raw": {
            "home_attack_minus_away_defense",
            "away_attack_minus_home_defense",
            "home_home_attack_minus_away_away_defense",
            "away_away_attack_minus_home_home_defense",
            "attack_balance_diff",
            "defense_balance_diff",
            "league_relative_attack_diff",
            "league_relative_defense_diff",
        },
        "matchup_shrunk": {
            "home_attack_minus_away_defense_shrunk",
            "away_attack_minus_home_defense_shrunk",
            "home_home_attack_minus_away_away_defense_shrunk",
            "away_away_attack_minus_home_home_defense_shrunk",
            "attack_balance_diff_shrunk",
            "defense_balance_diff_shrunk",
            "league_relative_attack_diff_shrunk",
            "league_relative_defense_diff_shrunk",
        },
        "rival_context": {
            "home_opponent_attack_overall_5",
            "away_opponent_attack_overall_5",
            "home_opponent_defense_overall_5",
            "away_opponent_defense_overall_5",
            "home_opponent_attack_side_5",
            "away_opponent_attack_side_5",
            "home_opponent_defense_side_5",
            "away_opponent_defense_side_5",
            "home_opponent_strength_overall_5",
            "away_opponent_strength_overall_5",
            "home_opponent_strength_side_5",
            "away_opponent_strength_side_5",
            "home_opponent_attack_overall_10",
            "away_opponent_attack_overall_10",
            "home_opponent_defense_overall_10",
            "away_opponent_defense_overall_10",
            "home_opponent_attack_side_10",
            "away_opponent_attack_side_10",
            "home_opponent_defense_side_10",
            "away_opponent_defense_side_10",
            "home_opponent_strength_overall_10",
            "away_opponent_strength_overall_10",
            "home_opponent_strength_side_10",
            "away_opponent_strength_side_10",
            "home_opponent_attack_overall_20",
            "away_opponent_attack_overall_20",
            "home_opponent_defense_overall_20",
            "away_opponent_defense_overall_20",
            "home_opponent_attack_side_20",
            "away_opponent_attack_side_20",
            "home_opponent_defense_side_20",
            "away_opponent_defense_side_20",
            "home_opponent_strength_overall_20",
            "away_opponent_strength_overall_20",
            "home_opponent_strength_side_20",
            "away_opponent_strength_side_20",
            "opponent_strength_diff_5",
            "opponent_strength_diff_10",
            "opponent_strength_diff_20",
            "fixture_congestion_weighted_diff_7d_5",
            "fixture_congestion_weighted_diff_14d_5",
            "fixture_congestion_weighted_diff_21d_5",
            "fixture_congestion_weighted_diff_7d_10",
            "fixture_congestion_weighted_diff_14d_10",
            "fixture_congestion_weighted_diff_21d_10",
            "fixture_congestion_weighted_diff_7d_20",
            "fixture_congestion_weighted_diff_14d_20",
            "fixture_congestion_weighted_diff_21d_20",
            "rest_vs_schedule_diff_5",
            "rest_vs_schedule_diff_10",
            "rest_vs_schedule_diff_20",
            "turnaround_diff_3d",
            "turnaround_diff_5d",
        },
        "uncertainty": {
            "home_overall_sample_size",
            "away_overall_sample_size",
            "home_side_sample_size",
            "away_side_sample_size",
            "home_long_overall_sample_size",
            "away_long_overall_sample_size",
            "home_long_side_sample_size",
            "away_long_side_sample_size",
            "home_sample_coverage_ratio_overall",
            "away_sample_coverage_ratio_overall",
            "home_sample_coverage_ratio_side",
            "away_sample_coverage_ratio_side",
            "home_sample_shortage_ratio_overall",
            "away_sample_shortage_ratio_overall",
            "home_sample_shortage_ratio_side",
            "away_sample_shortage_ratio_side",
            "home_shrinkage_ratio_overall",
            "away_shrinkage_ratio_overall",
            "home_shrinkage_ratio_side",
            "away_shrinkage_ratio_side",
            "home_sample_adequacy_overall_5",
            "away_sample_adequacy_overall_5",
            "home_sample_adequacy_overall_10",
            "away_sample_adequacy_overall_10",
            "home_sample_adequacy_overall_15",
            "away_sample_adequacy_overall_15",
            "home_sample_adequacy_overall_20",
            "away_sample_adequacy_overall_20",
            "home_sample_adequacy_side_5",
            "away_sample_adequacy_side_5",
            "home_sample_adequacy_side_10",
            "away_sample_adequacy_side_10",
            "home_sample_adequacy_side_15",
            "away_sample_adequacy_side_15",
            "home_sample_adequacy_side_20",
            "away_sample_adequacy_side_20",
            "home_low_confidence_overall",
            "away_low_confidence_overall",
            "home_low_confidence_side",
            "away_low_confidence_side",
            "home_low_confidence_overall_continuous",
            "away_low_confidence_overall_continuous",
            "home_low_confidence_side_continuous",
            "away_low_confidence_side_continuous",
            "home_low_confidence_match_score",
            "home_goals_for_volatility_5",
            "away_goals_for_volatility_5",
            "home_goals_against_volatility_5",
            "away_goals_against_volatility_5",
            "home_goals_for_volatility_10",
            "away_goals_for_volatility_10",
            "home_goals_against_volatility_10",
            "away_goals_against_volatility_10",
            "home_goals_for_volatility_20",
            "away_goals_for_volatility_20",
            "home_goals_against_volatility_20",
            "away_goals_against_volatility_20",
            "home_opponent_strength_volatility_5",
            "away_opponent_strength_volatility_5",
            "home_opponent_strength_volatility_10",
            "away_opponent_strength_volatility_10",
            "home_opponent_strength_volatility_20",
            "away_opponent_strength_volatility_20",
            "overall_confidence_balance",
            "side_confidence_balance",
            "low_confidence_match_flag",
        },
        "confidence_long": {
            "home_long_sample_ratio_overall",
            "away_long_sample_ratio_overall",
            "home_long_sample_ratio_side",
            "away_long_sample_ratio_side",
            "home_goal_volatility_20",
            "away_goal_volatility_20",
            "home_schedule_irregularity_20",
            "away_schedule_irregularity_20",
            "confidence_score_v2",
            "confidence_floor_continuous",
        },
        "interactions": {
            "rest_advantage_x_fixture_congestion_diff_7d",
            "season_phase_early_x_league_relative_attack_diff_shrunk",
            "season_phase_mid_x_league_relative_attack_diff_shrunk",
            "season_phase_late_x_league_relative_attack_diff_shrunk",
            "season_phase_early_x_league_relative_defense_diff_shrunk",
            "season_phase_mid_x_league_relative_defense_diff_shrunk",
            "season_phase_late_x_league_relative_defense_diff_shrunk",
            "matchup_balance_interaction_shrunk",
        },
        "regime_interactions": {
            "rest_advantage_x_season_progress_pct",
            "points_form_diff_5_x_season_progress_pct",
            "points_form_diff_10_x_season_progress_pct",
            "home_attack_minus_away_defense_shrunk_x_confidence_score_v2",
            "away_attack_minus_home_defense_shrunk_x_confidence_score_v2",
        },
        "season_phase": {
            "season_phase_early",
            "season_phase_mid",
            "season_phase_late",
        },
        "season_state": {
            "home_matches_played_pre",
            "away_matches_played_pre",
            "season_progress_pct",
            "matches_remaining_estimate",
            "home_points_last_5",
            "away_points_last_5",
            "home_points_last_10",
            "away_points_last_10",
            "home_points_per_match_last_5",
            "away_points_per_match_last_5",
            "home_points_per_match_last_10",
            "away_points_per_match_last_10",
            "points_form_diff_5",
            "points_form_diff_10",
        },
    }
    family_sets["baseline"] = set(columns) - set().union(*(family_sets.values()))
    ordered: dict[str, list[str]] = {}
    for family, family_columns in family_sets.items():
        ordered[family] = [column for column in columns if column in family_columns]
    return ordered


def variant_feature_manifest(feature_rows: pd.DataFrame, variants: Iterable[str] | None = None) -> list[dict[str, object]]:
    families = feature_family_columns(feature_rows)
    selected_variants = list(variants) if variants is not None else list(VARIANT_FEATURE_FAMILIES)
    manifest: list[dict[str, object]] = []
    for variant in selected_variants:
        feature_families = variant_feature_families(variant)
        feature_columns = [
            column
            for family in feature_families
            if family in families and family != "confidence_backoff"
            for column in families[family]
        ]
        manifest.append(
            {
                "variant_name": str(variant).lower(),
                "feature_families": feature_families,
                "feature_columns": feature_columns,
                "feature_count": int(len(feature_columns)),
            }
        )
    return manifest


def model_feature_columns_for_variant(feature_rows: pd.DataFrame, variant: str = "v3") -> list[str]:
    families = feature_family_columns(feature_rows)
    feature_families = variant_feature_families(variant)
    columns = []
    for family in feature_families:
        if family == "confidence_backoff":
            continue
        columns.extend(families.get(family, []))
    return columns
