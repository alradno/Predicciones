from __future__ import annotations

import csv
from difflib import SequenceMatcher
import json
import math
import re
import sqlite3
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from ..config import Settings
from ..contracts import MultiMarketCaptureResult, MultiMarketDiscoveryResult, RunContext
from ..core.lane_governance import (
    UNKNOWN_LANE_GOVERNANCE_REASON,
    LaneGovernance,
    LaneGovernanceError,
    get_lane_governance,
    require_can_build_predictions,
    require_can_capture,
)
from ..core.edge_hypothesis import EdgeHypothesisInputs, classify_edge_hypothesis
from ..core.promotion_state import (
    ForwardSampleInputs,
    ForwardSampleThresholds,
    PromotionStatus,
    SampleStatus,
    classify_forward_sample,
)
from ..data_sources import PolymarketClobClient, PolymarketGammaClient
from ..football.dataset import build_fixture_feature_rows
from ..models import outcome_probabilities_from_lambdas, score_matrix_from_lambdas
from ..reporting import create_run_context


MULTI_MARKET_DATABASE_FILENAME = "polymarket_multi_market.sqlite"
TARGET_FORWARD_DECISIONS_PER_DAY = 100
CONSERVATIVE_PICK_RATE_LOW = 0.15
CONSERVATIVE_PICK_RATE_HIGH = 0.25
MAX_SPORT_SERIES_DISCOVERY_CALLS = 80
LANE_MODEL_PREDICTIONS_FILENAME = "model_predictions.csv"
LANE_MODEL_PREDICTION_REPORT_FILENAME = "model_prediction_report.json"
LANE_BLOCKER_AUDIT_CSV_FILENAME = "lane_blocker_audit.csv"
LANE_BLOCKER_AUDIT_JSON_FILENAME = "lane_blocker_audit.json"
RAW_CAPTURE_HEALTH_REPORT_FILENAME = "raw_capture_health_report.json"
FOOTBALL_GOALS_POLICY_RESEARCH_REPORT_FILENAME = "football_goals_core_policy_research_report.json"
FOOTBALL_GOALS_POLICY_RESEARCH_ROWS_FILENAME = "football_goals_core_policy_research_rows.csv"
LANE_POLICY_BENCHMARK_REPORT_FILENAME = "lane_policy_benchmark_report.json"
RAW_CAPTURE_PLAN_CSV_FILENAME = "raw_capture_plan.csv"
RAW_CAPTURE_PLAN_JSON_FILENAME = "raw_capture_plan.json"
MIN_LANE_TEAM_HISTORY_ROWS = 5
DEFAULT_RAW_CAPTURE_FRESHNESS_SECONDS = 3600
CAPTURE_PRIORITY_VALUES = {"missing_or_stale", "missing_only", "stale_only", "all"}

FOOTBALL_GOALS_CORE_BOOTSTRAP_POLICY = {
    "policy_name": "football_goals_core_bootstrap_v1",
    "policy_family": "fixed_edge_ev_threshold",
    "policy_role": "forward_sample_collection",
    "edge_threshold": 0.02,
    "ev_threshold": 0.0,
    "min_odds": 1.2,
    "max_odds": 6.0,
    "allowed_outcomes": ["over", "under", "btts_yes", "btts_no"],
    "allowed_market_subtypes": ["total_goals", "btts"],
    "max_picks_per_event": 1,
}

LANE_MODEL_PREDICTION_COLUMNS = [
    "lane_id",
    "benchmark_id",
    "event_slug",
    "market_id",
    "market_subtype",
    "selection",
    "model_prob",
    "probability_source",
    "model_variant",
]

STATUS_DISCOVERY_ONLY = "discovery_only"
STATUS_CAPTURE_READY = "capture_ready"
STATUS_SHADOW_COLLECT_ONLY = "shadow_collect_only"
STATUS_CAPTURE_ONLY = "capture_only"
STATUS_MODEL_READY = "model_ready"
STATUS_SAMPLE_READY = "sample_ready"
STATUS_REJECTED = "rejected"
STATUS_PROMOTABLE_FOR_MANUAL_REVIEW = "promotable_for_manual_review"
STATUS_DEFERRED = "deferred"
STATUS_REFERENCE_ONLY = "reference_only"

LEGACY_FOOTBALL_1X2_LEAGUES = ("E0", "SP1", "D1")
GLOBAL_FOOTBALL_1X2_LEAGUES = ("E0", "SP1", "D1", "I1", "F1", "N1", "P1", "MEX", "USA")

ACTIVE_FAMILIES = (
    "football_1x2_global",
    "football_goals_core",
    "tennis_match_winner",
    "basketball_moneyline",
    "baseball_moneyline",
    "hockey_moneyline",
    "cricket_match_winner",
)

DEFERRED_FAMILIES = (
    "football_cards",
    "football_corners",
    "player_props",
    "spreads_handicaps",
    "non_football_totals",
)

ACTIVE_LANES = ACTIVE_FAMILIES
REFERENCE_LANES = ("football_1x2_canonical",)
DEFERRED_LANES = DEFERRED_FAMILIES

LANE_CANDIDATE_COLUMNS = [
    "candidate_id",
    "lane_id",
    "benchmark_id",
    "event_slug",
    "market_id",
    "market_subtype",
    "game_start_time",
    "selection",
    "contract_outcome",
    "asset_id",
    "book_timestamp",
    "top_ask",
    "quoted_odds",
    "model_prob",
    "probability_source",
    "edge",
    "ev",
    "candidate_status",
    "candidate_blocker",
    "policy_edge_threshold",
    "policy_ev_threshold",
    "policy_min_odds",
    "policy_max_odds",
    "legacy_slice",
]

FOOTBALL_SPORT_CODES = {
    "epl",
    "lal",
    "bun",
    "fl1",
    "sea",
    "ucl",
    "uel",
    "mls",
    "mex",
    "arg",
    "ere",
    "fif",
    "fifwc",
    "acn",
    "afc",
    "ofc",
    "itc",
    "cde",
    "rus",
    "ukr1",
}
BASKETBALL_SPORT_CODES = {"nba", "wnba", "ncaab", "bkarg"}
BASEBALL_SPORT_CODES = {"mlb", "npb", "kbo", "wbc"}
HOCKEY_SPORT_CODES = {"nhl", "snhl"}
TENNIS_SPORT_CODES = {"atp", "wta"}
CRICKET_EXPLICIT_SPORT_CODES = {"ipl", "odi", "t20", "test", "csa"}


def _football_discovery_queries() -> tuple[str, ...]:
    return (
        "Premier League",
        "La Liga",
        "Bundesliga",
        "Serie A",
        "Ligue 1",
        "Champions League",
        "Europa League",
        "MLS",
        "Liga MX",
        "Copa Libertadores",
        "Brazil Serie A",
        "Argentina Primera",
    )


def _sport_code_is_target(code: str) -> bool:
    normalized = code.strip().lower()
    return (
        normalized in FOOTBALL_SPORT_CODES
        or normalized in BASKETBALL_SPORT_CODES
        or normalized in BASEBALL_SPORT_CODES
        or normalized in HOCKEY_SPORT_CODES
        or normalized in TENNIS_SPORT_CODES
        or normalized in CRICKET_EXPLICIT_SPORT_CODES
        or normalized.startswith("cric")
    )


def _sport_from_sport_code(code: str) -> str | None:
    normalized = code.strip().lower()
    if normalized in FOOTBALL_SPORT_CODES:
        return "football"
    if normalized in BASKETBALL_SPORT_CODES:
        return "basketball"
    if normalized in BASEBALL_SPORT_CODES:
        return "baseball"
    if normalized in HOCKEY_SPORT_CODES:
        return "hockey"
    if normalized in TENNIS_SPORT_CODES:
        return "tennis"
    if normalized in CRICKET_EXPLICIT_SPORT_CODES or normalized.startswith("cric"):
        return "cricket"
    return None


def _default_sample_requirements() -> dict[str, Any]:
    return {
        "valid_forward_decisions": 100,
        "settled_unique_decisions": 40,
        "fresh_book_rate": 0.80,
    }


@dataclass(frozen=True)
class MarketFamilySpec:
    market_family: str
    sport: str
    market_type: str
    outcome_schema: tuple[str, ...]
    benchmark_id: str
    discovery_queries: tuple[str, ...]
    model_mode: str
    initial_state: str
    settlement_rule: str
    min_valid_decisions: int = 100
    min_settled_decisions: int = 40
    fresh_book_rate_min: float = 0.80
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome_schema"] = list(self.outcome_schema)
        payload["discovery_queries"] = list(self.discovery_queries)
        return payload


@dataclass(frozen=True)
class MarketLaneSpec:
    lane_id: str
    sport: str
    sport_id: str
    sport_merge_group: str
    decision_type: str
    market_subtype: str
    outcome_schema: tuple[str, ...]
    benchmark_id: str
    legacy_included: tuple[str, ...]
    discovery_queries: tuple[str, ...]
    model_mode: str
    policy_mode: str
    sample_requirements: dict[str, Any]
    status: str
    settlement_rule: str
    reference_only: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome_schema"] = list(self.outcome_schema)
        payload["legacy_included"] = list(self.legacy_included)
        payload["discovery_queries"] = list(self.discovery_queries)
        return payload


LaneSpec = MarketLaneSpec


@dataclass(frozen=True)
class ParsedMarket:
    market_family: str
    sport: str
    market_type: str
    outcome_schema: tuple[str, ...]
    benchmark_id: str
    status: str
    parse_status: str
    parse_reason: str
    lane_id: str = ""
    sport_id: str = ""
    sport_merge_group: str = ""
    decision_type: str = ""
    market_subtype: str = ""
    legacy_slice: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome_schema"] = list(self.outcome_schema)
        return payload


def market_family_registry() -> dict[str, MarketFamilySpec]:
    """Declarative registry for the isolated multi-market lane."""

    football_queries = _football_discovery_queries()
    return {
        "football_1x2_global": MarketFamilySpec(
            market_family="football_1x2_global",
            sport="football",
            market_type="1x2",
            outcome_schema=("home", "draw", "away"),
            benchmark_id="football_1x2_global_v1",
            discovery_queries=football_queries,
            model_mode="football_goal_model_adapter_ready",
            initial_state=STATUS_SHADOW_COLLECT_ONLY,
            settlement_rule="team_or_draw_yes_no_contracts_only",
            description="Global football 1X2 expansion. It is not the E0+SP1+D1 canonical benchmark.",
        ),
        "football_goals_core": MarketFamilySpec(
            market_family="football_goals_core",
            sport="football",
            market_type="goals",
            outcome_schema=("over", "under", "btts_yes", "btts_no"),
            benchmark_id="football_goals_core_v1",
            discovery_queries=football_queries + ("Total Goals", "Both Teams To Score", "BTTS"),
            model_mode="poisson_goal_lambdas_adapter_ready",
            initial_state=STATUS_SHADOW_COLLECT_ONLY,
            settlement_rule="parseable_goal_total_or_btts_binary_contracts_only",
            description="Football totals and BTTS, restricted to parseable lines and clean settlement.",
        ),
        "tennis_match_winner": MarketFamilySpec(
            market_family="tennis_match_winner",
            sport="tennis",
            market_type="moneyline",
            outcome_schema=("player_a", "player_b"),
            benchmark_id="tennis_match_winner_v1",
            discovery_queries=("Tennis", "ATP", "WTA", "Grand Slam", "Wimbledon", "US Open"),
            model_mode="capture_and_baseline_only",
            initial_state=STATUS_CAPTURE_ONLY,
            settlement_rule="match_winner_binary_contracts_only",
            description="Tennis match winner. Capture-only until a free-data model adapter exists.",
        ),
        "basketball_moneyline": MarketFamilySpec(
            market_family="basketball_moneyline",
            sport="basketball",
            market_type="moneyline",
            outcome_schema=("team_a", "team_b"),
            benchmark_id="basketball_moneyline_v1",
            discovery_queries=("NBA", "EuroLeague", "NCAA Basketball", "Basketball"),
            model_mode="capture_and_baseline_only",
            initial_state=STATUS_CAPTURE_ONLY,
            settlement_rule="match_winner_binary_contracts_only",
            description="Basketball moneyline. Capture-only until a free-data model adapter exists.",
        ),
        "baseball_moneyline": MarketFamilySpec(
            market_family="baseball_moneyline",
            sport="baseball",
            market_type="moneyline",
            outcome_schema=("team_a", "team_b"),
            benchmark_id="baseball_moneyline_v1",
            discovery_queries=("MLB", "Baseball", "NPB", "KBO"),
            model_mode="capture_and_baseline_only",
            initial_state=STATUS_CAPTURE_ONLY,
            settlement_rule="match_winner_binary_contracts_only",
            description="Baseball moneyline. Capture-only until a free-data model adapter exists.",
        ),
        "hockey_moneyline": MarketFamilySpec(
            market_family="hockey_moneyline",
            sport="hockey",
            market_type="moneyline",
            outcome_schema=("team_a", "team_b"),
            benchmark_id="hockey_moneyline_v1",
            discovery_queries=("NHL", "Ice Hockey", "Hockey"),
            model_mode="capture_and_baseline_only",
            initial_state=STATUS_CAPTURE_ONLY,
            settlement_rule="match_winner_binary_contracts_only",
            description="Hockey moneyline. Capture-only until a free-data model adapter exists.",
        ),
        "cricket_match_winner": MarketFamilySpec(
            market_family="cricket_match_winner",
            sport="cricket",
            market_type="moneyline",
            outcome_schema=("team_a", "team_b"),
            benchmark_id="cricket_match_winner_v1",
            discovery_queries=("Cricket", "IPL", "T20", "ODI", "Test Cricket"),
            model_mode="capture_and_baseline_only",
            initial_state=STATUS_CAPTURE_ONLY,
            settlement_rule="match_winner_binary_contracts_only",
            description="Cricket match winner, only if Polymarket coverage is sufficient.",
        ),
        "football_cards": MarketFamilySpec(
            market_family="football_cards",
            sport="football",
            market_type="cards",
            outcome_schema=("deferred",),
            benchmark_id="football_cards_deferred",
            discovery_queries=(),
            model_mode="deferred",
            initial_state=STATUS_DEFERRED,
            settlement_rule="deferred_until_data_and_settlement_are_clean",
            description="Deferred: card markets need richer data and settlement validation.",
        ),
        "football_corners": MarketFamilySpec(
            market_family="football_corners",
            sport="football",
            market_type="corners",
            outcome_schema=("deferred",),
            benchmark_id="football_corners_deferred",
            discovery_queries=(),
            model_mode="deferred",
            initial_state=STATUS_DEFERRED,
            settlement_rule="deferred_until_data_and_settlement_are_clean",
            description="Deferred: corner markets need richer data and settlement validation.",
        ),
        "player_props": MarketFamilySpec(
            market_family="player_props",
            sport="multi_sport",
            market_type="player_props",
            outcome_schema=("deferred",),
            benchmark_id="player_props_deferred",
            discovery_queries=(),
            model_mode="deferred",
            initial_state=STATUS_DEFERRED,
            settlement_rule="deferred_until_player_data_is_available",
            description="Deferred: player markets are too granular for the current data layer.",
        ),
        "spreads_handicaps": MarketFamilySpec(
            market_family="spreads_handicaps",
            sport="multi_sport",
            market_type="spread",
            outcome_schema=("deferred",),
            benchmark_id="spreads_handicaps_deferred",
            discovery_queries=(),
            model_mode="deferred",
            initial_state=STATUS_DEFERRED,
            settlement_rule="deferred_until_line_parsing_and_settlement_are_clean",
            description="Deferred: handicap/spread lines need separate benchmark rules.",
        ),
        "non_football_totals": MarketFamilySpec(
            market_family="non_football_totals",
            sport="multi_sport",
            market_type="totals",
            outcome_schema=("deferred",),
            benchmark_id="non_football_totals_deferred",
            discovery_queries=(),
            model_mode="deferred",
            initial_state=STATUS_DEFERRED,
            settlement_rule="deferred_until_sport_specific_total_models_exist",
            description="Deferred: totals outside football are not part of v1.",
        ),
    }


def get_market_family_spec(market_family: str) -> MarketFamilySpec:
    registry = market_family_registry()
    if market_family not in registry:
        raise KeyError(f"Familia multi-mercado no declarada: {market_family}")
    return registry[market_family]


def market_lane_registry() -> dict[str, MarketLaneSpec]:
    football_queries = _football_discovery_queries()
    sample_requirements = _default_sample_requirements()
    return {
        "football_1x2_canonical": MarketLaneSpec(
            lane_id="football_1x2_canonical",
            sport="football",
            sport_id="football",
            sport_merge_group="football",
            decision_type="1x2",
            market_subtype="canonical_1x2",
            outcome_schema=("home", "draw", "away"),
            benchmark_id="football_1x2_canonical_legacy",
            legacy_included=LEGACY_FOOTBALL_1X2_LEAGUES,
            discovery_queries=("Premier League", "La Liga", "Bundesliga"),
            model_mode="legacy_reference_only",
            policy_mode="legacy_reference_only",
            sample_requirements=sample_requirements,
            status=STATUS_REFERENCE_ONLY,
            settlement_rule="legacy_e0_sp1_d1_reference_only",
            reference_only=True,
            description="Reference-only legacy benchmark kept for parity checks, not a new operational lane.",
        ),
        "football_1x2_global": MarketLaneSpec(
            lane_id="football_1x2_global",
            sport="football",
            sport_id="football",
            sport_merge_group="football",
            decision_type="1x2",
            market_subtype="global_1x2",
            outcome_schema=("home", "draw", "away"),
            benchmark_id="football_1x2_global_v1",
            legacy_included=LEGACY_FOOTBALL_1X2_LEAGUES,
            discovery_queries=football_queries,
            model_mode="football_goal_model_adapter_ready",
            policy_mode="lane_policy_pending",
            sample_requirements=sample_requirements,
            status=STATUS_SHADOW_COLLECT_ONLY,
            settlement_rule="team_or_draw_yes_no_contracts_only",
            description="Global football 1X2 lane. Includes E0, SP1 and D1 as legacy slices.",
        ),
        "football_goals_core": MarketLaneSpec(
            lane_id="football_goals_core",
            sport="football",
            sport_id="football",
            sport_merge_group="football",
            decision_type="goals",
            market_subtype="total_goals_or_btts",
            outcome_schema=("over", "under", "btts_yes", "btts_no"),
            benchmark_id="football_goals_core_v1",
            legacy_included=(),
            discovery_queries=football_queries + ("Total Goals", "Both Teams To Score", "BTTS"),
            model_mode="poisson_goal_lambdas_adapter_ready",
            policy_mode="lane_policy_pending",
            sample_requirements=sample_requirements,
            status=STATUS_SHADOW_COLLECT_ONLY,
            settlement_rule="parseable_goal_total_or_btts_binary_contracts_only",
            description="Football goals lane for total-goals and BTTS subtypes, reported separately.",
        ),
        "tennis_match_winner": MarketLaneSpec(
            lane_id="tennis_match_winner",
            sport="tennis",
            sport_id="tennis",
            sport_merge_group="tennis",
            decision_type="match_winner",
            market_subtype="moneyline",
            outcome_schema=("player_a", "player_b"),
            benchmark_id="tennis_match_winner_v1",
            legacy_included=(),
            discovery_queries=("Tennis", "ATP", "WTA", "Grand Slam", "Wimbledon", "US Open"),
            model_mode="capture_and_baseline_only",
            policy_mode="policy_disabled_until_model_ready",
            sample_requirements=sample_requirements,
            status=STATUS_CAPTURE_ONLY,
            settlement_rule="match_winner_binary_contracts_only",
            description="Tennis match-winner lane, capture-only until a free-data model adapter exists.",
        ),
        "basketball_moneyline": MarketLaneSpec(
            lane_id="basketball_moneyline",
            sport="basketball",
            sport_id="basketball",
            sport_merge_group="basketball",
            decision_type="match_winner",
            market_subtype="moneyline",
            outcome_schema=("team_a", "team_b"),
            benchmark_id="basketball_moneyline_v1",
            legacy_included=(),
            discovery_queries=("NBA", "EuroLeague", "NCAA Basketball", "Basketball"),
            model_mode="capture_and_baseline_only",
            policy_mode="policy_disabled_until_model_ready",
            sample_requirements=sample_requirements,
            status=STATUS_CAPTURE_ONLY,
            settlement_rule="match_winner_binary_contracts_only",
            description="Basketball moneyline lane, capture-only until a free-data model adapter exists.",
        ),
        "baseball_moneyline": MarketLaneSpec(
            lane_id="baseball_moneyline",
            sport="baseball",
            sport_id="baseball",
            sport_merge_group="baseball",
            decision_type="match_winner",
            market_subtype="moneyline",
            outcome_schema=("team_a", "team_b"),
            benchmark_id="baseball_moneyline_v1",
            legacy_included=(),
            discovery_queries=("MLB", "Baseball", "NPB", "KBO"),
            model_mode="capture_and_baseline_only",
            policy_mode="policy_disabled_until_model_ready",
            sample_requirements=sample_requirements,
            status=STATUS_CAPTURE_ONLY,
            settlement_rule="match_winner_binary_contracts_only",
            description="Baseball moneyline lane, capture-only until a free-data model adapter exists.",
        ),
        "hockey_moneyline": MarketLaneSpec(
            lane_id="hockey_moneyline",
            sport="hockey",
            sport_id="hockey",
            sport_merge_group="hockey",
            decision_type="match_winner",
            market_subtype="moneyline",
            outcome_schema=("team_a", "team_b"),
            benchmark_id="hockey_moneyline_v1",
            legacy_included=(),
            discovery_queries=("NHL", "Ice Hockey", "Hockey"),
            model_mode="capture_and_baseline_only",
            policy_mode="policy_disabled_until_model_ready",
            sample_requirements=sample_requirements,
            status=STATUS_CAPTURE_ONLY,
            settlement_rule="match_winner_binary_contracts_only",
            description="Hockey moneyline lane, capture-only until a free-data model adapter exists.",
        ),
        "cricket_match_winner": MarketLaneSpec(
            lane_id="cricket_match_winner",
            sport="cricket",
            sport_id="cricket",
            sport_merge_group="cricket",
            decision_type="match_winner",
            market_subtype="moneyline",
            outcome_schema=("team_a", "team_b"),
            benchmark_id="cricket_match_winner_v1",
            legacy_included=(),
            discovery_queries=("Cricket", "IPL", "T20", "ODI", "Test Cricket"),
            model_mode="capture_and_baseline_only",
            policy_mode="policy_disabled_until_model_ready",
            sample_requirements=sample_requirements,
            status=STATUS_CAPTURE_ONLY,
            settlement_rule="match_winner_binary_contracts_only",
            description="Cricket match-winner lane, capture-only until coverage and data are sufficient.",
        ),
        "football_cards": MarketLaneSpec(
            lane_id="football_cards",
            sport="football",
            sport_id="football",
            sport_merge_group="football",
            decision_type="cards",
            market_subtype="cards",
            outcome_schema=("deferred",),
            benchmark_id="football_cards_deferred",
            legacy_included=(),
            discovery_queries=(),
            model_mode="deferred",
            policy_mode="deferred",
            sample_requirements=sample_requirements,
            status=STATUS_DEFERRED,
            settlement_rule="deferred_until_data_and_settlement_are_clean",
            description="Deferred football cards lane.",
        ),
        "football_corners": MarketLaneSpec(
            lane_id="football_corners",
            sport="football",
            sport_id="football",
            sport_merge_group="football",
            decision_type="corners",
            market_subtype="corners",
            outcome_schema=("deferred",),
            benchmark_id="football_corners_deferred",
            legacy_included=(),
            discovery_queries=(),
            model_mode="deferred",
            policy_mode="deferred",
            sample_requirements=sample_requirements,
            status=STATUS_DEFERRED,
            settlement_rule="deferred_until_data_and_settlement_are_clean",
            description="Deferred football corners lane.",
        ),
        "player_props": MarketLaneSpec(
            lane_id="player_props",
            sport="multi_sport",
            sport_id="multi_sport",
            sport_merge_group="multi_sport",
            decision_type="player_props",
            market_subtype="player_props",
            outcome_schema=("deferred",),
            benchmark_id="player_props_deferred",
            legacy_included=(),
            discovery_queries=(),
            model_mode="deferred",
            policy_mode="deferred",
            sample_requirements=sample_requirements,
            status=STATUS_DEFERRED,
            settlement_rule="deferred_until_player_data_is_available",
            description="Deferred player props lane.",
        ),
        "spreads_handicaps": MarketLaneSpec(
            lane_id="spreads_handicaps",
            sport="multi_sport",
            sport_id="multi_sport",
            sport_merge_group="multi_sport",
            decision_type="spread",
            market_subtype="spread_or_handicap",
            outcome_schema=("deferred",),
            benchmark_id="spreads_handicaps_deferred",
            legacy_included=(),
            discovery_queries=(),
            model_mode="deferred",
            policy_mode="deferred",
            sample_requirements=sample_requirements,
            status=STATUS_DEFERRED,
            settlement_rule="deferred_until_line_parsing_and_settlement_are_clean",
            description="Deferred spreads and handicaps lane.",
        ),
        "non_football_totals": MarketLaneSpec(
            lane_id="non_football_totals",
            sport="multi_sport",
            sport_id="multi_sport",
            sport_merge_group="multi_sport",
            decision_type="totals",
            market_subtype="non_football_totals",
            outcome_schema=("deferred",),
            benchmark_id="non_football_totals_deferred",
            legacy_included=(),
            discovery_queries=(),
            model_mode="deferred",
            policy_mode="deferred",
            sample_requirements=sample_requirements,
            status=STATUS_DEFERRED,
            settlement_rule="deferred_until_sport_specific_total_models_exist",
            description="Deferred non-football totals lane.",
        ),
    }


def get_market_lane_spec(lane_id: str) -> MarketLaneSpec:
    registry = market_lane_registry()
    if lane_id not in registry:
        raise KeyError(f"Carril de mercado no declarado: {lane_id}")
    return registry[lane_id]


def _lane_governance_payload(governance: LaneGovernance) -> dict[str, Any]:
    return {
        "lane_id": governance.lane_id,
        "mode": governance.mode.value,
        "can_capture": governance.can_capture,
        "can_build_predictions": governance.can_build_predictions,
        "can_emit_shadow_picks": governance.can_emit_shadow_picks,
        "can_emit_capital_picks": governance.can_emit_capital_picks,
        "reason": governance.reason,
    }


def _catalog_capture_governance_mask(catalog: pd.DataFrame) -> pd.Series:
    if catalog.empty:
        return pd.Series(dtype=bool)
    lanes = catalog["lane_id"].fillna(catalog["market_family"]).astype(str)
    return lanes.map(lambda lane: get_lane_governance(lane).can_capture)


def _family_spec_from_lane(lane: MarketLaneSpec) -> MarketFamilySpec:
    return MarketFamilySpec(
        market_family=lane.lane_id,
        sport=lane.sport,
        market_type=lane.decision_type,
        outcome_schema=lane.outcome_schema,
        benchmark_id=lane.benchmark_id,
        discovery_queries=lane.discovery_queries,
        model_mode=lane.model_mode,
        initial_state=lane.status,
        settlement_rule=lane.settlement_rule,
        min_valid_decisions=int(lane.sample_requirements["valid_forward_decisions"]),
        min_settled_decisions=int(lane.sample_requirements["settled_unique_decisions"]),
        fresh_book_rate_min=float(lane.sample_requirements["fresh_book_rate"]),
        description=lane.description,
    )


def market_family_manifest() -> dict[str, Any]:
    target_low = math.ceil(TARGET_FORWARD_DECISIONS_PER_DAY / CONSERVATIVE_PICK_RATE_HIGH)
    target_high = math.ceil(TARGET_FORWARD_DECISIONS_PER_DAY / CONSERVATIVE_PICK_RATE_LOW)
    return {
        "lane": "raw_market_capture",
        "architecture": "raw_market_capture_plus_isolated_market_lanes",
        "canonical_benchmark_untouched": True,
        "database_filename": MULTI_MARKET_DATABASE_FILENAME,
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        "target_forward_decisions_per_day": TARGET_FORWARD_DECISIONS_PER_DAY,
        "estimated_events_needed_per_day": {
            "low": target_low,
            "high": target_high,
            "assumed_pick_rate_range": [CONSERVATIVE_PICK_RATE_LOW, CONSERVATIVE_PICK_RATE_HIGH],
        },
        "reference_lanes": [market_lane_registry()[item].to_dict() for item in REFERENCE_LANES],
        "active_lanes": [market_lane_registry()[item].to_dict() for item in ACTIVE_LANES],
        "deferred_lanes": [market_lane_registry()[item].to_dict() for item in DEFERRED_LANES],
        "active_families": [market_family_registry()[item].to_dict() for item in ACTIVE_FAMILIES],
        "deferred_families": [market_family_registry()[item].to_dict() for item in DEFERRED_FAMILIES],
    }


def default_multi_market_db_path(settings: Settings) -> Path:
    return settings.paths.data_dir / MULTI_MARKET_DATABASE_FILENAME


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _as_bool_int(value: Any) -> int:
    return 1 if bool(value) else 0


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _lane_from_family(market_family: str) -> MarketLaneSpec:
    return get_market_lane_spec(market_family)


def _is_legacy_football_slice(event: dict[str, Any], market: dict[str, Any]) -> bool:
    text = _text_blob(event, market)
    return _contains_any(
        text,
        (
            "premier league",
            " epl",
            "la liga",
            " laliga",
            "bundesliga",
            " e0",
            " sp1",
            " d1",
        ),
    )


def _parsed_from_lane(
    lane: MarketLaneSpec,
    parse_status: str,
    parse_reason: str,
    market_subtype: str | None = None,
    legacy_slice: bool = False,
) -> ParsedMarket:
    return ParsedMarket(
        market_family=lane.lane_id,
        sport=lane.sport,
        market_type=lane.decision_type,
        outcome_schema=lane.outcome_schema,
        benchmark_id=lane.benchmark_id,
        status=lane.status,
        parse_status=parse_status,
        parse_reason=parse_reason,
        lane_id=lane.lane_id,
        sport_id=lane.sport_id,
        sport_merge_group=lane.sport_merge_group,
        decision_type=lane.decision_type,
        market_subtype=market_subtype or lane.market_subtype,
        legacy_slice=legacy_slice,
    )


def _text_blob(event: dict[str, Any], market: dict[str, Any]) -> str:
    parts = [
        event.get("title"),
        event.get("slug"),
        event.get("ticker"),
        event.get("seriesSlug"),
        event.get("sport"),
        event.get("sportSlug"),
        market.get("question"),
        market.get("slug"),
        market.get("sportsMarketType"),
        market.get("groupItemTitle"),
        market.get("description"),
    ]
    parts.extend(PolymarketGammaClient.parse_outcomes(market))
    return " ".join(str(part or "") for part in parts).lower()


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def _is_deferred_market(text: str) -> tuple[str, str] | None:
    if _contains_any(text, (" card", "cards", "yellow card", "red card", "booking")):
        return "football_cards", "cards_deferred"
    if _contains_any(text, (" corner", "corners")):
        return "football_corners", "corners_deferred"
    if _contains_any(
        text,
        (
            "player",
            "scorer",
            "assist",
            "shots",
            "rebounds",
            "points by",
            "strikeouts",
            "home runs",
            "touchdown",
        ),
    ):
        return "player_props", "player_props_deferred"
    if _contains_any(text, ("spread", "handicap", "asian handicap", "+1.5", "-1.5")):
        return "spreads_handicaps", "spread_deferred"
    if _contains_any(text, ("total points", "total runs", "total games", "total sets")):
        return "non_football_totals", "non_football_totals_deferred"
    return None


def _infer_sport(text: str) -> str | None:
    if _contains_any(text, ("tennis", " atp", " wta", "wimbledon", "us open", "australian open", "roland garros")):
        return "tennis"
    if _contains_any(text, ("basketball", " nba", "euroleague", "ncaa basketball", "wnba")):
        return "basketball"
    if _contains_any(text, ("baseball", " mlb", " npb", " kbo")):
        return "baseball"
    if _contains_any(text, ("hockey", " nhl", "ice hockey")):
        return "hockey"
    if _contains_any(text, ("cricket", " ipl", " t20", " odi", "test cricket")):
        return "cricket"
    if _contains_any(
        text,
        (
            "football",
            "soccer",
            "premier league",
            " la liga",
            "bundesliga",
            "serie a",
            "ligue 1",
            "champions league",
            "europa league",
            " mls",
            "liga mx",
            "libertadores",
            "brazil serie a",
        ),
    ):
        return "football"
    return None


def _binary_yes_no(outcomes: list[str]) -> bool:
    normalized = {item.strip().lower() for item in outcomes}
    return {"yes", "no"}.issubset(normalized) or len(outcomes) == 2


def _has_match_context(text: str) -> bool:
    return _contains_any(
        text,
        (
            " vs ",
            " vs. ",
            " v ",
            " v. ",
            " against ",
            " beat ",
            " on 20",
        ),
    )


def _is_football_outright_future(text: str) -> bool:
    if _has_match_context(text):
        return False
    return _contains_any(
        text,
        (
            " win the 20",
            " win premier league",
            " win the premier league",
            " win la liga",
            " win the la liga",
            " win bundesliga",
            " win the bundesliga",
            " win serie a",
            " win ligue 1",
            " league winner",
            " winner-",
            " season winner",
            " tournament winner",
            " win the title",
            " champion",
        ),
    )


def classify_market(event: dict[str, Any], market: dict[str, Any]) -> ParsedMarket | None:
    text = _text_blob(event, market)
    deferred = _is_deferred_market(text)
    if deferred is not None:
        family, reason = deferred
        return _parsed_from_lane(get_market_lane_spec(family), parse_status="deferred", parse_reason=reason)

    sport = _sport_from_sport_code(str(event.get("_discovery_sport_code") or "")) or _infer_sport(text)
    outcomes = PolymarketGammaClient.parse_outcomes(market)
    outcomes_lower = [item.strip().lower() for item in outcomes]
    sports_market_type = str(market.get("sportsMarketType") or "").lower()

    if sport == "football":
        if _is_football_outright_future(text):
            return None
        if _contains_any(text, ("both teams to score", "btts")):
            return _parsed_from_lane(
                get_market_lane_spec("football_goals_core"),
                parse_status="parseable",
                parse_reason="football_btts",
                market_subtype="btts",
            )
        if _contains_any(text, ("total goals", "over ", "under ")) and "goal" in text:
            return _parsed_from_lane(
                get_market_lane_spec("football_goals_core"),
                parse_status="parseable",
                parse_reason="football_total_goals",
                market_subtype="total_goals",
            )
        if "draw" in outcomes_lower and len(outcomes_lower) >= 3:
            return _parsed_from_lane(
                get_market_lane_spec("football_1x2_global"),
                parse_status="parseable",
                parse_reason="football_1x2_three_way",
                legacy_slice=_is_legacy_football_slice(event, market),
            )
        binary_draw = _contains_any(text, ("end in a draw", "draw?", " be a draw"))
        binary_win = _has_match_context(text) and _contains_any(text, (" win", "winner", " beat "))
        if _binary_yes_no(outcomes) and (binary_draw or binary_win):
            return _parsed_from_lane(
                get_market_lane_spec("football_1x2_global"),
                parse_status="parseable",
                parse_reason="football_1x2_binary_contract",
                legacy_slice=_is_legacy_football_slice(event, market),
            )

    moneyline_family = {
        "tennis": "tennis_match_winner",
        "basketball": "basketball_moneyline",
        "baseball": "baseball_moneyline",
        "hockey": "hockey_moneyline",
        "cricket": "cricket_match_winner",
    }.get(str(sport))
    if moneyline_family and (
        "moneyline" in sports_market_type
        or _contains_any(text, (" win", "winner", "match winner"))
        or _binary_yes_no(outcomes)
    ):
        return _parsed_from_lane(
            get_market_lane_spec(moneyline_family),
            parse_status="parseable",
            parse_reason=f"{sport}_moneyline",
        )

    return None


def init_multi_market_db(path: Path | str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS mm_market_catalog (
            market_id TEXT PRIMARY KEY,
            event_id TEXT,
            event_slug TEXT,
            event_title TEXT,
            market_slug TEXT,
            question TEXT,
            lane_id TEXT,
            market_family TEXT NOT NULL,
            benchmark_id TEXT NOT NULL,
            sport TEXT NOT NULL,
            sport_id TEXT,
            sport_merge_group TEXT,
            decision_type TEXT,
            market_type TEXT NOT NULL,
            market_subtype TEXT,
            outcome_schema_json TEXT NOT NULL,
            status TEXT NOT NULL,
            parse_status TEXT NOT NULL,
            parse_reason TEXT,
            legacy_slice INTEGER DEFAULT 0,
            active INTEGER,
            closed INTEGER,
            accepting_orders INTEGER,
            game_start_time TEXT,
            outcomes_json TEXT,
            clob_token_ids_json TEXT,
            top_ask REAL,
            raw_json TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mm_book_checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            market_id TEXT NOT NULL,
            event_slug TEXT,
            market_family TEXT NOT NULL,
            benchmark_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            outcome_name TEXT,
            top_ask REAL,
            asks_json TEXT,
            bids_json TEXT,
            source TEXT,
            raw_json TEXT
        );

        CREATE TABLE IF NOT EXISTS mm_forward_ledger (
            decision_id TEXT PRIMARY KEY,
            market_id TEXT NOT NULL,
            event_slug TEXT,
            market_family TEXT NOT NULL,
            benchmark_id TEXT NOT NULL,
            decision_time TEXT,
            decision_status TEXT NOT NULL,
            blocker TEXT,
            selected_outcome TEXT,
            model_mode TEXT NOT NULL,
            policy_mode TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mm_raw_events (
            event_id TEXT PRIMARY KEY,
            event_slug TEXT UNIQUE,
            event_title TEXT,
            sport_id TEXT,
            venue TEXT,
            raw_json TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mm_raw_markets (
            market_id TEXT PRIMARY KEY,
            event_id TEXT,
            event_slug TEXT,
            market_slug TEXT,
            question TEXT,
            sport_id TEXT,
            venue TEXT,
            active INTEGER,
            closed INTEGER,
            accepting_orders INTEGER,
            game_start_time TEXT,
            outcomes_json TEXT,
            clob_token_ids_json TEXT,
            raw_json TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mm_raw_orderbooks (
            checkpoint_id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            market_id TEXT NOT NULL,
            event_slug TEXT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            outcome_name TEXT,
            top_ask REAL,
            asks_json TEXT,
            bids_json TEXT,
            source TEXT,
            raw_json TEXT
        );

        CREATE TABLE IF NOT EXISTS mm_raw_settlements (
            settlement_id TEXT PRIMARY KEY,
            market_id TEXT,
            event_slug TEXT,
            resolved_at TEXT,
            winning_outcome TEXT,
            status TEXT,
            raw_json TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mm_lane_market_links (
            link_id TEXT PRIMARY KEY,
            lane_id TEXT NOT NULL,
            market_id TEXT NOT NULL,
            event_slug TEXT,
            benchmark_id TEXT NOT NULL,
            sport_id TEXT NOT NULL,
            sport_merge_group TEXT NOT NULL,
            decision_type TEXT NOT NULL,
            market_subtype TEXT NOT NULL,
            outcome_schema_json TEXT NOT NULL,
            legacy_slice INTEGER DEFAULT 0,
            link_status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mm_lane_forward_ledger (
            decision_id TEXT PRIMARY KEY,
            lane_id TEXT NOT NULL,
            market_id TEXT NOT NULL,
            event_slug TEXT,
            benchmark_id TEXT NOT NULL,
            decision_time TEXT,
            decision_status TEXT NOT NULL,
            blocker TEXT,
            selected_outcome TEXT,
            model_mode TEXT NOT NULL,
            policy_mode TEXT NOT NULL,
            market_subtype TEXT,
            legacy_slice INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
    )
    _ensure_columns(
        connection,
        "mm_market_catalog",
        {
            "lane_id": "TEXT",
            "sport_id": "TEXT",
            "sport_merge_group": "TEXT",
            "decision_type": "TEXT",
            "market_subtype": "TEXT",
            "legacy_slice": "INTEGER DEFAULT 0",
        },
    )
    _ensure_columns(
        connection,
        "mm_lane_forward_ledger",
        {
            "game_start_time": "TEXT",
            "contract_outcome": "TEXT",
            "asset_id": "TEXT",
            "book_timestamp": "TEXT",
            "top_ask": "REAL",
            "quoted_odds": "REAL",
            "model_prob": "REAL",
            "probability_source": "TEXT",
            "edge": "REAL",
            "ev": "REAL",
            "closing_top_ask": "REAL",
            "closing_timestamp": "TEXT",
        },
    )
    from ..markets.trading import ensure_trade_audit_tables

    ensure_trade_audit_tables(connection)
    connection.commit()
    return connection


def _ensure_columns(connection: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    for column, definition in columns.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    connection.commit()


def _upsert_rows(connection: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    assignments = ", ".join(f"{column}=excluded.{column}" for column in columns)
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) ON CONFLICT DO UPDATE SET {assignments}"
    connection.executemany(sql, [[row.get(column) for column in columns] for row in rows])
    connection.commit()


def _clear_market_classifications(connection: sqlite3.Connection, market_ids: Iterable[str]) -> None:
    ids = [str(market_id) for market_id in market_ids if str(market_id)]
    if not ids:
        return
    connection.executemany("DELETE FROM mm_lane_market_links WHERE market_id = ?", [(market_id,) for market_id in ids])
    connection.executemany("DELETE FROM mm_market_catalog WHERE market_id = ?", [(market_id,) for market_id in ids])
    connection.commit()


def _market_id(market: dict[str, Any]) -> str:
    return str(market.get("id") or market.get("conditionId") or market.get("marketId") or market.get("slug") or "")


def _event_slug(event: dict[str, Any], market: dict[str, Any]) -> str:
    return str(event.get("slug") or market.get("eventSlug") or market.get("event_slug") or "")


def _catalog_row(
    event: dict[str, Any],
    market: dict[str, Any],
    parsed: ParsedMarket,
    now: pd.Timestamp,
) -> dict[str, Any]:
    outcomes = PolymarketGammaClient.parse_outcomes(market)
    token_ids = PolymarketGammaClient.parse_clob_token_ids(market)
    yes_price = PolymarketGammaClient.extract_yes_price(market)
    return {
        "market_id": _market_id(market),
        "event_id": str(event.get("id") or market.get("eventId") or ""),
        "event_slug": _event_slug(event, market),
        "event_title": str(event.get("title") or market.get("eventTitle") or ""),
        "market_slug": str(market.get("slug") or ""),
        "question": str(market.get("question") or ""),
        "lane_id": parsed.lane_id or parsed.market_family,
        "market_family": parsed.market_family,
        "benchmark_id": parsed.benchmark_id,
        "sport": parsed.sport,
        "sport_id": parsed.sport_id or parsed.sport,
        "sport_merge_group": parsed.sport_merge_group or parsed.sport,
        "decision_type": parsed.decision_type or parsed.market_type,
        "market_type": parsed.market_type,
        "market_subtype": parsed.market_subtype or parsed.market_type,
        "outcome_schema_json": _json_dumps(list(parsed.outcome_schema)),
        "status": parsed.status,
        "parse_status": parsed.parse_status,
        "parse_reason": parsed.parse_reason,
        "legacy_slice": _as_bool_int(parsed.legacy_slice),
        "active": _as_bool_int(market.get("active", False)),
        "closed": _as_bool_int(market.get("closed", False)),
        "accepting_orders": _as_bool_int(market.get("acceptingOrders", market.get("accepting_orders", False))),
        "game_start_time": str(market.get("gameStartTime") or event.get("startDate") or event.get("startTime") or ""),
        "outcomes_json": _json_dumps(outcomes),
        "clob_token_ids_json": _json_dumps(token_ids),
        "top_ask": yes_price,
        "raw_json": _json_dumps({"event": event, "market": market}),
        "updated_at": now.isoformat(),
    }


def _raw_event_row(event: dict[str, Any], parsed: ParsedMarket | None, now: pd.Timestamp) -> dict[str, Any]:
    event_id = str(event.get("id") or event.get("slug") or "")
    event_slug = str(event.get("slug") or event_id)
    sport_id = parsed.sport_id if parsed is not None and parsed.sport_id else (_infer_sport(str(event)) or "unknown")
    return {
        "event_id": event_id,
        "event_slug": event_slug,
        "event_title": str(event.get("title") or ""),
        "sport_id": sport_id,
        "venue": "polymarket",
        "raw_json": _json_dumps(event),
        "updated_at": now.isoformat(),
    }


def _raw_market_row(event: dict[str, Any], market: dict[str, Any], parsed: ParsedMarket | None, now: pd.Timestamp) -> dict[str, Any]:
    outcomes = PolymarketGammaClient.parse_outcomes(market)
    token_ids = PolymarketGammaClient.parse_clob_token_ids(market)
    sport_id = parsed.sport_id if parsed is not None and parsed.sport_id else (_infer_sport(_text_blob(event, market)) or "unknown")
    return {
        "market_id": _market_id(market),
        "event_id": str(event.get("id") or market.get("eventId") or ""),
        "event_slug": _event_slug(event, market),
        "market_slug": str(market.get("slug") or ""),
        "question": str(market.get("question") or ""),
        "sport_id": sport_id,
        "venue": "polymarket",
        "active": _as_bool_int(market.get("active", False)),
        "closed": _as_bool_int(market.get("closed", False)),
        "accepting_orders": _as_bool_int(market.get("acceptingOrders", market.get("accepting_orders", False))),
        "game_start_time": str(market.get("gameStartTime") or event.get("startDate") or event.get("startTime") or ""),
        "outcomes_json": _json_dumps(outcomes),
        "clob_token_ids_json": _json_dumps(token_ids),
        "raw_json": _json_dumps({"event": event, "market": market}),
        "updated_at": now.isoformat(),
    }


def _lane_link_row(catalog_row: dict[str, Any], now: pd.Timestamp) -> dict[str, Any]:
    lane_id = str(catalog_row.get("lane_id") or catalog_row.get("market_family"))
    market_id = str(catalog_row.get("market_id"))
    return {
        "link_id": f"{lane_id}:{market_id}",
        "lane_id": lane_id,
        "market_id": market_id,
        "event_slug": str(catalog_row.get("event_slug") or ""),
        "benchmark_id": str(catalog_row.get("benchmark_id") or ""),
        "sport_id": str(catalog_row.get("sport_id") or catalog_row.get("sport") or ""),
        "sport_merge_group": str(catalog_row.get("sport_merge_group") or catalog_row.get("sport") or ""),
        "decision_type": str(catalog_row.get("decision_type") or catalog_row.get("market_type") or ""),
        "market_subtype": str(catalog_row.get("market_subtype") or catalog_row.get("market_type") or ""),
        "outcome_schema_json": str(catalog_row.get("outcome_schema_json") or "[]"),
        "legacy_slice": int(catalog_row.get("legacy_slice") or 0),
        "link_status": str(catalog_row.get("status") or STATUS_DISCOVERY_ONLY),
        "created_at": now.isoformat(),
    }


def _iter_event_markets(event: dict[str, Any]) -> Iterable[dict[str, Any]]:
    markets = event.get("markets", [])
    if isinstance(markets, list):
        for market in markets:
            if isinstance(market, dict):
                yield market


def _discover_events_from_sports_series(
    gamma_client: PolymarketGammaClient,
    limit_per_series: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not hasattr(gamma_client, "list_sports") or not hasattr(gamma_client, "list_events"):
        return [], {
            "source": "sports_series",
            "status": "client_does_not_support_sports_series",
            "series_queries": 0,
            "events": 0,
            "errors": [],
        }
    try:
        sports = gamma_client.list_sports()
    except Exception as exc:  # pragma: no cover - public API reliability
        return [], {
            "source": "sports_series",
            "status": "sports_endpoint_failed",
            "series_queries": 0,
            "events": 0,
            "errors": [str(exc)],
        }
    events_by_slug: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    series_queries = 0
    for sport in sports:
        if not isinstance(sport, dict):
            continue
        sport_code = str(sport.get("sport") or "").strip().lower()
        if not _sport_code_is_target(sport_code):
            continue
        series_id = str(sport.get("series") or "").strip()
        if not series_id:
            continue
        if series_queries >= MAX_SPORT_SERIES_DISCOVERY_CALLS:
            break
        series_queries += 1
        try:
            events = gamma_client.list_events(
                limit=limit_per_series,
                active=True,
                closed=False,
                series_id=series_id,
            )
        except Exception as exc:  # pragma: no cover - public API reliability
            errors.append(f"{sport_code}/{series_id}: {exc}")
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            slug = str(event.get("slug") or event.get("id") or "")
            if slug:
                enriched = dict(event)
                enriched["_discovery_source"] = "sports_series"
                enriched["_discovery_sport_code"] = sport_code
                enriched["_discovery_series_id"] = series_id
                events_by_slug[slug] = enriched
    return list(events_by_slug.values()), {
        "source": "sports_series",
        "status": "ok",
        "series_queries": series_queries,
        "events": len(events_by_slug),
        "errors": errors[:20],
        "truncated": series_queries >= MAX_SPORT_SERIES_DISCOVERY_CALLS,
    }


def _market_start_series(catalog: pd.DataFrame) -> pd.Series:
    values = catalog.get("game_start_time", pd.Series(dtype=str))
    try:
        return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    except TypeError:  # pragma: no cover - compatibilidad con pandas antiguo
        return pd.to_datetime(values, utc=True, errors="coerce")


def _active_market_mask(catalog: pd.DataFrame) -> pd.Series:
    if catalog.empty:
        return pd.Series(dtype=bool)
    return (catalog.get("active", pd.Series(dtype=int)) == 1) & (catalog.get("closed", pd.Series(dtype=int)) == 0)


def _forward_active_mask(catalog: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
    if catalog.empty:
        return pd.Series(dtype=bool)
    starts = _market_start_series(catalog)
    return _active_market_mask(catalog) & starts.notna() & (starts >= now)


def _forward_active_catalog(catalog: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    if catalog.empty:
        return catalog.copy()
    return catalog[_forward_active_mask(catalog, now)].copy()


def _inventory_quality(catalog: pd.DataFrame, now: pd.Timestamp) -> dict[str, Any]:
    starts = _market_start_series(catalog)
    active = _active_market_mask(catalog)
    future = starts.notna() & (starts >= now)
    historical = starts.notna() & (starts < now)
    next_24h = future & (starts <= now + pd.Timedelta(hours=24))
    return {
        "catalog_markets": int(len(catalog)),
        "historical_catalog_markets": int(historical.sum()) if not catalog.empty else 0,
        "missing_start_time_markets": int(starts.isna().sum()) if not catalog.empty else 0,
        "active_markets": int(active.sum()) if not catalog.empty else 0,
        "active_historical_markets": int((active & historical).sum()) if not catalog.empty else 0,
        "future_catalog_markets": int(future.sum()) if not catalog.empty else 0,
        "future_active_markets": int((active & future).sum()) if not catalog.empty else 0,
        "next_24h_markets": int((active & next_24h).sum()) if not catalog.empty else 0,
        "forward_inventory_ready": bool((active & future).sum() > 0) if not catalog.empty else False,
    }


def _raw_inventory_quality_rows(catalog: pd.DataFrame, now: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane_id in list(ACTIVE_LANES) + list(DEFERRED_LANES):
        subset = catalog[catalog["lane_id"].fillna(catalog["market_family"]) == lane_id] if not catalog.empty else pd.DataFrame()
        quality = _inventory_quality(subset, now)
        rows.append(
            {
                "lane_id": lane_id,
                "benchmark_id": market_lane_registry()[lane_id].benchmark_id,
                "sport": market_lane_registry()[lane_id].sport,
                **quality,
                "inventory_status": "forward_inventory_ready" if quality["forward_inventory_ready"] else "inventory_not_forward_ready",
            }
        )
    return rows


def _family_coverage(catalog: pd.DataFrame, now: pd.Timestamp) -> list[dict[str, Any]]:
    registry = market_family_registry()
    rows: list[dict[str, Any]] = []
    for family in ACTIVE_FAMILIES:
        spec = registry[family]
        subset = catalog[catalog["market_family"] == family] if not catalog.empty else pd.DataFrame()
        inventory = _inventory_quality(subset, now)
        upcoming_24h = int(inventory["next_24h_markets"])
        active_markets = int(inventory["active_markets"])
        parseable_markets = int((subset.get("parse_status", pd.Series(dtype=str)) == "parseable").sum()) if not subset.empty else 0
        markets_with_tokens = int(subset["clob_token_ids_json"].map(lambda value: len(_json_list(value)) > 0).sum()) if not subset.empty else 0
        coverage_status = "inventory_watch"
        if inventory["future_active_markets"] <= 0:
            coverage_status = "inventory_not_forward_ready"
        elif markets_with_tokens > 0:
            coverage_status = "capture_ready"
        if upcoming_24h >= math.ceil(TARGET_FORWARD_DECISIONS_PER_DAY / CONSERVATIVE_PICK_RATE_HIGH):
            coverage_status = "inventory_ready_for_100_decisions_day"
        rows.append(
            {
                "market_family": family,
                "benchmark_id": spec.benchmark_id,
                "sport": spec.sport,
                "market_type": spec.market_type,
                "model_mode": spec.model_mode,
                "state": spec.initial_state,
                "markets": int(len(subset)),
                "unique_events": int(subset["event_slug"].nunique()) if not subset.empty else 0,
                "active_markets": active_markets,
                "parseable_markets": parseable_markets,
                "markets_with_clob_tokens": markets_with_tokens,
                "upcoming_24h": upcoming_24h,
                **inventory,
                "coverage_status": coverage_status,
                "can_emit_picks": False,
                "pick_blocker": "family_scoring_not_enabled_in_v1",
                "legacy_slice_markets": int(subset.get("legacy_slice", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if not subset.empty else 0,
                "global_slice_markets": int((subset.get("legacy_slice", pd.Series(dtype=int)).fillna(0).astype(int) == 0).sum()) if not subset.empty else 0,
                "market_subtype_counts": subset["market_subtype"].value_counts().to_dict() if not subset.empty and "market_subtype" in subset.columns else {},
            }
        )
    for family in DEFERRED_FAMILIES:
        subset = catalog[catalog["market_family"] == family] if not catalog.empty else pd.DataFrame()
        spec = registry[family]
        rows.append(
            {
                "market_family": family,
                "benchmark_id": spec.benchmark_id,
                "sport": spec.sport,
                "market_type": spec.market_type,
                "model_mode": spec.model_mode,
                "state": STATUS_DEFERRED,
                "markets": int(len(subset)),
                "unique_events": int(subset["event_slug"].nunique()) if not subset.empty else 0,
                "active_markets": 0,
                "parseable_markets": 0,
                "markets_with_clob_tokens": 0,
                "upcoming_24h": 0,
                "coverage_status": "deferred",
                "can_emit_picks": False,
                "pick_blocker": "deferred_market_family",
            }
        )
    return rows


def discover_multi_market(
    settings: Settings,
    db_path: Path | str | None = None,
    gamma: PolymarketGammaClient | None = None,
    limit_per_query: int = 50,
    now: pd.Timestamp | None = None,
) -> MultiMarketDiscoveryResult:
    now = now or _utcnow()
    database_path = Path(db_path) if db_path else default_multi_market_db_path(settings)
    connection = init_multi_market_db(database_path)
    gamma_client = gamma or PolymarketGammaClient()
    run = create_run_context(settings.paths.runs_dir, "multi_market_discovery")

    events_by_slug: dict[str, dict[str, Any]] = {}
    public_search_event_count = 0
    for spec in market_family_registry().values():
        if spec.initial_state == STATUS_DEFERRED:
            continue
        for query in spec.discovery_queries:
            for event in gamma_client.search_events(query=query, limit=limit_per_query):
                if not isinstance(event, dict):
                    continue
                slug = str(event.get("slug") or event.get("id") or "")
                if slug:
                    enriched = dict(event)
                    enriched.setdefault("_discovery_source", "public_search")
                    public_search_event_count += 1
                    events_by_slug[slug] = enriched

    sports_events, sports_diagnostics = _discover_events_from_sports_series(gamma_client, limit_per_query)
    for event in sports_events:
        slug = str(event.get("slug") or event.get("id") or "")
        if slug:
            events_by_slug[slug] = event

    rows: list[dict[str, Any]] = []
    raw_event_rows: list[dict[str, Any]] = []
    raw_market_rows: list[dict[str, Any]] = []
    lane_link_rows: list[dict[str, Any]] = []
    unclassified = 0
    for event in events_by_slug.values():
        first_parsed: ParsedMarket | None = None
        for market in _iter_event_markets(event):
            parsed = classify_market(event, market)
            raw_market_rows.append(_raw_market_row(event, market, parsed, now))
            if parsed is None:
                unclassified += 1
                continue
            market_id = _market_id(market)
            if not market_id:
                unclassified += 1
                continue
            first_parsed = first_parsed or parsed
            row = _catalog_row(event, market, parsed, now)
            rows.append(row)
            lane_link_rows.append(_lane_link_row(row, now))
        raw_event_rows.append(_raw_event_row(event, first_parsed, now))

    rows = list({row["market_id"]: row for row in rows}.values())
    raw_event_rows = list({row["event_id"]: row for row in raw_event_rows if row["event_id"]}.values())
    raw_market_rows = list({row["market_id"]: row for row in raw_market_rows if row["market_id"]}.values())
    lane_link_rows = list({row["link_id"]: row for row in lane_link_rows}.values())
    _clear_market_classifications(connection, (row["market_id"] for row in raw_market_rows))
    _upsert_rows(connection, "mm_raw_events", raw_event_rows)
    _upsert_rows(connection, "mm_raw_markets", raw_market_rows)
    _upsert_rows(connection, "mm_market_catalog", rows)
    _upsert_rows(connection, "mm_lane_market_links", lane_link_rows)
    connection.close()
    catalog = pd.DataFrame(rows)
    coverage = _family_coverage(catalog, now)
    inventory_quality = _raw_inventory_quality_rows(catalog, now)

    manifest = market_family_manifest()
    discovery_report = {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "database_path": str(database_path),
        "created_at": now.isoformat(),
        "lane": "multi_market",
        "canonical_benchmark_untouched": True,
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        "events_discovered": int(len(events_by_slug)),
        "public_search_events_seen": int(public_search_event_count),
        "sports_series_events_seen": int(sports_diagnostics.get("events", 0)),
        "sports_series_queries": int(sports_diagnostics.get("series_queries", 0)),
        "discovery_sources": {
            "public_search": {
                "status": "ok",
                "events_seen": int(public_search_event_count),
            },
            "sports_series": sports_diagnostics,
        },
        "raw_events": int(len(raw_event_rows)),
        "raw_markets": int(len(raw_market_rows)),
        "lane_links": int(len(lane_link_rows)),
        "markets_classified": int(len(rows)),
        "markets_unclassified": int(unclassified),
        "active_families": list(ACTIVE_FAMILIES),
        "deferred_families": list(DEFERRED_FAMILIES),
        "note": "Discovery only. No family ROI is mixed with another family and no picks are emitted here.",
    }
    coverage_report = {
        "run_id": run.run_id,
        "created_at": now.isoformat(),
        "global_roi_actionable": False,
        "target_inventory": manifest["estimated_events_needed_per_day"],
        "families": coverage,
    }
    inventory_quality_report = {
        "run_id": run.run_id,
        "created_at": now.isoformat(),
        "database_path": str(database_path),
        "global_roi_actionable": False,
        "policy_reoptimized": False,
        "quality_rule": "Only active markets with game_start_time >= run time count as forward inventory.",
        "lanes": inventory_quality,
    }

    manifest_path = run.run_dir / "market_family_manifest.json"
    discovery_path = run.run_dir / "multi_market_discovery_report.json"
    coverage_path = run.run_dir / "market_family_coverage_report.json"
    inventory_quality_path = run.run_dir / "raw_inventory_quality_report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    discovery_path.write_text(json.dumps(discovery_report, indent=2, ensure_ascii=True), encoding="utf-8")
    coverage_path.write_text(json.dumps(coverage_report, indent=2, ensure_ascii=True), encoding="utf-8")
    inventory_quality_path.write_text(json.dumps(inventory_quality_report, indent=2, ensure_ascii=True), encoding="utf-8")
    if not catalog.empty:
        catalog.to_csv(run.run_dir / "multi_market_catalog.csv", index=False)

    artifacts = {
        "market_family_manifest": manifest_path,
        "multi_market_discovery_report": discovery_path,
        "market_family_coverage_report": coverage_path,
        "raw_inventory_quality_report": inventory_quality_path,
    }
    if not catalog.empty:
        artifacts["multi_market_catalog"] = run.run_dir / "multi_market_catalog.csv"
    return MultiMarketDiscoveryResult(
        run=run,
        database_path=database_path,
        catalog=catalog,
        coverage=pd.DataFrame(coverage),
        summary=discovery_report,
        artifacts=artifacts,
    )


def _frame_from_query(connection: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, connection)


def _best_ask_from_order_book(order_book: dict[str, Any]) -> tuple[float | None, list[Any], list[Any]]:
    asks = order_book.get("asks") or []
    bids = order_book.get("bids") or []
    best_ask: float | None = None
    if isinstance(asks, list):
        for level in asks:
            if not isinstance(level, dict):
                continue
            price = _safe_float(level.get("price"))
            size = _safe_float(level.get("size"))
            if price is None or size is None or size <= 0:
                continue
            if best_ask is None or price < best_ask:
                best_ask = price
    return best_ask, asks if isinstance(asks, list) else [], bids if isinstance(bids, list) else []


def _checkpoint_rows_for_market(
    row: pd.Series,
    clob: PolymarketClobClient,
    timestamp: pd.Timestamp,
    event_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    token_ids = [str(item) for item in _json_list(row.get("clob_token_ids_json"))]
    outcomes = [str(item) for item in _json_list(row.get("outcomes_json"))]
    if not token_ids:
        blockers.append(
            {
                "market_id": str(row.get("market_id")),
                "event_slug": str(row.get("event_slug")),
                "market_family": str(row.get("market_family")),
                "blocker": "missing_clob_token_ids",
            }
        )
        return checkpoints, blockers

    for index, token_id in enumerate(token_ids):
        outcome_name = outcomes[index] if index < len(outcomes) else ""
        try:
            order_book = clob.get_order_book(token_id)
        except Exception as exc:  # pragma: no cover - depends on public API reliability
            blockers.append(
                {
                    "market_id": str(row.get("market_id")),
                    "event_slug": str(row.get("event_slug")),
                    "market_family": str(row.get("market_family")),
                    "asset_id": token_id,
                    "blocker": "book_fetch_failed",
                    "detail": str(exc),
                }
            )
            continue
        best_ask, asks, bids = _best_ask_from_order_book(order_book)
        checkpoint_id = f"{token_id}:{timestamp.isoformat()}:{event_type}"
        checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "asset_id": token_id,
                "market_id": str(row.get("market_id")),
                "event_slug": str(row.get("event_slug")),
                "timestamp": timestamp.isoformat(),
                "event_type": event_type,
                "outcome_name": outcome_name,
                "top_ask": best_ask,
                "asks_json": _json_dumps(asks),
                "bids_json": _json_dumps(bids),
                "source": "clob_rest",
                "raw_json": _json_dumps(order_book),
            }
        )
    return checkpoints, blockers


def _ledger_rows(catalog: pd.DataFrame, now: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    registry = market_family_registry()
    for _, row in catalog.iterrows():
        family = str(row.get("market_family"))
        spec = registry.get(family)
        if spec is None:
            continue
        if spec.model_mode == "capture_and_baseline_only":
            status = "capture_only"
            blocker = "model_not_ready"
        elif spec.model_mode.endswith("_pending"):
            status = "shadow_collect_only"
            blocker = "family_scoring_adapter_pending"
        else:
            status = "shadow_collect_only"
            blocker = "policy_not_enabled_for_family"
        rows.append(
            {
                "decision_id": f"{family}:{row.get('market_id')}:capture-ledger",
                "market_id": str(row.get("market_id")),
                "event_slug": str(row.get("event_slug")),
                "market_family": family,
                "benchmark_id": str(row.get("benchmark_id")),
                "decision_time": "",
                "decision_status": status,
                "blocker": blocker,
                "selected_outcome": "",
                "model_mode": spec.model_mode,
                "policy_mode": "policy_disabled_until_family_model_and_sample_ready",
                "created_at": now.isoformat(),
            }
        )
    return rows


def _lane_ledger_rows(
    catalog: pd.DataFrame,
    now: pd.Timestamp,
    readiness_by_lane: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    registry = market_lane_registry()
    for _, row in catalog.iterrows():
        lane_id = str(row.get("lane_id") or row.get("market_family"))
        spec = registry.get(lane_id)
        if spec is None or spec.reference_only:
            continue
        readiness = readiness_by_lane.get(lane_id) if readiness_by_lane else None
        if spec.model_mode == "capture_and_baseline_only":
            status = "capture_only"
            blocker = "model_not_ready"
        elif spec.model_mode.endswith("_pending"):
            status = "shadow_collect_only"
            blocker = "family_scoring_adapter_pending"
        elif not _lane_can_emit_picks(spec, readiness):
            status = "shadow_collect_only"
            blocker = "lane_contract_not_ready"
        else:
            status = "candidate_ready"
            blocker = ""
        rows.append(
            {
                "decision_id": f"{lane_id}:{row.get('market_id')}:capture-ledger",
                "lane_id": lane_id,
                "market_id": str(row.get("market_id")),
                "event_slug": str(row.get("event_slug")),
                "benchmark_id": str(row.get("benchmark_id")),
                "decision_time": "",
                "decision_status": status,
                "blocker": blocker,
                "selected_outcome": "",
                "model_mode": spec.model_mode,
                "policy_mode": "policy_ready" if readiness and readiness.get("policy_ready") else spec.policy_mode,
                "market_subtype": str(row.get("market_subtype") or spec.market_subtype),
                "legacy_slice": int(row.get("legacy_slice") or 0),
                "created_at": now.isoformat(),
            }
        )
    return rows


def _selected_lane_ledger_rows(
    spec: MarketLaneSpec,
    candidate_rows: pd.DataFrame,
    now: pd.Timestamp,
    policy_mode: str,
) -> list[dict[str, Any]]:
    if not get_lane_governance(spec.lane_id).can_emit_shadow_picks:
        return []
    if candidate_rows.empty or "candidate_status" not in candidate_rows.columns:
        return []
    selected = candidate_rows[candidate_rows["candidate_status"].astype(str).eq("selected_candidate")].copy()
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        selection = str(row.get("selection") or "")
        market_id = str(row.get("market_id") or "")
        event_slug = str(row.get("event_slug") or "")
        rows.append(
            {
                "decision_id": f"{spec.lane_id}:{event_slug}:{market_id}:{selection}:valid-forward",
                "lane_id": spec.lane_id,
                "market_id": market_id,
                "event_slug": event_slug,
                "benchmark_id": spec.benchmark_id,
                "decision_time": now.isoformat(),
                "decision_status": "valid_forward_sample",
                "blocker": "",
                "selected_outcome": selection,
                "model_mode": spec.model_mode,
                "policy_mode": policy_mode,
                "market_subtype": str(row.get("market_subtype") or spec.market_subtype),
                "legacy_slice": int(row.get("legacy_slice") or 0),
                "created_at": now.isoformat(),
                "game_start_time": str(row.get("game_start_time") or ""),
                "contract_outcome": str(row.get("contract_outcome") or ""),
                "asset_id": str(row.get("asset_id") or ""),
                "book_timestamp": str(row.get("book_timestamp") or ""),
                "top_ask": _safe_float(row.get("top_ask")),
                "quoted_odds": _safe_float(row.get("quoted_odds")),
                "model_prob": _safe_float(row.get("model_prob")),
                "probability_source": str(row.get("probability_source") or ""),
                "edge": _safe_float(row.get("edge")),
                "ev": _safe_float(row.get("ev")),
            }
        )
    return rows


def _lane_model_ready(spec: MarketLaneSpec) -> bool:
    return spec.model_mode in {
        "model_ready",
        "football_goal_model_adapter_ready",
        "poisson_goal_lambdas_adapter_ready",
    }


def _lane_policy_ready(spec: MarketLaneSpec) -> bool:
    return spec.policy_mode == "policy_ready"


def _resolve_path_from_pointer(pointer_path: Path, settings: Settings) -> Path | None:
    if not pointer_path.exists():
        return None
    raw = pointer_path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        candidate = settings.paths.root / path
        path = candidate if candidate.exists() else settings.paths.outputs_dir / path
    return path if path.exists() else None


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _validate_native_lane_policy(spec: MarketLaneSpec, payload: dict[str, Any]) -> tuple[bool, str]:
    if not payload:
        return False, "empty_policy_bundle"
    if str(payload.get("lane_id") or "") != spec.lane_id:
        return False, "policy_lane_mismatch"
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    if not policy:
        return False, "policy_object_missing"
    allowed = {str(item) for item in policy.get("allowed_outcomes", []) or []}
    if allowed and not allowed.issubset(set(spec.outcome_schema)):
        return False, "policy_outcomes_not_in_lane_schema"
    market_subtypes = {str(item) for item in policy.get("allowed_market_subtypes", []) or []}
    if spec.lane_id == "football_goals_core" and market_subtypes and not market_subtypes.issubset({"total_goals", "btts"}):
        return False, "policy_market_subtypes_not_supported"
    return True, ""


def _lane_policy_context(settings: Settings, spec: MarketLaneSpec) -> dict[str, Any]:
    """Resolve a lane-specific frozen policy contract without optimizing anything."""

    if _lane_policy_ready(spec):
        return {
            "policy_ready": True,
            "policy_mode_effective": "policy_ready",
            "policy_activation_status": "policy_ready",
            "policy_transfer_mode": "native_lane_policy",
            "policy_bundle_path": "",
            "source_policy_bundle_path": "",
            "policy_reoptimized": False,
            "policy_blocker": "",
            "policy": {},
            "probability_source": "raw",
        }

    lane_policy_path = _lane_output_dir(settings, spec.lane_id) / "policy_bundle.json"
    lane_policy_payload = _load_json_file(lane_policy_path) if lane_policy_path.exists() else {}
    native_valid, native_blocker = _validate_native_lane_policy(spec, lane_policy_payload)
    if native_valid:
        policy = lane_policy_payload.get("policy") if isinstance(lane_policy_payload.get("policy"), dict) else {}
        return {
            "policy_ready": True,
            "policy_mode_effective": "policy_ready",
            "policy_activation_status": str(lane_policy_payload.get("policy_activation_status") or "policy_ready"),
            "policy_transfer_mode": str(lane_policy_payload.get("policy_transfer_mode") or "native_lane_policy"),
            "policy_bundle_path": str(lane_policy_path),
            "source_policy_bundle_path": str(lane_policy_path),
            "source_policy_bundle": lane_policy_payload,
            "policy_reoptimized": False,
            "policy_blocker": "",
            "policy": policy,
            "probability_source": str(lane_policy_payload.get("probability_source", "raw")),
        }
    if lane_policy_path.exists() and spec.lane_id != "football_1x2_global":
        return {
            "policy_ready": False,
            "policy_mode_effective": spec.policy_mode,
            "policy_activation_status": "native_policy_invalid",
            "policy_transfer_mode": "native_lane_policy",
            "policy_bundle_path": str(lane_policy_path),
            "source_policy_bundle_path": str(lane_policy_path),
            "source_policy_bundle": lane_policy_payload,
            "policy_reoptimized": False,
            "policy_blocker": native_blocker or "native_policy_invalid",
            "policy": lane_policy_payload.get("policy") if isinstance(lane_policy_payload.get("policy"), dict) else {},
            "probability_source": str(lane_policy_payload.get("probability_source", "raw")),
        }

    if spec.lane_id != "football_1x2_global":
        return {
            "policy_ready": False,
            "policy_mode_effective": spec.policy_mode,
            "policy_activation_status": "lane_policy_pending",
            "policy_transfer_mode": "none",
            "policy_bundle_path": "",
            "source_policy_bundle_path": "",
            "policy_reoptimized": False,
            "policy_blocker": "family_specific_policy_required",
            "policy": {},
            "probability_source": "raw",
        }

    pointer = settings.paths.outputs_dir / "latest_polymarket_policy.txt"
    source_path = _resolve_path_from_pointer(pointer, settings)
    if source_path is None:
        return {
            "policy_ready": False,
            "policy_mode_effective": spec.policy_mode,
            "policy_activation_status": "legacy_frozen_policy_missing",
            "policy_transfer_mode": "legacy_frozen_policy_bridge",
            "policy_bundle_path": "",
            "source_policy_bundle_path": "",
            "policy_reoptimized": False,
            "policy_blocker": "latest_polymarket_policy_missing",
            "policy": {},
            "probability_source": "raw",
        }
    source_payload = _load_json_file(source_path)
    policy = source_payload.get("policy") if isinstance(source_payload.get("policy"), dict) else {}
    if not policy:
        return {
            "policy_ready": False,
            "policy_mode_effective": spec.policy_mode,
            "policy_activation_status": "legacy_frozen_policy_invalid",
            "policy_transfer_mode": "legacy_frozen_policy_bridge",
            "policy_bundle_path": "",
            "source_policy_bundle_path": str(source_path),
            "policy_reoptimized": False,
            "policy_blocker": "source_policy_missing_policy_object",
            "policy": {},
            "probability_source": "raw",
        }

    allowed_outcomes = {str(item) for item in policy.get("allowed_outcomes", []) or []}
    if allowed_outcomes and not allowed_outcomes.issubset(set(spec.outcome_schema)):
        return {
            "policy_ready": False,
            "policy_mode_effective": spec.policy_mode,
            "policy_activation_status": "legacy_frozen_policy_incompatible",
            "policy_transfer_mode": "legacy_frozen_policy_bridge",
            "policy_bundle_path": "",
            "source_policy_bundle_path": str(source_path),
            "policy_reoptimized": False,
            "policy_blocker": "source_policy_outcomes_not_in_lane_schema",
            "policy": policy,
            "probability_source": str(source_payload.get("probability_source", "raw")),
        }

    return {
        "policy_ready": True,
        "policy_mode_effective": "policy_ready",
        "policy_activation_status": "policy_ready",
        "policy_transfer_mode": "legacy_frozen_policy_bridge",
        "policy_bundle_path": "",
        "source_policy_bundle_path": str(source_path),
        "source_policy_bundle": source_payload,
        "policy_reoptimized": False,
        "policy_blocker": "",
        "policy": policy,
        "probability_source": str(source_payload.get("probability_source", "raw")),
    }


def _lane_policy_bundle_payload(spec: MarketLaneSpec, policy_context: dict[str, Any]) -> dict[str, Any]:
    source_payload = policy_context.get("source_policy_bundle")
    return {
        "lane_id": spec.lane_id,
        "benchmark_id": spec.benchmark_id,
        "sport": spec.sport,
        "decision_type": spec.decision_type,
        "market_subtype": spec.market_subtype,
        "outcome_schema": list(spec.outcome_schema),
        "legacy_included": list(spec.legacy_included),
        "policy_ready": bool(policy_context.get("policy_ready")),
        "policy_mode": policy_context.get("policy_mode_effective", spec.policy_mode),
        "policy_activation_status": policy_context.get("policy_activation_status"),
        "policy_transfer_mode": policy_context.get("policy_transfer_mode"),
        "policy_reoptimized": False,
        "thresholds_changed": False,
        "scopes_changed": False,
        "global_roi_actionable": False,
        "source_policy_bundle_path": policy_context.get("source_policy_bundle_path", ""),
        "source_benchmark_id": "football_1x2_canonical_legacy" if policy_context.get("policy_transfer_mode") == "legacy_frozen_policy_bridge" else spec.benchmark_id,
        "probability_source": policy_context.get("probability_source", "raw"),
        "policy": policy_context.get("policy", {}),
        "source_policy_metadata": {
            "source_mode": source_payload.get("source_mode") if isinstance(source_payload, dict) else None,
            "model_variant": source_payload.get("model_variant") if isinstance(source_payload, dict) else None,
            "bundle_status": source_payload.get("bundle_status") if isinstance(source_payload, dict) else None,
            "research_status": source_payload.get("research_status") if isinstance(source_payload, dict) else None,
            "promotion_eligibility": source_payload.get("promotion_eligibility") if isinstance(source_payload, dict) else None,
        },
        "discipline": {
            "no_threshold_optimization": True,
            "no_scope_optimization": True,
            "no_cross_lane_roi": True,
            "global_roi_actionable": False,
            "legacy_benchmark_untouched": True,
        },
    }


def _lane_settlement_ready(spec: MarketLaneSpec) -> bool:
    return spec.status not in {STATUS_DEFERRED, STATUS_REFERENCE_ONLY} and "deferred" not in spec.settlement_rule


def _lane_sample_contract_ready(spec: MarketLaneSpec) -> bool:
    required = {"valid_forward_decisions", "settled_unique_decisions", "fresh_book_rate"}
    return required.issubset(set(spec.sample_requirements))


def _lane_can_emit_picks(spec: MarketLaneSpec, readiness: dict[str, Any] | None = None) -> bool:
    if not get_lane_governance(spec.lane_id).can_emit_shadow_picks:
        return False
    if spec.reference_only or spec.status in {STATUS_DEFERRED, STATUS_REFERENCE_ONLY}:
        return False
    if readiness is not None:
        return all(
            bool(readiness.get(key))
            for key in (
                "coverage_ready",
                "model_ready",
                "policy_ready",
                "settlement_ready",
                "sample_contract_ready",
                "decision_inventory_ready",
            )
        )
    return _lane_model_ready(spec) and _lane_policy_ready(spec) and _lane_settlement_ready(spec) and _lane_sample_contract_ready(spec)


def _raw_books_for_markets(connection: sqlite3.Connection, market_ids: set[str]) -> pd.DataFrame:
    if not market_ids:
        return pd.DataFrame()
    books = _frame_from_query(connection, "SELECT * FROM mm_raw_orderbooks")
    if books.empty or "market_id" not in books.columns:
        return pd.DataFrame()
    return books[books["market_id"].astype(str).isin(market_ids)].copy()


def _catalog_policy_ready_mask(settings: Settings, catalog: pd.DataFrame) -> pd.Series:
    if catalog.empty:
        return pd.Series(dtype=bool)
    ready_lanes: set[str] = set()
    for lane in catalog["lane_id"].fillna(catalog["market_family"]).dropna().astype(str).unique():
        try:
            spec = get_market_lane_spec(lane)
        except KeyError:
            continue
        policy_context = _lane_policy_context(settings, spec)
        if bool(policy_context.get("policy_ready")) and _lane_model_ready(spec):
            ready_lanes.add(lane)
    return catalog["lane_id"].fillna(catalog["market_family"]).astype(str).isin(ready_lanes)


def _build_raw_capture_plan(
    catalog: pd.DataFrame,
    existing_books: pd.DataFrame,
    now: pd.Timestamp,
    capture_priority: str,
    freshness_seconds: int,
    max_stale_age_seconds: int | None = None,
    max_events: int | None = None,
    max_markets: int | None = None,
) -> pd.DataFrame:
    if capture_priority not in CAPTURE_PRIORITY_VALUES:
        raise ValueError(f"capture_priority invalido: {capture_priority}. Usa uno de {sorted(CAPTURE_PRIORITY_VALUES)}")
    columns = [
        "market_id",
        "event_slug",
        "lane_id",
        "market_family",
        "market_subtype",
        "game_start_time",
        "capture_priority",
        "previous_book_status",
        "latest_book_timestamp",
        "latest_book_age_seconds",
        "has_clob_tokens",
        "clob_token_count",
        "eligible_by_priority",
        "selected_for_capture",
    ]
    if catalog.empty:
        return pd.DataFrame(columns=columns)

    latest_by_market: dict[str, pd.Series] = {}
    if not existing_books.empty and "market_id" in existing_books.columns:
        books = existing_books.copy()
        books["_timestamp"] = pd.to_datetime(books.get("timestamp", pd.Series(dtype=str)), utc=True, errors="coerce")
        for _, row in books.sort_values("_timestamp").drop_duplicates("market_id", keep="last").iterrows():
            latest_by_market[str(row.get("market_id") or "")] = row

    rows: list[dict[str, Any]] = []
    for _, market in catalog.iterrows():
        market_id = str(market.get("market_id") or "")
        tokens = _json_list(market.get("clob_token_ids_json"))
        latest = latest_by_market.get(market_id)
        latest_ts = pd.to_datetime(latest.get("_timestamp"), utc=True, errors="coerce") if latest is not None else pd.NaT
        age = float((now - latest_ts).total_seconds()) if latest is not None and not pd.isna(latest_ts) else None
        if not tokens:
            previous_status = "no_clob_token"
            priority_label = "not_captureable"
        elif latest is None:
            previous_status = "capture_not_attempted"
            priority_label = "missing"
        elif age is not None and age > int(freshness_seconds):
            previous_status = "stale_book"
            priority_label = "stale"
        else:
            previous_status = "fresh_book"
            priority_label = "fresh"

        eligible = False
        if tokens:
            if capture_priority == "all":
                eligible = True
            elif capture_priority == "missing_or_stale":
                eligible = previous_status in {"capture_not_attempted", "stale_book"}
            elif capture_priority == "missing_only":
                eligible = previous_status == "capture_not_attempted"
            elif capture_priority == "stale_only":
                eligible = previous_status == "stale_book"
        if previous_status == "stale_book" and max_stale_age_seconds is not None and age is not None:
            eligible = eligible and age >= int(max_stale_age_seconds)

        rows.append(
            {
                "market_id": market_id,
                "event_slug": str(market.get("event_slug") or ""),
                "lane_id": str(market.get("lane_id") or market.get("market_family") or ""),
                "market_family": str(market.get("market_family") or ""),
                "market_subtype": str(market.get("market_subtype") or ""),
                "game_start_time": str(market.get("game_start_time") or ""),
                "capture_priority": priority_label,
                "previous_book_status": previous_status,
                "latest_book_timestamp": "" if latest is None or pd.isna(latest_ts) else latest_ts.isoformat(),
                "latest_book_age_seconds": age,
                "has_clob_tokens": bool(tokens),
                "clob_token_count": int(len(tokens)),
                "eligible_by_priority": bool(eligible),
                "selected_for_capture": False,
            }
        )

    plan = pd.DataFrame(rows, columns=columns)
    if plan.empty:
        return plan
    starts = pd.to_datetime(plan["game_start_time"], utc=True, errors="coerce")
    priority_rank = plan["capture_priority"].map({"missing": 0, "stale": 1, "fresh": 2, "not_captureable": 9}).fillna(9)
    ordered = plan.assign(_start=starts, _priority_rank=priority_rank).sort_values(
        ["_priority_rank", "_start", "event_slug", "market_id"],
        na_position="last",
    )
    eligible = ordered[ordered["eligible_by_priority"].astype(bool)]
    selected_market_ids: set[str] = set()
    if not eligible.empty:
        selected = eligible
        if max_events is not None and int(max_events) > 0:
            selected_events = selected.drop_duplicates("event_slug").head(int(max_events))["event_slug"].astype(str).tolist()
            selected = selected[selected["event_slug"].astype(str).isin(set(selected_events))]
        if max_markets is not None and int(max_markets) > 0:
            selected = selected.head(int(max_markets))
        selected_market_ids = set(selected["market_id"].astype(str).tolist())
    plan.loc[plan["market_id"].astype(str).isin(selected_market_ids), "selected_for_capture"] = True
    return (
        plan.assign(_start=starts, _priority_rank=priority_rank)
        .sort_values(["_priority_rank", "_start", "event_slug", "market_id"], na_position="last")
        .drop(columns=["_start", "_priority_rank"])
        .reset_index(drop=True)
    )


def _raw_capture_plan_summary(plan: pd.DataFrame, checkpoints: pd.DataFrame, blockers: pd.DataFrame) -> dict[str, Any]:
    captured_ids = set(checkpoints.get("market_id", pd.Series(dtype=str)).dropna().astype(str)) if not checkpoints.empty else set()
    failed_ids = set(
        blockers[blockers.get("blocker", pd.Series(dtype=str)).astype(str).eq("book_fetch_failed")]["market_id"].dropna().astype(str)
    ) if not blockers.empty and "blocker" in blockers.columns and "market_id" in blockers.columns else set()
    selected = plan[plan["selected_for_capture"].astype(bool)] if not plan.empty else pd.DataFrame()
    lanes: list[dict[str, Any]] = []
    if not plan.empty:
        for lane, group in plan.groupby("lane_id", dropna=False):
            selected_group = group[group["selected_for_capture"].astype(bool)]
            lane_market_ids = set(group["market_id"].astype(str))
            selected_ids = set(selected_group["market_id"].astype(str))
            lanes.append(
                {
                    "lane_id": str(lane),
                    "planned_markets": int(len(selected_group)),
                    "planned_missing": int(selected_group["previous_book_status"].eq("capture_not_attempted").sum()),
                    "planned_stale": int(selected_group["previous_book_status"].eq("stale_book").sum()),
                    "planned_fresh": int(selected_group["previous_book_status"].eq("fresh_book").sum()),
                    "captured": int(len(captured_ids & lane_market_ids)),
                    "failed_fetch": int(len(failed_ids & lane_market_ids)),
                    "selected_without_checkpoint": int(len(selected_ids - captured_ids)),
                }
            )
    return {
        "planned_markets": int(len(selected)),
        "planned_missing": int(selected["previous_book_status"].eq("capture_not_attempted").sum()) if not selected.empty else 0,
        "planned_stale": int(selected["previous_book_status"].eq("stale_book").sum()) if not selected.empty else 0,
        "planned_fresh": int(selected["previous_book_status"].eq("fresh_book").sum()) if not selected.empty else 0,
        "captured_markets": int(len(captured_ids)),
        "failed_fetch_markets": int(len(failed_ids)),
        "lanes": lanes,
        "global_roi_actionable": False,
    }


def capture_multi_market(
    settings: Settings,
    db_path: Path | str | None = None,
    clob: PolymarketClobClient | None = None,
    stream_seconds: int = 0,
    lane_id: str | None = None,
    max_markets: int | None = None,
    max_events: int | None = None,
    capture_priority: str = "missing_or_stale",
    freshness_seconds: int = DEFAULT_RAW_CAPTURE_FRESHNESS_SECONDS,
    max_stale_age_seconds: int | None = None,
    include_policy_ready_only: bool = False,
    now: pd.Timestamp | None = None,
) -> MultiMarketCaptureResult:
    now = now or _utcnow()
    database_path = Path(db_path) if db_path else default_multi_market_db_path(settings)
    connection = init_multi_market_db(database_path)
    run = create_run_context(settings.paths.runs_dir, "multi_market_capture")
    clob_client = clob or PolymarketClobClient()

    catalog = _frame_from_query(
        connection,
        """
        SELECT * FROM mm_market_catalog
        WHERE status IN ('capture_ready', 'shadow_collect_only', 'capture_only', 'model_ready')
          AND active = 1
          AND closed = 0
        """,
    )
    catalog = _forward_active_catalog(catalog, now)
    if lane_id:
        require_can_capture(lane_id)
        get_market_lane_spec(lane_id)
        catalog = catalog[catalog["lane_id"].fillna(catalog["market_family"]) == lane_id].copy()
    if not catalog.empty:
        catalog = catalog[_catalog_capture_governance_mask(catalog)].copy()
    if include_policy_ready_only and not catalog.empty:
        catalog = catalog[_catalog_policy_ready_mask(settings, catalog)].copy()
    existing_books = _raw_books_for_markets(connection, set(catalog["market_id"].astype(str))) if not catalog.empty else pd.DataFrame()
    capture_plan = _build_raw_capture_plan(
        catalog=catalog,
        existing_books=existing_books,
        now=now,
        capture_priority=str(capture_priority or "missing_or_stale"),
        freshness_seconds=int(freshness_seconds or DEFAULT_RAW_CAPTURE_FRESHNESS_SECONDS),
        max_stale_age_seconds=max_stale_age_seconds,
        max_events=max_events,
        max_markets=max_markets,
    )
    selected_market_ids = (
        set(capture_plan[capture_plan["selected_for_capture"].astype(bool)]["market_id"].astype(str))
        if not capture_plan.empty
        else set()
    )
    capture_catalog = (
        catalog[catalog["market_id"].astype(str).isin(selected_market_ids)].copy()
        if selected_market_ids and not catalog.empty
        else pd.DataFrame(columns=catalog.columns)
    )
    if not capture_catalog.empty:
        starts = _market_start_series(capture_catalog)
        priority_rank = (
            capture_plan.set_index("market_id")["capture_priority"]
            .map({"missing": 0, "stale": 1, "fresh": 2, "not_captureable": 9})
            .to_dict()
        )
        capture_catalog = (
            capture_catalog.assign(
                _capture_sort_start=starts,
                _capture_priority_rank=capture_catalog["market_id"].astype(str).map(priority_rank).fillna(9),
            )
            .sort_values(["_capture_priority_rank", "_capture_sort_start", "event_slug", "market_id"], na_position="last")
            .drop(columns=["_capture_sort_start", "_capture_priority_rank"])
        )
    if capture_catalog.empty:
        blockers = pd.DataFrame(
            [{
                "blocker": "empty_forward_catalog" if catalog.empty else "nothing_selected_by_capture_priority",
                "detail": "Run discover-market-raw close to upcoming fixtures; historical active markets are ignored."
                if catalog.empty
                else "No forward markets matched capture_priority/freshness filters.",
            }]
        )
        summary = _capture_summary(settings, run, database_path, capture_catalog, pd.DataFrame(), blockers, now)
        summary["capture_lane_filter"] = lane_id or "all"
        summary["capture_max_markets"] = int(max_markets) if max_markets is not None else None
        summary["capture_max_events"] = int(max_events) if max_events is not None else None
        summary["capture_priority"] = str(capture_priority or "missing_or_stale")
        summary["capture_freshness_seconds"] = int(freshness_seconds or DEFAULT_RAW_CAPTURE_FRESHNESS_SECONDS)
        summary["capture_max_stale_age_seconds"] = int(max_stale_age_seconds) if max_stale_age_seconds is not None else None
        summary["include_policy_ready_only"] = bool(include_policy_ready_only)
        summary["raw_capture_plan"] = _raw_capture_plan_summary(capture_plan, pd.DataFrame(), blockers)
        connection.close()
        return _write_capture_artifacts(run, database_path, capture_catalog, pd.DataFrame(), blockers, summary, capture_plan=capture_plan)

    all_checkpoints: list[dict[str, Any]] = []
    all_blockers: list[dict[str, Any]] = []
    deadline = time.monotonic() + max(0, int(stream_seconds))
    sleep_seconds = max(1, min(int(settings.polymarket.checkpoint_interval_seconds), max(1, int(stream_seconds) or 1)))
    first_pass = True
    while first_pass or (stream_seconds > 0 and time.monotonic() < deadline):
        first_pass = False
        timestamp = _utcnow()
        for _, market_row in capture_catalog.iterrows():
            checkpoints, blockers = _checkpoint_rows_for_market(market_row, clob_client, timestamp, "multi_market_rest_checkpoint")
            all_checkpoints.extend(checkpoints)
            all_blockers.extend(blockers)
        if stream_seconds <= 0 or time.monotonic() >= deadline:
            break
        time.sleep(sleep_seconds)

    _upsert_rows(connection, "mm_raw_orderbooks", all_checkpoints)
    ledger_rows = _ledger_rows(capture_catalog, now)
    _upsert_rows(connection, "mm_forward_ledger", ledger_rows)
    lane_ledger_rows = _lane_ledger_rows(capture_catalog, now)
    _upsert_rows(connection, "mm_lane_forward_ledger", lane_ledger_rows)
    connection.close()
    checkpoints_frame = pd.DataFrame(all_checkpoints)
    blockers_frame = pd.DataFrame(all_blockers)
    summary = _capture_summary(settings, run, database_path, capture_catalog, checkpoints_frame, blockers_frame, now)
    summary["capture_lane_filter"] = lane_id or "all"
    summary["capture_max_markets"] = int(max_markets) if max_markets is not None else None
    summary["capture_max_events"] = int(max_events) if max_events is not None else None
    summary["capture_priority"] = str(capture_priority or "missing_or_stale")
    summary["capture_freshness_seconds"] = int(freshness_seconds or DEFAULT_RAW_CAPTURE_FRESHNESS_SECONDS)
    summary["capture_max_stale_age_seconds"] = int(max_stale_age_seconds) if max_stale_age_seconds is not None else None
    summary["include_policy_ready_only"] = bool(include_policy_ready_only)
    summary["raw_capture_plan"] = _raw_capture_plan_summary(capture_plan, checkpoints_frame, blockers_frame)
    return _write_capture_artifacts(
        run,
        database_path,
        capture_catalog,
        checkpoints_frame,
        blockers_frame,
        summary,
        capture_plan=capture_plan,
    )


def _capture_summary(
    settings: Settings,
    run: RunContext,
    database_path: Path,
    catalog: pd.DataFrame,
    checkpoints: pd.DataFrame,
    blockers: pd.DataFrame,
    now: pd.Timestamp,
) -> dict[str, Any]:
    families: list[dict[str, Any]] = []
    registry = market_family_registry()
    for family in ACTIVE_FAMILIES:
        spec = registry[family]
        lane_spec = get_market_lane_spec(family)
        policy_context = _lane_policy_context(settings, lane_spec)
        family_catalog = catalog[catalog["lane_id"].fillna(catalog["market_family"]) == family] if not catalog.empty else pd.DataFrame()
        family_market_ids = set(family_catalog["market_id"].astype(str)) if not family_catalog.empty else set()
        family_books = checkpoints[checkpoints["market_id"].astype(str).isin(family_market_ids)] if not checkpoints.empty else pd.DataFrame()
        family_blockers = blockers[blockers["market_family"] == family] if not blockers.empty and "market_family" in blockers.columns else pd.DataFrame()
        status = spec.initial_state
        if spec.model_mode == "capture_and_baseline_only":
            status = STATUS_CAPTURE_ONLY
        if not family_catalog.empty and not family_books.empty:
            status = STATUS_SHADOW_COLLECT_ONLY
        families.append(
            {
                "market_family": family,
                "lane_id": family,
                "benchmark_id": spec.benchmark_id,
                "sport": spec.sport,
                "sport_id": lane_spec.sport_id,
                "sport_merge_group": lane_spec.sport_merge_group,
                "market_type": spec.market_type,
                "decision_type": lane_spec.decision_type,
                "market_subtype_counts": family_catalog["market_subtype"].value_counts().to_dict() if not family_catalog.empty and "market_subtype" in family_catalog.columns else {},
                "model_mode": spec.model_mode,
                "policy_mode": str(policy_context.get("policy_mode_effective", lane_spec.policy_mode)),
                "policy_activation_status": str(policy_context.get("policy_activation_status", lane_spec.policy_mode)),
                "policy_transfer_mode": str(policy_context.get("policy_transfer_mode", "none")),
                "sample_status": status,
                "markets_in_catalog": int(len(family_catalog)),
                "book_checkpoints": int(len(family_books)),
                "blockers": int(len(family_blockers)),
                "valid_decisions": 0,
                "settled_decisions": 0,
                "fresh_book_rate": None,
                "roi": None,
                "hit_rate": None,
                "can_emit_picks": False,
                "pick_blocker": "raw_capture_only_run_market_lane_next"
                if bool(policy_context.get("policy_ready")) and _lane_model_ready(lane_spec)
                else "family_model_or_policy_not_enabled_in_v1",
                "legacy_slice_markets": int(family_catalog.get("legacy_slice", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if not family_catalog.empty else 0,
            }
        )
    return {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "database_path": str(database_path),
        "created_at": now.isoformat(),
        "lane": "multi_market",
        "canonical_benchmark_untouched": True,
        "architecture": "raw_market_capture_plus_isolated_market_lanes",
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        "global_roi": None,
        "markets_considered": int(len(catalog)),
        "book_checkpoints": int(len(checkpoints)),
        "blockers": int(len(blockers)),
        "families": families,
        "portfolio_readiness": "blocked_until_two_or_more_families_sample_ready",
    }


def _write_capture_artifacts(
    run: RunContext,
    database_path: Path,
    catalog: pd.DataFrame,
    checkpoints: pd.DataFrame,
    blockers: pd.DataFrame,
    summary: dict[str, Any],
    capture_plan: pd.DataFrame | None = None,
) -> MultiMarketCaptureResult:
    sample_path = run.run_dir / "multi_market_forward_sample_report.json"
    ledger_path = run.run_dir / "multi_market_forward_ledger.csv"
    blockers_path = run.run_dir / "multi_market_blockers.csv"
    portfolio_path = run.run_dir / "portfolio_readiness_report.json"
    capture_plan_csv_path = run.run_dir / RAW_CAPTURE_PLAN_CSV_FILENAME
    capture_plan_json_path = run.run_dir / RAW_CAPTURE_PLAN_JSON_FILENAME
    sample_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    portfolio_payload = {
        "run_id": run.run_id,
        "global_roi_actionable": False,
        "ready_families": [
            family["market_family"] for family in summary["families"] if family.get("sample_status") == STATUS_SAMPLE_READY
        ],
        "status": summary["portfolio_readiness"],
    }
    portfolio_path.write_text(json.dumps(portfolio_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    ledger_rows = _lane_ledger_rows(catalog, _utcnow()) if not catalog.empty else []
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "decision_id",
            "lane_id",
            "market_id",
            "event_slug",
            "benchmark_id",
            "decision_time",
            "decision_status",
            "blocker",
            "selected_outcome",
            "model_mode",
            "policy_mode",
            "market_subtype",
            "legacy_slice",
            "created_at",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ledger_rows)
    if blockers.empty:
        blockers = pd.DataFrame(columns=["market_id", "event_slug", "market_family", "asset_id", "blocker", "detail"])
    blockers.to_csv(blockers_path, index=False)
    if capture_plan is None:
        capture_plan = pd.DataFrame(
            columns=[
                "market_id",
                "event_slug",
                "lane_id",
                "capture_priority",
                "previous_book_status",
                "latest_book_age_seconds",
                "has_clob_tokens",
                "selected_for_capture",
            ]
        )
    capture_plan.to_csv(capture_plan_csv_path, index=False)
    capture_plan_json_path.write_text(
        json.dumps(summary.get("raw_capture_plan", {}), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if not checkpoints.empty:
        checkpoints.to_csv(run.run_dir / "multi_market_book_checkpoints.csv", index=False)

    artifacts = {
        "multi_market_forward_sample_report": sample_path,
        "multi_market_forward_ledger": ledger_path,
        "multi_market_blockers": blockers_path,
        "portfolio_readiness_report": portfolio_path,
        "raw_capture_plan_csv": capture_plan_csv_path,
        "raw_capture_plan_json": capture_plan_json_path,
    }
    if not checkpoints.empty:
        artifacts["multi_market_book_checkpoints"] = run.run_dir / "multi_market_book_checkpoints.csv"
    return MultiMarketCaptureResult(
        run=run,
        database_path=database_path,
        catalog=catalog,
        checkpoints=checkpoints,
        blockers=blockers,
        summary=summary,
        artifacts=artifacts,
    )


def report_multi_market(run_dir: Path | str) -> tuple[dict[str, Any], str]:
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"No encuentro el run multi-market: {root}")
    sample_path = root / "multi_market_forward_sample_report.json"
    coverage_path = root / "market_family_coverage_report.json"
    discovery_path = root / "multi_market_discovery_report.json"
    if sample_path.exists():
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        lines = [
            "Multi-market capture report",
            f"- run: {payload.get('run_dir', root)}",
            f"- database: {payload.get('database_path')}",
            "- global_roi_actionable: false",
            f"- markets_considered: {payload.get('markets_considered', 0)}",
            f"- book_checkpoints: {payload.get('book_checkpoints', 0)}",
            f"- portfolio_readiness: {payload.get('portfolio_readiness')}",
        ]
        for family in payload.get("families", []):
            lines.append(
                "- "
                f"{family.get('market_family')}: status={family.get('sample_status')} "
                f"markets={family.get('markets_in_catalog')} books={family.get('book_checkpoints')} "
                f"valid_decisions={family.get('valid_decisions')} blocker={family.get('pick_blocker')}"
            )
        return payload, "\n".join(lines)
    if coverage_path.exists():
        payload = json.loads(coverage_path.read_text(encoding="utf-8"))
        lines = [
            "Multi-market discovery coverage",
            f"- run: {payload.get('run_id')}",
            "- global_roi_actionable: false",
            f"- target_inventory: {payload.get('target_inventory')}",
        ]
        for family in payload.get("families", []):
            lines.append(
                "- "
                f"{family.get('market_family')}: status={family.get('coverage_status')} "
                f"markets={family.get('markets')} events={family.get('unique_events')} "
                f"upcoming_24h={family.get('upcoming_24h')} blocker={family.get('pick_blocker')}"
            )
        return payload, "\n".join(lines)
    if discovery_path.exists():
        payload = json.loads(discovery_path.read_text(encoding="utf-8"))
        return payload, json.dumps(payload, indent=2)
    raise FileNotFoundError(f"El run no contiene artefactos multi-market reconocidos: {root}")


def _lane_output_dir(settings: Settings, lane_id: str) -> Path:
    lane_dir = settings.paths.outputs_dir / "lanes" / lane_id
    lane_dir.mkdir(parents=True, exist_ok=True)
    return lane_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _lane_catalog(connection: sqlite3.Connection, lane_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT c.*
        FROM mm_lane_market_links l
        JOIN mm_market_catalog c ON c.market_id = l.market_id
        WHERE l.lane_id = ?
        """,
        connection,
        params=(lane_id,),
    )


def _lane_raw_orderbooks(connection: sqlite3.Connection, lane_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT b.*
        FROM mm_raw_orderbooks b
        JOIN mm_lane_market_links l ON l.market_id = b.market_id
        WHERE l.lane_id = ?
        """,
        connection,
        params=(lane_id,),
    )


def _lane_raw_settlements(connection: sqlite3.Connection, lane_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT s.*
        FROM mm_raw_settlements s
        JOIN mm_lane_market_links l ON l.market_id = s.market_id
        WHERE l.lane_id = ?
        """,
        connection,
        params=(lane_id,),
    )


def _lane_forward_ledger(connection: sqlite3.Connection, lane_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM mm_lane_forward_ledger WHERE lane_id = ?",
        connection,
        params=(lane_id,),
    )


def _latest_book_coverage(catalog: pd.DataFrame, books: pd.DataFrame) -> dict[str, Any]:
    market_count = int(len(catalog))
    if catalog.empty or books.empty:
        return {
            "raw_orderbook_checkpoints": int(len(books)),
            "markets_with_raw_books": 0,
            "book_market_coverage_rate": 0.0,
            "latest_book_timestamp": "",
        }
    markets_with_books = int(books["market_id"].astype(str).nunique())
    latest = pd.to_datetime(books.get("timestamp", pd.Series(dtype=str)), utc=True, errors="coerce").max()
    return {
        "raw_orderbook_checkpoints": int(len(books)),
        "markets_with_raw_books": markets_with_books,
        "book_market_coverage_rate": float(markets_with_books / market_count) if market_count else 0.0,
        "latest_book_timestamp": "" if pd.isna(latest) else latest.isoformat(),
    }


def _lane_subtype_readiness(spec: MarketLaneSpec, catalog: pd.DataFrame) -> dict[str, Any]:
    counts = catalog["market_subtype"].value_counts().to_dict() if not catalog.empty and "market_subtype" in catalog.columns else {}
    required: tuple[str, ...] = ()
    if spec.lane_id == "football_goals_core":
        required = ("total_goals", "btts")
    missing = [item for item in required if int(counts.get(item, 0)) <= 0]
    return {
        "market_subtype_counts": counts,
        "required_subtypes": list(required),
        "missing_subtypes": missing,
        "subtype_status": "ready" if not missing else "partial_coverage",
    }


def _row_text(row: pd.Series) -> str:
    return " ".join(
        str(row.get(column) or "")
        for column in ("event_title", "event_slug", "market_slug", "question", "market_subtype")
    ).lower()


def _clean_match_team(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("-", " ")
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def _clean_event_team_segment(value: str) -> str:
    text = str(value or "")
    for separator in (" - ", " | "):
        if separator in text:
            text = text.split(separator, 1)[0]
    text = re.sub(r"\b(more markets|halftime result|half time result)\b", "", text, flags=re.IGNORECASE)
    return _clean_match_team(text)


def _event_teams_from_row(row: pd.Series) -> tuple[str, str] | tuple[None, None]:
    raw = str(row.get("event_title") or row.get("event_slug") or "")
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    text = raw.replace("-", " ")
    for separator in (" vs. ", " vs ", " v. ", " v ", " against "):
        if separator in text.lower():
            pattern = separator.strip()
            left, right = text.lower().split(separator, 1)
            # Re-split the original-ish text to preserve multi-word teams before cleaning.
            parts = text.split(pattern, 1) if pattern in text else [left, right]
            if len(parts) != 2:
                parts = [left, right]
            home = _clean_event_team_segment(parts[0])
            away = _clean_event_team_segment(parts[1])
            if home and away:
                return home, away
    return None, None


def _football_1x2_market_role(row: pd.Series) -> str:
    text = _row_text(row)
    outcomes = [str(item).strip().lower() for item in _json_list(row.get("outcomes_json"))]
    if "draw" in outcomes and len(outcomes) >= 3:
        return "three_way_1x2"
    if _binary_yes_no(outcomes) and _contains_any(text, ("end in a draw", "draw?", " be a draw")):
        return "draw_binary"
    if _binary_yes_no(outcomes) and _has_match_context(text) and _contains_any(text, (" win", "winner", " beat ")):
        return "team_win_binary"
    return "unresolved_role"


def _football_1x2_selection(row: pd.Series, contract_outcome: str) -> str:
    role = _football_1x2_market_role(row)
    outcome = _clean_match_team(contract_outcome)
    if role == "draw_binary":
        return "draw" if outcome == "yes" else ""
    home, away = _event_teams_from_row(row)
    if role == "team_win_binary":
        if outcome != "yes":
            return ""
        question = _clean_match_team(str(row.get("question") or row.get("market_slug") or ""))
        if home and (question.startswith(f"will {home} win") or f"{home} win against" in question):
            return "home"
        if away and (question.startswith(f"will {away} win") or f"{away} win against" in question):
            return "away"
        if home and away and (home in question) ^ (away in question):
            return "home" if home in question else "away"
        return ""
    if role == "three_way_1x2":
        if outcome == "draw":
            return "draw"
        if home and home in outcome:
            return "home"
        if away and away in outcome:
            return "away"
    return ""


def _football_goals_selection(row: pd.Series, contract_outcome: str) -> str:
    subtype = str(row.get("market_subtype") or "").strip().lower()
    outcome = _clean_match_team(contract_outcome)
    text = _row_text(row)
    if subtype == "btts":
        if outcome == "yes":
            return "btts_yes"
        if outcome == "no":
            return "btts_no"
    if subtype == "total_goals":
        if outcome in {"over", "under"}:
            return outcome
        if "over" in text:
            return "over" if outcome == "yes" else "under" if outcome == "no" else ""
        if "under" in text:
            return "under" if outcome == "yes" else "over" if outcome == "no" else ""
    return ""


def _selection_for_lane(spec: MarketLaneSpec, row: pd.Series, contract_outcome: str) -> str:
    if spec.lane_id == "football_1x2_global":
        return _football_1x2_selection(row, contract_outcome)
    if spec.lane_id == "football_goals_core":
        return _football_goals_selection(row, contract_outcome)
    return ""


def _lane_decision_inventory(spec: MarketLaneSpec, catalog: pd.DataFrame, now: pd.Timestamp) -> dict[str, Any]:
    future = _forward_active_catalog(catalog, now)
    if future.empty:
        return {
            "lane_id": spec.lane_id,
            "future_active_markets": 0,
            "decision_opportunities_ready": 0,
            "decision_inventory_status": "no_forward_inventory",
            "blockers": ["inventory_not_forward_ready"],
        }

    if spec.lane_id == "football_1x2_global":
        working = future.copy()
        working["decision_role"] = working.apply(_football_1x2_market_role, axis=1)
        rows: list[dict[str, Any]] = []
        for event_slug, group in working.groupby("event_slug", observed=True):
            role_counts = group["decision_role"].value_counts().to_dict()
            has_three_way = int(role_counts.get("three_way_1x2", 0)) > 0
            complete_binary = int(role_counts.get("team_win_binary", 0)) >= 2 and int(role_counts.get("draw_binary", 0)) >= 1
            status = "complete_1x2_decision" if has_three_way or complete_binary else "partial_1x2_decision"
            rows.append(
                {
                    "event_slug": str(event_slug),
                    "event_title": str(group.iloc[0].get("event_title", "")),
                    "markets": int(len(group)),
                    "role_counts": role_counts,
                    "decision_status": status,
                    "legacy_slice": int(group.get("legacy_slice", pd.Series(dtype=int)).fillna(0).astype(int).max()) if "legacy_slice" in group.columns else 0,
                    "game_start_time": str(group.iloc[0].get("game_start_time", "")),
                }
            )
        complete = [row for row in rows if row["decision_status"] == "complete_1x2_decision"]
        partial = [row for row in rows if row["decision_status"] != "complete_1x2_decision"]
        return {
            "lane_id": spec.lane_id,
            "future_active_markets": int(len(future)),
            "future_events": int(len(rows)),
            "decision_opportunities_ready": int(len(complete)),
            "partial_decision_events": int(len(partial)),
            "legacy_slice_decision_events": int(sum(row["legacy_slice"] for row in complete)),
            "decision_inventory_status": "decision_ready" if complete else "partial_market_coverage",
            "blockers": [] if complete else ["missing_complete_1x2_event_groups"],
            "role_counts": working["decision_role"].value_counts().to_dict(),
            "sample_partial_events": partial[:10],
        }

    if spec.lane_id == "football_goals_core":
        subtype_counts = future["market_subtype"].value_counts().to_dict() if "market_subtype" in future.columns else {}
        return {
            "lane_id": spec.lane_id,
            "future_active_markets": int(len(future)),
            "future_events": int(future["event_slug"].nunique()) if "event_slug" in future.columns else 0,
            "decision_opportunities_ready": int(len(future)),
            "market_subtype_counts": subtype_counts,
            "decision_inventory_status": "decision_ready" if len(future) else "no_forward_inventory",
            "blockers": [],
        }

    return {
        "lane_id": spec.lane_id,
        "future_active_markets": int(len(future)),
        "future_events": int(future["event_slug"].nunique()) if "event_slug" in future.columns else 0,
        "decision_opportunities_ready": int(len(future)) if _lane_model_ready(spec) else 0,
        "decision_inventory_status": "model_pending" if not _lane_model_ready(spec) else "decision_ready",
        "blockers": [] if _lane_model_ready(spec) else ["model_not_ready"],
    }


def _lane_model_predictions_path(settings: Settings, spec: MarketLaneSpec) -> Path:
    return _lane_output_dir(settings, spec.lane_id) / LANE_MODEL_PREDICTIONS_FILENAME


def _load_lane_model_predictions(
    settings: Settings,
    spec: MarketLaneSpec,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], str, int]:
    path = _lane_model_predictions_path(settings, spec)
    if not path.exists():
        return {}, str(path), 0
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}, str(path), 0
    required = {"event_slug", "selection"}
    if frame.empty or not required.issubset(set(frame.columns)):
        return {}, str(path), 0
    probability_column = "model_prob" if "model_prob" in frame.columns else "probability" if "probability" in frame.columns else ""
    if not probability_column:
        return {}, str(path), 0

    predictions: dict[tuple[str, str, str], dict[str, Any]] = {}
    loaded_rows = 0
    for _, row in frame.iterrows():
        event_slug = str(row.get("event_slug") or "").strip()
        selection = str(row.get("selection") or "").strip()
        if not event_slug or not selection:
            continue
        prob = _safe_float(row.get(probability_column))
        if prob is None or prob < 0.0 or prob > 1.0:
            continue
        market_id = str(row.get("market_id") or "").strip()
        payload = {
            "model_prob": prob,
            "probability_source": str(row.get("probability_source") or "lane_model"),
            "model_variant": str(row.get("model_variant") or spec.model_mode),
        }
        loaded_rows += 1
        predictions[(event_slug, market_id, selection)] = payload
        predictions[(event_slug, "", selection)] = payload
    return predictions, str(path), loaded_rows


def _latest_books_by_market_outcome(books: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    if books.empty:
        return {}
    frame = books.copy()
    frame["_timestamp"] = pd.to_datetime(frame.get("timestamp", pd.Series(dtype=str)), utc=True, errors="coerce")
    frame = frame.sort_values("_timestamp")
    latest: dict[tuple[str, str], pd.Series] = {}
    for _, row in frame.iterrows():
        key = (str(row.get("market_id") or ""), str(row.get("outcome_name") or "").strip().lower())
        latest[key] = row
    return latest


def _policy_thresholds(policy_context: dict[str, Any]) -> dict[str, Any]:
    policy = policy_context.get("policy") if isinstance(policy_context.get("policy"), dict) else {}
    def _number(name: str, default: float) -> float:
        value = _safe_float(policy.get(name))
        return default if value is None else float(value)

    return {
        "edge_threshold": _number("edge_threshold", 0.0),
        "ev_threshold": _number("ev_threshold", 0.0),
        "min_odds": _number("min_odds", 0.0),
        "max_odds": _number("max_odds", 999.0),
        "allowed_outcomes": {str(item) for item in policy.get("allowed_outcomes", []) or []},
        "allowed_market_subtypes": {str(item) for item in policy.get("allowed_market_subtypes", []) or []},
    }


def _build_lane_candidate_rows(
    settings: Settings,
    spec: MarketLaneSpec,
    catalog: pd.DataFrame,
    books: pd.DataFrame,
    now: pd.Timestamp,
    policy_context: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forward = _forward_active_catalog(catalog, now)
    governance = get_lane_governance(spec.lane_id)
    predictions, predictions_path, prediction_rows_loaded = _load_lane_model_predictions(settings, spec)
    latest_books = _latest_books_by_market_outcome(books)
    thresholds = _policy_thresholds(policy_context)
    rows: list[dict[str, Any]] = []
    if forward.empty:
        empty = pd.DataFrame(columns=LANE_CANDIDATE_COLUMNS)
        return empty, {
            "lane_id": spec.lane_id,
            "candidate_scoring_status": "no_forward_inventory",
            "model_predictions_path": predictions_path,
            "model_predictions_loaded": 0,
            "model_prediction_lookup_entries": 0,
            "candidate_rows": 0,
            "quote_ready_candidates": 0,
            "scorable_candidates": 0,
            "selected_candidates": 0,
            "blockers": ["empty_forward_catalog"],
        }

    for _, market in forward.iterrows():
        outcomes = [str(item) for item in _json_list(market.get("outcomes_json"))]
        for outcome in outcomes:
            selection = _selection_for_lane(spec, market, outcome)
            if not selection or selection not in set(spec.outcome_schema):
                continue
            book = latest_books.get((str(market.get("market_id") or ""), outcome.strip().lower()))
            top_ask = _safe_float(book.get("top_ask")) if book is not None else None
            quoted_odds = (1.0 / top_ask) if top_ask and top_ask > 0.0 else None
            prediction = predictions.get(
                (
                    str(market.get("event_slug") or ""),
                    str(market.get("market_id") or ""),
                    selection,
                )
            ) or predictions.get((str(market.get("event_slug") or ""), "", selection))
            model_prob = prediction.get("model_prob") if prediction else None
            edge = float(model_prob) - float(top_ask) if model_prob is not None and top_ask is not None else None
            ev = float(model_prob) * float(quoted_odds) - 1.0 if model_prob is not None and quoted_odds is not None else None
            status = "quote_ready"
            blocker = ""
            if top_ask is None or quoted_odds is None:
                status = "blocked"
                blocker = "book_missing"
            elif model_prob is None:
                status = "blocked"
                blocker = "model_probability_missing"
            elif not bool(policy_context.get("policy_ready")):
                status = "blocked"
                blocker = "policy_not_ready"
            elif thresholds["allowed_outcomes"] and selection not in thresholds["allowed_outcomes"]:
                status = "policy_rejected"
                blocker = "outcome_not_allowed"
            elif thresholds["allowed_market_subtypes"] and str(market.get("market_subtype") or spec.market_subtype) not in thresholds["allowed_market_subtypes"]:
                status = "policy_rejected"
                blocker = "market_subtype_not_allowed"
            elif quoted_odds < thresholds["min_odds"] or quoted_odds > thresholds["max_odds"]:
                status = "policy_rejected"
                blocker = "odds_out_of_policy_range"
            elif edge is None or edge < thresholds["edge_threshold"]:
                status = "policy_rejected"
                blocker = "edge_below_threshold"
            elif ev is None or ev < thresholds["ev_threshold"]:
                status = "policy_rejected"
                blocker = "ev_below_threshold"
            elif not governance.can_emit_shadow_picks:
                status = "blocked"
                blocker = "lane_governance_blocks_shadow_picks"
            else:
                status = "selected_candidate"

            rows.append(
                {
                    "candidate_id": f"{spec.lane_id}:{market.get('event_slug')}:{market.get('market_id')}:{selection}:{outcome}",
                    "lane_id": spec.lane_id,
                    "benchmark_id": spec.benchmark_id,
                    "event_slug": str(market.get("event_slug") or ""),
                    "market_id": str(market.get("market_id") or ""),
                    "market_subtype": str(market.get("market_subtype") or spec.market_subtype),
                    "game_start_time": str(market.get("game_start_time") or ""),
                    "selection": selection,
                    "contract_outcome": outcome,
                    "asset_id": str(book.get("asset_id") or "") if book is not None else "",
                    "book_timestamp": str(book.get("timestamp") or "") if book is not None else "",
                    "top_ask": top_ask,
                    "quoted_odds": quoted_odds,
                    "model_prob": model_prob,
                    "probability_source": prediction.get("probability_source") if prediction else "",
                    "edge": edge,
                    "ev": ev,
                    "candidate_status": status,
                    "candidate_blocker": blocker,
                    "policy_edge_threshold": thresholds["edge_threshold"],
                    "policy_ev_threshold": thresholds["ev_threshold"],
                    "policy_min_odds": thresholds["min_odds"],
                    "policy_max_odds": thresholds["max_odds"],
                    "legacy_slice": int(market.get("legacy_slice") or 0),
                }
            )

    candidates = pd.DataFrame(rows, columns=LANE_CANDIDATE_COLUMNS)
    if not candidates.empty and "selected_candidate" in set(candidates["candidate_status"]):
        selected = candidates[candidates["candidate_status"] == "selected_candidate"].copy()
        selected["_ev_sort"] = pd.to_numeric(selected["ev"], errors="coerce").fillna(-999.0)
        selected["_edge_sort"] = pd.to_numeric(selected["edge"], errors="coerce").fillna(-999.0)
        selected["_odds_sort"] = pd.to_numeric(selected["quoted_odds"], errors="coerce").fillna(999.0)
        keep = (
            selected.sort_values(["event_slug", "_ev_sort", "_edge_sort", "_odds_sort"], ascending=[True, False, False, True])
            .drop_duplicates("event_slug")["candidate_id"]
            .astype(str)
            .tolist()
        )
        one_pick_mask = candidates["candidate_status"].eq("selected_candidate") & ~candidates["candidate_id"].astype(str).isin(set(keep))
        candidates.loc[one_pick_mask, "candidate_status"] = "policy_rejected"
        candidates.loc[one_pick_mask, "candidate_blocker"] = "one_pick_per_event"

    status_counts = candidates["candidate_status"].value_counts().to_dict() if not candidates.empty else {}
    blocker_counts = (
        candidates["candidate_blocker"].replace("", pd.NA).dropna().value_counts().to_dict()
        if not candidates.empty
        else {}
    )
    selected_count = int(status_counts.get("selected_candidate", 0))
    scorable_count = int(candidates["model_prob"].notna().sum()) if not candidates.empty else 0
    quote_ready_count = int(candidates["top_ask"].notna().sum()) if not candidates.empty else 0
    horizon_hours = (24, 48, 72, 168, 336)
    horizon_counts = {f"next_{hours}h": 0 for hours in horizon_hours}
    scorable_horizon_counts = {f"next_{hours}h": 0 for hours in horizon_hours}
    if not candidates.empty and "game_start_time" in candidates.columns:
        starts = pd.to_datetime(candidates["game_start_time"], utc=True, errors="coerce")
        selected_mask = candidates["candidate_status"].eq("selected_candidate")
        scorable_mask = candidates["model_prob"].notna()
        for hours in horizon_hours:
            within = starts.ge(now) & starts.le(now + pd.Timedelta(hours=hours))
            horizon_counts[f"next_{hours}h"] = int((selected_mask & within).sum())
            scorable_horizon_counts[f"next_{hours}h"] = int((scorable_mask & within).sum())
    if selected_count > 0:
        scoring_status = "selected_candidates_ready"
    elif scorable_count > 0 and not bool(policy_context.get("policy_ready")):
        scoring_status = "policy_not_ready"
    elif scorable_count > 0:
        scoring_status = "no_candidate_passed_frozen_policy"
    elif quote_ready_count > 0:
        scoring_status = "model_probability_missing"
    else:
        scoring_status = "book_missing"
    report = {
        "lane_id": spec.lane_id,
        "candidate_scoring_status": scoring_status,
        "model_predictions_path": predictions_path,
        "model_predictions_loaded": int(prediction_rows_loaded),
        "model_prediction_lookup_entries": int(len(predictions)),
        "candidate_rows": int(len(candidates)),
        "quote_ready_candidates": quote_ready_count,
        "scorable_candidates": scorable_count,
        "selected_candidates": selected_count,
        "candidate_status_counts": status_counts,
        "candidate_blocker_counts": blocker_counts,
        "selected_candidate_horizon_counts": horizon_counts,
        "scorable_candidate_horizon_counts": scorable_horizon_counts,
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        "one_pick_per_event_enforced": True,
        "lane_governance": _lane_governance_payload(governance),
    }
    return candidates, report


def _lane_model_prediction_template(spec: MarketLaneSpec, candidate_rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "lane_id",
        "benchmark_id",
        "event_slug",
        "market_id",
        "market_subtype",
        "selection",
        "model_prob",
        "probability_source",
        "model_variant",
    ]
    if candidate_rows.empty:
        return pd.DataFrame(columns=columns)
    frame = candidate_rows[
        ["lane_id", "benchmark_id", "event_slug", "market_id", "market_subtype", "selection"]
    ].copy()
    if spec.lane_id == "football_1x2_global":
        frame["market_id"] = ""
        frame["market_subtype"] = spec.market_subtype
    frame = frame.drop_duplicates(["event_slug", "market_id", "selection"]).sort_values(
        ["event_slug", "market_id", "selection"]
    )
    frame["model_prob"] = ""
    frame["probability_source"] = "raw"
    frame["model_variant"] = spec.model_mode
    return frame.loc[:, columns]


def _resolve_lane_model_bundle_path(settings: Settings, model_path: Path | str | None = None, lane_id: str | None = None) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if not path.is_absolute():
            path = settings.paths.root / path
        if path.exists():
            return path
        raise FileNotFoundError(f"No encuentro el modelo indicado: {path}")
    if lane_id:
        lane_pointer = settings.paths.outputs_dir / "lanes" / lane_id / "latest_model.txt"
        resolved = _resolve_path_from_pointer(lane_pointer, settings)
        if resolved is not None:
            return resolved
    for pointer_name in ("latest_niche_model.txt", "latest_model.txt"):
        pointer = settings.paths.outputs_dir / pointer_name
        resolved = _resolve_path_from_pointer(pointer, settings)
        if resolved is not None:
            return resolved
    raise FileNotFoundError("No encuentro latest_niche_model.txt ni latest_model.txt para generar predicciones de carril.")


def _league_code_from_market_row(row: pd.Series) -> str:
    slug = str(row.get("event_slug") or "").lower()
    title = str(row.get("event_title") or "").lower()
    if slug.startswith("epl-") or "premier league" in title:
        return "E0"
    if slug.startswith("lal-") or "la liga" in title:
        return "SP1"
    if slug.startswith("bun-") or "bundesliga" in title:
        return "D1"
    if slug.startswith("sea-") or "serie a" in title:
        return "I1"
    if slug.startswith("fl1-") or "ligue 1" in title:
        return "F1"
    if slug.startswith("ere-") or "eredivisie" in title:
        return "N1"
    if slug.startswith("ppl-") or "primeira liga" in title or "liga portugal" in title:
        return "P1"
    if slug.startswith("mex-") or "liga mx" in title:
        return "MEX"
    if slug.startswith("mls-") or "major league soccer" in title or " mls" in title:
        return "USA"
    return ""


def _league_name_for_code(league_code: str) -> str:
    return {
        "E0": "Premier League",
        "SP1": "La Liga",
        "D1": "Bundesliga",
        "I1": "Serie A",
        "F1": "Ligue 1",
        "N1": "Eredivisie",
        "P1": "Primeira Liga",
        "MEX": "Liga MX",
        "USA": "MLS",
    }.get(str(league_code), str(league_code))


def _season_code_for_date(date: pd.Timestamp) -> str:
    year = int(date.year)
    start = year if int(date.month) >= 7 else year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def _team_alias_values(team: str) -> set[str]:
    cleaned = _clean_match_team(team)
    aliases = {cleaned}
    suffix_stripped = re.sub(r"\b(fc|afc|cf|sc|sd|ac|bc|club)\b", "", cleaned).strip()
    if suffix_stripped:
        aliases.add(" ".join(suffix_stripped.split()))
    manual = {
        "man city": ("manchester city", "manchester city fc"),
        "man united": ("manchester united", "manchester united fc", "man utd"),
        "nottm forest": ("nottingham forest", "nottingham forest fc", "nott m forest"),
        "nott'm forest": ("nottingham forest", "nottingham forest fc", "nottm forest"),
        "wolves": ("wolverhampton wanderers", "wolverhampton wanderers fc"),
        "brighton": ("brighton hove albion", "brighton hove albion fc"),
        "bournemouth": ("afc bournemouth", "bournemouth fc"),
        "west ham": ("west ham united", "west ham united fc"),
        "leeds": ("leeds united", "leeds united fc"),
        "tottenham": ("tottenham hotspur", "tottenham hotspur fc"),
        "newcastle": ("newcastle united", "newcastle united fc"),
        "leicester": ("leicester city", "leicester city fc"),
        "ath madrid": ("atletico madrid", "club atletico de madrid"),
        "ath bilbao": ("athletic bilbao", "athletic club", "athletic club bilbao"),
        "betis": ("real betis", "real betis balompie"),
        "sociedad": ("real sociedad", "real sociedad de futbol"),
        "celta": ("celta vigo", "rc celta"),
        "vallecano": ("rayo vallecano", "rayo vallecano de madrid"),
        "alaves": ("deportivo alaves",),
        "espanyol": ("espanol", "rcd espanyol"),
        "bayern munich": ("bayern", "fc bayern munich", "bayern munchen"),
        "heidenheim": ("1 fc heidenheim", "1 fc heidenheim 1846", "fc heidenheim"),
        "leverkusen": ("bayer leverkusen", "bayer 04 leverkusen"),
        "ein frankfurt": ("eintracht frankfurt", "frankfurt"),
        "rb leipzig": ("leipzig", "r b leipzig"),
        "dortmund": ("borussia dortmund", "bv borussia 09 dortmund", "bvb"),
        "m gladbach": ("monchengladbach", "borussia monchengladbach", "borussia m gladbach"),
        "m'gladbach": ("monchengladbach", "borussia monchengladbach", "borussia m gladbach"),
        "stuttgart": ("vfb stuttgart",),
        "freiburg": ("sc freiburg",),
        "mainz": ("mainz 05", "1 fsv mainz 05"),
        "hoffenheim": ("tsg hoffenheim", "tsg 1899 hoffenheim"),
        "wolfsburg": ("vfl wolfsburg",),
        "fc koln": ("1 fc koln", "koln"),
        "st pauli": ("fc st pauli", "fc st pauli 1910"),
        "union berlin": ("1 fc union berlin",),
        "barcelona": ("fc barcelona",),
        "celta": ("celta vigo", "rc celta", "rc celta de vigo"),
        "mallorca": ("rcd mallorca",),
        "osasuna": ("ca osasuna",),
        "espanyol": ("espanol", "rcd espanyol", "rcd espanyol de barcelona"),
        "levante ud": ("levante",),
        "real oviedo": ("oviedo",),
        "hamburger sv": ("hamburg",),
        "paris saint germain fc": ("paris sg", "psg"),
        "stade brestois 29": ("brest",),
        "racing club de lens": ("lens",),
        "olympique lyonnais": ("lyon",),
        "aj auxerre": ("auxerre",),
        "angers sco": ("angers",),
        "as monaco fc": ("monaco",),
        "rc strasbourg alsace": ("strasbourg",),
        "lille osc": ("lille",),
        "stade rennais fc 1901": ("rennes",),
        "olympique de marseille": ("marseille",),
        "ogc nice": ("nice",),
        "us lecce": ("lecce",),
        "acf fiorentina": ("fiorentina",),
        "ssc napoli": ("napoli",),
        "us cremonese": ("cremonese",),
        "parma calcio 1913": ("parma",),
        "bologna fc 1909": ("bologna",),
        "as roma": ("roma",),
        "hellas verona fc": ("verona",),
        "us sassuolo calcio": ("sassuolo",),
        "genoa cfc": ("genoa",),
        "fc internazionale milano": ("inter",),
        "cagliari calcio": ("cagliari",),
        "ss lazio": ("lazio",),
        "udinese calcio": ("udinese",),
        "como 1907": ("como",),
        "atalanta bc": ("atalanta",),
        "ac milan": ("milan",),
        "az": ("az alkmaar",),
        "pec zwolle": ("zwolle",),
        "psv": ("psv eindhoven",),
        "afc ajax": ("ajax",),
        "feyenoord rotterdam": ("feyenoord",),
        "fc twente 65": ("twente",),
        "fc utrecht": ("utrecht",),
        "fc groningen": ("groningen",),
        "fc volendam": ("volendam",),
        "sc heerenveen": ("heerenveen",),
        "fortuna sittard": ("for sittard",),
        "heracles almelo": ("heracles",),
        "sparta rotterdam": ("sparta rotterdam",),
        "go ahead eagles": ("go ahead eagles",),
        "sbv excelsior": ("excelsior",),
        "nac breda": ("nac breda",),
        "nec": ("nijmegen",),
        "telstar 1963": ("telstar",),
        "unam pumas": ("pumas de la unam", "pumas"),
        "juarez": ("fc juarez",),
        "queretaro": ("queretaro fc",),
        "cruz azul": ("cf cruz azul",),
        "monterrey": ("cf monterrey",),
        "puebla": ("club puebla",),
        "club leon": ("club leon fc",),
        "club america": ("cf america",),
        "atlas": ("atlas fc",),
        "tigres uanl": ("tigres de la uanl",),
        "toluca": ("deportivo toluca fc",),
        "atl san luis": ("atletico san luis", "atl. san luis"),
        "santos laguna": ("club santos laguna",),
        "necaxa": ("club necaxa",),
        "guadalajara chivas": ("cd guadalajara", "guadalajara"),
        "pachuca": ("cf pachuca",),
        "club tijuana": ("tijuana",),
        "seattle sounders": ("seattle sounders fc",),
        "orlando city": ("orlando city sc",),
        "charlotte": ("charlotte fc",),
        "new york city": ("new york city fc",),
        "atlanta utd": ("atlanta united", "atlanta united fc"),
        "minnesota united": ("minnesota united fc",),
        "dc united": ("d c united sc", "d.c. united", "d c united"),
    }
    if cleaned in manual:
        aliases.update(_clean_match_team(item) for item in manual[cleaned])
    for canonical, values in manual.items():
        normalized_values = {_clean_match_team(item) for item in values}
        if cleaned in normalized_values:
            aliases.add(_clean_match_team(canonical))
            aliases.update(normalized_values)
    return {item for item in aliases if item}


def _team_lookup_for_league(history_matches: pd.DataFrame, league_code: str) -> dict[str, str]:
    league_history = history_matches[history_matches["league_code"].astype(str).eq(str(league_code))].copy()
    teams = sorted(set(league_history.get("HomeTeam", pd.Series(dtype=str)).dropna().astype(str)).union(
        set(league_history.get("AwayTeam", pd.Series(dtype=str)).dropna().astype(str))
    ))
    candidates: dict[str, set[str]] = {}
    for team in teams:
        for alias in _team_alias_values(team):
            candidates.setdefault(alias, set()).add(team)
    return {alias: next(iter(values)) for alias, values in candidates.items() if len(values) == 1}


def _resolve_team_name(raw_team: str | None, lookup: dict[str, str]) -> str | None:
    if not raw_team:
        return None
    aliases = _team_alias_values(raw_team)
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    best_team: str | None = None
    best_score = 0.0
    cleaned = _clean_match_team(raw_team)
    for alias, team in lookup.items():
        score = SequenceMatcher(None, cleaned, alias).ratio()
        if score > best_score:
            best_score = score
            best_team = team
    return best_team if best_score >= 0.88 else None


def _team_history_count(history_matches: pd.DataFrame, league_code: str, team: str) -> int:
    league_history = history_matches[history_matches["league_code"].astype(str).eq(str(league_code))]
    return int(
        (
            league_history.get("HomeTeam", pd.Series(dtype=str)).astype(str).eq(str(team))
            | league_history.get("AwayTeam", pd.Series(dtype=str)).astype(str).eq(str(team))
        ).sum()
    )


def _model_variant_name(model_path: Path, payload: dict[str, Any]) -> str:
    variant = payload.get("model_variant") or payload.get("variant_name")
    if variant:
        return str(variant)
    return model_path.parent.name or model_path.stem


def _fixture_rows_from_template(
    template: pd.DataFrame,
    catalog: pd.DataFrame,
    history_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, int]]:
    contexts: dict[str, dict[str, Any]] = {}
    blockers: dict[str, int] = {}
    fixtures: list[dict[str, Any]] = []
    if template.empty:
        return pd.DataFrame(), contexts, blockers
    event_slugs = template["event_slug"].dropna().astype(str).unique().tolist()
    for match_id, event_slug in enumerate(event_slugs, start=len(history_matches) + 1):
        event_catalog = catalog[catalog["event_slug"].astype(str).eq(str(event_slug))] if not catalog.empty else pd.DataFrame()
        row = event_catalog.iloc[0] if not event_catalog.empty else pd.Series({"event_slug": event_slug})
        league_code = _league_code_from_market_row(row)
        row_count = int(len(template[template["event_slug"].astype(str).eq(str(event_slug))]))
        if league_code not in set(GLOBAL_FOOTBALL_1X2_LEAGUES):
            contexts[event_slug] = {
                "status": "blocked",
                "blocker": "unsupported_league",
                "league_code": league_code,
                "legacy_slice": int(event_catalog.get("legacy_slice", pd.Series([0])).fillna(0).astype(int).max()) if not event_catalog.empty else 0,
            }
            blockers["unsupported_league"] = blockers.get("unsupported_league", 0) + int(len(template[template["event_slug"].astype(str).eq(str(event_slug))]))
            continue
        league_history_rows = int(history_matches["league_code"].astype(str).eq(str(league_code)).sum()) if "league_code" in history_matches.columns else 0
        if league_history_rows <= 0:
            raw_home, raw_away = _event_teams_from_row(row)
            contexts[event_slug] = {
                "status": "blocked",
                "blocker": "unsupported_league",
                "league_code": league_code,
                "raw_home": raw_home,
                "raw_away": raw_away,
                "league_history_rows": league_history_rows,
                "legacy_slice": int(event_catalog.get("legacy_slice", pd.Series([0])).fillna(0).astype(int).max()) if not event_catalog.empty else 0,
            }
            blockers["unsupported_league"] = blockers.get("unsupported_league", 0) + row_count
            continue
        game_start = pd.to_datetime(row.get("game_start_time"), utc=True, errors="coerce")
        if pd.isna(game_start):
            contexts[event_slug] = {"status": "blocked", "blocker": "date_parse_failed", "league_code": league_code}
            blockers["date_parse_failed"] = blockers.get("date_parse_failed", 0) + row_count
            continue
        raw_home, raw_away = _event_teams_from_row(row)
        lookup = _team_lookup_for_league(history_matches, league_code)
        home_team = _resolve_team_name(raw_home, lookup)
        away_team = _resolve_team_name(raw_away, lookup)
        if home_team is None or away_team is None or home_team == away_team:
            contexts[event_slug] = {
                "status": "blocked",
                "blocker": "team_mapping_failed",
                "league_code": league_code,
                "raw_home": raw_home,
                "raw_away": raw_away,
                "home_team": home_team or "",
                "away_team": away_team or "",
                "league_history_rows": league_history_rows,
                "legacy_slice": int(event_catalog.get("legacy_slice", pd.Series([0])).fillna(0).astype(int).max()) if not event_catalog.empty else 0,
            }
            blockers["team_mapping_failed"] = blockers.get("team_mapping_failed", 0) + row_count
            continue
        home_samples = _team_history_count(history_matches, league_code, home_team)
        away_samples = _team_history_count(history_matches, league_code, away_team)
        if min(home_samples, away_samples) < MIN_LANE_TEAM_HISTORY_ROWS:
            contexts[event_slug] = {
                "status": "blocked",
                "blocker": "history_missing",
                "league_code": league_code,
                "raw_home": raw_home,
                "raw_away": raw_away,
                "home_team": home_team,
                "away_team": away_team,
                "home_history_rows": home_samples,
                "away_history_rows": away_samples,
                "league_history_rows": league_history_rows,
                "legacy_slice": int(event_catalog.get("legacy_slice", pd.Series([0])).fillna(0).astype(int).max()) if not event_catalog.empty else 0,
            }
            blockers["history_missing"] = blockers.get("history_missing", 0) + row_count
            continue
        fixture_date = game_start.tz_convert(None).normalize()
        fixture = {
            "match_id": match_id,
            "Date": fixture_date,
            "league_code": league_code,
            "league_name": _league_name_for_code(league_code),
            "season": _season_code_for_date(fixture_date),
            "HomeTeam": home_team,
            "AwayTeam": away_team,
        }
        fixtures.append(fixture)
        contexts[event_slug] = {
            "status": "ready",
            "blocker": "",
            "match_id": match_id,
            "league_code": league_code,
            "raw_home": raw_home,
            "raw_away": raw_away,
            "home_team": home_team,
            "away_team": away_team,
            "home_history_rows": home_samples,
            "away_history_rows": away_samples,
            "legacy_slice": int(event_catalog.get("legacy_slice", pd.Series([0])).fillna(0).astype(int).max()) if not event_catalog.empty else 0,
        }
    return pd.DataFrame(fixtures), contexts, blockers


def _parse_total_goals_line(row: pd.Series) -> float | None:
    text = _row_text(row)
    match = re.search(r"\b(?:over|under|o/u)\s+(\d+(?:\.\d+)?)\b", text)
    if match:
        line = _safe_float(match.group(1))
    else:
        slug = str(row.get("market_slug") or "").lower()
        slug_match = re.search(r"\b(?:over|under|total)-(\d)(?:pt|p)?(\d)\b", slug)
        line = float(f"{slug_match.group(1)}.{slug_match.group(2)}") if slug_match else None
    if line is None:
        return None
    return line if abs((line % 1.0) - 0.5) < 1e-9 else None


def _goal_market_probability(selection: str, row: pd.Series, home_lambda: float, away_lambda: float, rho: float, max_goals: int) -> tuple[float | None, str]:
    matrix = score_matrix_from_lambdas(home_lambda, away_lambda, rho=rho, max_goals=max_goals)
    if selection in {"btts_yes", "btts_no"}:
        yes = float(matrix[1:, 1:].sum())
        return (yes if selection == "btts_yes" else 1.0 - yes), ""
    if selection in {"over", "under"}:
        line = _parse_total_goals_line(row)
        if line is None:
            return None, "line_parse_failed"
        totals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        over = float(matrix[totals > float(line)].sum())
        return (over if selection == "over" else 1.0 - over), ""
    return None, "unsupported_selection"


def build_market_lane_predictions(
    settings: Settings,
    lane_id: str,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
    now: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    now = now or _utcnow()
    governance = require_can_build_predictions(lane_id)
    spec = get_market_lane_spec(lane_id)
    if spec.reference_only or spec.status in {STATUS_REFERENCE_ONLY, STATUS_DEFERRED}:
        raise ValueError(f"El carril {lane_id} no admite predicciones operativas.")
    if spec.lane_id not in {"football_1x2_global", "football_goals_core"}:
        raise ValueError(f"El carril {lane_id} sigue en capture_only/model_pending y no tiene adaptador de prediccion v1.")

    lane_dir = _lane_output_dir(settings, lane_id)
    template_path = lane_dir / "model_prediction_template.csv"
    if not template_path.exists():
        raise FileNotFoundError(
            f"No encuentro {template_path}. Ejecuta primero predicciones lane run-shadow --lane-id {lane_id}."
        )
    template = pd.read_csv(template_path)
    if template.empty:
        predictions = pd.DataFrame(columns=LANE_MODEL_PREDICTION_COLUMNS)
        output_path = lane_dir / LANE_MODEL_PREDICTIONS_FILENAME
        report_path = lane_dir / LANE_MODEL_PREDICTION_REPORT_FILENAME
        predictions.to_csv(output_path, index=False)
        summary = {
            "lane_id": lane_id,
            "template_rows": 0,
            "predicted_rows": 0,
            "prediction_coverage_rate": 0.0,
            "blocker_counts": {"empty_template": 0},
            "supported_legacy_rows": 0,
            "unsupported_global_rows": 0,
            "policy_reoptimized": False,
            "global_roi_actionable": False,
            "model_predictions_path": str(output_path),
            "lane_governance": _lane_governance_payload(governance),
        }
        _write_json(report_path, summary)
        return summary, {"model_predictions": output_path, "model_prediction_report": report_path}

    database_path = Path(db_path) if db_path else default_multi_market_db_path(settings)
    connection = init_multi_market_db(database_path)
    catalog = _lane_catalog(connection, lane_id)
    connection.close()
    bundle_path = _resolve_lane_model_bundle_path(settings, model_path, lane_id=spec.lane_id)
    payload = joblib.load(bundle_path)
    if not isinstance(payload, dict) or "model" not in payload or "history_matches" not in payload:
        raise ValueError(f"Bundle de modelo invalido para carril: {bundle_path}")
    history_matches = payload["history_matches"].copy()
    model = payload["model"]
    rho = float(payload.get("rho", 0.0) or 0.0)
    max_goals = int(payload.get("max_poisson_goals", 10) or 10)
    feature_columns = list(payload.get("feature_columns") or getattr(model, "feature_columns", []))
    fixtures, contexts, context_blockers = _fixture_rows_from_template(template, catalog, history_matches)
    prediction_rows: list[dict[str, Any]] = []
    row_blockers: dict[str, int] = dict(context_blockers)
    model_variant = _model_variant_name(bundle_path, payload)
    lambda_by_event: dict[str, tuple[float, float]] = {}

    if not fixtures.empty:
        try:
            feature_rows = build_fixture_feature_rows(
                history_matches=history_matches,
                fixtures=fixtures,
                rolling_window=int(payload.get("rolling_window", 8) or 8),
            )
            for column in feature_columns:
                if column not in feature_rows.columns:
                    feature_rows[column] = np.nan
            home_lambdas, away_lambdas = model.predict_lambdas(feature_rows[feature_columns])
            match_id_to_event = {
                int(context["match_id"]): event_slug
                for event_slug, context in contexts.items()
                if context.get("status") == "ready" and context.get("match_id") is not None
            }
            for index, row in feature_rows.reset_index(drop=True).iterrows():
                event_slug = match_id_to_event.get(int(row["match_id"]))
                if event_slug:
                    lambda_by_event[event_slug] = (float(home_lambdas[index]), float(away_lambdas[index]))
        except Exception:
            for event_slug, context in list(contexts.items()):
                if context.get("status") == "ready":
                    context["status"] = "blocked"
                    context["blocker"] = "feature_build_failed"
            row_blockers["feature_build_failed"] = row_blockers.get("feature_build_failed", 0) + int(
                template["event_slug"].astype(str).isin([event for event, context in contexts.items() if context.get("blocker") == "feature_build_failed"]).sum()
            )

    if spec.lane_id == "football_1x2_global" and lambda_by_event:
        ordered_events = list(lambda_by_event.keys())
        home = np.array([lambda_by_event[event][0] for event in ordered_events], dtype=float)
        away = np.array([lambda_by_event[event][1] for event in ordered_events], dtype=float)
        probs = outcome_probabilities_from_lambdas(home, away, rho=rho, max_goals=max_goals)
        for event_slug, row_probs in zip(ordered_events, probs):
            context = contexts.get(event_slug, {})
            for selection, prob in {"away": row_probs[0], "draw": row_probs[1], "home": row_probs[2]}.items():
                prediction_rows.append(
                    {
                        "lane_id": spec.lane_id,
                        "benchmark_id": spec.benchmark_id,
                        "event_slug": event_slug,
                        "market_id": "",
                        "market_subtype": spec.market_subtype,
                        "selection": selection,
                        "model_prob": float(prob),
                        "probability_source": "raw",
                        "model_variant": model_variant,
                    }
                )
            if int(context.get("legacy_slice", 0)):
                row_blockers.setdefault("_supported_legacy_event_rows", 0)
                row_blockers["_supported_legacy_event_rows"] += int(len(template[template["event_slug"].astype(str).eq(event_slug)]))

    if spec.lane_id == "football_goals_core":
        catalog_by_market = {str(row.get("market_id")): row for _, row in catalog.iterrows()}
        for _, template_row in template.iterrows():
            event_slug = str(template_row.get("event_slug") or "")
            selection = str(template_row.get("selection") or "")
            market_id = str(template_row.get("market_id") or "")
            context = contexts.get(event_slug, {})
            if context.get("status") != "ready" or event_slug not in lambda_by_event:
                continue
            market_row = catalog_by_market.get(market_id, pd.Series({"market_id": market_id, "market_subtype": template_row.get("market_subtype")}))
            home_lambda, away_lambda = lambda_by_event[event_slug]
            prob, blocker = _goal_market_probability(selection, market_row, home_lambda, away_lambda, rho, max_goals)
            if prob is None:
                row_blockers[blocker or "poisson_probability_failed"] = row_blockers.get(blocker or "poisson_probability_failed", 0) + 1
                continue
            prediction_rows.append(
                {
                    "lane_id": spec.lane_id,
                    "benchmark_id": spec.benchmark_id,
                    "event_slug": event_slug,
                    "market_id": market_id,
                    "market_subtype": str(template_row.get("market_subtype") or ""),
                    "selection": selection,
                    "model_prob": float(prob),
                    "probability_source": "raw",
                    "model_variant": model_variant,
                }
            )

    predictions = pd.DataFrame(prediction_rows, columns=LANE_MODEL_PREDICTION_COLUMNS)
    output_path = lane_dir / LANE_MODEL_PREDICTIONS_FILENAME
    report_path = lane_dir / LANE_MODEL_PREDICTION_REPORT_FILENAME
    lane_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    predicted_rows = int(len(predictions))
    template_rows = int(len(template))
    supported_legacy_rows = int(sum(
        len(template[template["event_slug"].astype(str).eq(event_slug)])
        for event_slug, context in contexts.items()
        if context.get("status") == "ready" and int(context.get("legacy_slice", 0))
    ))
    unsupported_global_rows = int(sum(
        len(template[template["event_slug"].astype(str).eq(event_slug)])
        for event_slug, context in contexts.items()
        if context.get("blocker") == "unsupported_league"
    ))
    blocker_counts = {key: int(value) for key, value in row_blockers.items() if not key.startswith("_") and int(value) > 0}
    summary = {
        "lane_id": spec.lane_id,
        "benchmark_id": spec.benchmark_id,
        "template_rows": template_rows,
        "predicted_rows": predicted_rows,
        "prediction_coverage_rate": float(predicted_rows / template_rows) if template_rows else 0.0,
        "blocker_counts": blocker_counts,
        "supported_legacy_rows": supported_legacy_rows,
        "unsupported_global_rows": unsupported_global_rows,
        "model_path": str(bundle_path),
        "model_variant": model_variant,
        "model_predictions_path": str(output_path),
        "model_prediction_report_path": str(report_path),
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        "generated_at": now.isoformat(),
        "lane_governance": _lane_governance_payload(governance),
    }
    _write_json(report_path, summary)
    return summary, {"model_predictions": output_path, "model_prediction_report": report_path}


def _build_lane_readiness(
    spec: MarketLaneSpec,
    catalog: pd.DataFrame,
    books: pd.DataFrame,
    ledger: pd.DataFrame,
    now: pd.Timestamp,
    policy_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy_context = policy_context or {"policy_ready": _lane_policy_ready(spec), "policy_mode_effective": spec.policy_mode}
    governance = get_lane_governance(spec.lane_id)
    market_count = int(len(catalog))
    active_markets = int(((catalog.get("active", pd.Series(dtype=int)) == 1) & (catalog.get("closed", pd.Series(dtype=int)) == 0)).sum()) if not catalog.empty else 0
    markets_with_tokens = int(catalog["clob_token_ids_json"].map(lambda value: len(_json_list(value)) > 0).sum()) if not catalog.empty else 0
    parseable_markets = int((catalog.get("parse_status", pd.Series(dtype=str)) == "parseable").sum()) if not catalog.empty else 0
    inventory = _inventory_quality(catalog, now)
    forward_catalog = _forward_active_catalog(catalog, now)
    forward_market_ids = set(forward_catalog["market_id"].astype(str)) if not forward_catalog.empty else set()
    forward_books = books[books["market_id"].astype(str).isin(forward_market_ids)] if not books.empty and forward_market_ids else pd.DataFrame()
    book_coverage = _latest_book_coverage(forward_catalog, forward_books)
    decision_inventory = _lane_decision_inventory(spec, catalog, now)
    forward_inventory_ready = bool(inventory["future_active_markets"] > 0)
    decision_inventory_ready = bool(int(decision_inventory.get("decision_opportunities_ready", 0)) > 0)
    coverage_ready = bool(forward_inventory_ready and markets_with_tokens > 0 and book_coverage["markets_with_raw_books"] > 0)
    model_ready = _lane_model_ready(spec)
    policy_ready = bool(policy_context.get("policy_ready"))
    settlement_ready = _lane_settlement_ready(spec)
    sample_contract_ready = _lane_sample_contract_ready(spec)
    blockers: list[str] = []
    if spec.reference_only:
        blockers.append("reference_only_lane")
    if spec.status == STATUS_DEFERRED:
        blockers.append("deferred_lane")
    if not governance.can_emit_shadow_picks:
        blockers.append("lane_governance_blocks_shadow_picks")
    if not forward_inventory_ready:
        blockers.append("inventory_not_forward_ready")
    if not coverage_ready:
        blockers.append("coverage_not_ready")
    if not decision_inventory_ready:
        blockers.append("decision_inventory_not_ready")
    if not model_ready:
        blockers.append("model_not_ready")
    if not policy_ready:
        blockers.append(str(policy_context.get("policy_blocker") or "policy_not_ready"))
    if not settlement_ready:
        blockers.append("settlement_not_ready")
    if not sample_contract_ready:
        blockers.append("sample_contract_not_ready")
    readiness = {
        "lane_id": spec.lane_id,
        "benchmark_id": spec.benchmark_id,
        "sport": spec.sport,
        "sport_id": spec.sport_id,
        "sport_merge_group": spec.sport_merge_group,
        "decision_type": spec.decision_type,
        "market_subtype": spec.market_subtype,
        "status": spec.status,
        "reference_only": spec.reference_only,
        "lane_governance": _lane_governance_payload(governance),
        "governance_mode": governance.mode.value,
        "governance_reason": governance.reason,
        "can_capture": governance.can_capture,
        "can_build_predictions": governance.can_build_predictions,
        "can_emit_shadow_picks": governance.can_emit_shadow_picks,
        "can_emit_capital_picks": governance.can_emit_capital_picks,
        "coverage_ready": coverage_ready,
        "model_ready": model_ready,
        "policy_ready": policy_ready,
        "decision_inventory_ready": decision_inventory_ready,
        "policy_mode_effective": str(policy_context.get("policy_mode_effective", spec.policy_mode)),
        "policy_activation_status": str(policy_context.get("policy_activation_status", "policy_ready" if policy_ready else spec.policy_mode)),
        "policy_transfer_mode": str(policy_context.get("policy_transfer_mode", "native_lane_policy" if policy_ready else "none")),
        "policy_bundle_path": str(policy_context.get("policy_bundle_path", "")),
        "source_policy_bundle_path": str(policy_context.get("source_policy_bundle_path", "")),
        "settlement_ready": settlement_ready,
        "sample_contract_ready": sample_contract_ready,
        "can_emit_picks": False,
        "readiness_status": "blocked",
        "readiness_blockers": blockers,
        "markets_in_lane": market_count,
        "active_markets": active_markets,
        "parseable_markets": parseable_markets,
        "markets_with_clob_tokens": markets_with_tokens,
        "decision_opportunities_ready": int(decision_inventory.get("decision_opportunities_ready", 0)),
        "decision_inventory_status": str(decision_inventory.get("decision_inventory_status", "")),
        "decision_inventory_blockers": list(decision_inventory.get("blockers", [])),
        **inventory,
        "legacy_slice_markets": int(catalog.get("legacy_slice", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if not catalog.empty else 0,
        "global_slice_markets": int((catalog.get("legacy_slice", pd.Series(dtype=int)).fillna(0).astype(int) == 0).sum()) if not catalog.empty else 0,
        "valid_decisions": int((ledger.get("decision_status", pd.Series(dtype=str)) == "valid_forward_sample").sum()) if not ledger.empty else 0,
        "settled_decisions": int((ledger.get("decision_status", pd.Series(dtype=str)) == "settled").sum()) if not ledger.empty else 0,
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        **book_coverage,
        **_lane_subtype_readiness(spec, catalog),
    }
    readiness["can_emit_picks"] = _lane_can_emit_picks(spec, readiness)
    if readiness["can_emit_picks"]:
        readiness["readiness_status"] = "ready_to_emit_shadow_candidates"
        readiness["readiness_blockers"] = []
    elif spec.reference_only:
        readiness["readiness_status"] = "reference_only"
    elif spec.status == STATUS_DEFERRED:
        readiness["readiness_status"] = "deferred"
    elif not forward_inventory_ready:
        readiness["readiness_status"] = "inventory_not_forward_ready"
    elif not coverage_ready:
        readiness["readiness_status"] = "waiting_for_raw_capture"
    elif model_ready and not policy_ready:
        readiness["readiness_status"] = "model_ready_policy_pending"
    elif not model_ready:
        readiness["readiness_status"] = "capture_only_model_pending"
    return readiness


def _legacy_parity_report(spec: MarketLaneSpec, catalog: pd.DataFrame, readiness: dict[str, Any]) -> dict[str, Any]:
    legacy_slice = catalog[catalog.get("legacy_slice", pd.Series(dtype=int)).fillna(0).astype(int) == 1] if not catalog.empty else pd.DataFrame()
    global_slice = catalog[catalog.get("legacy_slice", pd.Series(dtype=int)).fillna(0).astype(int) == 0] if not catalog.empty else pd.DataFrame()
    declared = tuple(spec.legacy_included)
    structural_ready = bool(spec.lane_id == "football_1x2_global" and declared == LEGACY_FOOTBALL_1X2_LEAGUES and len(legacy_slice) > 0)
    return {
        "lane_id": spec.lane_id,
        "reference_lane_id": "football_1x2_canonical" if spec.lane_id == "football_1x2_global" else None,
        "reference_status": "reference_only",
        "legacy_included": list(spec.legacy_included),
        "required_legacy_leagues": list(LEGACY_FOOTBALL_1X2_LEAGUES),
        "legacy_slice_markets": int(len(legacy_slice)),
        "legacy_slice_events": int(legacy_slice["event_slug"].nunique()) if not legacy_slice.empty else 0,
        "global_slice_markets": int(len(global_slice)),
        "new_lane_markets": int(len(catalog)),
        "structural_parity_ready": structural_ready,
        "coverage_ready": bool(readiness.get("coverage_ready")),
        "parity_status": "structural_parity_pending_forward_sample" if structural_ready else "legacy_slice_missing_or_incomplete",
        "can_replace_legacy": False,
        "replacement_blockers": [
            "forward_sample_not_ready",
            "legacy_benchmark_kept_for_comparison",
        ],
    }


def _lane_model_contexts_for_audit(
    settings: Settings,
    spec: MarketLaneSpec,
    catalog: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    if spec.lane_id not in {"football_1x2_global", "football_goals_core"} or candidate_rows.empty:
        return {}
    try:
        prediction_report_path = _lane_output_dir(settings, spec.lane_id) / LANE_MODEL_PREDICTION_REPORT_FILENAME
        prediction_report = json.loads(prediction_report_path.read_text(encoding="utf-8")) if prediction_report_path.exists() else {}
        reported_model_path = prediction_report.get("model_path") if isinstance(prediction_report, dict) else None
        bundle_path = _resolve_lane_model_bundle_path(settings, reported_model_path, lane_id=spec.lane_id)
        payload = joblib.load(bundle_path)
    except Exception:
        return {
            str(event_slug): {"status": "blocked", "blocker": "model_bundle_missing"}
            for event_slug in candidate_rows.get("event_slug", pd.Series(dtype=str)).dropna().astype(str).unique()
        }
    history_matches = payload.get("history_matches") if isinstance(payload, dict) else None
    if not isinstance(history_matches, pd.DataFrame):
        return {
            str(event_slug): {"status": "blocked", "blocker": "model_history_missing"}
            for event_slug in candidate_rows.get("event_slug", pd.Series(dtype=str)).dropna().astype(str).unique()
        }
    template = _lane_model_prediction_template(spec, candidate_rows)
    _, contexts, _ = _fixture_rows_from_template(template, catalog, history_matches)
    return contexts


def _blocker_next_action(blocker: str) -> str:
    return {
        "predicted": "collect_forward_sample",
        "policy_rejected": "no_action_policy_applied",
        "book_missing": "capture_raw_books_for_forward_markets",
        "capture_not_attempted": "capture_raw_books_for_forward_markets",
        "stale_book": "refresh_raw_books_near_decision_time",
        "no_clob_token": "refresh_discovery_or_keep_blocked",
        "unsupported_league": "add_historical_support_before_predicting",
        "team_mapping_failed": "add_legacy_safe_alias_if_team_exists_in_history",
        "history_missing": "keep_blocked_until_history_support_exists",
        "feature_build_failed": "inspect_feature_builder_for_fixture",
        "date_parse_failed": "fix_start_time_or_discovery_metadata",
        "line_parse_failed": "improve_goals_line_parser",
        "policy_not_ready": "create_family_policy_benchmark_before_picks",
        "model_bundle_missing": "build_or_point_latest_model_bundle",
        "model_history_missing": "rebuild_model_bundle_with_history_matches",
        "model_predictions_file_missing_or_empty": "run_build_market_lane_predictions",
        "prediction_row_missing_for_selection": "regenerate_model_predictions_and_audit_mapping",
    }.get(str(blocker or ""), "inspect_lane_blocker")


def _build_lane_blocker_audit(
    settings: Settings,
    spec: MarketLaneSpec,
    catalog: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    candidate_report: dict[str, Any],
    policy_context: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    catalog_by_market = {str(row.get("market_id") or ""): row for _, row in catalog.iterrows()}
    contexts = _lane_model_contexts_for_audit(settings, spec, catalog, candidate_rows)
    rows: list[dict[str, Any]] = []
    if candidate_rows.empty:
        audit = pd.DataFrame()
        return audit, {
            "lane_id": spec.lane_id,
            "audit_rows": 0,
            "final_blocker_counts": {},
            "legacy_final_blocker_counts": {},
            "policy_reoptimized": False,
            "global_roi_actionable": False,
        }

    for _, candidate in candidate_rows.iterrows():
        market_id = str(candidate.get("market_id") or "")
        event_slug = str(candidate.get("event_slug") or "")
        market = catalog_by_market.get(market_id, pd.Series(dtype=object))
        context = contexts.get(event_slug, {})
        top_ask = _safe_float(candidate.get("top_ask"))
        model_prob = _safe_float(candidate.get("model_prob"))
        candidate_status = str(candidate.get("candidate_status") or "")
        candidate_blocker = str(candidate.get("candidate_blocker") or "")
        book_available = top_ask is not None
        prediction_available = model_prob is not None
        model_blocker = ""
        if not prediction_available:
            model_blocker = str(context.get("blocker") or "")
            if not model_blocker and int(candidate_report.get("model_predictions_loaded", 0) or 0) <= 0:
                model_blocker = "model_predictions_file_missing_or_empty"
            if not model_blocker and spec.lane_id == "football_goals_core":
                selection = str(candidate.get("selection") or "")
                if selection in {"over", "under"} and _parse_total_goals_line(market) is None:
                    model_blocker = "line_parse_failed"
            if not model_blocker:
                model_blocker = "prediction_row_missing_for_selection"

        if candidate_status == "selected_candidate":
            final_blocker = "predicted"
        elif candidate_status == "policy_rejected":
            final_blocker = "policy_rejected"
        elif candidate_blocker == "model_probability_missing":
            final_blocker = model_blocker
        elif candidate_blocker:
            final_blocker = candidate_blocker
        elif prediction_available:
            final_blocker = "predicted"
        else:
            final_blocker = model_blocker or "prediction_row_missing_for_selection"

        raw_home, raw_away = _event_teams_from_row(market) if not market.empty else (None, None)
        league_code = str(context.get("league_code") or _league_code_from_market_row(market) if not market.empty else "")
        rows.append(
            {
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "lane_id": spec.lane_id,
                "benchmark_id": spec.benchmark_id,
                "event_slug": event_slug,
                "market_id": market_id,
                "market_subtype": str(candidate.get("market_subtype") or ""),
                "selection": str(candidate.get("selection") or ""),
                "contract_outcome": str(candidate.get("contract_outcome") or ""),
                "legacy_slice": int(candidate.get("legacy_slice") or 0),
                "league_code": league_code,
                "raw_home": str(context.get("raw_home") or raw_home or ""),
                "raw_away": str(context.get("raw_away") or raw_away or ""),
                "mapped_home": str(context.get("home_team") or ""),
                "mapped_away": str(context.get("away_team") or ""),
                "home_history_rows": int(context.get("home_history_rows") or 0),
                "away_history_rows": int(context.get("away_history_rows") or 0),
                "book_available": bool(book_available),
                "book_timestamp": str(candidate.get("book_timestamp") or ""),
                "top_ask": top_ask,
                "quoted_odds": _safe_float(candidate.get("quoted_odds")),
                "prediction_available": bool(prediction_available),
                "model_prob": model_prob,
                "edge": _safe_float(candidate.get("edge")),
                "ev": _safe_float(candidate.get("ev")),
                "policy_ready": bool(policy_context.get("policy_ready")),
                "candidate_status": candidate_status,
                "candidate_blocker": candidate_blocker,
                "model_blocker": model_blocker,
                "final_blocker": final_blocker,
                "next_action": _blocker_next_action(final_blocker),
            }
        )

    audit = pd.DataFrame(rows)
    summary = {
        "lane_id": spec.lane_id,
        "audit_rows": int(len(audit)),
        "final_blocker_counts": audit["final_blocker"].value_counts().to_dict() if not audit.empty else {},
        "candidate_blocker_counts": audit["candidate_blocker"].replace("", pd.NA).dropna().value_counts().to_dict() if not audit.empty else {},
        "model_blocker_counts": audit["model_blocker"].replace("", pd.NA).dropna().value_counts().to_dict() if not audit.empty else {},
        "legacy_final_blocker_counts": audit[audit["legacy_slice"].astype(int).eq(1)]["final_blocker"].value_counts().to_dict() if not audit.empty else {},
        "model_probability_missing_actionable": bool(
            audit[audit["candidate_blocker"].eq("model_probability_missing")]["final_blocker"].replace("", pd.NA).notna().all()
        ) if not audit.empty else True,
        "policy_reoptimized": False,
        "global_roi_actionable": False,
    }
    return audit, summary


def _raw_capture_health_report(spec: MarketLaneSpec, catalog: pd.DataFrame, books: pd.DataFrame, now: pd.Timestamp) -> dict[str, Any]:
    forward = _forward_active_catalog(catalog, now)
    if forward.empty:
        return {
            "lane_id": spec.lane_id,
            "future_active_markets": 0,
            "fresh_book_markets": 0,
            "stale_book_markets": 0,
            "capture_not_attempted_markets": 0,
            "fresh_book_rate": 0.0,
            "next_24h_fresh_book_rate": 0.0,
            "capture_status_counts": {},
            "book_market_coverage_rate": 0.0,
            "policy_reoptimized": False,
            "global_roi_actionable": False,
        }
    freshness_seconds = DEFAULT_RAW_CAPTURE_FRESHNESS_SECONDS
    forward_market_ids = set(forward["market_id"].dropna().astype(str))
    forward_books = (
        books[books["market_id"].astype(str).isin(forward_market_ids)].copy()
        if not books.empty and "market_id" in books.columns
        else pd.DataFrame()
    )
    latest_by_market = pd.DataFrame()
    if not forward_books.empty:
        working = forward_books.copy()
        working["_timestamp"] = pd.to_datetime(working.get("timestamp", pd.Series(dtype=str)), utc=True, errors="coerce")
        latest_by_market = working.sort_values("_timestamp").drop_duplicates("market_id", keep="last")
    latest_map = {
        str(row.get("market_id") or ""): row
        for _, row in latest_by_market.iterrows()
    } if not latest_by_market.empty else {}
    rows: list[dict[str, Any]] = []
    for _, market in forward.iterrows():
        market_id = str(market.get("market_id") or "")
        tokens = _json_list(market.get("clob_token_ids_json"))
        latest = latest_map.get(market_id)
        latest_ts = pd.to_datetime(latest.get("_timestamp"), utc=True, errors="coerce") if latest is not None else pd.NaT
        age = (now - latest_ts).total_seconds() if latest is not None and not pd.isna(latest_ts) else None
        if not tokens:
            status = "no_clob_token"
        elif latest is None:
            status = "capture_not_attempted"
        elif age is not None and age > freshness_seconds:
            status = "stale_book"
        else:
            status = "fresh_book"
        rows.append(
            {
                "market_id": market_id,
                "event_slug": str(market.get("event_slug") or ""),
                "market_subtype": str(market.get("market_subtype") or ""),
                "legacy_slice": int(market.get("legacy_slice") or 0),
                "capture_status": status,
                "latest_book_timestamp": "" if latest is None or pd.isna(latest_ts) else latest_ts.isoformat(),
                "latest_book_age_seconds": None if age is None else float(age),
                "clob_token_count": int(len(tokens)),
            }
        )
    health_rows = pd.DataFrame(rows)
    status_counts = health_rows["capture_status"].value_counts().to_dict() if not health_rows.empty else {}
    fresh_book_markets = int(status_counts.get("fresh_book", 0))
    stale_book_markets = int(status_counts.get("stale_book", 0))
    capture_not_attempted_markets = int(status_counts.get("capture_not_attempted", 0))
    starts = _market_start_series(forward)
    next_24h_market_ids = set(forward.loc[(starts >= now) & (starts <= now + pd.Timedelta(hours=24)), "market_id"].astype(str))
    next_24h_rows = health_rows[health_rows["market_id"].astype(str).isin(next_24h_market_ids)] if next_24h_market_ids else pd.DataFrame()
    next_24h_fresh_rate = (
        float(next_24h_rows["capture_status"].eq("fresh_book").sum() / len(next_24h_rows))
        if not next_24h_rows.empty
        else 0.0
    )
    forward_book_market_count = int(forward_books["market_id"].astype(str).nunique()) if not forward_books.empty else 0
    timestamps = pd.to_datetime(forward_books.get("timestamp", pd.Series(dtype=str)), utc=True, errors="coerce") if not forward_books.empty else pd.Series(dtype="datetime64[ns, UTC]")
    latest_book = timestamps.max() if len(timestamps) else pd.NaT
    return {
        "lane_id": spec.lane_id,
        "future_active_markets": int(len(forward)),
        "raw_orderbook_checkpoints": int(len(forward_books)),
        "markets_with_raw_books": forward_book_market_count,
        "book_market_coverage_rate": float(forward_book_market_count / len(forward)) if len(forward) else 0.0,
        "fresh_book_markets": fresh_book_markets,
        "stale_book_markets": stale_book_markets,
        "capture_not_attempted_markets": capture_not_attempted_markets,
        "fresh_book_rate": float(fresh_book_markets / len(forward)) if len(forward) else 0.0,
        "next_24h_fresh_book_rate": next_24h_fresh_rate,
        "capture_status_counts": status_counts,
        "freshness_reference_seconds": freshness_seconds,
        "checkpoints_last_1h": int((timestamps >= now - pd.Timedelta(hours=1)).sum()) if len(timestamps) else 0,
        "checkpoints_last_6h": int((timestamps >= now - pd.Timedelta(hours=6)).sum()) if len(timestamps) else 0,
        "latest_book_timestamp": "" if pd.isna(latest_book) else latest_book.isoformat(),
        "latest_book_age_seconds": None if pd.isna(latest_book) else float((now - latest_book).total_seconds()),
        "markets_needing_capture": int(health_rows["capture_status"].isin(["capture_not_attempted", "stale_book"]).sum()) if not health_rows.empty else 0,
        "next_action": "capture_raw_books_for_missing_or_stale_markets",
        "policy_reoptimized": False,
        "global_roi_actionable": False,
    }


def _football_goals_policy_research(
    spec: MarketLaneSpec,
    catalog: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    settlements: pd.DataFrame,
    policy_context: Mapping[str, Any] | None = None,
    readiness: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if spec.lane_id != "football_goals_core":
        return pd.DataFrame(), {}
    policy_ready = bool((policy_context or {}).get("policy_ready"))
    can_emit_picks = bool((readiness or {}).get("can_emit_picks")) if policy_ready else False
    policy_blocker = str((policy_context or {}).get("policy_blocker") or "")
    catalog_by_market = {
        str(row.get("market_id") or ""): row
        for _, row in catalog.iterrows()
    } if not catalog.empty else {}
    settlement_by_market = {
        str(row.get("market_id") or ""): row
        for _, row in settlements.iterrows()
    } if not settlements.empty else {}
    rows: list[dict[str, Any]] = []
    for _, candidate in candidate_rows.iterrows():
        market_id = str(candidate.get("market_id") or "")
        market = catalog_by_market.get(market_id, pd.Series(dtype=object))
        settlement = settlement_by_market.get(market_id, pd.Series(dtype=object))
        rows.append(
            {
                "lane_id": spec.lane_id,
                "benchmark_id": spec.benchmark_id,
                "event_slug": str(candidate.get("event_slug") or ""),
                "market_id": market_id,
                "game_start_time": str(candidate.get("game_start_time") or ""),
                "market_subtype": str(candidate.get("market_subtype") or ""),
                "line": _parse_total_goals_line(market) if str(candidate.get("market_subtype") or "") == "total_goals" else None,
                "selection": str(candidate.get("selection") or ""),
                "contract_outcome": str(candidate.get("contract_outcome") or ""),
                "model_prob": _safe_float(candidate.get("model_prob")),
                "top_ask": _safe_float(candidate.get("top_ask")),
                "quoted_odds": _safe_float(candidate.get("quoted_odds")),
                "edge": _safe_float(candidate.get("edge")),
                "ev": _safe_float(candidate.get("ev")),
                "candidate_status": str(candidate.get("candidate_status") or ""),
                "candidate_blocker": str(candidate.get("candidate_blocker") or ""),
                "settlement_status": str(settlement.get("status") or "") if not settlement.empty else "",
                "winning_outcome": str(settlement.get("winning_outcome") or "") if not settlement.empty else "",
                "policy_ready": policy_ready,
                "can_emit_picks": can_emit_picks,
            }
        )
    research = pd.DataFrame(rows)
    summary = {
        "lane_id": spec.lane_id,
        "benchmark_id": spec.benchmark_id,
        "research_rows": int(len(research)),
        "market_subtype_counts": research["market_subtype"].value_counts().to_dict() if not research.empty else {},
        "quote_ready_rows": int(research["top_ask"].notna().sum()) if not research.empty else 0,
        "scorable_rows": int(research["model_prob"].notna().sum()) if not research.empty else 0,
        "line_parseable_rows": int(research["line"].notna().sum()) if not research.empty and "line" in research.columns else 0,
        "candidate_blocker_counts": research["candidate_blocker"].replace("", pd.NA).dropna().value_counts().to_dict() if not research.empty else {},
        "policy_ready": policy_ready,
        "can_emit_picks": can_emit_picks,
        "policy_activation_status": str((policy_context or {}).get("policy_activation_status") or ("policy_ready" if policy_ready else "lane_policy_pending")),
        "policy_transfer_mode": str((policy_context or {}).get("policy_transfer_mode") or ("native_lane_policy" if policy_ready else "none")),
        "policy_blocker": policy_blocker,
        "roi_status": "not_actionable_until_forward_settled",
        "policy_reoptimized": False,
        "global_roi_actionable": False,
    }
    return research, summary


def _policy_selected_rows(rows: pd.DataFrame, policy: dict[str, Any], now: pd.Timestamp | None = None) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    frame = rows.copy()
    edge = float(policy.get("edge_threshold", 0.0) or 0.0)
    ev = float(policy.get("ev_threshold", 0.0) or 0.0)
    min_odds = float(policy.get("min_odds", 0.0) or 0.0)
    max_odds = float(policy.get("max_odds", 999.0) or 999.0)
    allowed_outcomes = {str(item) for item in policy.get("allowed_outcomes", []) or []}
    allowed_subtypes = {str(item) for item in policy.get("allowed_market_subtypes", []) or []}
    mask = (
        pd.to_numeric(frame.get("model_prob", pd.Series(dtype=float)), errors="coerce").notna()
        & pd.to_numeric(frame.get("top_ask", pd.Series(dtype=float)), errors="coerce").notna()
        & pd.to_numeric(frame.get("edge", pd.Series(dtype=float)), errors="coerce").ge(edge)
        & pd.to_numeric(frame.get("ev", pd.Series(dtype=float)), errors="coerce").ge(ev)
        & pd.to_numeric(frame.get("quoted_odds", pd.Series(dtype=float)), errors="coerce").ge(min_odds)
        & pd.to_numeric(frame.get("quoted_odds", pd.Series(dtype=float)), errors="coerce").le(max_odds)
    )
    if allowed_outcomes:
        mask &= frame.get("selection", pd.Series(dtype=str)).astype(str).isin(allowed_outcomes)
    if allowed_subtypes:
        mask &= frame.get("market_subtype", pd.Series(dtype=str)).astype(str).isin(allowed_subtypes)
    selected = frame[mask].copy()
    if selected.empty:
        return selected
    selected["_ev_sort"] = pd.to_numeric(selected.get("ev", pd.Series(dtype=float)), errors="coerce").fillna(-999.0)
    selected["_edge_sort"] = pd.to_numeric(selected.get("edge", pd.Series(dtype=float)), errors="coerce").fillna(-999.0)
    selected["_odds_sort"] = pd.to_numeric(selected.get("quoted_odds", pd.Series(dtype=float)), errors="coerce").fillna(999.0)
    selected = (
        selected.sort_values(["event_slug", "_ev_sort", "_edge_sort", "_odds_sort"], ascending=[True, False, False, True])
        .drop_duplicates("event_slug")
        .drop(columns=[column for column in ["_ev_sort", "_edge_sort", "_odds_sort"] if column in selected.columns])
    )
    return selected


def _policy_horizon_counts(rows: pd.DataFrame, now: pd.Timestamp) -> dict[str, int]:
    counts = {f"next_{hours}h": 0 for hours in (24, 48, 72, 168, 336)}
    if rows.empty or "game_start_time" not in rows.columns:
        return counts
    starts = pd.to_datetime(rows["game_start_time"], utc=True, errors="coerce")
    for hours in (24, 48, 72, 168, 336):
        counts[f"next_{hours}h"] = int((starts.ge(now) & starts.le(now + pd.Timedelta(hours=hours))).sum())
    return counts


def _lane_sample_status(spec: MarketLaneSpec, valid_decisions: int, settled_decisions: int, fresh_book_rate: float) -> dict[str, Any]:
    required_valid = int(spec.sample_requirements.get("valid_forward_decisions", 100))
    required_settled = int(spec.sample_requirements.get("settled_unique_decisions", 40))
    required_fresh = float(spec.sample_requirements.get("fresh_book_rate", 0.8))
    decision = classify_forward_sample(
        ForwardSampleInputs(
            valid_forward_decisions=int(valid_decisions),
            settled_decisions=int(settled_decisions),
            fresh_book_rate=float(fresh_book_rate),
        ),
        ForwardSampleThresholds(
            min_valid_forward_decisions=required_valid,
            min_settled_decisions=required_settled,
            min_fresh_book_rate=required_fresh,
        ),
    )
    status = decision.sample_status.value
    roi_display_mode = (
        "hidden_until_sample_ready"
        if decision.sample_status != SampleStatus.sample_ready
        else ("actionable" if decision.actionable_roi else "diagnostic_only")
    )
    return {
        "sample_status": status,
        "sample_ready": decision.sample_status == SampleStatus.sample_ready,
        "sample_blockers": list(decision.sample_blockers),
        "valid_forward_decisions": int(valid_decisions),
        "settled_decisions": int(settled_decisions),
        "settled_unique_decisions": int(settled_decisions),
        "fresh_book_rate": float(fresh_book_rate),
        "actionable_roi": bool(decision.actionable_roi),
        "can_reopen_decision_region_analysis": bool(decision.can_reopen_decision_region_analysis),
        "roi_display_mode": roi_display_mode,
        "capital_promotion_allowed": False,
        "required_valid_forward_decisions": required_valid,
        "required_settled_unique_decisions": required_settled,
        "required_fresh_book_rate": required_fresh,
        "roi_45_hypothesis": classify_edge_hypothesis(
            EdgeHypothesisInputs(
                lane_id=spec.lane_id,
                valid_forward_decisions=int(valid_decisions),
                settled_decisions=int(settled_decisions),
                fresh_book_rate=float(fresh_book_rate),
                observed_roi=None,
                roi_lower_bound_95=None,
                avg_clv=None,
                rolling_roi_min=None,
                promotion_status=PromotionStatus.not_actionable,
            )
        ).to_dict(),
    }


def create_market_lane_policy(
    settings: Settings,
    lane_id: str,
    overwrite: bool = False,
    now: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    now = now or _utcnow()
    spec = get_market_lane_spec(lane_id)
    if spec.lane_id != "football_goals_core":
        raise ValueError("La policy bootstrap v1 solo esta definida para football_goals_core.")

    lane_dir = _lane_output_dir(settings, spec.lane_id)
    lane_dir.mkdir(parents=True, exist_ok=True)
    policy_path = lane_dir / "policy_bundle.json"
    benchmark_path = lane_dir / LANE_POLICY_BENCHMARK_REPORT_FILENAME
    if policy_path.exists() and not overwrite:
        payload = _load_json_file(policy_path)
        benchmark = _load_json_file(benchmark_path)
        return {**payload, "benchmark_report": benchmark}, {
            "policy_bundle": policy_path,
            "lane_policy_benchmark_report": benchmark_path,
        }

    policy = dict(FOOTBALL_GOALS_CORE_BOOTSTRAP_POLICY)
    research_path = lane_dir / FOOTBALL_GOALS_POLICY_RESEARCH_ROWS_FILENAME
    research = pd.read_csv(research_path) if research_path.exists() else pd.DataFrame()
    selected = _policy_selected_rows(research, policy, now=now) if not research.empty else pd.DataFrame()
    policy_bundle = {
        "lane_id": spec.lane_id,
        "benchmark_id": spec.benchmark_id,
        "sport": spec.sport,
        "decision_type": spec.decision_type,
        "market_subtype": spec.market_subtype,
        "outcome_schema": list(spec.outcome_schema),
        "legacy_included": list(spec.legacy_included),
        "policy_ready": True,
        "policy_mode": "policy_ready",
        "policy_activation_status": "policy_ready",
        "policy_transfer_mode": "native_lane_policy",
        "policy_reoptimized": False,
        "thresholds_changed": False,
        "scopes_changed": False,
        "source_policy_bundle_path": "",
        "source_benchmark_id": spec.benchmark_id,
        "probability_source": "raw",
        "policy": policy,
        "research_status": "bootstrap_forward_sample_policy",
        "promotion_eligibility": "not_evaluated_forward_sample_required",
        "discipline": {
            "no_cross_lane_roi": True,
            "no_legacy_policy_inheritance": True,
            "no_threshold_optimization": True,
            "global_roi_actionable": False,
            "requires_forward_settlement_before_roi_claim": True,
        },
        "generated_at": now.isoformat(),
    }
    benchmark = {
        "lane_id": spec.lane_id,
        "benchmark_id": spec.benchmark_id,
        "policy_name": policy["policy_name"],
        "benchmark_status": "forward_sample_collection_ready",
        "policy_ready": True,
        "can_emit_picks_after_run_market_lane": True,
        "research_rows": int(len(research)),
        "quote_ready_rows": int(research["top_ask"].notna().sum()) if not research.empty and "top_ask" in research.columns else 0,
        "scorable_rows": int(research["model_prob"].notna().sum()) if not research.empty and "model_prob" in research.columns else 0,
        "selected_rows_after_policy": int(len(selected)),
        "selected_unique_events_after_policy": int(selected["event_slug"].nunique()) if not selected.empty and "event_slug" in selected.columns else 0,
        "selected_by_market_subtype": selected["market_subtype"].value_counts().to_dict() if not selected.empty and "market_subtype" in selected.columns else {},
        "selected_by_outcome": selected["selection"].value_counts().to_dict() if not selected.empty and "selection" in selected.columns else {},
        "selected_horizon_counts": _policy_horizon_counts(selected, now),
        "policy": policy,
        "sample_requirements": spec.sample_requirements,
        "roi_status": "not_actionable_until_forward_settled",
        "policy_reoptimized": False,
        "global_roi_actionable": False,
    }
    _write_json(policy_path, policy_bundle)
    _write_json(benchmark_path, benchmark)
    return {**policy_bundle, "benchmark_report": benchmark}, {
        "policy_bundle": policy_path,
        "lane_policy_benchmark_report": benchmark_path,
    }


def run_market_lane(
    settings: Settings,
    lane_id: str,
    db_path: Path | str | None = None,
    now: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    now = now or _utcnow()
    governance = get_lane_governance(lane_id)
    if governance.reason == UNKNOWN_LANE_GOVERNANCE_REASON:
        raise LaneGovernanceError(governance.reason)
    spec = get_market_lane_spec(lane_id)
    database_path = Path(db_path) if db_path else default_multi_market_db_path(settings)
    connection = init_multi_market_db(database_path)
    catalog = _lane_catalog(connection, lane_id)
    books = _lane_raw_orderbooks(connection, lane_id)
    settlements = _lane_raw_settlements(connection, lane_id)
    existing_ledger = _lane_forward_ledger(connection, lane_id)
    connection.close()

    run = create_run_context(settings.paths.runs_dir, f"{lane_id}_shadow")
    lane_dir = _lane_output_dir(settings, lane_id)
    policy_context = _lane_policy_context(settings, spec)
    forward_catalog = _forward_active_catalog(catalog, now)
    decision_inventory = _lane_decision_inventory(spec, catalog, now)
    readiness = _build_lane_readiness(spec, catalog, books, existing_ledger, now, policy_context=policy_context)
    if policy_context.get("policy_ready"):
        lane_policy_path = lane_dir / "policy_bundle.json"
        run_policy_path = run.run_dir / "policy_bundle.json"
        policy_context = {**policy_context, "policy_bundle_path": str(lane_policy_path)}
        policy_bundle = _lane_policy_bundle_payload(spec, policy_context)
        _write_json(lane_policy_path, policy_bundle)
        _write_json(run_policy_path, policy_bundle)
        readiness = {
            **readiness,
            "policy_bundle_path": str(lane_policy_path),
            "run_policy_bundle_path": str(run_policy_path),
        }
    candidate_rows, candidate_report = _build_lane_candidate_rows(
        settings=settings,
        spec=spec,
        catalog=catalog,
        books=books,
        now=now,
        policy_context=policy_context,
    )
    prediction_template = _lane_model_prediction_template(spec, candidate_rows)
    blocker_audit, blocker_audit_report = _build_lane_blocker_audit(
        settings=settings,
        spec=spec,
        catalog=catalog,
        candidate_rows=candidate_rows,
        candidate_report=candidate_report,
        policy_context=policy_context,
    )
    selected_ledger_rows = _selected_lane_ledger_rows(
        spec=spec,
        candidate_rows=candidate_rows,
        now=now,
        policy_mode=str(readiness.get("policy_mode_effective", spec.policy_mode)),
    )
    ledger_rows = selected_ledger_rows or _lane_ledger_rows(forward_catalog, now, readiness_by_lane={lane_id: readiness})
    ledger = pd.DataFrame(ledger_rows)
    if ledger.empty:
        ledger = pd.DataFrame(
            columns=[
                "decision_id",
                "lane_id",
                "market_id",
                "event_slug",
                "benchmark_id",
                "decision_time",
                "decision_status",
                "blocker",
                "selected_outcome",
                "model_mode",
                "policy_mode",
                "market_subtype",
                "legacy_slice",
                "created_at",
            ]
        )
    raw_capture_health = _raw_capture_health_report(spec, catalog, books, now)
    goals_research, goals_research_report = _football_goals_policy_research(
        spec,
        catalog,
        candidate_rows,
        settlements,
        policy_context=policy_context,
        readiness=readiness,
    )
    if selected_ledger_rows:
        connection = init_multi_market_db(database_path)
        _upsert_rows(connection, "mm_lane_forward_ledger", selected_ledger_rows)
        connection.close()

    can_emit_picks = bool(readiness["can_emit_picks"])
    ledger_status_counts = ledger["decision_status"].value_counts().to_dict() if not ledger.empty and "decision_status" in ledger.columns else {}
    valid_forward_decisions = int(ledger_status_counts.get("valid_forward_sample", 0))
    settled_decisions = int(ledger_status_counts.get("settled", 0))
    sample_status = _lane_sample_status(
        spec=spec,
        valid_decisions=valid_forward_decisions,
        settled_decisions=settled_decisions,
        fresh_book_rate=float(raw_capture_health.get("fresh_book_rate", 0.0) or 0.0),
    )
    manifest = {
        "lane_id": spec.lane_id,
        "benchmark_id": spec.benchmark_id,
        "sport": spec.sport,
        "sport_id": spec.sport_id,
        "sport_merge_group": spec.sport_merge_group,
        "decision_type": spec.decision_type,
        "market_subtype": spec.market_subtype,
        "outcome_schema": list(spec.outcome_schema),
        "legacy_included": list(spec.legacy_included),
        "reference_only": spec.reference_only,
        "lane_governance": _lane_governance_payload(governance),
        "governance_mode": governance.mode.value,
        "governance_reason": governance.reason,
        "model_mode": spec.model_mode,
        "policy_mode": readiness.get("policy_mode_effective", spec.policy_mode),
        "policy_activation_status": readiness.get("policy_activation_status"),
        "policy_transfer_mode": readiness.get("policy_transfer_mode"),
        "policy_bundle_path": readiness.get("policy_bundle_path", ""),
        "source_policy_bundle_path": readiness.get("source_policy_bundle_path", ""),
        "status": spec.status,
        "can_emit_picks": can_emit_picks,
        "can_emit_shadow_picks": governance.can_emit_shadow_picks and can_emit_picks,
        "can_emit_capital_picks": governance.can_emit_capital_picks,
        "pick_blocker": "" if can_emit_picks else ",".join(readiness["readiness_blockers"]),
        "policy_reoptimized": False,
        "global_roi_actionable": False,
        "readiness_status": readiness["readiness_status"],
        "readiness_blockers": readiness["readiness_blockers"],
    }
    sample_report = {
        **manifest,
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "database_path": str(database_path),
        "markets_in_lane": int(len(catalog)),
        "legacy_slice_markets": int(catalog.get("legacy_slice", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if not catalog.empty else 0,
        "global_slice_markets": int((catalog.get("legacy_slice", pd.Series(dtype=int)).fillna(0).astype(int) == 0).sum()) if not catalog.empty else 0,
        "market_subtype_counts": catalog["market_subtype"].value_counts().to_dict() if not catalog.empty and "market_subtype" in catalog.columns else {},
        "valid_decisions": valid_forward_decisions,
        "settled_decisions": settled_decisions,
        **sample_status,
        "forward_ledger_status_counts": ledger_status_counts,
        "forward_ledger_rows": int(len(ledger)),
        "decision_inventory": decision_inventory,
        "candidate_scoring": candidate_report,
        "blocker_audit": blocker_audit_report,
        "raw_capture_health": raw_capture_health,
        "lane_roi": None,
        "diagnostic_only": {
            "raw_roi": None,
            "settled_roi": None,
            "lane_roi": None,
            "reason": "ROI is not actionable until sample_ready"
            if not sample_status["sample_ready"]
            else "sample_ready allows decision-region analysis only; it is not capital promotion",
        },
        "promotion_status": "blocked_until_lane_sample_ready"
        if not sample_status["sample_ready"]
        else "sample_ready_analysis_only_no_capital_promotion",
        "portfolio_eligible": False,
        "lane_readiness": readiness,
    }
    legacy_parity = _legacy_parity_report(spec, catalog, readiness)

    lane_manifest_path = lane_dir / "lane_manifest.json"
    lane_ledger_path = lane_dir / "forward_ledger.csv"
    lane_sample_path = lane_dir / "sample_report.json"
    lane_readiness_path = lane_dir / "lane_readiness_report.json"
    lane_decision_inventory_path = lane_dir / "lane_decision_inventory_report.json"
    lane_candidate_rows_path = lane_dir / "lane_candidate_rows.csv"
    lane_candidate_report_path = lane_dir / "lane_candidate_report.json"
    lane_prediction_template_path = lane_dir / "model_prediction_template.csv"
    lane_blocker_audit_csv_path = lane_dir / LANE_BLOCKER_AUDIT_CSV_FILENAME
    lane_blocker_audit_json_path = lane_dir / LANE_BLOCKER_AUDIT_JSON_FILENAME
    lane_raw_capture_health_path = lane_dir / RAW_CAPTURE_HEALTH_REPORT_FILENAME
    run_manifest_path = run.run_dir / "lane_manifest.json"
    run_ledger_path = run.run_dir / "forward_ledger.csv"
    run_sample_path = run.run_dir / "sample_report.json"
    run_readiness_path = run.run_dir / "lane_readiness_report.json"
    run_decision_inventory_path = run.run_dir / "lane_decision_inventory_report.json"
    run_candidate_rows_path = run.run_dir / "lane_candidate_rows.csv"
    run_candidate_report_path = run.run_dir / "lane_candidate_report.json"
    run_prediction_template_path = run.run_dir / "model_prediction_template.csv"
    run_blocker_audit_csv_path = run.run_dir / LANE_BLOCKER_AUDIT_CSV_FILENAME
    run_blocker_audit_json_path = run.run_dir / LANE_BLOCKER_AUDIT_JSON_FILENAME
    run_raw_capture_health_path = run.run_dir / RAW_CAPTURE_HEALTH_REPORT_FILENAME
    _write_json(lane_manifest_path, manifest)
    _write_json(lane_sample_path, sample_report)
    _write_json(lane_readiness_path, readiness)
    _write_json(lane_decision_inventory_path, decision_inventory)
    _write_json(lane_candidate_report_path, candidate_report)
    _write_json(lane_blocker_audit_json_path, blocker_audit_report)
    _write_json(lane_raw_capture_health_path, raw_capture_health)
    _write_json(run_manifest_path, manifest)
    _write_json(run_sample_path, sample_report)
    _write_json(run_readiness_path, readiness)
    _write_json(run_decision_inventory_path, decision_inventory)
    _write_json(run_candidate_report_path, candidate_report)
    _write_json(run_blocker_audit_json_path, blocker_audit_report)
    _write_json(run_raw_capture_health_path, raw_capture_health)
    ledger.to_csv(lane_ledger_path, index=False)
    ledger.to_csv(run_ledger_path, index=False)
    candidate_rows.to_csv(lane_candidate_rows_path, index=False)
    candidate_rows.to_csv(run_candidate_rows_path, index=False)
    prediction_template.to_csv(lane_prediction_template_path, index=False)
    prediction_template.to_csv(run_prediction_template_path, index=False)
    blocker_audit.to_csv(lane_blocker_audit_csv_path, index=False)
    blocker_audit.to_csv(run_blocker_audit_csv_path, index=False)
    artifacts = {
        "lane_manifest": lane_manifest_path,
        "forward_ledger": lane_ledger_path,
        "sample_report": lane_sample_path,
        "lane_readiness_report": lane_readiness_path,
        "lane_decision_inventory_report": lane_decision_inventory_path,
        "lane_candidate_rows": lane_candidate_rows_path,
        "lane_candidate_report": lane_candidate_report_path,
        "model_prediction_template": lane_prediction_template_path,
        "lane_blocker_audit_csv": lane_blocker_audit_csv_path,
        "lane_blocker_audit_json": lane_blocker_audit_json_path,
        "raw_capture_health_report": lane_raw_capture_health_path,
        "run_lane_manifest": run_manifest_path,
        "run_forward_ledger": run_ledger_path,
        "run_sample_report": run_sample_path,
        "run_lane_readiness_report": run_readiness_path,
        "run_lane_decision_inventory_report": run_decision_inventory_path,
        "run_lane_candidate_rows": run_candidate_rows_path,
        "run_lane_candidate_report": run_candidate_report_path,
        "run_model_prediction_template": run_prediction_template_path,
        "run_lane_blocker_audit_csv": run_blocker_audit_csv_path,
        "run_lane_blocker_audit_json": run_blocker_audit_json_path,
        "run_raw_capture_health_report": run_raw_capture_health_path,
    }
    if readiness.get("policy_bundle_path"):
        artifacts["policy_bundle"] = Path(str(readiness["policy_bundle_path"]))
    if readiness.get("run_policy_bundle_path"):
        artifacts["run_policy_bundle"] = Path(str(readiness["run_policy_bundle_path"]))
    if spec.lane_id == "football_1x2_global":
        parity_path = lane_dir / "legacy_parity_report.json"
        run_parity_path = run.run_dir / "legacy_parity_report.json"
        _write_json(parity_path, legacy_parity)
        _write_json(run_parity_path, legacy_parity)
        artifacts["legacy_parity_report"] = parity_path
        artifacts["run_legacy_parity_report"] = run_parity_path
    if spec.lane_id == "football_goals_core":
        research_rows_path = lane_dir / FOOTBALL_GOALS_POLICY_RESEARCH_ROWS_FILENAME
        research_report_path = lane_dir / FOOTBALL_GOALS_POLICY_RESEARCH_REPORT_FILENAME
        run_research_rows_path = run.run_dir / FOOTBALL_GOALS_POLICY_RESEARCH_ROWS_FILENAME
        run_research_report_path = run.run_dir / FOOTBALL_GOALS_POLICY_RESEARCH_REPORT_FILENAME
        goals_research.to_csv(research_rows_path, index=False)
        goals_research.to_csv(run_research_rows_path, index=False)
        goals_research_report = {
            **goals_research_report,
            "research_rows_path": str(research_rows_path),
            "policy_blocker": str(policy_context.get("policy_blocker") or ""),
        }
        _write_json(research_report_path, goals_research_report)
        _write_json(run_research_report_path, goals_research_report)
        artifacts["football_goals_policy_research_rows"] = research_rows_path
        artifacts["football_goals_policy_research_report"] = research_report_path
        artifacts["run_football_goals_policy_research_rows"] = run_research_rows_path
        artifacts["run_football_goals_policy_research_report"] = run_research_report_path
    return sample_report, artifacts


def report_market_lane(
    settings: Settings,
    lane_id: str,
    run_dir: Path | str | None = None,
) -> tuple[dict[str, Any], str]:
    root = Path(run_dir) if run_dir else _lane_output_dir(settings, lane_id)
    sample_path = root / "sample_report.json"
    if not sample_path.exists():
        sample_path = root / "sample_report.json" if root.name == lane_id else root / "sample_report.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"No encuentro sample_report.json para el carril {lane_id}: {root}")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    candidate_scoring = payload.get("candidate_scoring", {})
    capture_health = payload.get("raw_capture_health", {})
    blocker_audit = payload.get("blocker_audit", {})
    primary_blockers = blocker_audit.get("final_blocker_counts") or candidate_scoring.get("candidate_blocker_counts") or {}
    if payload.get("lane_id") == "football_goals_core" and not payload.get("can_emit_picks"):
        next_action = "create_family_policy_benchmark_before_picks"
    elif int(candidate_scoring.get("quote_ready_candidates", 0) or 0) <= 0:
        next_action = "capture_raw_books_for_forward_markets"
    elif int(candidate_scoring.get("scorable_candidates", 0) or 0) <= 0:
        next_action = "run_build_market_lane_predictions_and_fix_mapping"
    elif int(candidate_scoring.get("selected_candidates", 0) or 0) <= 0:
        next_action = "collect_more_forward_sample_or_review_policy_rejections"
    else:
        next_action = "collect_forward_sample_with_frozen_policy"
    lines = [
        "Market lane report",
        f"- lane_id: {payload.get('lane_id')}",
        f"- benchmark_id: {payload.get('benchmark_id')}",
        f"- sport: {payload.get('sport')}",
        f"- readiness_status: {payload.get('readiness_status')}",
        f"- markets_in_lane: {payload.get('markets_in_lane', 0)}",
        f"- future_active_markets: {payload.get('lane_readiness', {}).get('future_active_markets', 0)}",
        f"- legacy_slice_markets: {payload.get('legacy_slice_markets', 0)}",
        f"- global_slice_markets: {payload.get('global_slice_markets', 0)}",
        f"- decision_opportunities_ready: {payload.get('lane_readiness', {}).get('decision_opportunities_ready', 0)}",
        f"- decision_inventory_status: {payload.get('lane_readiness', {}).get('decision_inventory_status')}",
        f"- candidate_scoring_status: {candidate_scoring.get('candidate_scoring_status')}",
        f"- prediction_coverage: {candidate_scoring.get('scorable_candidates', 0)}/{candidate_scoring.get('candidate_rows', 0)}",
        f"- book_coverage_rate: {capture_health.get('book_market_coverage_rate', payload.get('lane_readiness', {}).get('book_market_coverage_rate', 0.0))}",
        f"- fresh_book_rate: {capture_health.get('fresh_book_rate', 0.0)}",
        f"- fresh_book_markets: {capture_health.get('fresh_book_markets', 0)}",
        f"- stale_book_markets: {capture_health.get('stale_book_markets', 0)}",
        f"- capture_not_attempted_markets: {capture_health.get('capture_not_attempted_markets', 0)}",
        f"- next_24h_fresh_book_rate: {capture_health.get('next_24h_fresh_book_rate', 0.0)}",
        f"- quote_ready_candidates: {candidate_scoring.get('quote_ready_candidates', 0)}",
        f"- scorable_candidates: {candidate_scoring.get('scorable_candidates', 0)}",
        f"- selected_candidates: {candidate_scoring.get('selected_candidates', 0)}",
        f"- selected_candidate_horizon_counts: {candidate_scoring.get('selected_candidate_horizon_counts', {})}",
        f"- scorable_candidate_horizon_counts: {candidate_scoring.get('scorable_candidate_horizon_counts', {})}",
        f"- valid_forward_decisions: {payload.get('valid_decisions', 0)}",
        f"- settled_decisions: {payload.get('settled_decisions', 0)}",
        f"- sample_status: {payload.get('sample_status')}",
        f"- sample_blockers: {payload.get('sample_blockers', [])}",
        f"- actionable_roi: {str(payload.get('actionable_roi', False)).lower()}",
        f"- roi_display_mode: {payload.get('roi_display_mode')}",
        f"- can_reopen_decision_region_analysis: {str(payload.get('can_reopen_decision_region_analysis', False)).lower()}",
        f"- forward_ledger_status_counts: {payload.get('forward_ledger_status_counts', {})}",
        f"- primary_blockers: {primary_blockers}",
        f"- next_action: {next_action}",
        f"- policy_activation_status: {payload.get('policy_activation_status')}",
        f"- policy_transfer_mode: {payload.get('policy_transfer_mode')}",
        f"- can_emit_picks: {payload.get('can_emit_picks')}",
        f"- readiness_blockers: {','.join(payload.get('readiness_blockers', []))}",
        f"- promotion_status: {payload.get('promotion_status')}",
        f"- roi_45_hypothesis: {payload.get('roi_45_hypothesis', {}).get('status')}",
        f"- roi_45_blockers: {payload.get('roi_45_hypothesis', {}).get('blockers', [])}",
        "- global_roi_actionable: false",
    ]
    return payload, "\n".join(lines)


def report_sport_merge(
    settings: Settings,
    sport: str,
    db_path: Path | str | None = None,
    now: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    now = now or _utcnow()
    database_path = Path(db_path) if db_path else default_multi_market_db_path(settings)
    connection = init_multi_market_db(database_path)
    links = pd.read_sql_query(
        """
        SELECT l.*, c.event_title, c.market_slug, c.question, c.game_start_time
        FROM mm_lane_market_links l
        LEFT JOIN mm_market_catalog c ON c.market_id = l.market_id
        WHERE l.sport_merge_group = ?
        """,
        connection,
        params=(sport,),
    )
    connection.close()
    run = create_run_context(settings.paths.runs_dir, f"{sport}_sport_merge")
    if links.empty:
        features = pd.DataFrame(
            columns=[
                "fixture_key",
                "event_slug",
                "source_event_slugs",
                "sport_merge_group",
                "linked_lane_ids",
                "market_subtypes",
                "has_football_1x2_global",
                "has_football_goals_core",
                "legacy_slice",
                "global_roi_actionable",
                "promotion_allowed",
            ]
        )
    else:
        working = links.copy()
        fixture_keys: list[str] = []
        for _, row in working.iterrows():
            home, away = _event_teams_from_row(row)
            start = pd.to_datetime(row.get("game_start_time"), utc=True, errors="coerce")
            date_key = "" if pd.isna(start) else start.strftime("%Y-%m-%d")
            if str(sport) == "football" and home and away and date_key:
                fixture_keys.append(f"{date_key}|{home}|{away}")
            else:
                fixture_keys.append(str(row.get("event_slug") or ""))
        working["fixture_key"] = fixture_keys
        grouped = working.groupby("fixture_key", dropna=False)
        features = grouped.agg(
            event_slug=("event_slug", "first"),
            source_event_slugs=("event_slug", lambda values: ",".join(sorted(set(map(str, values))))),
            sport_merge_group=("sport_merge_group", "first"),
            linked_lane_ids=("lane_id", lambda values: ",".join(sorted(set(map(str, values))))),
            market_subtypes=("market_subtype", lambda values: ",".join(sorted(set(map(str, values))))),
            legacy_slice=("legacy_slice", "max"),
        ).reset_index()
        features["has_football_1x2_global"] = features["linked_lane_ids"].str.contains("football_1x2_global", regex=False)
        features["has_football_goals_core"] = features["linked_lane_ids"].str.contains("football_goals_core", regex=False)
        features["global_roi_actionable"] = False
        features["promotion_allowed"] = False

    features_path = run.run_dir / f"{sport}_sport_context_features.csv"
    report_path = run.run_dir / f"{sport}_sport_merge_report.json"
    stable_features_path = settings.paths.outputs_dir / "lanes" / f"{sport}_sport_context_features.csv"
    stable_report_path = settings.paths.outputs_dir / "lanes" / f"{sport}_sport_merge_report.json"
    stable_features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    features.to_csv(stable_features_path, index=False)
    common_lane_events = int(
        (features.get("has_football_1x2_global", pd.Series(dtype=bool)).astype(bool)
         & features.get("has_football_goals_core", pd.Series(dtype=bool)).astype(bool)).sum()
    ) if not features.empty else 0
    report = {
        "sport": sport,
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "database_path": str(database_path),
        "created_at": now.isoformat(),
        "sport_merge_group": sport,
        "events": int(len(features)),
        "lanes": sorted(set(links["lane_id"].astype(str))) if not links.empty else [],
        "common_lane_events": common_lane_events,
        "merge_key": "fixture_key",
        "feature_store_only": True,
        "global_roi_actionable": False,
        "portfolio_eligible": False,
        "promotion_allowed": False,
    }
    _write_json(report_path, report)
    _write_json(stable_report_path, report)
    return report, {
        "sport_context_features": features_path,
        "sport_merge_report": report_path,
        "stable_sport_context_features": stable_features_path,
        "stable_sport_merge_report": stable_report_path,
    }


def latest_multi_market_run(settings: Settings) -> Path | None:
    for pointer_name in ("latest_multi_market_capture.txt", "latest_multi_market_discovery.txt"):
        pointer = settings.paths.outputs_dir / pointer_name
        if pointer.exists():
            value = pointer.read_text(encoding="utf-8").strip()
            if value:
                path = Path(value)
                if path.exists():
                    return path
    return None
