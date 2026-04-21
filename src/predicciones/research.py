from __future__ import annotations

from .research_candidates import (
    _evaluate_policy,
    _probability_array,
    _roi_by_fold,
    build_candidate_rows,
    choose_probability_source,
    optimize_research_policy,
    select_candidate_bets,
    simulate_execution,
    summarize_execution,
)
from .research_niches import build_promotion_report, discover_niches
from .research_snapshots import (
    _categorize_time_bucket,
    _default_kickoff,
    _edge_band,
    _odds_band,
    _prepare_snapshot_frame,
    _season_phase,
    build_market_snapshots,
    capture_odds_for_dataset,
    load_snapshot_file,
)
from .research_summary import _build_research_summary, format_net_summary, load_research_run
from .research_workflows import run_net_backtest, shadow_run_from_bundle, train_niche_bundle

