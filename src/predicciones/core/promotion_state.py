from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SampleStatus(str, Enum):
    collecting_forward_sample = "collecting_forward_sample"
    coverage_blocked = "coverage_blocked"
    settlement_pending = "settlement_pending"
    sample_ready = "sample_ready"


class PromotionStatus(str, Enum):
    not_actionable = "not_actionable"
    analysis_ready = "analysis_ready"
    paper_promotable = "paper_promotable"
    capital_promotable = "capital_promotable"
    suspended = "suspended"


@dataclass(frozen=True)
class ForwardSampleInputs:
    valid_forward_decisions: int
    settled_decisions: int
    fresh_book_rate: float
    selected_candidates: int = 0


@dataclass(frozen=True)
class ForwardSampleThresholds:
    min_valid_forward_decisions: int = 100
    min_settled_decisions: int = 40
    min_fresh_book_rate: float = 0.80


@dataclass(frozen=True)
class ForwardSampleDecision:
    sample_status: SampleStatus
    sample_blockers: tuple[str, ...]
    actionable_roi: bool
    can_reopen_decision_region_analysis: bool


@dataclass(frozen=True)
class PromotionInputs:
    valid_forward_decisions: int
    settled_decisions: int
    fresh_book_rate: float
    roi_lower_bound_95: float | None = None
    avg_clv: float | None = None
    max_drawdown_units: float | None = None


@dataclass(frozen=True)
class PromotionThresholds:
    analysis_min_valid_forward_decisions: int = 100
    analysis_min_settled_decisions: int = 40
    analysis_min_fresh_book_rate: float = 0.80
    paper_min_settled_decisions: int = 150
    paper_min_roi_lower_bound: float = 0.0
    paper_min_avg_clv: float = 0.0
    capital_min_settled_decisions: int = 300
    capital_min_roi_lower_bound: float = 0.0
    capital_min_avg_clv: float = 0.0
    capital_max_drawdown_units: float = 50.0


@dataclass(frozen=True)
class PromotionDecision:
    promotion_status: PromotionStatus
    promotion_blockers: tuple[str, ...]
    actionable_roi: bool
    paper_stake_allowed: bool
    capital_stake_allowed: bool


def classify_forward_sample(
    inputs: ForwardSampleInputs,
    thresholds: ForwardSampleThresholds = ForwardSampleThresholds(),
) -> ForwardSampleDecision:
    blockers: list[str] = []

    if inputs.fresh_book_rate < thresholds.min_fresh_book_rate:
        blockers.append(
            f"fresh_book_rate_below_minimum:{inputs.fresh_book_rate:.4f}<"
            f"{thresholds.min_fresh_book_rate:.4f}"
        )
        return ForwardSampleDecision(
            sample_status=SampleStatus.coverage_blocked,
            sample_blockers=tuple(blockers),
            actionable_roi=False,
            can_reopen_decision_region_analysis=False,
        )

    if inputs.valid_forward_decisions < thresholds.min_valid_forward_decisions:
        blockers.append(
            f"valid_forward_decisions_below_minimum:{inputs.valid_forward_decisions}<"
            f"{thresholds.min_valid_forward_decisions}"
        )
        return ForwardSampleDecision(
            sample_status=SampleStatus.collecting_forward_sample,
            sample_blockers=tuple(blockers),
            actionable_roi=False,
            can_reopen_decision_region_analysis=False,
        )

    if inputs.settled_decisions < thresholds.min_settled_decisions:
        blockers.append(
            f"settled_decisions_below_minimum:{inputs.settled_decisions}<"
            f"{thresholds.min_settled_decisions}"
        )
        return ForwardSampleDecision(
            sample_status=SampleStatus.settlement_pending,
            sample_blockers=tuple(blockers),
            actionable_roi=False,
            can_reopen_decision_region_analysis=False,
        )

    return ForwardSampleDecision(
        sample_status=SampleStatus.sample_ready,
        sample_blockers=(),
        actionable_roi=False,
        can_reopen_decision_region_analysis=True,
    )


def _paper_promotion_blockers(
    inputs: PromotionInputs,
    thresholds: PromotionThresholds,
) -> tuple[str, ...]:
    blockers: list[str] = []

    if inputs.settled_decisions < thresholds.paper_min_settled_decisions:
        blockers.append(
            f"settled_decisions_below_paper_minimum:{inputs.settled_decisions}<"
            f"{thresholds.paper_min_settled_decisions}"
        )

    if inputs.roi_lower_bound_95 is None:
        blockers.append("roi_lower_bound_95_missing")
    elif inputs.roi_lower_bound_95 <= thresholds.paper_min_roi_lower_bound:
        blockers.append(
            f"roi_lower_bound_95_below_paper_minimum:{inputs.roi_lower_bound_95:.4f}<="
            f"{thresholds.paper_min_roi_lower_bound:.4f}"
        )

    if inputs.avg_clv is None:
        blockers.append("avg_clv_missing")
    elif inputs.avg_clv < thresholds.paper_min_avg_clv:
        blockers.append(
            f"avg_clv_below_paper_minimum:{inputs.avg_clv:.4f}<"
            f"{thresholds.paper_min_avg_clv:.4f}"
        )

    return tuple(blockers)


def _capital_promotion_blockers(
    inputs: PromotionInputs,
    thresholds: PromotionThresholds,
) -> tuple[str, ...]:
    blockers: list[str] = []

    if inputs.settled_decisions < thresholds.capital_min_settled_decisions:
        blockers.append(
            f"settled_decisions_below_capital_minimum:{inputs.settled_decisions}<"
            f"{thresholds.capital_min_settled_decisions}"
        )

    if inputs.roi_lower_bound_95 is None:
        blockers.append("roi_lower_bound_95_missing")
    elif inputs.roi_lower_bound_95 <= thresholds.capital_min_roi_lower_bound:
        blockers.append(
            f"roi_lower_bound_95_below_capital_minimum:{inputs.roi_lower_bound_95:.4f}<="
            f"{thresholds.capital_min_roi_lower_bound:.4f}"
        )

    if inputs.avg_clv is None:
        blockers.append("avg_clv_missing")
    elif inputs.avg_clv < thresholds.capital_min_avg_clv:
        blockers.append(
            f"avg_clv_below_capital_minimum:{inputs.avg_clv:.4f}<"
            f"{thresholds.capital_min_avg_clv:.4f}"
        )

    if inputs.max_drawdown_units is None:
        blockers.append("max_drawdown_units_missing")
    elif inputs.max_drawdown_units > thresholds.capital_max_drawdown_units:
        blockers.append(
            f"max_drawdown_units_above_capital_maximum:{inputs.max_drawdown_units:.4f}>"
            f"{thresholds.capital_max_drawdown_units:.4f}"
        )

    return tuple(blockers)


def classify_promotion_status(
    inputs: PromotionInputs,
    thresholds: PromotionThresholds = PromotionThresholds(),
) -> PromotionDecision:
    sample_decision = classify_forward_sample(
        ForwardSampleInputs(
            valid_forward_decisions=inputs.valid_forward_decisions,
            settled_decisions=inputs.settled_decisions,
            fresh_book_rate=inputs.fresh_book_rate,
        ),
        ForwardSampleThresholds(
            min_valid_forward_decisions=thresholds.analysis_min_valid_forward_decisions,
            min_settled_decisions=thresholds.analysis_min_settled_decisions,
            min_fresh_book_rate=thresholds.analysis_min_fresh_book_rate,
        ),
    )

    if sample_decision.sample_status != SampleStatus.sample_ready:
        return PromotionDecision(
            promotion_status=PromotionStatus.not_actionable,
            promotion_blockers=sample_decision.sample_blockers,
            actionable_roi=False,
            paper_stake_allowed=False,
            capital_stake_allowed=False,
        )

    paper_blockers = _paper_promotion_blockers(inputs, thresholds)
    if paper_blockers:
        promotion_blockers = list(paper_blockers)
        if inputs.settled_decisions >= thresholds.capital_min_settled_decisions:
            if inputs.max_drawdown_units is None:
                promotion_blockers.append("max_drawdown_units_missing")
            elif inputs.max_drawdown_units > thresholds.capital_max_drawdown_units:
                promotion_blockers.append(
                    f"max_drawdown_units_above_capital_maximum:{inputs.max_drawdown_units:.4f}>"
                    f"{thresholds.capital_max_drawdown_units:.4f}"
                )

        return PromotionDecision(
            promotion_status=PromotionStatus.analysis_ready,
            promotion_blockers=tuple(promotion_blockers),
            actionable_roi=False,
            paper_stake_allowed=False,
            capital_stake_allowed=False,
        )

    capital_blockers = _capital_promotion_blockers(inputs, thresholds)
    if capital_blockers:
        return PromotionDecision(
            promotion_status=PromotionStatus.paper_promotable,
            promotion_blockers=capital_blockers,
            actionable_roi=True,
            paper_stake_allowed=True,
            capital_stake_allowed=False,
        )

    return PromotionDecision(
        promotion_status=PromotionStatus.capital_promotable,
        promotion_blockers=(),
        actionable_roi=True,
        paper_stake_allowed=True,
        capital_stake_allowed=True,
    )
