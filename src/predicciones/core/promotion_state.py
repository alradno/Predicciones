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
