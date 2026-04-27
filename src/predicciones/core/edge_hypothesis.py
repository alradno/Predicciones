from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .promotion_state import PromotionStatus


EDGE_TARGET_ROI = 0.45
EDGE_TARGET_MODE = "honest_forward_hypothesis"


class EdgeHypothesisStatus(str, Enum):
    collecting = "collecting"
    rejected = "rejected"
    supported = "supported"
    live_micro_allowed = "live_micro_allowed"


@dataclass(frozen=True)
class EdgeHypothesisInputs:
    lane_id: str
    valid_forward_decisions: int
    settled_decisions: int
    fresh_book_rate: float
    observed_roi: float | None
    roi_lower_bound_95: float | None
    avg_clv: float | None
    rolling_roi_min: float | None
    promotion_status: PromotionStatus | str = PromotionStatus.not_actionable
    live_gate_passed: bool = False


@dataclass(frozen=True)
class EdgeHypothesisDecision:
    lane_id: str
    target_roi: float
    target_mode: str
    status: EdgeHypothesisStatus
    blockers: tuple[str, ...]
    observed_roi: float | None
    roi_lower_bound_95: float | None
    avg_clv: float | None
    rolling_roi_min: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "lane_id": self.lane_id,
            "target_roi": self.target_roi,
            "target_mode": self.target_mode,
            "status": self.status.value,
            "blockers": list(self.blockers),
            "observed_roi": self.observed_roi,
            "roi_lower_bound_95": self.roi_lower_bound_95,
            "avg_clv": self.avg_clv,
            "rolling_roi_min": self.rolling_roi_min,
            "actionable_as_capital": self.status == EdgeHypothesisStatus.live_micro_allowed,
        }


def _value(value: object) -> str:
    return getattr(value, "value", str(value))


def classify_edge_hypothesis(
    inputs: EdgeHypothesisInputs,
    *,
    target_roi: float = EDGE_TARGET_ROI,
    min_valid_forward_decisions: int = 100,
    min_settled_decisions: int = 300,
    min_fresh_book_rate: float = 0.80,
) -> EdgeHypothesisDecision:
    blockers: list[str] = []

    if inputs.fresh_book_rate < min_fresh_book_rate:
        blockers.append(
            f"fresh_book_rate_below_minimum:{inputs.fresh_book_rate:.4f}<"
            f"{min_fresh_book_rate:.4f}"
        )
    if inputs.valid_forward_decisions < min_valid_forward_decisions:
        blockers.append(
            f"valid_forward_decisions_below_minimum:{inputs.valid_forward_decisions}<"
            f"{min_valid_forward_decisions}"
        )
    if inputs.settled_decisions < min_settled_decisions:
        blockers.append(
            f"settled_decisions_below_roi45_minimum:{inputs.settled_decisions}<"
            f"{min_settled_decisions}"
        )

    if blockers:
        status = EdgeHypothesisStatus.collecting
        return EdgeHypothesisDecision(
            lane_id=inputs.lane_id,
            target_roi=target_roi,
            target_mode=EDGE_TARGET_MODE,
            status=status,
            blockers=tuple(blockers),
            observed_roi=inputs.observed_roi,
            roi_lower_bound_95=inputs.roi_lower_bound_95,
            avg_clv=inputs.avg_clv,
            rolling_roi_min=inputs.rolling_roi_min,
        )

    if inputs.observed_roi is None:
        blockers.append("observed_roi_missing")
    elif inputs.observed_roi < target_roi:
        blockers.append(f"observed_roi_below_target:{inputs.observed_roi:.4f}<{target_roi:.4f}")

    if inputs.roi_lower_bound_95 is None:
        blockers.append("roi_lower_bound_95_missing")
    elif inputs.roi_lower_bound_95 <= 0.0:
        blockers.append(f"roi_lower_bound_95_not_positive:{inputs.roi_lower_bound_95:.4f}<=0.0000")

    if inputs.avg_clv is None:
        blockers.append("avg_clv_missing")
    elif inputs.avg_clv < 0.0:
        blockers.append(f"avg_clv_negative:{inputs.avg_clv:.4f}<0.0000")

    if inputs.rolling_roi_min is None:
        blockers.append("rolling_roi_min_missing")
    elif inputs.rolling_roi_min < 0.0:
        blockers.append(f"rolling_roi_min_negative:{inputs.rolling_roi_min:.4f}<0.0000")

    if blockers:
        status = EdgeHypothesisStatus.rejected
    elif _value(inputs.promotion_status) == PromotionStatus.capital_promotable.value and inputs.live_gate_passed:
        status = EdgeHypothesisStatus.live_micro_allowed
    else:
        status = EdgeHypothesisStatus.supported

    return EdgeHypothesisDecision(
        lane_id=inputs.lane_id,
        target_roi=target_roi,
        target_mode=EDGE_TARGET_MODE,
        status=status,
        blockers=tuple(blockers),
        observed_roi=inputs.observed_roi,
        roi_lower_bound_95=inputs.roi_lower_bound_95,
        avg_clv=inputs.avg_clv,
        rolling_roi_min=inputs.rolling_roi_min,
    )
