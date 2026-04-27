from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


UNKNOWN_LANE_GOVERNANCE_REASON = "Unknown lane is blocked by governance."
ACTIVE_VALIDATION_REASON = "Active forward validation lane; capital blocked until promotion."
CAPTURE_ONLY_REASON = "Capture-only lane; predictions and pick emission are blocked until promotion."


class LaneGovernanceError(KeyError):
    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class LaneMode(str, Enum):
    roi_active = "roi_active"
    capture_only = "capture_only"
    reference_only = "reference_only"
    paused = "paused"


@dataclass(frozen=True)
class LaneGovernance:
    lane_id: str
    mode: LaneMode
    can_capture: bool
    can_build_predictions: bool
    can_emit_shadow_picks: bool
    can_emit_capital_picks: bool
    reason: str


LANE_GOVERNANCE: dict[str, LaneGovernance] = {
    "football_1x2_global": LaneGovernance(
        lane_id="football_1x2_global",
        mode=LaneMode.roi_active,
        can_capture=True,
        can_build_predictions=True,
        can_emit_shadow_picks=True,
        can_emit_capital_picks=False,
        reason=ACTIVE_VALIDATION_REASON,
    ),
    "football_goals_core": LaneGovernance(
        lane_id="football_goals_core",
        mode=LaneMode.roi_active,
        can_capture=True,
        can_build_predictions=True,
        can_emit_shadow_picks=True,
        can_emit_capital_picks=False,
        reason=ACTIVE_VALIDATION_REASON,
    ),
    "football_1x2_canonical": LaneGovernance(
        lane_id="football_1x2_canonical",
        mode=LaneMode.reference_only,
        can_capture=False,
        can_build_predictions=False,
        can_emit_shadow_picks=False,
        can_emit_capital_picks=False,
        reason="Legacy reference only.",
    ),
    "tennis_match_winner": LaneGovernance(
        lane_id="tennis_match_winner",
        mode=LaneMode.capture_only,
        can_capture=True,
        can_build_predictions=False,
        can_emit_shadow_picks=False,
        can_emit_capital_picks=False,
        reason=CAPTURE_ONLY_REASON,
    ),
    "basketball_moneyline": LaneGovernance(
        lane_id="basketball_moneyline",
        mode=LaneMode.capture_only,
        can_capture=True,
        can_build_predictions=False,
        can_emit_shadow_picks=False,
        can_emit_capital_picks=False,
        reason=CAPTURE_ONLY_REASON,
    ),
    "baseball_moneyline": LaneGovernance(
        lane_id="baseball_moneyline",
        mode=LaneMode.capture_only,
        can_capture=True,
        can_build_predictions=False,
        can_emit_shadow_picks=False,
        can_emit_capital_picks=False,
        reason=CAPTURE_ONLY_REASON,
    ),
    "hockey_moneyline": LaneGovernance(
        lane_id="hockey_moneyline",
        mode=LaneMode.capture_only,
        can_capture=True,
        can_build_predictions=False,
        can_emit_shadow_picks=False,
        can_emit_capital_picks=False,
        reason=CAPTURE_ONLY_REASON,
    ),
    "cricket_match_winner": LaneGovernance(
        lane_id="cricket_match_winner",
        mode=LaneMode.capture_only,
        can_capture=True,
        can_build_predictions=False,
        can_emit_shadow_picks=False,
        can_emit_capital_picks=False,
        reason=CAPTURE_ONLY_REASON,
    ),
}


def _unknown_lane_governance(lane_id: str) -> LaneGovernance:
    return LaneGovernance(
        lane_id=lane_id,
        mode=LaneMode.paused,
        can_capture=False,
        can_build_predictions=False,
        can_emit_shadow_picks=False,
        can_emit_capital_picks=False,
        reason=UNKNOWN_LANE_GOVERNANCE_REASON,
    )


def get_lane_governance(lane_id: str) -> LaneGovernance:
    return LANE_GOVERNANCE.get(str(lane_id), _unknown_lane_governance(str(lane_id)))


def _require_permission(lane_id: str, permission: str) -> LaneGovernance:
    governance = get_lane_governance(lane_id)
    if not bool(getattr(governance, permission)):
        raise LaneGovernanceError(governance.reason)
    return governance


def require_can_capture(lane_id: str) -> LaneGovernance:
    return _require_permission(lane_id, "can_capture")


def require_can_build_predictions(lane_id: str) -> LaneGovernance:
    return _require_permission(lane_id, "can_build_predictions")


def require_can_emit_shadow_picks(lane_id: str) -> LaneGovernance:
    return _require_permission(lane_id, "can_emit_shadow_picks")


def require_can_emit_capital_picks(lane_id: str) -> LaneGovernance:
    return _require_permission(lane_id, "can_emit_capital_picks")
