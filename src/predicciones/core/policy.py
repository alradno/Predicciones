from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EdgePolicySpec:
    policy_id: str
    lane_id: str
    edge_threshold: float
    ev_threshold: float
    min_odds: float
    max_odds: float
    max_picks_per_event: int = 1
    frozen: bool = True
    locked_holdout_used_for_training: bool = False

    def validate(self) -> None:
        if self.max_picks_per_event != 1:
            raise ValueError("EdgePolicySpec requires max_picks_per_event=1")
        if not self.frozen:
            raise ValueError("EdgePolicySpec must be frozen before forward use")
        if self.locked_holdout_used_for_training:
            raise ValueError("locked_holdout cannot be used to train policy gates")
        if self.edge_threshold < 0.0:
            raise ValueError("edge_threshold must be non-negative")
        if self.ev_threshold < 0.0:
            raise ValueError("ev_threshold must be non-negative")
        if self.min_odds <= 1.0:
            raise ValueError("min_odds must be greater than 1.0")
        if self.max_odds < self.min_odds:
            raise ValueError("max_odds must be greater than or equal to min_odds")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)
