from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ModelVariantRole(str, Enum):
    champion = "champion"
    challenger = "challenger"


@dataclass(frozen=True)
class ModelVariantSpec:
    variant_id: str
    role: ModelVariantRole
    lane_ids: tuple[str, ...]
    estimator: str
    calibration: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["role"] = self.role.value
        payload["lane_ids"] = list(self.lane_ids)
        return payload


MODEL_VARIANTS: dict[str, ModelVariantSpec] = {
    "champion_poisson_elo_calibrated": ModelVariantSpec(
        variant_id="champion_poisson_elo_calibrated",
        role=ModelVariantRole.champion,
        lane_ids=("football_1x2_global", "football_goals_core"),
        estimator="current_poisson_elo_goal_model",
        calibration="isotonic",
        description="Current champion adapter: Poisson goal lambdas with Elo/form features and calibration.",
    ),
    "challenger_poisson_features_v2": ModelVariantSpec(
        variant_id="challenger_poisson_features_v2",
        role=ModelVariantRole.challenger,
        lane_ids=("football_1x2_global", "football_goals_core"),
        estimator="poisson_goal_model_with_existing_feature_set",
        calibration="isotonic",
        description="Poisson challenger using the leakage-free feature matrix already present in football.dataset.",
    ),
    "challenger_hgb_poisson_calibrated": ModelVariantSpec(
        variant_id="challenger_hgb_poisson_calibrated",
        role=ModelVariantRole.challenger,
        lane_ids=("football_1x2_global", "football_goals_core"),
        estimator="HistGradientBoostingRegressor(loss='poisson')",
        calibration="isotonic",
        description="Nonlinear Poisson challenger evaluated against log-loss, Brier and CLV.",
    ),
}


def get_model_variant_spec(variant_id: str) -> ModelVariantSpec:
    try:
        return MODEL_VARIANTS[variant_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model variant: {variant_id}") from exc


def choose_probability_source(
    raw_metrics: dict[str, float],
    calibrated_metrics: dict[str, float],
    *,
    min_log_loss_delta: float = 0.0,
    require_non_worse_brier: bool = True,
) -> str:
    raw_log_loss = float(raw_metrics.get("log_loss", float("inf")))
    calibrated_log_loss = float(calibrated_metrics.get("log_loss", float("inf")))
    raw_brier = float(raw_metrics.get("brier_score", float("inf")))
    calibrated_brier = float(calibrated_metrics.get("brier_score", float("inf")))

    log_loss_delta = raw_log_loss - calibrated_log_loss
    brier_ok = calibrated_brier <= raw_brier if require_non_worse_brier else True
    if log_loss_delta >= min_log_loss_delta and brier_ok:
        return "calibrated"
    return "raw"


def validate_model_candidate_contract(payload: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    if bool(payload.get("locked_holdout_used_for_training")):
        blockers.append("locked_holdout_used_for_training")
    if bool(payload.get("policy_reoptimized")):
        blockers.append("policy_reoptimized")
    if str(payload.get("lane_id") or "") not in {"football_1x2_global", "football_goals_core"}:
        blockers.append("lane_not_in_initial_model_scope")
    return not blockers, tuple(blockers)
