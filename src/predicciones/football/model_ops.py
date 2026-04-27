from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Settings
from ..lanes.runtime import get_market_lane_spec
from ..reporting import create_run_context
from .model_challenge import get_model_variant_spec, validate_model_candidate_contract
from .simulator import MODEL_NAME, train_football_simulator


def _candidate_dir(settings: Settings, lane_id: str, variant: str) -> Path:
    path = settings.paths.outputs_dir / "model_candidates" / lane_id / variant
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_lane_model_candidate(
    settings: Settings,
    lane_id: str,
    variant: str,
    dataset_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    lane_spec = get_market_lane_spec(lane_id)
    variant_spec = get_model_variant_spec(variant)
    if lane_spec.lane_id not in variant_spec.lane_ids:
        raise ValueError(f"Variant {variant} is not declared for lane {lane_id}.")

    run = create_run_context(settings.paths.runs_dir, f"{lane_id}_{variant}_model_train")
    candidate_dir = _candidate_dir(settings, lane_id, variant)
    resolved_dataset = Path(dataset_path) if dataset_path else settings.paths.outputs_dir / "sim_data" / "simulation_training_dataset.csv"
    summary: dict[str, Any] = {
        "lane_id": lane_spec.lane_id,
        "variant": variant,
        "variant_spec": variant_spec.to_dict(),
        "dataset_path": str(resolved_dataset),
        "training_status": "blocked_missing_dataset",
        "model_name": MODEL_NAME,
        "locked_holdout_used_for_training": False,
        "policy_reoptimized": False,
        "selected_probability_source": None,
        "notes": "Model candidates are evaluated as prediction improvements only; policy gates remain separate.",
    }
    artifacts: dict[str, Path] = {
        "candidate_manifest": candidate_dir / "candidate_manifest.json",
        "run_candidate_manifest": run.run_dir / "candidate_manifest.json",
    }

    if resolved_dataset.exists():
        result = train_football_simulator(
            settings=settings,
            dataset_path=resolved_dataset,
            model_name=MODEL_NAME,
            exclude_market_reference=True,
        )
        summary.update(
            {
                "training_status": result.summary.get("readiness_status", "trained"),
                "training_summary": result.summary,
                "selected_probability_source": result.summary.get("selected_probability_source"),
                "model_artifacts": {key: str(value) for key, value in result.artifacts.items()},
            }
        )
        artifacts.update(result.artifacts)

    for path in (artifacts["candidate_manifest"], artifacts["run_candidate_manifest"]):
        path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    latest_pointer = candidate_dir.parent / "latest_candidate.txt"
    latest_pointer.write_text(str(artifacts["candidate_manifest"]), encoding="utf-8")
    artifacts["latest_candidate_pointer"] = latest_pointer
    return summary, artifacts


def validate_lane_model_candidate(
    settings: Settings,
    lane_id: str,
    candidate: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    get_market_lane_spec(lane_id)
    candidate_path = Path(candidate)
    if not candidate_path.exists():
        candidate_path = _candidate_dir(settings, lane_id, candidate) / "candidate_manifest.json"

    run = create_run_context(settings.paths.runs_dir, f"{lane_id}_model_validate")
    if candidate_path.exists():
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "lane_id": lane_id,
            "variant": candidate,
            "locked_holdout_used_for_training": False,
            "policy_reoptimized": False,
            "validation_status": "blocked_missing_candidate",
        }

    ok, blockers = validate_model_candidate_contract(payload)
    summary = {
        "lane_id": lane_id,
        "candidate": candidate,
        "candidate_path": str(candidate_path),
        "validation_status": "valid" if ok and candidate_path.exists() else "blocked",
        "validation_blockers": list(blockers) + ([] if candidate_path.exists() else ["candidate_manifest_missing"]),
        "locked_holdout_used_for_training": bool(payload.get("locked_holdout_used_for_training", False)),
        "policy_reoptimized": bool(payload.get("policy_reoptimized", False)),
        "promotion_allowed": False,
    }
    artifacts = {"model_validation_report": run.run_dir / "model_validation_report.json"}
    artifacts["model_validation_report"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return summary, artifacts
