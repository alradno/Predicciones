from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, Self


def _take_first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.pop(key)
    return default


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    raise TypeError(f"Expected a mapping-compatible value, got {type(value)!r}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _as_bool_or_none(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "passed", "promotable", "approved"}:
            return True
        if normalized in {"false", "0", "no", "n", "blocked", "rejected"}:
            return False
    return bool(value)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        payload = {}
        for field in fields(value):
            if field.name == "extra":
                continue
            payload[field.name] = _jsonable(getattr(value, field.name))
        extra = getattr(value, "extra", None)
        if isinstance(extra, Mapping):
            payload.update({str(key): _jsonable(item) for key, item in extra.items()})
        return payload
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return value


class JsonSummaryMixin:
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field in fields(self):
            if field.name == "extra":
                continue
            payload[field.name] = _jsonable(getattr(self, field.name))
        payload.update({key: _jsonable(value) for key, value in self.extra.items()})
        return payload

    def to_json(self, *, indent: int = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=sort_keys)

    def evolve(self, **changes: Any) -> Self:
        return replace(self, **changes)

    def with_extra(self, **updates: Any) -> Self:
        merged = {**self.extra, **updates}
        return replace(self, extra=merged)

    @classmethod
    def ensure(cls, value: Any) -> Self:
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()  # type: ignore[call-arg]
        if isinstance(value, Mapping):
            return cls.from_dict(value)  # type: ignore[attr-defined]
        raise TypeError(f"Cannot coerce {type(value)!r} into {cls.__name__}")


@dataclass(frozen=True, slots=True)
class SignalSummary(JsonSummaryMixin):
    selected_probability_source: str | None = None
    selected_source_strategy: str | None = None
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    calibrated_metrics: dict[str, Any] = field(default_factory=dict)
    feature_sanitization: dict[str, Any] = field(default_factory=dict)
    quality_gate: dict[str, Any] = field(default_factory=dict)
    regional_diagnostics: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Self:
        payload = dict(data or {})
        return cls(
            selected_probability_source=_take_first(
                payload,
                "selected_probability_source",
                "selected_source",
                "probability_source",
            ),
            selected_source_strategy=_take_first(
                payload,
                "selected_source_strategy",
                "probability_strategy",
                "source_strategy",
            ),
            baseline_metrics=_as_dict(
                _take_first(payload, "baseline_metrics", "baseline", "market_baseline", default={})
            ),
            raw_metrics=_as_dict(_take_first(payload, "raw_metrics", "model_raw", "raw", default={})),
            calibrated_metrics=_as_dict(
                _take_first(payload, "calibrated_metrics", "model_calibrated", "calibrated", default={})
            ),
            feature_sanitization=_as_dict(
                _take_first(payload, "feature_sanitization", "sanitization", "feature_cleanup", default={})
            ),
            quality_gate=_as_dict(_take_first(payload, "quality_gate", "signal_gate", default={})),
            regional_diagnostics=_as_dict(
                _take_first(payload, "regional_diagnostics", "regional_source_diagnostics", default={})
            ),
            verdict=_take_first(payload, "verdict", "truth", "assessment"),
            extra=payload,
        )


@dataclass(frozen=True, slots=True)
class PolicySummary(JsonSummaryMixin):
    selected_policy: dict[str, Any] = field(default_factory=dict)
    candidate_policies: list[Any] = field(default_factory=list)
    training_metrics: dict[str, Any] = field(default_factory=dict)
    holdout_metrics: dict[str, Any] = field(default_factory=dict)
    objective: dict[str, Any] = field(default_factory=dict)
    inner_validation: dict[str, Any] = field(default_factory=dict)
    conservative_score_breakdown: dict[str, Any] = field(default_factory=dict)
    candidate_rankings: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Self:
        payload = dict(data or {})
        return cls(
            selected_policy=_as_dict(_take_first(payload, "selected_policy", "policy", "best_policy", default={})),
            candidate_policies=_as_list(
                _take_first(payload, "candidate_policies", "candidates", "policy_candidates", default=[])
            ),
            training_metrics=_as_dict(
                _take_first(payload, "training_metrics", "training_overall", "train", "training", default={})
            ),
            holdout_metrics=_as_dict(
                _take_first(payload, "holdout_metrics", "holdout_overall", "validation", "holdout", default={})
            ),
            objective=_as_dict(_take_first(payload, "objective", "policy_objective", default={})),
            inner_validation=_as_dict(_take_first(payload, "inner_validation", "validation_plan", default={})),
            conservative_score_breakdown=_as_dict(
                _take_first(payload, "conservative_score_breakdown", "score_breakdown", default={})
            ),
            candidate_rankings=_as_dict(_take_first(payload, "candidate_rankings", "rankings", default={})),
            verdict=_take_first(payload, "verdict", "truth", "assessment"),
            extra=payload,
        )


@dataclass(frozen=True, slots=True)
class ExecutionSummary(JsonSummaryMixin):
    coverage: dict[str, Any] = field(default_factory=dict)
    blockers: dict[str, Any] = field(default_factory=dict)
    fills: dict[str, Any] = field(default_factory=dict)
    decisions: dict[str, Any] = field(default_factory=dict)
    clv_summary: dict[str, Any] = field(default_factory=dict)
    fill_model_summary: dict[str, Any] = field(default_factory=dict)
    fill_adjusted_ev_summary: dict[str, Any] = field(default_factory=dict)
    sizing_summary: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Self:
        payload = dict(data or {})
        return cls(
            coverage=_as_dict(
                _take_first(payload, "coverage", "coverage_summary", "decision_coverage", "execution_viability", default={})
            ),
            blockers=_as_dict(_take_first(payload, "blockers", "blocker_counts", "failure_reasons", default={})),
            fills=_as_dict(_take_first(payload, "fills", "fill_summary", "execution", default={})),
            decisions=_as_dict(_take_first(payload, "decisions", "decision_summary", "quotes", default={})),
            clv_summary=_as_dict(_take_first(payload, "clv_summary", "clv", default={})),
            fill_model_summary=_as_dict(_take_first(payload, "fill_model_summary", "fill_model", default={})),
            fill_adjusted_ev_summary=_as_dict(
                _take_first(payload, "fill_adjusted_ev_summary", "fill_adjusted_ev", default={})
            ),
            sizing_summary=_as_dict(_take_first(payload, "sizing_summary", "sizing", default={})),
            verdict=_take_first(payload, "verdict", "truth", "assessment"),
            extra=payload,
        )


@dataclass(frozen=True, slots=True)
class PromotionGate(JsonSummaryMixin):
    passed: bool | None = None
    stage: str | None = None
    blockers: list[Any] = field(default_factory=list)
    required_inputs: list[Any] = field(default_factory=list)
    reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Self:
        payload = dict(data or {})
        return cls(
            passed=_as_bool_or_none(_take_first(payload, "passed", "promotable", "is_promotable", "approved")),
            stage=_take_first(payload, "stage", "validation_stage"),
            blockers=_as_list(_take_first(payload, "blockers", "reasons", "blocking_reasons", default=[])),
            required_inputs=_as_list(_take_first(payload, "required_inputs", "requirements", default=[])),
            reason=_take_first(payload, "reason", "message"),
            evidence=_as_dict(_take_first(payload, "evidence", "supporting_evidence", "signals", default={})),
            extra=payload,
        )
