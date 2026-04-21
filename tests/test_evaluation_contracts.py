from __future__ import annotations

import json
import unittest
from pathlib import Path

from predicciones.evaluation import ExecutionSummary, PolicySummary, PromotionGate, SignalSummary


class EvaluationContractTests(unittest.TestCase):
    def test_signal_summary_accepts_legacy_dicts_and_exports_json(self) -> None:
        summary = SignalSummary.from_dict(
            {
                "selected_source": "calibrated",
                "baseline": {"accuracy": 0.52},
                "model_raw": {"log_loss": 1.12},
                "model_calibrated": {"log_loss": 1.01},
                "feature_cleanup": {"dropped": ["empty_feature"]},
                "artifact_path": Path("signal.json"),
                "note": "legacy payload",
            }
        )

        self.assertEqual(summary.selected_probability_source, "calibrated")
        self.assertEqual(summary.baseline_metrics["accuracy"], 0.52)
        self.assertEqual(summary.raw_metrics["log_loss"], 1.12)
        self.assertEqual(summary.calibrated_metrics["log_loss"], 1.01)
        self.assertEqual(summary.feature_sanitization["dropped"], ["empty_feature"])
        self.assertEqual(summary.extra["note"], "legacy payload")
        self.assertEqual(summary.to_dict()["artifact_path"], "signal.json")
        self.assertEqual(json.loads(summary.to_json()), summary.to_dict())

    def test_policy_summary_round_trips_unknown_keys(self) -> None:
        summary = PolicySummary.from_dict(
            {
                "policy": {"name": "kelly"},
                "policy_candidates": [{"name": "flat"}, {"name": "kelly"}],
                "training_overall": {"roi": 0.12},
                "holdout_overall": {"roi": 0.08},
                "policy_objective": {"roi": 0.10, "drawdown": 0.02},
                "tuning_manifest": {"run_id": "abc123"},
            }
        )

        self.assertEqual(summary.selected_policy["name"], "kelly")
        self.assertEqual(len(summary.candidate_policies), 2)
        self.assertEqual(summary.training_metrics["roi"], 0.12)
        self.assertEqual(summary.holdout_metrics["roi"], 0.08)
        self.assertEqual(summary.objective["drawdown"], 0.02)
        self.assertEqual(summary.extra["tuning_manifest"], {"run_id": "abc123"})
        self.assertEqual(PolicySummary.from_dict(summary.to_dict()), summary)

    def test_execution_summary_and_promotion_gate_aliases(self) -> None:
        execution = ExecutionSummary.from_dict(
            {
                "decision_coverage": {"ready": 7, "blocked": 3},
                "blocker_counts": {"stale_book": 2},
                "fill_summary": {"fills": 5},
                "decision_summary": {"selected": 4},
            }
        )
        gate = PromotionGate.from_dict(
            {
                "promotable": "yes",
                "validation_stage": "shadow",
                "reasons": ["coverage_ready"],
                "requirements": ["shadow_summary"],
                "supporting_evidence": {"execution": execution.to_dict()},
                "opaque": True,
            }
        )

        self.assertEqual(execution.coverage["ready"], 7)
        self.assertEqual(execution.blockers["stale_book"], 2)
        self.assertEqual(execution.fills["fills"], 5)
        self.assertEqual(execution.decisions["selected"], 4)
        self.assertTrue(gate.passed)
        self.assertEqual(gate.stage, "shadow")
        self.assertEqual(gate.blockers, ["coverage_ready"])
        self.assertEqual(gate.required_inputs, ["shadow_summary"])
        self.assertEqual(gate.evidence["execution"]["coverage"]["ready"], 7)
        self.assertTrue(gate.extra["opaque"])
        self.assertEqual(PromotionGate.ensure(gate), gate)


if __name__ == "__main__":
    unittest.main()
