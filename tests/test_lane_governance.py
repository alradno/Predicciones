from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from predicciones.config import BacktestConfig, ExecutionConfig, PolymarketConfig, ProjectPaths, ResearchConfig, Settings, SnapshotConfig
from predicciones.core.lane_governance import (
    ACTIVE_VALIDATION_REASON,
    CAPTURE_ONLY_REASON,
    LANE_GOVERNANCE,
    UNKNOWN_LANE_GOVERNANCE_REASON,
    LaneGovernanceError,
    LaneMode,
    get_lane_governance,
    require_can_build_predictions,
    require_can_capture,
    require_can_emit_capital_picks,
    require_can_emit_shadow_picks,
)
from predicciones.lanes.runtime import build_market_lane_predictions, default_multi_market_db_path, run_market_lane
from predicciones.app.pipeline import capture_market_raw_pipeline


def _settings(tmpdir: str) -> Settings:
    root = Path(tmpdir)
    paths = ProjectPaths(
        root=root,
        data_dir=root / "data",
        outputs_dir=root / "outputs",
        runs_dir=root / "outputs" / "runs",
        models_dir=root / "outputs" / "models",
        benchmarks_dir=root / "benchmarks",
    )
    for path in (paths.data_dir, paths.outputs_dir, paths.runs_dir, paths.models_dir, paths.benchmarks_dir):
        path.mkdir(parents=True, exist_ok=True)
    return Settings(
        paths=paths,
        anthropic_api_key=None,
        claude_model="none",
        default_leagues=("E0",),
        default_seasons=("2526",),
        benchmark_dir_name="legacy_v1",
        snapshot=SnapshotConfig(),
        execution=ExecutionConfig(),
        research=ResearchConfig(),
        polymarket=PolymarketConfig(checkpoint_interval_seconds=1),
        backtest=BacktestConfig(),
    )


class LaneGovernanceTests(unittest.TestCase):
    def test_core_football_lanes_can_emit_shadow_but_not_capital(self) -> None:
        for lane_id in ("football_1x2_global", "football_goals_core"):
            governance = get_lane_governance(lane_id)

            self.assertEqual(governance.mode, LaneMode.roi_active)
            self.assertTrue(governance.can_capture)
            self.assertTrue(governance.can_build_predictions)
            self.assertTrue(governance.can_emit_shadow_picks)
            self.assertFalse(governance.can_emit_capital_picks)
            self.assertEqual(governance.reason, ACTIVE_VALIDATION_REASON)
            self.assertEqual(require_can_emit_shadow_picks(lane_id), governance)
            with self.assertRaisesRegex(LaneGovernanceError, ACTIVE_VALIDATION_REASON):
                require_can_emit_capital_picks(lane_id)

        self.assertFalse(any(governance.can_emit_capital_picks for governance in LANE_GOVERNANCE.values()))

    def test_capture_only_lanes_can_capture_but_not_emit_picks(self) -> None:
        capture_only_lanes = (
            "tennis_match_winner",
            "basketball_moneyline",
            "baseball_moneyline",
            "hockey_moneyline",
            "cricket_match_winner",
        )

        for lane_id in capture_only_lanes:
            governance = get_lane_governance(lane_id)

            self.assertEqual(governance.mode, LaneMode.capture_only)
            self.assertTrue(governance.can_capture)
            self.assertFalse(governance.can_build_predictions)
            self.assertFalse(governance.can_emit_shadow_picks)
            self.assertFalse(governance.can_emit_capital_picks)
            self.assertEqual(require_can_capture(lane_id), governance)
            with self.assertRaisesRegex(LaneGovernanceError, CAPTURE_ONLY_REASON):
                require_can_build_predictions(lane_id)
            with self.assertRaisesRegex(LaneGovernanceError, CAPTURE_ONLY_REASON):
                require_can_emit_shadow_picks(lane_id)

    def test_reference_only_lane_cannot_emit_picks(self) -> None:
        governance = get_lane_governance("football_1x2_canonical")

        self.assertEqual(governance.mode, LaneMode.reference_only)
        self.assertFalse(governance.can_capture)
        self.assertFalse(governance.can_build_predictions)
        self.assertFalse(governance.can_emit_shadow_picks)
        self.assertFalse(governance.can_emit_capital_picks)
        with self.assertRaisesRegex(LaneGovernanceError, "Legacy reference only."):
            require_can_emit_shadow_picks("football_1x2_canonical")

    def test_unknown_lane_is_blocked_by_default(self) -> None:
        governance = get_lane_governance("unknown_lane")

        self.assertEqual(governance.mode, LaneMode.paused)
        self.assertFalse(governance.can_capture)
        self.assertFalse(governance.can_build_predictions)
        self.assertFalse(governance.can_emit_shadow_picks)
        self.assertFalse(governance.can_emit_capital_picks)
        self.assertEqual(governance.reason, UNKNOWN_LANE_GOVERNANCE_REASON)
        with self.assertRaisesRegex(LaneGovernanceError, UNKNOWN_LANE_GOVERNANCE_REASON):
            require_can_capture("unknown_lane")

    def test_capture_market_raw_allows_capture_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            result = capture_market_raw_pipeline(
                settings=settings,
                db_path=default_multi_market_db_path(settings),
                lane_id="tennis_match_winner",
            )

            self.assertEqual(result.summary["capture_lane_filter"], "tennis_match_winner")

    def test_build_predictions_blocks_capture_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            with self.assertRaisesRegex(LaneGovernanceError, CAPTURE_ONLY_REASON):
                build_market_lane_predictions(settings=settings, lane_id="tennis_match_winner")

    def test_run_market_lane_blocks_capture_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            summary, _ = run_market_lane(
                settings=settings,
                lane_id="tennis_match_winner",
                db_path=default_multi_market_db_path(settings),
            )

            self.assertFalse(summary["can_emit_picks"])
            self.assertFalse(summary["can_emit_shadow_picks"])
            self.assertFalse(summary["can_emit_capital_picks"])
            self.assertEqual(summary["lane_governance"]["mode"], "capture_only")
            self.assertIn("lane_governance_blocks_shadow_picks", summary["readiness_blockers"])


if __name__ == "__main__":
    unittest.main()
