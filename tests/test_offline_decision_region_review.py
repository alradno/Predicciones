from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from predicciones.validation.offline_review import (
    build_forward_capture_status,
    run_offline_decision_region_review,
)


class OfflineDecisionRegionReviewTests(unittest.TestCase):
    def test_monitor_reads_sqlite_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "shadow.sqlite"
            now = datetime.now(UTC)
            con = sqlite3.connect(db_path)
            try:
                con.execute("CREATE TABLE pm_book_checkpoints (timestamp TEXT, event_type TEXT)")
                con.execute("CREATE TABLE pm_book_best (timestamp TEXT)")
                con.execute("CREATE TABLE pm_trades (timestamp TEXT)")
                con.execute("CREATE TABLE pm_shadow_decisions (decision_id TEXT)")
                con.execute("CREATE TABLE pm_shadow_fills (fill_id TEXT)")
                con.execute(
                    "CREATE TABLE pm_market_groups ("
                    "group_key TEXT, league_code TEXT, home_team TEXT, away_team TEXT, "
                    "game_start_time TEXT, mapping_status TEXT)"
                )
                con.execute(
                    "INSERT INTO pm_book_checkpoints VALUES (?, ?)",
                    ((now - timedelta(seconds=30)).isoformat(), "decision_checkpoint"),
                )
                con.execute(
                    "INSERT INTO pm_market_groups VALUES (?, ?, ?, ?, ?, ?)",
                    ("g1", "E0", "Home", "Away", (now + timedelta(hours=2)).isoformat(), "complete"),
                )
                con.commit()
            finally:
                con.close()

            status = build_forward_capture_status(db_path)

            self.assertTrue(status["read_only"])
            self.assertFalse(status["writes_performed"])
            self.assertEqual(status["capture_liveness"], "active_recent_checkpoint")
            self.assertEqual(status["counts"]["book_checkpoints"], 1)
            self.assertEqual(status["counts"]["decision_checkpoints"], 1)
            self.assertEqual(len(status["upcoming_t45m_windows"]), 1)

    def test_review_generates_reports_and_rejects_holdout_only_luck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retro = root / "retro"
            outputs = root / "outputs"
            retro.mkdir()
            outputs.mkdir()
            self._write_policy(retro / "policy_bundle.json")
            self._write_retro_candidates(retro / "retro_candidate_rows.csv")
            pd.DataFrame(columns=["decision_id", "notional", "net_profit"]).to_csv(retro / "retro_fill_rows.csv", index=False)
            pd.DataFrame(columns=["decision_id"]).to_csv(retro / "retro_decision_rows.csv", index=False)
            active_pointer = outputs / "latest_polymarket_policy.txt"
            active_pointer.write_text("active-policy-before-review", encoding="utf-8")
            lane_policy = outputs / "lanes" / "football_goals_core" / "policy_bundle.json"
            lane_policy.parent.mkdir(parents=True, exist_ok=True)
            lane_policy.write_text("lane-policy-before-review", encoding="utf-8")

            result = run_offline_decision_region_review(retro, outputs, retro / "policy_bundle.json")

            self.assertTrue((result.run_dir / "decision_region_noise_report.json").exists())
            self.assertTrue((result.run_dir / "frozen_policy_comparison_report.json").exists())
            self.assertTrue((result.run_dir / "decision_region_reformulation_report.json").exists())
            self.assertTrue((outputs / "latest_offline_decision_region_review.txt").exists())
            self.assertEqual(active_pointer.read_text(encoding="utf-8"), "active-policy-before-review")
            self.assertEqual(lane_policy.read_text(encoding="utf-8"), "lane-policy-before-review")
            self.assertTrue(result.summary["diagnostic_only"])
            self.assertFalse(result.summary["policy_written"])

            comparison = json.loads((result.run_dir / "frozen_policy_comparison_report.json").read_text(encoding="utf-8"))
            self.assertFalse(comparison["policy_reoptimized"])
            self.assertFalse(comparison["thresholds_changed"])
            self.assertFalse(comparison["scopes_changed"])
            self.assertFalse(comparison["locked_holdout_used_for_training"])
            self.assertIn("rejected_oof_negative", {item["status"] for item in comparison["comparisons"]})

            reformulation_rows = pd.read_csv(result.run_dir / "decision_region_reformulation_report.csv")
            self.assertFalse(reformulation_rows["match_key"].duplicated().any())
            reformulation = json.loads((result.run_dir / "decision_region_reformulation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(reformulation["status"], "rejected_oof_negative")

    def test_offline_decision_region_review_never_writes_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retro = root / "retro"
            outputs = root / "outputs"
            retro.mkdir()
            outputs.mkdir()
            self._write_policy(retro / "policy_bundle.json")
            self._write_retro_candidates(retro / "retro_candidate_rows.csv")
            pd.DataFrame(columns=["decision_id", "notional", "net_profit"]).to_csv(retro / "retro_fill_rows.csv", index=False)
            pd.DataFrame(columns=["decision_id"]).to_csv(retro / "retro_decision_rows.csv", index=False)

            result = run_offline_decision_region_review(retro, outputs, retro / "policy_bundle.json")

            self.assertFalse((outputs / "latest_polymarket_policy.txt").exists())
            self.assertFalse((outputs / "lanes" / "football_goals_core" / "policy_bundle.json").exists())
            self.assertTrue(result.summary["diagnostic_only"])
            self.assertFalse(result.summary["policy_written"])

    def test_monitor_scripts_do_not_launch_shadow(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        self.assertFalse((repo / "scripts" / "report_forward_capture_status.ps1").exists())
        self.assertFalse((repo / "scripts" / "run_offline_decision_region_review.ps1").exists())
        text = (repo / "scripts" / "run_multi_market_lane_cycle.ps1").read_text(encoding="utf-8")
        self.assertNotIn("shadow-polymarket", text)
        self.assertNotIn("report-polymarket", text)
        self.assertIn('@("lane", "run-shadow")', text)

    @staticmethod
    def _write_policy(path: Path) -> None:
        payload = {
            "policy": {
                "edge_threshold": 0.03,
                "ev_threshold": 0.05,
                "min_odds": 1.2,
                "max_odds": 6.0,
                "family": "edge_ev_threshold",
                "scope_name": "global_all",
            }
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _write_retro_candidates(path: Path) -> None:
        rows = [
            ("m1", "discovery_train", 1, "", "E0", "home", "away", 2.0, 0.50, 0.60, 0.10, 0.20, 0.60),
            ("m2", "discovery_train", 2, "", "E0", "home", "away", 2.0, 0.50, 0.60, 0.10, 0.20, 0.60),
            ("m3", "selection_dev", 0, "pre_holdout_a", "E0", "home", "away", 2.0, 0.50, 0.60, 0.10, 0.20, 0.60),
            ("m4", "selection_dev", 0, "pre_holdout_b", "E0", "home", "home", 2.0, 0.50, 0.60, 0.10, 0.20, 0.60),
            ("m5", "locked_holdout", 0, "", "E0", "home", "home", 3.0, 0.30, 0.45, 0.15, 0.35, 0.60),
        ]
        df = pd.DataFrame(
            rows,
            columns=[
                "match_id",
                "retro_segment",
                "retro_fold_id",
                "pre_holdout_window",
                "league_code",
                "selection",
                "actual_outcome",
                "quoted_odds",
                "top_ask",
                "policy_prob",
                "policy_edge",
                "policy_ev",
                "policy_rank_score",
            ],
        )
        df["quote_status"] = "eligible"
        df["confidence_score_v2"] = 0.65
        df.to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
