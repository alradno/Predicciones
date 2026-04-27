from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from predicciones.config import (
    BacktestConfig,
    ExecutionConfig,
    PolymarketConfig,
    ProjectPaths,
    ResearchConfig,
    Settings,
    SnapshotConfig,
    TradingConfig,
)
from predicciones.core.edge_hypothesis import EdgeHypothesisStatus
from predicciones.core.promotion_state import PromotionStatus
from predicciones.lanes.evaluation import evaluate_forward_lane
from predicciones.lanes.runtime import init_multi_market_db
from predicciones.markets.execution import run_trade_cycle
from predicciones.markets.trading import (
    LimitOrderIntent,
    PolymarketGlobalVenue,
    PostedOrder,
    TradeMode,
    TradeRiskInputs,
    evaluate_trade_risk,
)


def _settings(tmpdir: str, trade: TradingConfig | None = None) -> Settings:
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
        polymarket=PolymarketConfig(decision_book_freshness_seconds=15),
        trade=trade or TradingConfig(),
        backtest=BacktestConfig(),
    )


def _seed_supported_forward_sample(db_path: Path) -> None:
    connection = init_multi_market_db(db_path)
    now = "2026-04-27T10:00:00+00:00"
    book = "2026-04-27T09:59:50+00:00"
    rows = []
    settlements = []
    for index in range(300):
        market_id = f"m{index}"
        rows.append(
            (
                f"d{index}",
                "football_1x2_global",
                market_id,
                f"event-{index}",
                "football_1x2_global_v1",
                now,
                "valid_forward_sample",
                "",
                "home",
                "football_goal_model_adapter_ready",
                "policy_ready",
                "global_1x2",
                0,
                now,
                "2026-04-27T12:00:00+00:00",
                "home",
                f"asset-{index}",
                book,
                0.50,
                2.0,
                0.70,
                "calibrated",
                0.20,
                0.40,
                0.60,
                "2026-04-27T11:55:00+00:00",
            )
        )
        settlements.append((f"s{index}", market_id, f"event-{index}", now, "home", "settled", "{}", now))
    connection.executemany(
        """
        INSERT INTO mm_lane_forward_ledger (
            decision_id, lane_id, market_id, event_slug, benchmark_id,
            decision_time, decision_status, blocker, selected_outcome, model_mode,
            policy_mode, market_subtype, legacy_slice, created_at, game_start_time,
            contract_outcome, asset_id, book_timestamp, top_ask, quoted_odds,
            model_prob, probability_source, edge, ev, closing_top_ask, closing_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.executemany(
        """
        INSERT INTO mm_raw_settlements (
            settlement_id, market_id, event_slug, resolved_at, winning_outcome,
            status, raw_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        settlements,
    )
    connection.commit()
    connection.close()


class TradeAutomationTests(unittest.TestCase):
    def test_forward_evaluation_marks_roi_45_supported_on_synthetic_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = settings.paths.data_dir / "forward.sqlite"
            _seed_supported_forward_sample(db_path)

            summary, _ = evaluate_forward_lane(settings, "football_1x2_global", db_path)

            self.assertEqual(summary["promotion_status"], PromotionStatus.capital_promotable.value)
            self.assertEqual(summary["roi_45_hypothesis"]["status"], EdgeHypothesisStatus.supported.value)
            self.assertGreaterEqual(summary["forward_metrics"]["flat_stake_roi"], 0.45)

    def test_live_risk_is_blocked_by_default_configuration(self) -> None:
        decision = evaluate_trade_risk(
            TradeRiskInputs(
                lane_id="football_1x2_global",
                mode=TradeMode.live,
                promotion_status=PromotionStatus.capital_promotable,
                edge_status=EdgeHypothesisStatus.supported,
                order_notional=1.0,
                daily_committed=0.0,
                open_exposure=0.0,
                quote_age_seconds=1.0,
                config=TradingConfig(),
            )
        )

        self.assertFalse(decision.allowed)
        self.assertIn("lane_capital_governance_disabled", decision.blockers)
        self.assertIn("live_trading_disabled", decision.blockers)
        self.assertIn("bankroll_usdc_missing", decision.blockers)
        self.assertIn("jurisdiction_not_confirmed", decision.blockers)

    def test_paper_cycle_writes_intents_after_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(tmpdir)
            db_path = settings.paths.data_dir / "forward.sqlite"
            _seed_supported_forward_sample(db_path)

            summary, _ = run_trade_cycle(settings, "football_1x2_global", TradeMode.paper, db_path)

            self.assertFalse(summary["blocked"])
            self.assertGreater(summary["intents_written"], 0)
            connection = sqlite3.connect(db_path)
            try:
                count = connection.execute("SELECT COUNT(*) FROM mm_trade_order_intents").fetchone()[0]
            finally:
                connection.close()
            self.assertGreater(count, 0)

    def test_mocked_global_venue_can_post_when_explicitly_enabled(self) -> None:
        class FakeClient:
            def place_limit_order(self, intent: LimitOrderIntent) -> PostedOrder:
                return PostedOrder(order_id=f"order-{intent.intent_id}", status="posted", raw={"ok": True})

        venue = PolymarketGlobalVenue(live_enabled=True, client=FakeClient())
        intent = LimitOrderIntent(
            intent_id="intent-1",
            lane_id="football_1x2_global",
            decision_id="decision-1",
            venue="global",
            token_id="asset-1",
            side="BUY",
            price=0.5,
            size=2.0,
            notional=1.0,
            time_in_force="GTD",
            expires_at="2026-04-27T12:00:00+00:00",
            mode=TradeMode.live,
        )

        preview = venue.preview_order(intent)
        posted = venue.place_limit_order(intent)

        self.assertTrue(preview.accepted)
        self.assertEqual(posted.status, "posted")


if __name__ == "__main__":
    unittest.main()
