from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from predicciones.polymarket_retro import report_polymarket_retro
from predicciones.polymarket_shadow_common import (
    BUNDLE_STATUS_PROVISIONAL,
    BUNDLE_STATUS_PROMOTABLE,
    PRICE_PROVENANCE_EXACT,
    PRICE_PROVENANCE_PROXY,
    PRICE_PROVENANCE_RESOLUTION_ONLY,
    SOURCE_MODE_FORWARD,
    SOURCE_MODE_RETRO,
    VALIDATION_STAGE_RETRO,
    VALIDATION_STAGE_SHADOW,
    build_polymarket_lifecycle_summary,
)
from predicciones.polymarket_shadow_reporting import _summarize_shadow, report_polymarket


class PolymarketLifecycleTests(unittest.TestCase):
    def test_lifecycle_summary_prefers_the_dominant_price_provenance(self) -> None:
        retro = build_polymarket_lifecycle_summary(
            source_mode=SOURCE_MODE_RETRO,
            bundle_status=BUNDLE_STATUS_PROVISIONAL,
            price_provenance_counts={
                PRICE_PROVENANCE_PROXY: 4,
                PRICE_PROVENANCE_EXACT: 1,
            },
            validation_stage=VALIDATION_STAGE_RETRO,
        )
        shadow = build_polymarket_lifecycle_summary(
            source_mode=SOURCE_MODE_FORWARD,
            bundle_status=BUNDLE_STATUS_PROMOTABLE,
            price_provenance_counts={
                PRICE_PROVENANCE_EXACT: 3,
                PRICE_PROVENANCE_RESOLUTION_ONLY: 1,
            },
            validation_stage=VALIDATION_STAGE_SHADOW,
        )

        self.assertEqual(retro["validation_stage"], VALIDATION_STAGE_RETRO)
        self.assertEqual(retro["price_provenance"], PRICE_PROVENANCE_PROXY)
        self.assertEqual(retro["bundle_readiness"], BUNDLE_STATUS_PROVISIONAL)
        self.assertEqual(retro["lifecycle_label"], "retro / proxy / provisional")

        self.assertEqual(shadow["validation_stage"], VALIDATION_STAGE_SHADOW)
        self.assertEqual(shadow["price_provenance"], PRICE_PROVENANCE_EXACT)
        self.assertEqual(shadow["bundle_readiness"], BUNDLE_STATUS_PROMOTABLE)
        self.assertEqual(shadow["lifecycle_label"], "shadow / exact / promotable_for_forward")

    def test_shadow_summary_exposes_lifecycle_fields_and_report_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            decisions = pd.DataFrame(
                [
                    {
                        "selection": "home",
                        "skip_reason": "",
                        "mapping_status": "complete",
                        "group_key": "g1",
                        "snapshot_time": "2026-04-19T09:15:00+00:00",
                        "book_age_seconds": 2.0,
                        "price_provenance": PRICE_PROVENANCE_EXACT,
                        "validation_stage": VALIDATION_STAGE_SHADOW,
                    },
                    {
                        "selection": "",
                        "skip_reason": "stale_book",
                        "mapping_status": "complete",
                        "group_key": "g2",
                        "snapshot_time": "2026-04-19T09:10:00+00:00",
                        "book_age_seconds": 12.0,
                        "price_provenance": PRICE_PROVENANCE_EXACT,
                        "validation_stage": VALIDATION_STAGE_SHADOW,
                    },
                    {
                        "selection": "",
                        "skip_reason": "no_book_checkpoint",
                        "mapping_status": "missing_book",
                        "group_key": "g3",
                        "snapshot_time": "",
                        "book_age_seconds": np.nan,
                        "price_provenance": PRICE_PROVENANCE_RESOLUTION_ONLY,
                        "validation_stage": VALIDATION_STAGE_SHADOW,
                    },
                ]
            )
            fills = pd.DataFrame(
                [
                    {
                        "fill_id": "f1",
                        "status": "settled",
                        "cost_basis": 10.0,
                        "net_profit": 3.0,
                        "notional": 10.0,
                        "fill_rate": 1.0,
                        "partial_fill": False,
                        "fee_paid": 0.2,
                        "raw_vwap": 0.48,
                        "effective_vwap": 0.50,
                        "top_ask": 0.46,
                        "created_at": pd.Timestamp("2026-04-19T09:16:00Z"),
                        "league_code": "E0",
                    }
                ]
            )
            mappings = pd.DataFrame(
                [
                    {"mapping_status": "complete"},
                    {"mapping_status": "complete"},
                    {"mapping_status": "missing"},
                ]
            )

            summary = _summarize_shadow(
                decisions,
                fills,
                mappings,
                bundle_status=BUNDLE_STATUS_PROMOTABLE,
            )
            (root / "shadow_summary.json").write_text(json.dumps(summary), encoding="utf-8")

            loaded_summary, text = report_polymarket(root)

            self.assertEqual(loaded_summary["validation_stage"], VALIDATION_STAGE_SHADOW)
            self.assertEqual(loaded_summary["price_provenance"], PRICE_PROVENANCE_EXACT)
            self.assertEqual(loaded_summary["bundle_readiness"], BUNDLE_STATUS_PROMOTABLE)
            self.assertIn("Lifecycle: shadow / exact / promotable_for_forward", text)
            self.assertIn("Validation stage: shadow", text)

    def test_retro_report_surfaces_lifecycle_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "source_mode": SOURCE_MODE_RETRO,
                "validation_stage": VALIDATION_STAGE_RETRO,
                "price_provenance": PRICE_PROVENANCE_PROXY,
                "bundle_readiness": BUNDLE_STATUS_PROVISIONAL,
                "bundle_status": BUNDLE_STATUS_PROVISIONAL,
                "coverage_status": "coverage_limited",
                "research_status": "coverage_limited",
                "promotion_eligibility": "coverage_limited",
                "model_variant": "v1",
                "probability_source": "raw",
                "scope_name": "global_all",
                "policy_metrics": {"bets": 1, "hit_rate": 0.5},
                "model_metrics": {"argmax_hit_rate": 0.5},
                "net_roi": 0.12,
                "net_pnl": 1.2,
                "promotion_decision": {"status": "not_promoted"},
                "approximate_only": True,
                "coverage_summary": {
                    "coverage_status": "coverage_limited",
                    "bundle_status": BUNDLE_STATUS_PROVISIONAL,
                    "mapped_matches": 1,
                },
            }
            (root / "retro_shadow_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (root / "retro_coverage_summary.json").write_text(
                json.dumps(summary["coverage_summary"]),
                encoding="utf-8",
            )

            loaded_summary, text = report_polymarket_retro(root)

            self.assertEqual(loaded_summary["validation_stage"], VALIDATION_STAGE_RETRO)
            self.assertEqual(loaded_summary["price_provenance"], PRICE_PROVENANCE_PROXY)
            self.assertEqual(loaded_summary["bundle_readiness"], BUNDLE_STATUS_PROVISIONAL)
            self.assertIn("Lifecycle: retro / proxy / provisional", text)
            self.assertIn("Validation stage: retro", text)

