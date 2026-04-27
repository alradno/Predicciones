from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path


class ModularCliTests(unittest.TestCase):
    def test_active_modules_do_not_import_archive(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "predicciones"
        offenders: list[str] = []
        for path in root.rglob("*.py"):
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] == "archive":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("predicciones.archive"):
                            offenders.append(str(relative))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.startswith("predicciones.archive"):
                        offenders.append(str(relative))
        self.assertEqual(offenders, [])

    def test_cli_exposes_new_lane_commands_and_rejects_legacy_backtest(self) -> None:
        lane_help = subprocess.run(
            [sys.executable, "-m", "predicciones.cli", "lane", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("capture-raw", lane_help.stdout)
        self.assertIn("build-predictions", lane_help.stdout)
        self.assertIn("run-shadow", lane_help.stdout)
        self.assertIn("evaluate-forward", lane_help.stdout)

        model_help = subprocess.run(
            [sys.executable, "-m", "predicciones.cli", "model", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        trade_help = subprocess.run(
            [sys.executable, "-m", "predicciones.cli", "trade", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("train", model_help.stdout)
        self.assertIn("validate", model_help.stdout)
        self.assertIn("paper", trade_help.stdout)
        self.assertIn("live", trade_help.stdout)
        self.assertIn("kill-switch", trade_help.stdout)

        legacy = subprocess.run(
            [sys.executable, "-m", "predicciones.cli", "backtest"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(legacy.returncode, 0)
        self.assertIn("invalid choice", legacy.stderr)

    def test_lane_spec_is_public_contract(self) -> None:
        from predicciones.lanes import LaneSpec, get_market_lane_spec

        spec = get_market_lane_spec("football_1x2_global")

        self.assertIsInstance(spec, LaneSpec)
        self.assertEqual(spec.lane_id, "football_1x2_global")
        self.assertEqual(spec.sport, "football")


if __name__ == "__main__":
    unittest.main()
