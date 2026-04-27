from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .app.pipeline import (
    archive_legacy_1x2_pipeline,
    build_market_lane_predictions_pipeline,
    build_sim_features_pipeline,
    capture_market_raw_pipeline,
    collect_sim_data_pipeline,
    discover_sim_data_sources_pipeline,
    evaluate_forward_lane_pipeline,
    export_sim_training_dataset_pipeline,
    normalize_sim_data_pipeline,
    report_market_lane_pipeline,
    report_sim_data_pipeline,
    report_sport_merge_pipeline,
    run_market_lane_pipeline,
    trade_kill_switch_pipeline,
    trade_live_pipeline,
    trade_paper_pipeline,
    trade_reconcile_pipeline,
    train_lane_model_candidate_pipeline,
    train_football_sim_pipeline,
    validate_lane_model_candidate_pipeline,
)
from .config import load_settings


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _print_summary_and_artifacts(summary: dict[str, Any], artifacts: dict[str, Path]) -> None:
    _print_json({"summary": summary, "artifacts": artifacts})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predicciones: lane-isolated sports market validation."
    )
    parser.add_argument("--config", help="Optional JSON config path.")
    subparsers = parser.add_subparsers(dest="domain", required=True)

    lane = subparsers.add_parser("lane", help="Forward lane capture, prediction, shadow, and reporting.")
    lane_sub = lane.add_subparsers(dest="action", required=True)

    capture = lane_sub.add_parser("capture-raw", help="Capture raw market books for one lane.")
    capture.add_argument("--lane-id", required=True)
    capture.add_argument("--db-path")
    capture.add_argument("--stream-seconds", type=int, default=0)
    capture.add_argument("--max-markets", type=int)
    capture.add_argument("--max-events", type=int)
    capture.add_argument("--capture-priority", default="missing_or_stale")
    capture.add_argument("--freshness-seconds", type=int, default=3600)
    capture.add_argument("--max-stale-age-seconds", type=int)
    capture.add_argument("--include-policy-ready-only", action="store_true")

    build_predictions = lane_sub.add_parser("build-predictions", help="Build lane model predictions.")
    build_predictions.add_argument("--lane-id", required=True)
    build_predictions.add_argument("--db-path")
    build_predictions.add_argument("--model-path")

    run_shadow = lane_sub.add_parser("run-shadow", help="Run frozen lane policy in shadow mode.")
    run_shadow.add_argument("--lane-id", required=True)
    run_shadow.add_argument("--db-path")

    lane_report = lane_sub.add_parser("report", help="Report one lane without cross-lane ROI aggregation.")
    lane_report.add_argument("--lane-id", required=True)
    lane_report.add_argument("--run-dir")

    lane_evaluate = lane_sub.add_parser("evaluate-forward", help="Evaluate forward ROI, CLV, promotion, and ROI 45 hypothesis.")
    lane_evaluate.add_argument("--lane-id", required=True)
    lane_evaluate.add_argument("--db-path")

    model = subparsers.add_parser("model", help="Champion/challenger football model workflow.")
    model_sub = model.add_subparsers(dest="action", required=True)
    model_train = model_sub.add_parser("train", help="Train or register a lane model candidate.")
    model_train.add_argument("--lane-id", required=True)
    model_train.add_argument("--variant", default="champion_poisson_elo_calibrated")
    model_train.add_argument("--dataset-path")
    model_validate = model_sub.add_parser("validate", help="Validate candidate model contracts without policy promotion.")
    model_validate.add_argument("--lane-id", required=True)
    model_validate.add_argument("--candidate", required=True)

    trade = subparsers.add_parser("trade", help="Paper/live execution, reconciliation, and kill switch.")
    trade_sub = trade.add_subparsers(dest="action", required=True)
    trade_paper = trade_sub.add_parser("paper", help="Create paper order intents for a promoted lane.")
    trade_paper.add_argument("--lane-id", required=True)
    trade_paper.add_argument("--db-path")
    trade_live = trade_sub.add_parser("live", help="Create capped live micro order intents; live sends are disabled by default.")
    trade_live.add_argument("--lane-id", required=True)
    trade_live.add_argument("--mode", default="micro")
    trade_live.add_argument("--db-path")
    trade_reconcile = trade_sub.add_parser("reconcile", help="Record venue reconciliation snapshot.")
    trade_reconcile.add_argument("--db-path")
    trade_kill = trade_sub.add_parser("kill-switch", help="Enable or disable the trading kill switch.")
    switch = trade_kill.add_mutually_exclusive_group(required=True)
    switch.add_argument("--enable", action="store_true")
    switch.add_argument("--disable", action="store_true")
    trade_kill.add_argument("--reason", default="")
    trade_kill.add_argument("--db-path")

    sport = subparsers.add_parser("sport", help="Sport-level reports without promotion decisions.")
    sport_sub = sport.add_subparsers(dest="action", required=True)
    sport_report = sport_sub.add_parser("report", help="Build a sport merge report.")
    sport_report.add_argument("--sport", default="football")
    sport_report.add_argument("--db-path")

    sim = subparsers.add_parser("sim", help="Football simulation data and model commands.")
    sim_sub = sim.add_subparsers(dest="action", required=True)

    sim_discover = sim_sub.add_parser("discover-sources", help="Audit registered free data sources.")
    sim_discover.add_argument("--db-path")

    sim_collect = sim_sub.add_parser("collect", help="Collect raw simulation source data.")
    sim_collect.add_argument("--source", default="football_data")
    sim_collect.add_argument("--leagues", nargs="+", required=True)
    sim_collect.add_argument("--seasons", nargs="+", required=True)
    sim_collect.add_argument("--db-path")
    sim_collect.add_argument("--profile")
    sim_collect.add_argument("--seasons-back", type=int)
    sim_collect.add_argument("--probe-missing", action="store_true")
    sim_collect.add_argument("--teams", default="mapped")
    sim_collect.add_argument("--max-statsbomb-matches", type=int)

    sim_normalize = sim_sub.add_parser("normalize", help="Normalize raw simulation data.")
    sim_normalize.add_argument("--db-path")

    sim_features = sim_sub.add_parser("build-features", help="Build leakage-free simulation features.")
    sim_features.add_argument("--as-of-date")
    sim_features.add_argument("--db-path")

    sim_report = sim_sub.add_parser("report", help="Report simulation data status.")
    sim_report.add_argument("--db-path")

    sim_export = sim_sub.add_parser("export-training", help="Export simulation training dataset.")
    sim_export.add_argument("--db-path")
    sim_export.add_argument("--include-market-reference", action="store_true")

    sim_train = sim_sub.add_parser("train", help="Train the football simulator.")
    sim_train.add_argument("--dataset-path")
    sim_train.add_argument("--model-name", default="football_sim_poisson_v1")
    sim_train.add_argument("--include-market-reference", action="store_true")

    archive = subparsers.add_parser("archive", help="Archived reference manifests.")
    archive_sub = archive.add_subparsers(dest="action", required=True)
    archive_sub.add_parser("legacy-1x2", help="Show the archived football 1X2 canonical manifest.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings(config_path=Path(args.config) if args.config else None)

    if args.domain == "lane" and args.action == "capture-raw":
        summary, artifacts = capture_market_raw_pipeline(
            settings=settings,
            db_path=args.db_path,
            stream_seconds=args.stream_seconds,
            lane_id=args.lane_id,
            max_markets=args.max_markets,
            max_events=args.max_events,
            capture_priority=args.capture_priority,
            freshness_seconds=args.freshness_seconds,
            max_stale_age_seconds=args.max_stale_age_seconds,
            include_policy_ready_only=args.include_policy_ready_only,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "lane" and args.action == "build-predictions":
        summary, artifacts = build_market_lane_predictions_pipeline(
            settings=settings,
            lane_id=args.lane_id,
            db_path=args.db_path,
            model_path=args.model_path,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "lane" and args.action == "run-shadow":
        summary, artifacts = run_market_lane_pipeline(
            settings=settings,
            lane_id=args.lane_id,
            db_path=args.db_path,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "lane" and args.action == "report":
        summary, text = report_market_lane_pipeline(
            settings=settings,
            lane_id=args.lane_id,
            run_dir=args.run_dir,
        )
        print(text)
        _print_json({"summary": summary})
        return

    if args.domain == "lane" and args.action == "evaluate-forward":
        summary, artifacts = evaluate_forward_lane_pipeline(
            settings=settings,
            lane_id=args.lane_id,
            db_path=args.db_path,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "model" and args.action == "train":
        summary, artifacts = train_lane_model_candidate_pipeline(
            settings=settings,
            lane_id=args.lane_id,
            variant=args.variant,
            dataset_path=args.dataset_path,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "model" and args.action == "validate":
        summary, artifacts = validate_lane_model_candidate_pipeline(
            settings=settings,
            lane_id=args.lane_id,
            candidate=args.candidate,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "trade" and args.action == "paper":
        summary, artifacts = trade_paper_pipeline(
            settings=settings,
            lane_id=args.lane_id,
            db_path=args.db_path,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "trade" and args.action == "live":
        summary, artifacts = trade_live_pipeline(
            settings=settings,
            lane_id=args.lane_id,
            mode=args.mode,
            db_path=args.db_path,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "trade" and args.action == "reconcile":
        summary, artifacts = trade_reconcile_pipeline(settings=settings, db_path=args.db_path)
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "trade" and args.action == "kill-switch":
        summary, artifacts = trade_kill_switch_pipeline(
            settings=settings,
            enabled=bool(args.enable),
            reason=args.reason,
            db_path=args.db_path,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "sport" and args.action == "report":
        summary, artifacts = report_sport_merge_pipeline(
            settings=settings,
            sport=args.sport,
            db_path=args.db_path,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "sim" and args.action == "discover-sources":
        summary, artifacts = discover_sim_data_sources_pipeline(settings=settings, db_path=args.db_path)
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "sim" and args.action == "collect":
        summary, artifacts = collect_sim_data_pipeline(
            settings=settings,
            source_id=args.source,
            leagues=args.leagues,
            seasons=args.seasons,
            db_path=args.db_path,
            profile=args.profile,
            seasons_back=args.seasons_back,
            probe_missing=args.probe_missing,
            teams=args.teams,
            max_statsbomb_matches=args.max_statsbomb_matches,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "sim" and args.action == "normalize":
        summary, artifacts = normalize_sim_data_pipeline(settings=settings, db_path=args.db_path)
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "sim" and args.action == "build-features":
        summary, artifacts = build_sim_features_pipeline(
            settings=settings,
            as_of_date=args.as_of_date,
            db_path=args.db_path,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "sim" and args.action == "report":
        summary, text, artifacts = report_sim_data_pipeline(settings=settings, db_path=args.db_path)
        print(text)
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "sim" and args.action == "export-training":
        summary, artifacts = export_sim_training_dataset_pipeline(
            settings=settings,
            db_path=args.db_path,
            exclude_market_reference=not args.include_market_reference,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "sim" and args.action == "train":
        summary, artifacts = train_football_sim_pipeline(
            settings=settings,
            dataset_path=args.dataset_path,
            model_name=args.model_name,
            exclude_market_reference=not args.include_market_reference,
        )
        _print_summary_and_artifacts(summary, artifacts)
        return

    if args.domain == "archive" and args.action == "legacy-1x2":
        summary, text = archive_legacy_1x2_pipeline(settings)
        print(text)
        _print_json({"summary": summary})
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()
