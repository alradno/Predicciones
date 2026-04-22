from __future__ import annotations

import argparse
from pathlib import Path

from .claude_client import ProbabilitySnapshot, build_analyst
from .config import load_settings
from .pipeline import (
    audit_polymarket_coverage_pipeline,
    backtest_pipeline,
    backfill_polymarket_history_pipeline,
    backtest_polymarket_retro_pipeline,
    backtest_net_pipeline,
    build_market_lane_predictions_pipeline,
    build_sim_features_pipeline,
    capture_odds_pipeline,
    collect_sim_data_pipeline,
    collect_polymarket_pipeline,
    capture_multi_market_pipeline,
    capture_market_raw_pipeline,
    create_market_lane_policy_pipeline,
    discover_market_raw_pipeline,
    discover_multi_market_pipeline,
    discover_sim_data_sources_pipeline,
    discover_niches_pipeline,
    export_sim_training_dataset_pipeline,
    ingest_pipeline,
    predict_pipeline,
    promotion_report_pipeline,
    report_polymarket_retro_pipeline,
    report_polymarket_pipeline,
    report_market_lane_pipeline,
    report_multi_market_pipeline,
    report_sport_merge_pipeline,
    report_sim_data_pipeline,
    report_pipeline,
    search_polymarket,
    shadow_polymarket_pipeline,
    shadow_run_pipeline,
    normalize_sim_data_pipeline,
    run_market_lane_pipeline,
    tune_polymarket_policy_pipeline,
    train_football_sim_pipeline,
    train_market_lane_model_pipeline,
    train_niche_pipeline,
    train_final_pipeline,
)
from .reporting import format_summary
from .research import format_net_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sistema de prediccion de futbol con baseline, goal model y edge.")
    parser.add_argument("--config", help="Ruta a un JSON de configuracion opcional.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Descarga, normaliza y guarda un dataset canonico.")
    ingest_parser.add_argument("--seasons", nargs="+", help="Temporadas, por ejemplo: 2425 2324 2223")
    ingest_parser.add_argument("--leagues", nargs="+", help="Ligas, por ejemplo: E0 SP1 D1")
    ingest_parser.add_argument("--dataset-name", help="Nombre opcional para la carpeta del dataset.")

    sim_discover_parser = subparsers.add_parser(
        "discover-sim-data-sources",
        help="Audita fuentes gratuitas registradas para el carril football_sim_data.",
    )
    sim_discover_parser.add_argument("--db-path", help="Ruta opcional a data/football_sim_data.sqlite.")

    sim_collect_parser = subparsers.add_parser(
        "collect-sim-data",
        help="Recolecta raw snapshots reproducibles para football_sim_data.",
    )
    sim_collect_parser.add_argument("--source", default="football_data", help="Fuente declarada, por ejemplo football_data.")
    sim_collect_parser.add_argument("--leagues", nargs="+", help="Ligas, por ejemplo: E0 SP1 D1 I1 F1.")
    sim_collect_parser.add_argument("--seasons", nargs="+", help="Temporadas, por ejemplo: 2526 2425 2324.")
    sim_collect_parser.add_argument("--profile", help="Perfil de cobertura, por ejemplo max_free_v1 u open_data_all.")
    sim_collect_parser.add_argument("--seasons-back", type=int, help="Numero de temporadas hacia atras para football_data.")
    sim_collect_parser.add_argument("--probe-missing", action="store_true", help="Guarda fallos de fetch como raw blockers.")
    sim_collect_parser.add_argument("--teams", default="mapped", help="Modo de equipos para clubelo; default mapped.")
    sim_collect_parser.add_argument("--max-statsbomb-matches", type=int, help="Limite opcional para smoke tests de StatsBomb.")
    sim_collect_parser.add_argument("--db-path", help="Ruta opcional a data/football_sim_data.sqlite.")

    sim_normalize_parser = subparsers.add_parser(
        "normalize-sim-data",
        help="Normaliza bronze raw a entidades canonicas silver para el simulador.",
    )
    sim_normalize_parser.add_argument("--db-path", help="Ruta opcional a data/football_sim_data.sqlite.")

    sim_features_parser = subparsers.add_parser(
        "build-sim-features",
        help="Construye gold_sim_features leakage-free para simulacion.",
    )
    sim_features_parser.add_argument("--as-of-date", help="Fecha maxima YYYY-MM-DD para construir features.")
    sim_features_parser.add_argument("--db-path", help="Ruta opcional a data/football_sim_data.sqlite.")

    sim_report_parser = subparsers.add_parser(
        "report-sim-data",
        help="Resume cobertura, entidades, leakage y readiness de football_sim_data.",
    )
    sim_report_parser.add_argument("--db-path", help="Ruta opcional a data/football_sim_data.sqlite.")

    sim_export_parser = subparsers.add_parser(
        "export-sim-training-dataset",
        help="Exporta dataset entrenable leakage-free para el futuro simulador.",
    )
    sim_export_parser.add_argument("--db-path", help="Ruta opcional a data/football_sim_data.sqlite.")
    sim_export_parser.add_argument(
        "--exclude-market-reference",
        action="store_true",
        default=True,
        help="Excluye cuotas y probabilidades de mercado del dataset entrenable.",
    )

    sim_train_parser = subparsers.add_parser(
        "train-football-sim",
        help="Entrena football_sim_poisson_v1 desde simulation_training_dataset.csv sin emitir picks.",
    )
    sim_train_parser.add_argument(
        "--dataset-path",
        default="outputs/sim_data/simulation_training_dataset.csv",
        help="Ruta al dataset entrenable exportado por export-sim-training-dataset.",
    )
    sim_train_parser.add_argument("--model-name", default="football_sim_poisson_v1")
    sim_train_parser.add_argument(
        "--exclude-market-reference",
        action="store_true",
        default=True,
        help="Rechaza datasets con cuotas/probabilidades de mercado.",
    )

    backtest_parser = subparsers.add_parser("backtest", help="Ejecuta baseline, goal model, calibracion y politica.")
    backtest_parser.add_argument("--dataset-dir", help="Ruta a un dataset ya ingerido.")
    backtest_parser.add_argument("--seasons", nargs="+", help="Temporadas si quieres ingerir automaticamente.")
    backtest_parser.add_argument("--leagues", nargs="+", help="Ligas si quieres ingerir automaticamente.")

    capture_parser = subparsers.add_parser("capture-odds", help="Genera market snapshots y opcionalmente agrega snapshots externos.")
    capture_parser.add_argument("--dataset-dir", help="Ruta a un dataset ya ingerido.")
    capture_parser.add_argument("--snapshot-file", action="append", help="CSV externo con snapshots reales. Repetible.")

    backtest_net_parser = subparsers.add_parser(
        "backtest-net",
        help="Ejecuta el research backtest con snapshots ejecutables, holdout y promotion report.",
    )
    backtest_net_parser.add_argument("--dataset-dir", help="Ruta a un dataset ya ingerido.")
    backtest_net_parser.add_argument("--snapshot-file", action="append", help="CSV externo con snapshots reales. Repetible.")
    backtest_net_parser.add_argument("--seasons", nargs="+", help="Temporadas si quieres ingerir automaticamente.")
    backtest_net_parser.add_argument("--leagues", nargs="+", help="Ligas si quieres ingerir automaticamente.")

    discover_parser = subparsers.add_parser("discover-niches", help="Busca nichos robustos a partir del ultimo backtest neto.")
    discover_parser.add_argument("--run-dir", help="Ruta a un run de backtest neto.")

    train_final_parser = subparsers.add_parser("train-final", help="Entrena el modelo final sobre todo el historico.")
    train_final_parser.add_argument("--dataset-dir", help="Ruta a un dataset ya ingerido.")
    train_final_parser.add_argument("--seasons", nargs="+")
    train_final_parser.add_argument("--leagues", nargs="+")

    train_niche_parser = subparsers.add_parser("train-niche", help="Empaqueta el modelo final del nicho elegido y su politica.")
    train_niche_parser.add_argument("--dataset-dir", help="Ruta a un dataset ya ingerido.")
    train_niche_parser.add_argument("--research-run-dir", help="Run de backtest neto del que sacar policy/niche.")
    train_niche_parser.add_argument("--seasons", nargs="+")
    train_niche_parser.add_argument("--leagues", nargs="+")

    predict_parser = subparsers.add_parser("predict", help="Genera probabilidades y picks para fixtures nuevos.")
    predict_parser.add_argument("--fixtures-file", required=True, help="CSV con Date, league_code, HomeTeam, AwayTeam y cuotas.")
    predict_parser.add_argument("--model-path", help="Ruta al modelo final. Si no se indica usa el ultimo.")

    shadow_parser = subparsers.add_parser("shadow-run", help="Genera picks planificados con el niche model para partidos futuros.")
    shadow_parser.add_argument("--fixtures-file", required=True, help="CSV con Date, league_code, HomeTeam, AwayTeam y cuotas.")
    shadow_parser.add_argument("--model-path", help="Ruta al niche model. Si no se indica usa el ultimo.")
    shadow_parser.add_argument("--snapshot-file", action="append", help="CSV externo con snapshots reales. Repetible.")

    collect_pm_parser = subparsers.add_parser(
        "collect-polymarket",
        help="Descubre mercados 1X2 de futbol en Polymarket y guarda libros/resoluciones en SQLite.",
    )
    collect_pm_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite.")
    collect_pm_parser.add_argument("--stream-seconds", type=int, default=0, help="Tiempo opcional de streaming WS.")

    discover_raw_parser = subparsers.add_parser(
        "discover-market-raw",
        help="Descubre inventario raw comun para carriles atomicos sin emitir picks.",
    )
    discover_raw_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite multi-market raw.")
    discover_raw_parser.add_argument("--limit-per-query", type=int, default=50, help="Limite de eventos por query de discovery.")

    capture_raw_parser = subparsers.add_parser(
        "capture-market-raw",
        help="Captura orderbooks raw comunes sin duplicarlos por carril.",
    )
    capture_raw_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite multi-market raw.")
    capture_raw_parser.add_argument("--stream-seconds", type=int, default=0, help="Duracion opcional de polling REST.")
    capture_raw_parser.add_argument("--lane-id", help="Captura solo un carril declarado, por ejemplo football_1x2_global.")
    capture_raw_parser.add_argument("--max-markets", type=int, help="Limita mercados futuros activos para smoke tests seguros.")
    capture_raw_parser.add_argument("--max-events", type=int, help="Limita eventos futuros activos y captura todos sus mercados del carril.")
    capture_raw_parser.add_argument(
        "--capture-priority",
        choices=("missing_or_stale", "missing_only", "stale_only", "all"),
        default="missing_or_stale",
        help="Prioridad de captura raw; por defecto captura primero books ausentes o stale.",
    )
    capture_raw_parser.add_argument(
        "--freshness-seconds",
        type=int,
        default=3600,
        help="Ventana para considerar fresco un book en los reportes raw.",
    )
    capture_raw_parser.add_argument(
        "--max-stale-age-seconds",
        type=int,
        help="Si se indica, no recaptura books stale hasta que superen esta edad.",
    )
    capture_raw_parser.add_argument(
        "--include-policy-ready-only",
        action="store_true",
        help="Captura solo carriles con modelo y policy listos; prioriza football_1x2_global frente a research-only.",
    )

    run_lane_parser = subparsers.add_parser(
        "run-market-lane",
        help="Ejecuta un carril atomico aislado desde el inventario raw comun.",
    )
    run_lane_parser.add_argument("--lane-id", required=True, help="Carril, por ejemplo football_1x2_global.")
    run_lane_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite multi-market raw.")

    build_lane_predictions_parser = subparsers.add_parser(
        "build-market-lane-predictions",
        help="Genera model_predictions.csv para un carril atomico desde su plantilla.",
    )
    build_lane_predictions_parser.add_argument("--lane-id", required=True, help="Carril, por ejemplo football_1x2_global.")
    build_lane_predictions_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite multi-market raw.")
    build_lane_predictions_parser.add_argument("--model-path", help="Ruta opcional al bundle de modelo; por defecto usa latest_niche_model y luego latest_model.")

    train_lane_model_parser = subparsers.add_parser(
        "train-market-lane-model",
        help="Entrena un bundle de modelo aislado para un carril sin mover punteros legacy.",
    )
    train_lane_model_parser.add_argument("--lane-id", required=True, help="Carril, por ejemplo football_1x2_global.")
    train_lane_model_parser.add_argument("--dataset-dir", help="Dataset existente opcional.")
    train_lane_model_parser.add_argument("--dataset-name", help="Nombre opcional si hay que ingerir un dataset de carril.")
    train_lane_model_parser.add_argument("--seasons", nargs="+", help="Temporadas, por ejemplo 2526 2425 2324.")
    train_lane_model_parser.add_argument("--leagues", nargs="+", help="Ligas, por ejemplo E0 SP1 D1 I1 F1.")

    create_lane_policy_parser = subparsers.add_parser(
        "create-market-lane-policy",
        help="Crea una policy nativa bootstrap para un carril con benchmark separado.",
    )
    create_lane_policy_parser.add_argument("--lane-id", required=True, help="Carril, por ejemplo football_goals_core.")
    create_lane_policy_parser.add_argument("--overwrite", action="store_true", help="Recrear la policy aunque ya exista.")

    report_lane_parser = subparsers.add_parser(
        "report-market-lane",
        help="Resume un carril atomico sin mezclar ROI con otros carriles.",
    )
    report_lane_parser.add_argument("--lane-id", required=True, help="Carril, por ejemplo football_1x2_global.")
    report_lane_parser.add_argument("--run-dir", help="Ruta opcional al run del carril.")

    report_sport_merge_parser = subparsers.add_parser(
        "report-sport-merge",
        help="Crea feature store por deporte sin ROI agregado ni promocion.",
    )
    report_sport_merge_parser.add_argument("--sport", required=True, help="Sport merge group, por ejemplo football.")
    report_sport_merge_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite multi-market raw.")

    discover_mm_parser = subparsers.add_parser(
        "discover-multi-market",
        help="Descubre mercados multi-deporte en un carril aislado con benchmarks separados.",
    )
    discover_mm_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite multi-market.")
    discover_mm_parser.add_argument("--limit-per-query", type=int, default=50, help="Limite de eventos por query de discovery.")

    capture_mm_parser = subparsers.add_parser(
        "capture-multi-market",
        help="Captura books forward multi-mercado en la SQLite aislada sin emitir picks.",
    )
    capture_mm_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite multi-market.")
    capture_mm_parser.add_argument("--stream-seconds", type=int, default=0, help="Duracion opcional de polling REST.")
    capture_mm_parser.add_argument("--lane-id", help="Alias temporal: captura solo un carril declarado.")
    capture_mm_parser.add_argument("--max-markets", type=int, help="Alias temporal: limita mercados futuros activos.")
    capture_mm_parser.add_argument("--max-events", type=int, help="Alias temporal: limita eventos futuros activos.")
    capture_mm_parser.add_argument(
        "--capture-priority",
        choices=("missing_or_stale", "missing_only", "stale_only", "all"),
        default="missing_or_stale",
        help="Alias temporal: prioridad de captura raw.",
    )
    capture_mm_parser.add_argument("--freshness-seconds", type=int, default=3600, help="Alias temporal: segundos para freshness raw.")
    capture_mm_parser.add_argument("--max-stale-age-seconds", type=int, help="Alias temporal: edad minima para recapturar stale.")
    capture_mm_parser.add_argument(
        "--include-policy-ready-only",
        action="store_true",
        help="Alias temporal: captura solo carriles con modelo y policy listos.",
    )

    backfill_pm_parser = subparsers.add_parser(
        "backfill-polymarket-history",
        help="Hace backfill historico de catalogo, grupos, resoluciones, aliases y prices-history en SQLite.",
    )
    backfill_pm_parser.add_argument("--dataset-dir", help="Ruta opcional a un dataset ya ingerido.")
    backfill_pm_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite.")
    backfill_pm_parser.add_argument("--model-path", help="Ruta opcional a un modelo para reutilizar su historico.")

    audit_pm_parser = subparsers.add_parser(
        "audit-polymarket-coverage",
        help="Audita todos los submercados historicos de Polymarket y separa 1X2 real, binarios, duplicados y no parseables.",
    )
    audit_pm_parser.add_argument("--dataset-dir", help="Ruta opcional a un dataset ya ingerido.")
    audit_pm_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite.")
    audit_pm_parser.add_argument("--model-path", help="Ruta opcional a un modelo para derivar el historico.")

    retro_pm_parser = subparsers.add_parser(
        "backtest-polymarket-retro",
        help="Evalua retrospectivamente el modelo en Polymarket con pricing aproximado sobre eventos cerrados.",
    )
    retro_pm_parser.add_argument("--dataset-dir", help="Ruta opcional a un dataset ya ingerido.")
    retro_pm_parser.add_argument("--db-path", help="SQLite opcional para reutilizar checkpoints reales ya capturados.")
    retro_pm_parser.add_argument("--model-path", help="Ruta opcional al modelo o niche model.")
    retro_pm_parser.add_argument("--policy-bundle", help="Bundle opcional de politica para re-evaluar el retro backtest.")

    tune_pm_parser = subparsers.add_parser(
        "tune-polymarket-policy",
        help="Afina edge y politica sobre el carril retrospectivo aproximado en modo diagnostico.",
    )
    tune_pm_parser.add_argument("--dataset-dir", help="Ruta opcional a un dataset ya ingerido.")
    tune_pm_parser.add_argument("--db-path", help="SQLite opcional para aprovechar checkpoints historicos capturados.")
    tune_pm_parser.add_argument("--model-path", help="Ruta opcional al modelo o niche model.")
    tune_pm_parser.add_argument(
        "--write-policy",
        action="store_true",
        help="Escribe el puntero de politica activo solo si la muestra forward esta sample_ready.",
    )

    shadow_pm_parser = subparsers.add_parser(
        "shadow-polymarket",
        help="Evalua el modelo contra orderbooks reales de Polymarket con fills taker simulados.",
    )
    shadow_pm_parser.add_argument("--fixtures-file", help="CSV opcional con fixtures. Si no se indica, usa los grupos completos de la base.")
    shadow_pm_parser.add_argument("--db-path", help="Ruta opcional a la base SQLite.")
    shadow_pm_parser.add_argument("--model-path", help="Ruta opcional al niche model.")
    shadow_pm_parser.add_argument("--policy-bundle", help="Bundle opcional de politica generado desde el modo retro.")

    report_pm_parser = subparsers.add_parser(
        "report-polymarket",
        help="Resume un shadow run de Polymarket ya completado.",
    )
    report_pm_parser.add_argument("--run-dir", help="Ruta al run de shadow Polymarket.")

    report_pm_retro_parser = subparsers.add_parser(
        "report-polymarket-retro",
        help="Resume un run retrospectivo aproximado de Polymarket ya completado.",
    )
    report_pm_retro_parser.add_argument("--run-dir", help="Ruta al run retrospectivo de Polymarket.")

    report_mm_parser = subparsers.add_parser(
        "report-multi-market",
        help="Resume un run multi-market sin mezclar ROI entre familias.",
    )
    report_mm_parser.add_argument("--run-dir", help="Ruta al run multi-market.")

    report_parser = subparsers.add_parser("report", help="Resume un run ya completado.")
    report_parser.add_argument("--run-dir", help="Ruta al run que quieres resumir.")

    promote_parser = subparsers.add_parser("promote-report", help="Muestra el estado de promocion por etapas del ultimo backtest neto.")
    promote_parser.add_argument("--run-dir", help="Ruta a un run de backtest neto.")

    legacy_parser = subparsers.add_parser("train", help="Alias heredado de backtest.")
    legacy_parser.add_argument("--dataset-dir")
    legacy_parser.add_argument("--seasons", nargs="+")
    legacy_parser.add_argument("--leagues", nargs="+")

    markets_parser = subparsers.add_parser("markets", help="Busca mercados de Polymarket.")
    markets_parser.add_argument("--query", required=True)
    markets_parser.add_argument("--limit", type=int, default=10)

    claude_parser = subparsers.add_parser("claude-report", help="Genera una nota con Claude o con fallback heuristico.")
    claude_parser.add_argument("--match", required=True)
    claude_parser.add_argument("--bookmaker", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"), required=True)
    claude_parser.add_argument("--model", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"), required=True)
    claude_parser.add_argument("--polymarket", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings(config_path=Path(args.config) if args.config else None)

    if args.command == "ingest":
        leagues = tuple(args.leagues or settings.default_leagues)
        seasons = tuple(args.seasons or settings.default_seasons)
        summary = ingest_pipeline(settings=settings, leagues=leagues, seasons=seasons, dataset_name=args.dataset_name)
        print("Ingesta completada.")
        print(f"- Filas descargadas: {summary.rows_downloaded}")
        print(f"- Filas canonicas: {summary.rows_canonical}")
        if summary.failures:
            print("- Descargas fallidas:")
            for failure in summary.failures:
                print(f"  - {failure}")
        print("- Artefactos:")
        for name, path in summary.artifacts.items():
            print(f"  - {name}: {path}")
        return

    if args.command == "discover-sim-data-sources":
        summary, artifacts = discover_sim_data_sources_pipeline(
            settings=settings,
            db_path=Path(args.db_path) if args.db_path else None,
        )
        print("Discovery de fuentes de simulacion completado.")
        print(f"- raw_payloads: {summary['raw_payloads']}")
        print(f"- matches: {summary['matches']}")
        print(f"- match_level_status: {summary['match_level_status']}")
        print(f"- source_coverage_report: {artifacts['source_coverage_report']}")
        print(f"- source_license_manifest: {artifacts['source_license_manifest']}")
        print("- picks_emitidos: 0")
        print("- global_roi_actionable: false")
        return

    if args.command == "collect-sim-data":
        leagues = tuple(args.leagues or settings.default_leagues)
        seasons = tuple(args.seasons or settings.default_seasons)
        summary, artifacts = collect_sim_data_pipeline(
            settings=settings,
            source_id=str(args.source),
            leagues=leagues,
            seasons=seasons,
            db_path=Path(args.db_path) if args.db_path else None,
            profile=str(args.profile) if args.profile else None,
            seasons_back=int(args.seasons_back) if args.seasons_back else None,
            probe_missing=bool(args.probe_missing),
            teams=str(args.teams),
            max_statsbomb_matches=int(args.max_statsbomb_matches) if args.max_statsbomb_matches else None,
        )
        print("Collect sim-data completado.")
        print(f"- source_id: {summary['source_id']}")
        print(f"- profile: {summary.get('profile')}")
        print(f"- raw_payloads_inserted: {summary['raw_payloads_inserted']}")
        print(f"- failures: {len(summary.get('failures', []))}")
        print(f"- source_coverage_report: {artifacts['source_coverage_report']}")
        print("- picks_emitidos: 0")
        print("- global_roi_actionable: false")
        return

    if args.command == "normalize-sim-data":
        summary, artifacts = normalize_sim_data_pipeline(
            settings=settings,
            db_path=Path(args.db_path) if args.db_path else None,
        )
        print("Normalize sim-data completado.")
        print(f"- teams: {summary['teams']}")
        print(f"- matches: {summary['matches']}")
        print(f"- entity_resolution_report: {artifacts['entity_resolution_report']}")
        print("- picks_emitidos: 0")
        print("- global_roi_actionable: false")
        return

    if args.command == "build-sim-features":
        summary, artifacts = build_sim_features_pipeline(
            settings=settings,
            as_of_date=str(args.as_of_date) if args.as_of_date else None,
            db_path=Path(args.db_path) if args.db_path else None,
        )
        print("Build sim-features completado.")
        print(f"- gold_feature_rows: {summary['gold_feature_rows']}")
        print(f"- match_level_status: {summary['match_level_status']}")
        print(f"- leakage_audit_report: {artifacts['leakage_audit_report']}")
        print(f"- simulation_feature_manifest: {artifacts['simulation_feature_manifest']}")
        print("- picks_emitidos: 0")
        print("- global_roi_actionable: false")
        return

    if args.command == "report-sim-data":
        _, text, _ = report_sim_data_pipeline(
            settings=settings,
            db_path=Path(args.db_path) if args.db_path else None,
        )
        print(text)
        return

    if args.command == "export-sim-training-dataset":
        summary, artifacts = export_sim_training_dataset_pipeline(
            settings=settings,
            db_path=Path(args.db_path) if args.db_path else None,
            exclude_market_reference=bool(args.exclude_market_reference),
        )
        print("Export sim-training-dataset completado.")
        print(f"- training_rows: {summary['training_rows']}")
        print(f"- feature_columns: {summary['feature_columns']}")
        print(f"- simulation_training_dataset: {artifacts['simulation_training_dataset']}")
        print(f"- simulation_training_manifest: {artifacts['simulation_training_manifest']}")
        print("- picks_emitidos: 0")
        print("- global_roi_actionable: false")
        return

    if args.command == "train-football-sim":
        summary, artifacts = train_football_sim_pipeline(
            settings=settings,
            dataset_path=Path(args.dataset_path) if args.dataset_path else None,
            model_name=str(args.model_name),
            exclude_market_reference=bool(args.exclude_market_reference),
        )
        print("Football simulator entrenado.")
        print(f"- model_name: {summary['model_name']}")
        print(f"- rows: {summary['rows']}")
        print(f"- split_counts: {summary['split_counts']}")
        print(f"- feature_count: {summary['feature_count']}")
        print(f"- selected_probability_source: {summary['selected_probability_source']}")
        print(f"- readiness_status: {summary['readiness_status']}")
        print(f"- readiness_blockers: {summary['readiness_blockers']}")
        print(f"- model_bundle: {artifacts['model_bundle']}")
        print(f"- simulation_model_report: {artifacts['simulation_model_report']}")
        print(f"- latest_football_sim_model: {artifacts['latest_football_sim_model']}")
        print("- picks_emitidos: 0")
        print("- global_roi_actionable: false")
        return

    if args.command == "capture-odds":
        result = capture_odds_pipeline(
            settings=settings,
            dataset_dir=args.dataset_dir,
            snapshot_files=[Path(path) for path in args.snapshot_file] if args.snapshot_file else None,
        )
        print("Snapshots preparados.")
        print(f"- Archivo: {result.snapshots_path}")
        print(f"- Filas: {result.summary['rows']}")
        print(f"- Sources: {', '.join(result.summary['sources'])}")
        return

    if args.command == "collect-polymarket":
        result = collect_polymarket_pipeline(
            settings=settings,
            db_path=Path(args.db_path) if args.db_path else None,
            stream_seconds=int(args.stream_seconds),
        )
        print("Collector de Polymarket completado.")
        print(f"- SQLite: {result.database_path}")
        print(f"- Summary: {result.summary_path}")
        print(f"- Complete groups: {result.summary['complete_groups']}")
        return

    if args.command == "discover-market-raw":
        result = discover_market_raw_pipeline(
            settings=settings,
            db_path=Path(args.db_path) if args.db_path else None,
            limit_per_query=int(args.limit_per_query),
        )
        print("Discovery raw de mercados completado.")
        print(f"- run: {result.run.run_dir}")
        print(f"- SQLite raw: {result.database_path}")
        print(f"- manifest: {result.artifacts['market_family_manifest']}")
        print(f"- coverage: {result.artifacts['market_family_coverage_report']}")
        print(f"- raw_inventory_quality: {result.artifacts['raw_inventory_quality_report']}")
        print("- picks_emitidos: 0")
        print("- global_roi_actionable: false")
        return

    if args.command == "capture-market-raw":
        result = capture_market_raw_pipeline(
            settings=settings,
            db_path=Path(args.db_path) if args.db_path else None,
            stream_seconds=int(args.stream_seconds),
            lane_id=str(args.lane_id) if args.lane_id else None,
            max_markets=int(args.max_markets) if args.max_markets else None,
            max_events=int(args.max_events) if args.max_events else None,
            capture_priority=str(args.capture_priority),
            freshness_seconds=int(args.freshness_seconds),
            max_stale_age_seconds=int(args.max_stale_age_seconds) if args.max_stale_age_seconds else None,
            include_policy_ready_only=bool(args.include_policy_ready_only),
        )
        print("Captura raw de mercados completada.")
        print(f"- run: {result.run.run_dir}")
        print(f"- SQLite raw: {result.database_path}")
        print(f"- sample_report: {result.artifacts['multi_market_forward_sample_report']}")
        print(f"- ledger agregado: {result.artifacts['multi_market_forward_ledger']}")
        print(f"- raw_capture_plan_csv: {result.artifacts['raw_capture_plan_csv']}")
        print(f"- raw_capture_plan_json: {result.artifacts['raw_capture_plan_json']}")
        print(f"- lane_filter: {result.summary.get('capture_lane_filter')}")
        print(f"- max_markets: {result.summary.get('capture_max_markets')}")
        print(f"- max_events: {result.summary.get('capture_max_events')}")
        print(f"- capture_priority: {result.summary.get('capture_priority')}")
        print(f"- freshness_seconds: {result.summary.get('capture_freshness_seconds')}")
        print(f"- include_policy_ready_only: {str(result.summary.get('include_policy_ready_only')).lower()}")
        plan = result.summary.get("raw_capture_plan", {})
        print(f"- planned_markets: {plan.get('planned_markets', 0)}")
        print(f"- captured_markets: {plan.get('captured_markets', 0)}")
        print(f"- failed_fetch_markets: {plan.get('failed_fetch_markets', 0)}")
        print("- picks_emitidos: 0")
        print("- global_roi_actionable: false")
        return

    if args.command == "run-market-lane":
        summary, artifacts = run_market_lane_pipeline(
            settings=settings,
            lane_id=str(args.lane_id),
            db_path=Path(args.db_path) if args.db_path else None,
        )
        print("Carril atomico ejecutado.")
        print(f"- lane_id: {summary['lane_id']}")
        print(f"- benchmark_id: {summary['benchmark_id']}")
        print(f"- readiness_status: {summary['readiness_status']}")
        print(f"- readiness_blockers: {','.join(summary.get('readiness_blockers', []))}")
        print(f"- lane_manifest: {artifacts['lane_manifest']}")
        print(f"- lane_readiness_report: {artifacts['lane_readiness_report']}")
        print(f"- lane_decision_inventory_report: {artifacts['lane_decision_inventory_report']}")
        print(f"- lane_candidate_report: {artifacts['lane_candidate_report']}")
        print(f"- lane_candidate_rows: {artifacts['lane_candidate_rows']}")
        print(f"- model_prediction_template: {artifacts['model_prediction_template']}")
        print(f"- forward_ledger: {artifacts['forward_ledger']}")
        print(f"- sample_report: {artifacts['sample_report']}")
        print(f"- candidate_scoring_status: {summary.get('candidate_scoring', {}).get('candidate_scoring_status')}")
        if "policy_bundle" in artifacts:
            print(f"- policy_bundle: {artifacts['policy_bundle']}")
        if "legacy_parity_report" in artifacts:
            print(f"- legacy_parity_report: {artifacts['legacy_parity_report']}")
        print("- global_roi_actionable: false")
        return

    if args.command == "build-market-lane-predictions":
        summary, artifacts = build_market_lane_predictions_pipeline(
            settings=settings,
            lane_id=str(args.lane_id),
            db_path=Path(args.db_path) if args.db_path else None,
            model_path=Path(args.model_path) if args.model_path else None,
        )
        print("Predicciones de carril generadas.")
        print(f"- lane_id: {summary['lane_id']}")
        print(f"- template_rows: {summary['template_rows']}")
        print(f"- predicted_rows: {summary['predicted_rows']}")
        print(f"- prediction_coverage_rate: {summary['prediction_coverage_rate']:.4f}")
        print(f"- blocker_counts: {summary.get('blocker_counts', {})}")
        print(f"- model_predictions: {artifacts['model_predictions']}")
        print(f"- model_prediction_report: {artifacts['model_prediction_report']}")
        print("- policy_reoptimized: false")
        print("- global_roi_actionable: false")
        return

    if args.command == "train-market-lane-model":
        summary, artifacts = train_market_lane_model_pipeline(
            settings=settings,
            lane_id=str(args.lane_id),
            dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            leagues=tuple(args.leagues) if args.leagues else None,
            seasons=tuple(args.seasons) if args.seasons else None,
            dataset_name=args.dataset_name,
        )
        print("Modelo de carril entrenado.")
        print(f"- lane_id: {summary['lane_id']}")
        print(f"- leagues: {summary['leagues']}")
        print(f"- seasons: {summary['seasons']}")
        print(f"- rows: {summary['rows']}")
        print(f"- model: {artifacts['model']}")
        print(f"- lane_model_manifest: {artifacts['lane_model_manifest']}")
        print("- latest_model_pointer_updated: false")
        print("- policy_reoptimized: false")
        print("- global_roi_actionable: false")
        return

    if args.command == "create-market-lane-policy":
        summary, artifacts = create_market_lane_policy_pipeline(
            settings=settings,
            lane_id=str(args.lane_id),
            overwrite=bool(args.overwrite),
        )
        benchmark = summary.get("benchmark_report", {}) if isinstance(summary.get("benchmark_report"), dict) else {}
        print("Policy de carril creada.")
        print(f"- lane_id: {summary['lane_id']}")
        print(f"- policy_name: {summary.get('policy', {}).get('policy_name')}")
        print(f"- policy_bundle: {artifacts['policy_bundle']}")
        print(f"- lane_policy_benchmark_report: {artifacts['lane_policy_benchmark_report']}")
        print(f"- selected_rows_after_policy: {benchmark.get('selected_rows_after_policy')}")
        print(f"- selected_horizon_counts: {benchmark.get('selected_horizon_counts')}")
        print("- policy_reoptimized: false")
        print("- global_roi_actionable: false")
        return

    if args.command == "report-market-lane":
        _, text = report_market_lane_pipeline(
            settings=settings,
            lane_id=str(args.lane_id),
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
        print(text)
        return

    if args.command == "report-sport-merge":
        report, artifacts = report_sport_merge_pipeline(
            settings=settings,
            sport=str(args.sport),
            db_path=Path(args.db_path) if args.db_path else None,
        )
        print("Sport merge report generado.")
        print(f"- sport: {report['sport']}")
        print(f"- events: {report['events']}")
        print(f"- features: {artifacts['sport_context_features']}")
        print(f"- report: {artifacts['sport_merge_report']}")
        print("- global_roi_actionable: false")
        return

    if args.command == "discover-multi-market":
        result = discover_multi_market_pipeline(
            settings=settings,
            db_path=Path(args.db_path) if args.db_path else None,
            limit_per_query=int(args.limit_per_query),
        )
        print("Discovery multi-market completado.")
        print(f"- run: {result.run.run_dir}")
        print(f"- SQLite: {result.database_path}")
        print(f"- manifest: {result.artifacts['market_family_manifest']}")
        print(f"- coverage: {result.artifacts['market_family_coverage_report']}")
        print("- global_roi_actionable: false")
        return

    if args.command == "capture-multi-market":
        result = capture_multi_market_pipeline(
            settings=settings,
            db_path=Path(args.db_path) if args.db_path else None,
            stream_seconds=int(args.stream_seconds),
            lane_id=str(args.lane_id) if args.lane_id else None,
            max_markets=int(args.max_markets) if args.max_markets else None,
            max_events=int(args.max_events) if args.max_events else None,
            capture_priority=str(args.capture_priority),
            freshness_seconds=int(args.freshness_seconds),
            max_stale_age_seconds=int(args.max_stale_age_seconds) if args.max_stale_age_seconds else None,
            include_policy_ready_only=bool(args.include_policy_ready_only),
        )
        print("Captura multi-market completada.")
        print(f"- run: {result.run.run_dir}")
        print(f"- SQLite: {result.database_path}")
        print(f"- sample_report: {result.artifacts['multi_market_forward_sample_report']}")
        print(f"- ledger: {result.artifacts['multi_market_forward_ledger']}")
        print(f"- raw_capture_plan_csv: {result.artifacts['raw_capture_plan_csv']}")
        print(f"- raw_capture_plan_json: {result.artifacts['raw_capture_plan_json']}")
        print(f"- capture_priority: {result.summary.get('capture_priority')}")
        print(f"- freshness_seconds: {result.summary.get('capture_freshness_seconds')}")
        print(f"- include_policy_ready_only: {str(result.summary.get('include_policy_ready_only')).lower()}")
        plan = result.summary.get("raw_capture_plan", {})
        print(f"- planned_markets: {plan.get('planned_markets', 0)}")
        print(f"- captured_markets: {plan.get('captured_markets', 0)}")
        print(f"- failed_fetch_markets: {plan.get('failed_fetch_markets', 0)}")
        print("- picks_emitidos: 0")
        print("- global_roi_actionable: false")
        return

    if args.command == "backfill-polymarket-history":
        result = backfill_polymarket_history_pipeline(
            settings=settings,
            dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            db_path=Path(args.db_path) if args.db_path else None,
            model_path=Path(args.model_path) if args.model_path else None,
        )
        print("Backfill historico de Polymarket completado.")
        print(f"- SQLite: {result.database_path}")
        print(f"- Summary: {result.summary_path}")
        print(f"- Events discovered: {result.summary['events_discovered']}")
        print(f"- Complete groups: {result.summary['complete_groups']}")
        print(f"- Price history rows: {result.summary['price_history_rows']}")
        return

    if args.command == "audit-polymarket-coverage":
        result = audit_polymarket_coverage_pipeline(
            settings=settings,
            dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            db_path=Path(args.db_path) if args.db_path else None,
            model_path=Path(args.model_path) if args.model_path else None,
        )
        print("Auditoria historica de cobertura Polymarket completada.")
        print(f"- SQLite: {result.database_path}")
        print(f"- Summary: {result.summary_path}")
        print(f"- Raw markets: {result.summary['raw_market_rows']}")
        print(f"- True 1X2 complete groups: {result.summary['true_1x2_complete_groups']}")
        print(f"- Binary/non-1X2 match markets: {result.summary['binary_match_markets']}")
        return

    if args.command == "backtest-polymarket-retro":
        result = backtest_polymarket_retro_pipeline(
            settings=settings,
            dataset_dir=args.dataset_dir,
            db_path=Path(args.db_path) if args.db_path else None,
            model_path=Path(args.model_path) if args.model_path else None,
            policy_bundle_path=Path(args.policy_bundle) if args.policy_bundle else None,
        )
        print("Backtest retro de Polymarket completado.")
        print(f"- run: {result.run.run_dir}")
        print(f"- summary: {result.artifacts['retro_shadow_summary']}")
        print(f"- coverage: {result.artifacts['retro_coverage_summary']}")
        print(f"- model_variant: {result.summary.get('model_variant', 'v1')}")
        print(f"- probability_source: {result.summary.get('probability_source', 'raw')}")
        print(f"- diagnostic_only: {str(result.summary.get('diagnostic_only', True)).lower()}")
        print(f"- policy_written: {str(result.summary.get('policy_written', False)).lower()}")
        print(f"- candidates: {result.coverage_summary.get('candidate_rows', len(result.candidates))}")
        print(f"- mapped_matches: {result.coverage_summary['mapped_matches']}")
        print(f"- holdout_bets: {result.summary.get('policy_metrics', {}).get('bets', 0)}")
        print(
            "- lifecycle: "
            f"{result.summary.get('validation_stage', 'retro')} / "
            f"{result.summary.get('price_provenance', 'unknown')} / "
            f"{result.summary.get('bundle_readiness', result.summary.get('bundle_status', 'provisional'))}"
        )
        print(f"- coverage_status: {result.coverage_summary['coverage_status']}")
        print(f"- promotion_status: {result.summary.get('promotion_decision', {}).get('status', 'pending')}")
        print(f"- net_roi: {result.summary['net_roi']:.4f}")
        return

    if args.command == "tune-polymarket-policy":
        result = tune_polymarket_policy_pipeline(
            settings=settings,
            dataset_dir=args.dataset_dir,
            db_path=Path(args.db_path) if args.db_path else None,
            model_path=Path(args.model_path) if args.model_path else None,
            write_policy=bool(args.write_policy),
        )
        print("Politica de Polymarket afinada.")
        print(f"- run: {result.run.run_dir}")
        if result.policy_bundle_path:
            print(f"- policy_bundle: {result.policy_bundle_path}")
        print(f"- diagnostic_only: {str(result.summary.get('diagnostic_only', True)).lower()}")
        print(f"- policy_written: {str(result.summary.get('policy_written', False)).lower()}")
        if result.summary.get("policy_write_blocked_reason"):
            print(f"- policy_write_blocked_reason: {result.summary['policy_write_blocked_reason']}")
        print(
            "- lifecycle: "
            f"{result.summary.get('validation_stage', 'retro')} / "
            f"{result.summary.get('price_provenance', 'unknown')} / "
            f"{result.summary.get('bundle_readiness', result.summary.get('bundle_status', 'provisional'))}"
        )
        print(f"- bundle_status: {result.summary['bundle_status']}")
        print(f"- net_roi: {result.summary['net_roi']:.4f}")
        return

    if args.command in {"backtest", "train"}:
        result = backtest_pipeline(
            settings=settings,
            dataset_dir=args.dataset_dir,
            leagues=tuple(args.leagues or settings.default_leagues),
            seasons=tuple(args.seasons or settings.default_seasons),
        )
        print("Backtest completado.")
        print(format_summary(result.summary))
        print("- Artefactos:")
        for name, path in result.artifacts.items():
            print(f"  - {name}: {path}")
        return

    if args.command == "backtest-net":
        result = backtest_net_pipeline(
            settings=settings,
            dataset_dir=args.dataset_dir,
            snapshot_files=[Path(path) for path in args.snapshot_file] if args.snapshot_file else None,
            leagues=tuple(args.leagues or settings.default_leagues),
            seasons=tuple(args.seasons or settings.default_seasons),
        )
        print("Backtest neto completado.")
        print(format_net_summary(result.summary))
        print("- Artefactos:")
        for name, path in result.artifacts.items():
            print(f"  - {name}: {path}")
        return

    if args.command == "discover-niches":
        niches, path = discover_niches_pipeline(
            settings=settings,
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
        print("Nichos calculados.")
        print(f"- CSV: {path}")
        if niches.empty:
            print("- No encontre segmentos robustos con los filtros actuales.")
            return
        for index, row in niches.head(10).iterrows():
            print(
                f"[{index + 1}] roi={row['roi']:.4f} bets={int(row['bets'])} score={row['score']:.4f} filters={row['filters_json']}"
            )
        return

    if args.command == "train-final":
        result = train_final_pipeline(
            settings=settings,
            dataset_dir=args.dataset_dir,
            leagues=tuple(args.leagues or settings.default_leagues),
            seasons=tuple(args.seasons or settings.default_seasons),
        )
        print("Entrenamiento final completado.")
        print(f"- Modelo: {result.model_path}")
        print(f"- Summary: {result.summary_path}")
        return

    if args.command == "train-niche":
        result = train_niche_pipeline(
            settings=settings,
            dataset_dir=args.dataset_dir,
            research_run_dir=Path(args.research_run_dir) if args.research_run_dir else None,
            leagues=tuple(args.leagues or settings.default_leagues),
            seasons=tuple(args.seasons or settings.default_seasons),
        )
        print("Niche model entrenado.")
        print(f"- Modelo: {result.model_path}")
        print(f"- Summary: {result.summary_path}")
        return

    if args.command == "predict":
        result = predict_pipeline(
            settings=settings,
            fixtures_path=Path(args.fixtures_file),
            model_path=Path(args.model_path) if args.model_path else None,
        )
        print("Prediccion completada.")
        print(f"- predictions: {result.artifacts['predictions']}")
        print(f"- bets: {result.artifacts['bets']}")
        return

    if args.command == "shadow-run":
        result = shadow_run_pipeline(
            settings=settings,
            fixtures_path=Path(args.fixtures_file),
            model_path=Path(args.model_path) if args.model_path else None,
            snapshot_files=[Path(path) for path in args.snapshot_file] if args.snapshot_file else None,
        )
        print("Shadow run preparado.")
        print(f"- predictions: {result.artifacts['predictions']}")
        print(f"- planned_bets: {result.artifacts['planned_bets']}")
        return

    if args.command == "shadow-polymarket":
        result = shadow_polymarket_pipeline(
            settings=settings,
            fixtures_path=Path(args.fixtures_file) if args.fixtures_file else None,
            db_path=Path(args.db_path) if args.db_path else None,
            model_path=Path(args.model_path) if args.model_path else None,
            policy_bundle_path=Path(args.policy_bundle) if args.policy_bundle else None,
        )
        print("Shadow Polymarket completado.")
        print(f"- run: {result.run.run_dir}")
        print(f"- decision_rows: {result.artifacts['decision_rows']}")
        print(f"- fill_rows: {result.artifacts['fill_rows']}")
        print(f"- forward_sample_report: {result.artifacts.get('forward_sample_report', '')}")
        print(f"- forward_sample_manifest: {result.artifacts.get('forward_sample_manifest', '')}")
        print(
            "- lifecycle: "
            f"{result.summary.get('validation_stage', 'shadow')} / "
            f"{result.summary.get('price_provenance', 'exact')} / "
            f"{result.summary.get('bundle_readiness', result.summary.get('policy_bundle_status', 'unknown'))}"
        )
        return

    if args.command == "report":
        _, text = report_pipeline(run_dir=Path(args.run_dir) if args.run_dir else None)
        print(text)
        return

    if args.command == "report-polymarket":
        _, text = report_polymarket_pipeline(
            settings=settings,
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
        print(text)
        return

    if args.command == "report-polymarket-retro":
        _, text = report_polymarket_retro_pipeline(
            settings=settings,
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
        print(text)
        return

    if args.command == "report-multi-market":
        _, text = report_multi_market_pipeline(
            settings=settings,
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
        print(text)
        return

    if args.command == "promote-report":
        _, text = promotion_report_pipeline(run_dir=Path(args.run_dir) if args.run_dir else None)
        print(text)
        return

    if args.command == "markets":
        results = search_polymarket(query=args.query, limit=args.limit)
        if not results:
            print("No encontre mercados que encajen con esa busqueda.")
            return

        for index, market in enumerate(results, start=1):
            print(f"[{index}] {market.get('question')}")
            print(f"    slug={market.get('slug')}")
            print(f"    event={market.get('eventSlug')}")
            print(f"    active={market.get('active')} closed={market.get('closed')}")
        return

    if args.command == "claude-report":
        analyst = build_analyst(
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
        )
        bookmaker = ProbabilitySnapshot(*args.bookmaker)
        model_probs = ProbabilitySnapshot(*args.model)
        polymarket = ProbabilitySnapshot(*args.polymarket) if args.polymarket else None
        report = analyst.analyze_probability_divergence(
            match_name=args.match,
            bookmaker=bookmaker,
            model_probs=model_probs,
            polymarket=polymarket,
        )
        print(f"Reporter: {analyst.provider_name}")
        print()
        print(report)
        return

    raise SystemExit(f"Comando no soportado: {args.command}")


if __name__ == "__main__":
    main()
