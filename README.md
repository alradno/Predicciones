# Predicciones

Predicciones is a lane-isolated sports market validation project. The current
focus is not optimization; it is honest forward evidence collection for football
lanes under frozen policies.

## Operating State

- Active ROI validation lanes: `football_1x2_global` and `football_goals_core`.
- `football_1x2_canonical` is archived as `reference_only`.
- Other sports remain capture-only or paused until they have model and settlement
  contracts.
- ROI is diagnostic-only before `sample_ready`.
- No capital picks are emitted before `capital_promotable`.

Read the operating rules in `docs/OPERATING_RULES.md`.

## Installation

```powershell
cd C:\Users\Alberto\Desktop\Predicciones
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Optional local config:

```powershell
Copy-Item config.example.json config.local.json
notepad config.local.json
```

The CLI accepts `--config config.local.json`.

## Daily Workflow

Use the single lane cycle wrapper:

```powershell
.\scripts\run_multi_market_lane_cycle.ps1 -LaneId football_1x2_global -MaxEvents 25
.\scripts\run_multi_market_lane_cycle.ps1 -LaneId football_goals_core -MaxEvents 25
```

Equivalent CLI commands:

```powershell
predicciones lane capture-raw --lane-id football_1x2_global --capture-priority missing_or_stale --freshness-seconds 3600 --max-events 25
predicciones lane build-predictions --lane-id football_1x2_global
predicciones lane run-shadow --lane-id football_1x2_global
predicciones lane evaluate-forward --lane-id football_1x2_global
predicciones lane report --lane-id football_1x2_global
predicciones sport report --sport football
```

Model candidates and trading automation are staged behind evidence gates:

```powershell
predicciones model train --lane-id football_1x2_global --variant champion_poisson_elo_calibrated
predicciones model validate --lane-id football_1x2_global --candidate champion_poisson_elo_calibrated
predicciones trade paper --lane-id football_1x2_global
predicciones trade live --lane-id football_1x2_global --mode micro
predicciones trade reconcile
predicciones trade kill-switch --enable --reason "manual stop"
```

Live trading is disabled by default. It also remains blocked unless lane promotion,
ROI 45 hypothesis support, venue, bankroll, jurisdiction confirmation, fresh books,
risk caps, kill switch, and reconciliation checks all pass.

Simulation data commands live under `sim`:

```powershell
predicciones sim discover-sources
predicciones sim collect --source football_data --leagues E0 SP1 D1 --seasons 2526 2425 2324
predicciones sim normalize
predicciones sim build-features
predicciones sim report
predicciones sim export-training
predicciones sim train
```

Archived reference manifest:

```powershell
predicciones archive legacy-1x2
```

## Modular Layout

- `predicciones.core`: governance, promotion, forward metrics.
- `predicciones.football`: football data, features, simulation, training.
- `predicciones.markets`: Polymarket storage, capture, shadow primitives, and gated execution adapters.
- `predicciones.lanes`: lane registry, prediction, frozen-policy execution.
- `predicciones.validation`: retro validation and diagnostic analysis.
- `predicciones.reports`: lane and market reporting.
- `predicciones.archive`: reference-only material; active modules must not import it.

The removed root modules `multi_market`, `polymarket_retro`,
`polymarket_shadow*`, `football_sim_data`, `football_simulator`, `dataset`,
`features`, `modeling`, and `pipeline` are intentionally not public entrypoints.

## Tests

Always validate with the project virtualenv:

```powershell
.\.venv\Scripts\python.exe -m compileall src tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The global Python installation is not expected to have the project dependencies.

## Artifacts

Local artifacts are reproducible and ignored by git. `outputs/`, generated
datasets, local SQLite databases, bytecode caches, and old benchmark bundles are
purged from the clean working base. Retention rules live in
`docs/ARTIFACT_RETENTION.md`.
