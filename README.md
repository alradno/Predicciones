# Predicciones

Proyecto Python para evaluar y explotar predicciones de futbol prepartido con cuatro capas separadas:

- baseline de mercado a partir de cuotas
- goal model independiente sin usar cuotas como features
- motor determinista de edge y EV
- politica de apuesta seleccionada dentro del entrenamiento

El foco ya no es solo acertar `1X2`, sino medir si existe ventaja real frente al mercado con backtests temporales sin leakage.

## 1. Preparacion

```powershell
cd C:\Users\Alberto\Desktop\Predicciones
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Configuracion opcional por entorno:

```powershell
Copy-Item .env.example .env
notepad .env
```

Configuracion opcional por archivo JSON:

```powershell
Copy-Item config.example.json config.local.json
notepad config.local.json
```

El CLI acepta `--config config.local.json`.

## 1.1 Pruebas locales

La suite real del proyecto usa `unittest`, no `pytest`:

```powershell
python -m unittest discover -s tests -v
```

Ese es el comando que conviene usar para validar cambios en local.

## 1.2 Wrappers locales

Para no memorizar comandos largos, hay wrappers PowerShell en `scripts\` que apuntan al CLI real del proyecto:

```powershell
.\scripts\run_backtest.ps1
.\scripts\run_backtest_net.ps1
.\scripts\run_shadow_polymarket.ps1
.\scripts\run_forward_sample_cycle.ps1
.\scripts\report_forward_capture_status.ps1
.\scripts\run_offline_decision_region_review.ps1
.\scripts\report_latest.ps1
.\scripts\cleanup_old_runs.ps1
```

Todos aceptan argumentos extra del CLI cuando tiene sentido. Por ejemplo:

```powershell
.\scripts\run_backtest.ps1 --dataset-dir C:\Users\Alberto\Desktop\Predicciones\data\dataset_E0-SP1-D1_2425-2324-2223
.\scripts\report_latest.ps1
.\scripts\report_forward_capture_status.ps1
.\scripts\run_offline_decision_region_review.ps1
.\scripts\cleanup_old_runs.ps1 -Days 30
.\scripts\cleanup_old_runs.ps1 -Days 30 -Delete
```

Los wrappers intentan usar primero `.venv\Scripts\predicciones.exe` y, si no existe, caen a `python -m predicciones.cli`.
Si tu PowerShell bloquea la ejecucion de scripts, usa `-ExecutionPolicy Bypass` al invocarlos:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_backtest.ps1
```

## 1.3 Registro De Experimentos Rechazados

Esta seccion deja constancia de lineas de trabajo que **si se probaron** dentro del benchmark canonico pero que se descartaron por no mejorar el edge de forma honesta.

Regla del proyecto:

- si una idea no mejora `OOF`, `pre_holdout` y `locked_holdout` bajo la misma policy congelada, se rechaza
- los artefactos del run se conservan en `outputs\runs\...` como evidencia
- la integracion experimental se retira del codigo activo para no acumular deuda tecnica innecesaria

### `penaltyblog` como challenger externo `1X2`

Estado: rechazado y retirado del pipeline activo.

Que se probo:

- variantes `pb_dixon_coles` y `pb_bivariate_poisson`
- mismo universo `E0 + SP1 + D1`
- mismo flujo `OOF -> pre_holdout -> locked_holdout`
- misma policy congelada
- sin usar cuotas ni senales de Polymarket para entrenar el modelo externo

Baseline congelado de comparacion:

- `outputs\runs\backtest_polymarket_retro_20260418_181928`
- campeon baseline: `v5/raw`
- `oof_aggregate_roi = -0.1784`
- `pre_holdout_aggregate_roi = -0.0186`
- `locked_holdout_frozen_policy_roi = -0.0379`

Run experimental conservado:

- `outputs\runs\backtest_polymarket_retro_20260418_190930`

Resultado:

- `pb_dixon_coles` mejoro `OOF ROI` a `+0.0649`, pero empeoro mucho `log_loss`, `pre_holdout` (`-0.3601`) y `holdout` (`-0.2925`)
- `pb_bivariate_poisson` quedo muy parecido: `OOF ROI +0.0501`, `pre_holdout -0.3716`, `holdout -0.2925`
- el campeon siguio siendo `v5/raw`, asi que no hubo mejora real del benchmark

Decision:

- se conserva la evidencia en artefactos
- se elimina la integracion del codigo y la dependencia
- no se abre la fase `MAPIE` porque ninguna variante `pb_*` supero a `v5/raw` en las capas exigidas

### `v6/v7` con confianza larga, backoff `v2` y decision-region scorers

Estado: implementado y rechazado como linea valida para promocion offline.

Que se probo:

- variantes `v6`, `v6b`, `v7`, `v7b`
- nueva capa de confianza larga desacoplada de la forma corta
- backoff de probabilidades hacia `v1/raw`
- scorers de segunda capa para la `decision region`:
  - `heuristic`
  - `dr_logit`
  - `dr_hgb`
- rama secundaria entrenando el scorer solo sobre `argmax_all_bets`
- mismo benchmark canonico:
  - `E0 + SP1 + D1`
  - `1X2`
  - `OOF -> pre_holdout -> locked_holdout`
  - policy congelada
  - sin tocar `locked_holdout`

Baseline congelado de comparacion:

- `outputs\runs\backtest_polymarket_retro_20260418_192807`
- campeon baseline: `v5/raw`
- `oof_aggregate_roi = -0.1784`
- `pre_holdout_aggregate_roi = -0.0186`
- `locked_holdout_frozen_policy_roi = -0.0379`

Run experimental conservado:

- `outputs\runs\backtest_polymarket_retro_20260418_201858`

Resultado:

- el campeon del run paso a `v6/raw`, pero solo con `locked_holdout_frozen_policy_roi = +0.1108`, `oof_aggregate_roi = -0.2130` y `pre_holdout_aggregate_roi = -0.0379`
- las variantes con backoff `v6b` y `v7b` llegaron a `locked_holdout_frozen_policy_roi = +0.4667`, superando el objetivo nominal del `45%`
- aun asi quedaron rechazadas porque mantenian `oof_aggregate_roi = -0.4388` y `pre_holdout_aggregate_roi = -0.0425`
- `v3r` tambien enseño un holdout alto (`+0.4623`), pero con `OOF` claramente negativo y `pre_holdout` aun por debajo de cero
- los scorers `dr_logit` y `dr_hgb` no cambiaron el veredicto del benchmark; bajo la policy congelada no consiguieron volver positivos `OOF` ni `pre_holdout`

Decision:

- se conserva la evidencia completa del run y de todas las ramas evaluadas
- no se considera alcanzado el objetivo de `45%` porque la mejora solo aparecio en `locked_holdout`
- la conclusion oficial de esta fase es que el cuello de botella actual es `estabilidad/muestra`, no falta de tuning adicional
- cualquier nueva iteracion debe partir de esta conclusion y no reabrir esta rama como si no se hubiera probado

### Penalizacion regional OOF por `selection x odds_band`

Estado: implementado y rechazado como correccion suficiente del cuello de botella.

Que se probo:

- ajuste regional entrenado solo con OOF por region `selection|odds_band`
- aplicacion crossfit por `retro_fold_id` para no filtrar informacion futura
- sustitucion de `policy_prob`, `policy_edge` y `policy_ev` por sus versiones regionalmente ajustadas cuando existiesen
- evaluacion sobre las ramas nuevas `v6`, `v6b`, `v7`, `v7b` y sobre la rama secundaria `argmax_all_bets`

Run experimental conservado:

- `outputs\runs\backtest_polymarket_retro_20260418_211612`

Resultado:

- el benchmark no cambio materialmente frente a `20260418_201858`
- el campeon siguio siendo `v6/raw` con `locked_holdout_frozen_policy_roi = +0.1108`, `oof_aggregate_roi = -0.2130` y `pre_holdout_aggregate_roi = -0.0379`
- las ramas con `regional_adjustment = oof_region_shrink` quedaron practicamente iguales que sus equivalentes sin ajuste
- conclusion: el problema ya no es solo castigar regiones malas a posteriori, sino que la policy congelada cruza umbrales en una muestra demasiado pequena e inestable para que ese ajuste regional cambie la seleccion de forma significativa

Decision:

- se conserva la evidencia completa del run
- esta rama no resuelve el cuello de botella de `estabilidad/muestra`
- el siguiente paso no debe ser mas scoring cosmetico, sino atacar la definicion de muestra/region elegible antes de que la policy umbralizada dispare apuestas

### `stability_gate` jerarquico antes de la policy

Estado: implementado, auditado y rechazado como gate actual para promocion.

Que se probo:

- gate aplicado despues de `candidate_score_columns` y antes de `select_candidate_rows`
- entrenamiento solo con filas OOF que ya pasaban la policy congelada base
- aplicacion crossfit por `retro_fold_id` en OOF
- aplicacion forward a `pre_holdout` y `locked_holdout` usando solo OOF disponible
- regiones jerarquicas:
  - `league_code | selection | odds_band`
  - `selection | odds_band`
  - `selection`
  - `odds_band`
  - `global`
- probabilidades conservadoras `gated_prob`, `gated_edge` y `gated_ev`
- ramas nuevas `v6/v6b/v7/v7b + hierarchical_oof_gate`

Baseline congelado de comparacion:

- `outputs\runs\backtest_polymarket_retro_20260418_211612`
- campeon baseline: `v6/raw`
- `oof_aggregate_roi = -0.2130`
- `pre_holdout_aggregate_roi = -0.0379`
- `locked_holdout_frozen_policy_roi = +0.1108`
- `locked_holdout_bets = 37`

Run experimental conservado:

- `outputs\runs\backtest_polymarket_retro_20260418_233516`

Resultado:

- el campeon del run fue `v6/raw + hierarchical_oof_gate`
- el gate bloqueo `76.0%` de candidatos y `75.8%` de candidatos que habrian pasado la policy base
- `pre_holdout` mejoro de `-3.9%` a `+82.1%`, pero con solo `6` apuestas
- `OOF` empeoro ligeramente de `-21.3%` a `-21.9%`, y quedo con solo `11` apuestas
- `locked_holdout` empeoro de `+11.1%` a `-40.6%`, bajando de `37` a `16` apuestas
- las ramas `v6b/v7b + hierarchical_oof_gate` bloquearon el `100%` de picks y por tanto no aportaron evidencia operativa

Decision:

- se conserva el gate y sus artefactos porque son utiles como diagnostico de estabilidad
- no se promociona ni se considera mejora del benchmark
- el problema detectado no es que falte otro filtro mas fuerte, sino que el gate global/parental bloquea demasiado con evidencia OOF pequena
- la siguiente iteracion, si se mantiene esta linea, debe convertir los niveles globales en ajuste suave y reservar bloqueos duros para regiones especificas con soporte suficiente
- si esa version tampoco vuelve positivo `OOF + pre_holdout` sin destruir muestra, la salida honesta sigue siendo ampliar muestra historica/forward antes de mas tuning

### `stability_gate` corregido con padres suaves

Estado: implementado y rechazado como mejora promocionable, pero valido como correccion metodologica del gate anterior.

Que se corrigio:

- se mantiene `hierarchical_oof_gate` como evidencia historica del filtro duro rechazado
- se anade `hierarchical_soft_parent_gate`
- los niveles `global`, `selection` y `odds_band` ya no pueden bloquear duro
- esos niveles parentales solo aplican un ajuste suave de probabilidad
- los bloqueos duros quedan reservados para regiones especificas con soporte suficiente:
  - `league_code | selection | odds_band`
  - `selection | odds_band`

Run experimental conservado:

- `outputs\runs\backtest_polymarket_retro_20260419_020641`

Resultado:

- el campeon final volvio a ser `v6/raw` sin gate
- `locked_holdout_frozen_policy_roi = +0.1108`
- `oof_aggregate_roi = -0.2130`
- `pre_holdout_aggregate_roi = -0.0379`
- la rama corregida `v6 + hierarchical_soft_parent_gate` mejoro `OOF` hasta `-0.1624` y dejo `pre_holdout_aggregate_roi = +0.0949`
- aun asi no paso gates porque `OOF` siguio negativo y `pre_holdout_positive_window_ratio = 1/3`
- ademas el holdout de esa rama bajo a `-0.2911`
- el bloqueo bajo de forma importante frente al gate duro: `blocked_candidate_share = 20.6%` y `blocked_base_policy_pick_share = 3.4%`

Decision:

- la correccion arregla el sobrebloqueo mecanico del gate anterior
- no genera una mejora robusta suficiente para promocion
- el champion sigue siendo el baseline sin gate porque es menos malo bajo el ranking disciplinado
- la senal mas importante es que suavizar el gate mejora OOF/pre-holdout, pero no lo bastante; el siguiente cuello de botella sigue siendo muestra/estabilidad de la zona apostada

### Historia larga gratuita y bug de `match_id` directo obsoleto

Estado: bug metodologico corregido; la historia larga queda como experimento valido solo despues del fix.

Que se probo:

- mantener el mismo universo `E0 + SP1 + D1`
- ampliar solo la historia gratuita de entrenamiento de `4.088` a `9.418` partidos canonicos
- no cambiar ligas, mercados, policy, thresholds, scopes ni `T-45m`
- conservar el benchmark Polymarket retro con `locked_holdout` intacto

Bug detectado:

- el primer run con historia larga (`outputs\runs\backtest_polymarket_retro_20260419_165417`) mostro `mapped_matches = 1005` pero solo `734` grupos completos
- eso era imposible como evidencia limpia
- causa: algunos `match_id` numericos de football-data no eran estables al cambiar el rango de temporadas y se enlazaban a grupos Polymarket de otras temporadas
- ejemplo observado: un partido de 2019 quedaba enlazado por `match_id` directo a un mercado de 2024 con otros equipos

Correccion:

- el mapping directo por `match_id` ahora exige equipos plausibles y ventana temporal razonable
- los partidos reprogramados siguen aceptandose si los equipos encajan y el desplazamiento temporal es plausible
- se anadio test unitario para rechazar IDs directos obsoletos con equipos incorrectos

Run limpio conservado:

- `outputs\runs\backtest_polymarket_retro_20260419_172523`

Resultado limpio:

- `mapped_matches = 733`
- `unique_groups = 733`
- `duplicated_group_rows = 0`
- campeon: `v3r/raw`
- `oof_aggregate_roi = +0.0145`, pero `oof_positive_fold_ratio = 0.25`
- `pre_holdout_aggregate_roi = -0.3314`
- `locked_holdout_frozen_policy_roi = -0.0049`
- estado: `overfit_rejected`

Decision:

- la historia larga ayuda a que algunas variantes dejen de ser tan negativas en OOF y holdout
- no resuelve la estabilidad de pre-holdout
- no se promociona
- el run contaminado queda descartado como evidencia y el fix queda como guardrail permanente

### Ajuste regional entrenado de verdad sobre la zona apostada

Estado: corregido, auditado y rechazado como mejora suficiente.

Que se corrigio:

- `oof_region_shrink` ya no aprende sobre todo el universo elegible, sino sobre la region que habria pasado la policy congelada base
- el ajuste regional ahora se aplica despues de calcular `policy_prob`, `policy_edge` y `policy_ev`
- se corrigio otro bug: el flujo real no tenia columna `won`, asi que el ajuste no estaba entrenando; ahora infiere victoria con `selection == actual_outcome`
- `ablation_report.json` y `decision_scorer_ablation.json` guardan filas y scope real de entrenamiento regional

Runs conservados:

- `outputs\runs\backtest_polymarket_retro_20260419_175242`
- `outputs\runs\backtest_polymarket_retro_20260419_182604`

Resultado:

- el shrink duro ahora entrena de verdad (`regional_adjustment_training_scope = base_policy_region`, `training_rows ~= 102-106`)
- pero destruye muestra: por ejemplo `v6 + oof_region_shrink` baja a `27` apuestas OOF, `10` pre-holdout y `10` holdout
- `v6 + oof_region_shrink`: `oof_aggregate_roi = -0.4388`, `pre_holdout_aggregate_roi = -0.1914`, `holdout = -0.0171`
- se anadio una variante fija mas suave, `oof_region_soft_shrink`, para preservar muestra sin abrir grid
- `v6 + oof_region_soft_shrink`: `oof_aggregate_roi = -0.0919`, `pre_holdout_aggregate_roi = -0.1514`, `holdout = +0.0248`, `holdout_bets = 40`
- `v6b/v7b + oof_region_soft_shrink` mejoran pre-holdout (`+0.3394`) pero fallan OOF (`-0.2932`) y holdout (`-0.1433`)

Decision:

- el ajuste regional funcional confirma la hipotesis: el cuello no era solo el tipo de gate
- cuando el filtro es duro, mata muestra
- cuando es suave, conserva muestra pero no vuelve positivos OOF y pre-holdout a la vez
- la conclusion sigue siendo que la zona apostada tiene estabilidad insuficiente bajo la policy congelada actual
- siguiente trabajo honesto: mas muestra forward/historica fiable o reformular la decision region, no seguir apilando gates parecidos

### Reformulacion de la decision region y crossfit temporal

Estado: implementado, auditado y rechazado como mejora suficiente.

Que se corrigio:

- se anadieron scorers que no solo reordenan candidatos, sino que recalibran de forma conservadora la probabilidad que ve la policy congelada:
  - `dr_logit_prob`
  - `dr_hgb_prob`
  - `dr_reliability_prob`
- estas ramas escriben `decision_adjusted_prob`, `decision_adjusted_edge` y `decision_adjusted_ev`
- la policy congelada usa esos valores solo como endurecimiento: nunca pueden subir la probabilidad base ni crear apuestas nuevas
- el calibrador de fiabilidad `dr_reliability_prob` usa tasas empiricas shrinked por bandas amplias de seleccion, odds y probabilidad, sin librerias nuevas
- el segundo modelo descarta columnas numericas totalmente vacias antes de entrenar, evitando ruido y warnings de imputacion
- se corrigio el crossfit de `decision_region` y `stability_gate`: cada fold OOF ahora entrena solo con folds anteriores (`past_folds_only`), no con futuro

Runs conservados:

- `outputs\runs\backtest_polymarket_retro_20260419_191454`
- `outputs\runs\backtest_polymarket_retro_20260419_194530`
- `outputs\runs\backtest_polymarket_retro_20260419_201606`

Resultado final mas honesto (`20260419_201606`):

- cobertura: `734` grupos completos, `733` partidos mapeados, `2109` candidates, `34` picks finales
- campeon: `v3r/raw`, `decision_scorer = heuristic`, sin gate ni ajuste regional
- `locked_holdout_frozen_policy_roi = -0.0049`
- `oof_aggregate_roi = +0.0145`
- `oof_positive_fold_ratio = 0.25`
- `pre_holdout_aggregate_roi = -0.3316`
- estado: `overfit_rejected`, `coverage_limited`

Lecturas importantes de la ablacion:

- `v3r + dr_logit_prob` mejora OOF (`+0.1128`) y holdout (`+0.1594`), pero falla pre-holdout (`-0.5527`) y solo deja `18` apuestas en holdout
- `v4 + dr_reliability_prob` logra OOF positivo (`+0.1602`) y holdout alto (`+0.4080`), pero con solo `5` apuestas holdout y pre-holdout negativo (`-0.4607`)
- las ramas `argmax_all_bets + dr_logit_prob` que parecian positivas antes del crossfit temporal quedaron invalidadas: al usar solo pasado en OOF, OOF cae a negativo o la muestra queda inutil
- tras el crossfit temporal no queda ninguna combinacion con `OOF > 0` y `pre_holdout > 0`

Decision:

- la reformulacion profunda de la decision region no alcanza estabilidad transferible
- los ROIs altos aparecen solo con muestra demasiado pequena, que es exactamente el patron que queremos rechazar
- no se promociona ninguna rama
- el siguiente cuello de botella ya no debe atacarse con mas scorers/gates offline parecidos
- siguiente paso honesto: acumular mas muestra forward fiable de la zona apostada y/o ampliar historico de mercados realmente mapeables, manteniendo policy congelada y sin tocar `T-45m`

### Ledger forward de muestra fiable

Estado: implementado como guardrail operativo para dejar de sobreinterpretar el retro offline.

Cambios:

- `shadow-polymarket` mantiene la policy congelada y no modifica `T-45m`
- cada run genera ahora `forward_sample_report.json`, `forward_sample_ledger.csv` y `forward_sample_blockers.csv`
- el ledger separa decisiones que realmente cuentan como muestra forward de:
  - fallos de mapping
  - libros ausentes
  - libros stale
  - rechazos por policy
  - precios no exactos o invalidos
- una decision solo cuenta como `valid_forward_sample` si:
  - fue seleccionada por la policy congelada
  - tiene mapping completo
  - tiene snapshot de libro
  - el libro esta fresco segun `decision_book_freshness_seconds`
  - usa precio exacto de orderbook
  - `model_prob` y `top_ask` son finitos
- el reporte acumula muestra historica desde `pm_shadow_decisions` y `pm_shadow_fills`, no solo el run actual

Objetivos de muestra antes de volver a tocar la decision region:

- `valid_forward_decisions >= 100`
- `settled_unique_decisions >= 40`
- `fresh_book_rate >= 0.80`

Estados posibles:

- `collecting_forward_sample`: seguir acumulando shadow con la policy congelada
- `coverage_blocked`: mejorar captura de orderbooks antes de sacar conclusiones de ROI
- `settlement_pending`: ya hay decisiones validas, pero falta que se resuelvan partidos
- `sample_ready`: ya hay muestra suficiente para reabrir analisis de decision region

Decision:

- no se considera esto una mejora de ROI por si misma
- si el ledger marca `coverage_blocked`, el trabajo correcto es mejorar captura forward, no volver a tunear offline
- si llega a `sample_ready`, la siguiente iteracion debe evaluar la zona apostada con decisiones forward reales antes de proponer otro cambio de modelo

### Ciclo autonomo de muestra forward

Estado: implementado para operar la acumulacion sin tocar modelo, policy ni `T-45m`.

Comando recomendado:

```powershell
.\scripts\run_forward_sample_cycle.ps1 -MaxCycles 1
```

Modo de validacion sin ejecutar comandos:

```powershell
.\scripts\run_forward_sample_cycle.ps1 -DryRun
```

Modo verboso con pulso frecuente:

```powershell
.\scripts\run_forward_sample_cycle.ps1 -MaxCycles 1 -HeartbeatSeconds 30
```

Modo largo seguro hasta la ventana `T-45m`:

```powershell
.\scripts\run_forward_sample_cycle.ps1 -MaxCycles 1 -AllowLongStream -HeartbeatSeconds 60 -CaptureChunkSeconds 1800
```

Reglas del ciclo:

- resuelve la policy congelada desde `outputs\latest_polymarket_policy.txt`
- si no existe policy bundle, cae al bundle del `latest_niche_model`
- calcula la proxima ventana `T-45m` conocida y ajusta `--stream-seconds`
- por defecto capa capturas largas a `7200s` para evitar dejar una terminal bloqueada durante muchas horas
- para cubrir toda la ventana calculada, usar `-AllowLongStream`; incluso en ese modo, la captura se divide en chunks reanudables
- cada chunk usa `-CaptureChunkSeconds` y se reintenta hasta `-MaxCollectRetries` antes de continuar con lo ya capturado
- durante `collect-polymarket`, imprime un heartbeat con fase, tiempo transcurrido, filas nuevas de checkpoints/books/trades, decisiones/fills y edad del ultimo checkpoint
- si el heartbeat muestra `capturing_new_rows`, la captura esta avanzando; si muestra `alive_no_new_rows_yet`, el proceso sigue vivo pero todavia no entran libros nuevos
- ejecuta `collect-polymarket`, luego `shadow-polymarket`, luego `report-polymarket`
- si aparece `coverage_blocked`, escribe `capture_blocker_report.json`
- si aparece `settlement_pending`, el siguiente ciclo solo refresca resoluciones y mantiene la misma policy
- si aparece `sample_ready`, se puede abrir el analisis forward de decision region

Artefactos nuevos:

- `forward_sample_manifest.json`: contrato congelado del run, con policy, modelo, DB, `T-45m`, freshness y `policy_reoptimized=false`
- `forward_decision_region_report.json`: solo se genera con contenido accionable cuando `sample_status = sample_ready`
- `capture_blocker_report.json`: se escribe cuando el problema operativo es captura/cobertura

Correccion operativa importante:

- la simulacion forward usa ahora `decision_book_freshness_seconds` para validar el checkpoint de decision
- esto alinea captura, shadow y ledger en la misma definicion de frescura para `T-45m`

### Auditoria offline de zona apostada mientras corre forward

Estado: implementado como herramienta read-only para aprovechar horas de captura sin tocar modelo, policy, `T-45m` ni la SQLite operativa en escritura.

Comandos:

```powershell
.\scripts\report_forward_capture_status.ps1
.\scripts\report_forward_capture_status.ps1 -Json
.\scripts\run_offline_decision_region_review.ps1
```

Garantias:

- `report_forward_capture_status.ps1` abre `data\polymarket_shadow.sqlite` en modo solo lectura y no ejecuta `shadow-polymarket`
- `run_offline_decision_region_review.ps1` lee el ultimo `backtest_polymarket_retro_*` y crea un run nuevo `offline_decision_region_review_*`
- no cambia thresholds, scopes, outcomes, policy bundle ni `T-45m`
- no usa `locked_holdout` para entrenar filtros ni decidir hipotesis
- cualquier mejora que falle `OOF` o `pre_holdout` queda rechazada aunque el holdout salga alto

Artefactos:

- `decision_region_noise_report.json/csv`: picks que pasan la policy y banderas de ruido por split, outcome, liga, odds band y confianza
- `frozen_policy_comparison_report.json/csv`: comparacion de policy congelada, probabilidad conservadora existente y exclusion por estabilidad pre-holdout
- `decision_region_reformulation_report.json/csv`: ranking estable experimental, siempre con una apuesta maxima por partido

Run inicial conservado:

- `outputs\runs\offline_decision_region_review_20260420_000344`

Resultado inicial:

- `177` picks pasan la policy congelada en el retro analizado
- `168` picks quedan marcados como `EV > 0` pero region inestable, baja confianza u odds alta
- `noise_positive_ev_but_unstable_share = 0.9492`
- la mejor comparacion offline mejora OOF y holdout, pero falla `pre_holdout`, por tanto queda `rejected_pre_holdout_negative`
- la reformulacion de ranking estable tambien queda `rejected_pre_holdout_negative`

Conclusion: esto no es una promocion. Es evidencia adicional de que el problema actual es ruido en la zona apostada y que la siguiente evidencia decisiva debe venir del ledger forward real, no de otro gate offline parecido.

## 2. Flujo principal

### Ingesta canonica

```powershell
predicciones ingest --seasons 2425 2324 2223 --leagues E0 SP1 D1
```

Genera un dataset canonico en `data\dataset_<ligas>_<temporadas>\` con:

- `matches.csv`
- `market_odds.csv`
- `market_snapshots.csv`
- `feature_rows.csv`
- `manifest.json`

### Backtest completo

```powershell
predicciones backtest
```

O contra un dataset concreto:

```powershell
predicciones backtest --dataset-dir C:\Users\Alberto\Desktop\Predicciones\data\dataset_E0-SP1-D1_2425-2324-2223
```

Cada run deja artefactos en `outputs\runs\backtest_<timestamp>\`:

- `prediction_rows.csv`
- `bet_rows.csv`
- `summary.json`
- `calibration_curves.png`
- `bank_curve.png`
- `feature_importance.csv`
- `feature_importance.png`

### Research Backtest Neto

```powershell
predicciones backtest-net --dataset-dir C:\Users\Alberto\Desktop\Predicciones\data\dataset_E0-SP1-D1_2425-2324-2223
```

Este flujo es el importante para evaluar ROI con supuestos de ejecucion:

- usa `market_snapshots.csv`
- compara `raw` vs `calibrated` y se queda con el ganador OOF
- optimiza la politica sobre OOF train
- congela esa politica y la mide en holdout forward
- genera discovery de nichos y `promotion_report`

Artefactos principales:

- `prediction_rows.csv`
- `candidate_rows.csv`
- `execution_rows.csv`
- `net_bet_rows.csv`
- `niches.csv`
- `promotion_report.json`

### Descubrir Nichos

```powershell
predicciones discover-niches
```

Ordena segmentos por ROI neto, drawdown y estabilidad por folds/recencia.

### Entrenar El Bundle Del Nicho

```powershell
predicciones train-niche --dataset-dir C:\Users\Alberto\Desktop\Predicciones\data\dataset_E0-SP1-D1_2425-2324-2223
```

Guarda el bundle promocionable en `outputs\models\train_niche_<timestamp>\model_bundle.joblib`.

### Shadow Run

```powershell
predicciones shadow-run --fixtures-file C:\ruta\fixtures.csv
```

Puntua fixtures futuros con el `niche model` y deja picks planificados sin marcar beneficio todavia.

### Polymarket Collector

```powershell
predicciones collect-polymarket
```

Descubre grupos `1X2` de futbol en Polymarket y guarda en `data\polymarket_shadow.sqlite`:

- catalogo de mercados
- grupos `home/draw/away`
- top of book
- checkpoints completos de asks/bids
- ultimos trades observados
- resoluciones oficiales

Si quieres escuchar WebSocket y capturar checkpoints periodicos/forzados:

```powershell
predicciones collect-polymarket --stream-seconds 1800
```

### Polymarket Retro Aproximado

```powershell
predicciones backtest-polymarket-retro
```

Este carril usa eventos cerrados de Polymarket para iterar rapido la politica sin esperar jornadas nuevas:

- mantiene congelado el modelo actual
- intenta usar `history_exact` si ya existe checkpoint local capturado
- cae a `history_proxy` con haircut conservador cuando no hay libro historico usable
- no cuenta como prueba final de `20% ROI real`

Artefactos por run:

- `retro_shadow_summary.json`
- `retro_candidate_rows.csv`
- `retro_decision_rows.csv`
- `retro_fill_rows.csv`
- `retro_skip_reasons.csv`
- `market_mapping.csv`
- `mapping_audit.csv`
- `retro_coverage_summary.json`
- `bank_curve.png`

Antes de confiar en este carril, conviene poblar la base historica local:

```powershell
predicciones backfill-polymarket-history
```

Ese backfill:

- recorre eventos cerrados por chunks de fecha y ligas soportadas
- guarda catalogo, grupos `1X2`, resoluciones y `prices-history` si la API los devuelve
- deja un `backfill_summary.json` por run para auditar cobertura real del venue

El retro backtest consume primero esa base local y solo cae a llamadas online para huecos. Ademas, ahora separa de forma explicita:

- cobertura (`retro_coverage_summary.json`)
- modelo puro sobre partidos mapeados
- politica sobre picks seleccionados
- calidad de precio `history_exact` vs `history_proxy`
- estado de muestra: `coverage_limited` o `coverage_ready`

### Afinar Politica Polymarket

```powershell
predicciones tune-polymarket-policy
```

Genera un `policy_bundle.json` a partir del carril retro aproximado. Ese bundle contiene:

- `policy`
- `probability_source`
- `source_mode = retro_approx`
- assumptions del retro pricing usadas para el afinado

El objetivo es corregir `edge` y filtros rapido antes de pasar al `T-45m` real.

El bundle ya no se considera automaticamente promocionable. Ahora incluye:

- `bundle_status = provisional` o `promotable_for_forward`
- `coverage_status`
- minimos exigidos de muestra
- tier minimo de calidad admitido para el tuning

Si la cobertura historica no supera los minimos, el bundle queda marcado como `provisional` y no reemplaza el puntero principal de politica forward.

### Polymarket Shadow Trading

```powershell
predicciones shadow-polymarket
```

Si no pasas `--fixtures-file`, el comando construye fixtures desde los grupos completos de la base SQLite y usa el `niche model` actual para:

- mapear fixture -> group `home/draw/away`
- buscar el checkpoint mas cercano antes de `T-45m`
- exigir libro fresco `<= 5s`
- seleccionar como maximo un `BUY YES`
- simular fills `taker` con ladder real, comision y slippage cushion
- evaluar la escalera fija `10/25/50/100 USDC`

Tambien puedes pasar un CSV de fixtures propio:

```powershell
predicciones shadow-polymarket --fixtures-file C:\ruta\fixtures.csv
```

Y si quieres usar una politica afinada en el carril retro:

```powershell
predicciones shadow-polymarket --policy-bundle C:\ruta\policy_bundle.json
```

El resumen forward conserva la metadata del bundle retro congelado para que quede claro si estas validando una politica `provisional` o una ya `promotable_for_forward`.

### Reporte Polymarket

```powershell
predicciones report-polymarket
```

### Reporte Polymarket Retro

```powershell
predicciones report-polymarket-retro
```

Este reporte muestra primero:

- funnel de cobertura
- estado de cobertura y del bundle
- tamaño de muestra util
- calidad de precio usada

Y solo despues el ROI. Si la muestra no supera los minimos, el run se etiqueta como `insufficient_sample` y el ROI queda marcado como no accionable.

Cada run deja artefactos en `outputs\runs\shadow_polymarket_<timestamp>\`:

- `shadow_summary.json`
- `decision_rows.csv`
- `fill_rows.csv`
- `forward_sample_report.json`
- `forward_sample_ledger.csv`
- `forward_sample_blockers.csv`
- `forward_sample_manifest.json`
- `forward_decision_region_report.json`
- `capacity_curve.csv`
- `market_mapping.csv`
- `skip_reasons.csv`
- `bank_curve.png`

### Arquitectura Nueva: Raw Comun + Carriles Atomicos

```powershell
predicciones discover-market-raw
predicciones capture-market-raw
predicciones capture-market-raw --lane-id football_1x2_global --capture-priority missing_or_stale --freshness-seconds 3600 --max-markets 50
predicciones capture-market-raw --lane-id football_1x2_global --capture-priority missing_or_stale --freshness-seconds 3600 --max-events 10
predicciones train-market-lane-model --lane-id football_1x2_global --leagues E0 SP1 D1 I1 F1 N1 P1 MEX USA --seasons 2526 2425 2324
predicciones train-market-lane-model --lane-id football_goals_core --leagues E0 SP1 D1 I1 F1 N1 P1 MEX USA --seasons 2526 2425 2324
predicciones build-market-lane-predictions --lane-id football_1x2_global
predicciones build-market-lane-predictions --lane-id football_goals_core
predicciones create-market-lane-policy --lane-id football_goals_core
predicciones run-market-lane --lane-id football_1x2_global
predicciones run-market-lane --lane-id football_goals_core
predicciones report-market-lane --lane-id football_1x2_global
predicciones report-sport-merge --sport football
.\scripts\run_multi_market_lane_cycle.ps1 -MaxEvents 5
.\scripts\run_multi_market_lane_cycle.ps1 -MaxEvents 25 -ModelPath outputs\models\<global_football_model>\model_bundle.joblib
```

La arquitectura nueva separa dos capas:

- `raw_market_capture`: captura comun de eventos, mercados, books y settlements en `data\polymarket_multi_market.sqlite`
- `market_lanes`: carriles atomicos por tipo de decision, cada uno con su benchmark, ledger, reportes, policy y sample gates

El benchmark canonico actual `football_1x2_canonical = E0 + SP1 + D1` sigue intacto en `data\polymarket_shadow.sqlite` y queda como referencia comparativa. El nuevo carril `football_1x2_global` incluye explicitamente `E0`, `SP1` y `D1` como `legacy_slice`, ademas de la expansion global. La meta a largo plazo es retirar el legacy solo cuando exista paridad estructural y validacion forward suficiente.

Objetivo:

- acelerar muestra forward sin mezclar metricas incompatibles
- mantener ledgers, modelos, policies y estados por `lane_id`
- reportar ROI solo por carril cuando haya muestra suficiente
- prohibir un `global_roi` accionable en v1

Carriles activos v1:

- `football_1x2_global`: futbol global `home/draw/away`, incluye `E0 + SP1 + D1` como slice legacy y reconoce `I1 + F1 + N1 + P1 + MEX + USA` como expansion global si el bundle indicado trae historico suficiente
- `football_goals_core`: futbol `over/under goals` y `BTTS`, con subtipos reportados por separado
- `tennis_match_winner`: ganador de partido, inicialmente `capture_only`
- `basketball_moneyline`: ganador de partido, inicialmente `capture_only`
- `baseball_moneyline`: ganador de partido, inicialmente `capture_only`
- `hockey_moneyline`: ganador de partido, inicialmente `capture_only`
- `cricket_match_winner`: ganador de partido si hay cobertura suficiente, inicialmente `capture_only`

Carriles diferidos explicitamente:

- tarjetas
- corners
- player props
- handicaps/spreads
- totales de deportes no futbol

Aliases temporales compatibles:

```powershell
predicciones discover-multi-market
predicciones capture-multi-market
predicciones report-multi-market
```

Artefactos raw:

- `market_family_manifest.json`
- `multi_market_discovery_report.json`
- `market_family_coverage_report.json`
- `raw_inventory_quality_report.json`
- `raw_capture_plan.csv`
- `raw_capture_plan.json`
- `multi_market_catalog.csv`
- tablas raw comunes `mm_raw_events`, `mm_raw_markets`, `mm_raw_orderbooks`, `mm_raw_settlements`
- enlaces `mm_lane_market_links` para proyectar un raw market a cero, uno o varios carriles

Calidad de inventario forward:

- solo cuentan como inventario forward los mercados activos con `game_start_time >= run_time`
- los mercados activos historicos quedan visibles como `active_historical_markets`, pero no desbloquean `coverage_ready`
- si un carril solo tiene mercados historicos o sin fecha, queda `inventory_not_forward_ready`
- `forward_ledger.csv` solo proyecta mercados futuros activos; los slugs antiguos no entran en muestra forward
- para smoke tests seguros, `capture-market-raw --lane-id <lane_id> --max-markets <n>` captura solo un carril y limita llamadas al CLOB
- para no partir una decision 1X2 por la mitad, `capture-market-raw --lane-id <lane_id> --max-events <n>` captura todos los mercados futuros de los primeros `n` eventos del carril
- la captura priorizada usa por defecto `--capture-priority missing_or_stale` y `--freshness-seconds 3600`: primero mercados futuros activos sin book, luego books stale, y solo books frescos si se pide `--capture-priority all`
- `--include-policy-ready-only` limita la captura a carriles con modelo y policy listos; sirve para priorizar carriles que ya pueden emitir shadow picks sin abrir familias capture-only
- `raw_capture_plan.csv/json` deja auditable cada intento: `previous_book_status`, edad del book, tokens CLOB, prioridad y si fue seleccionado para captura
- `train-market-lane-model` crea un bundle aislado bajo `outputs\lanes\<lane_id>\models\...` y escribe `outputs\lanes\<lane_id>\latest_model.txt`; no mueve `outputs\latest_model.txt`, no mueve `data\latest_dataset.txt` y no cambia ninguna policy
- `create-market-lane-policy --lane-id football_goals_core` crea la policy nativa `football_goals_core_bootstrap_v1` y su benchmark separado; no hereda thresholds de `1X2`, no mezcla ROI y queda marcada como `forward_sample_collection`
- `run_multi_market_lane_cycle.ps1` refresca primero la plantilla del carril con `run-market-lane`, luego genera `model_predictions.csv` y vuelve a ejecutar el carril; esto evita que mercados nuevos de discovery queden como `prediction_row_missing_for_selection`

Artefactos por carril:

- `outputs\lanes\<lane_id>\lane_manifest.json`
- `outputs\lanes\<lane_id>\lane_readiness_report.json`
- `outputs\lanes\<lane_id>\lane_decision_inventory_report.json`
- `outputs\lanes\<lane_id>\lane_candidate_report.json`
- `outputs\lanes\<lane_id>\lane_candidate_rows.csv`
- `outputs\lanes\<lane_id>\lane_blocker_audit.csv`
- `outputs\lanes\<lane_id>\lane_blocker_audit.json`
- `outputs\lanes\<lane_id>\raw_capture_health_report.json`
- `outputs\lanes\<lane_id>\lane_policy_benchmark_report.json` si el carril tiene policy nativa
- `outputs\lanes\<lane_id>\model_prediction_template.csv`
- `outputs\lanes\<lane_id>\model_predictions.csv`
- `outputs\lanes\<lane_id>\model_prediction_report.json`
- `outputs\lanes\<lane_id>\policy_bundle.json` si el carril tiene una policy congelada propia o un bridge auditado
- `outputs\lanes\<lane_id>\forward_ledger.csv`
- `outputs\lanes\<lane_id>\sample_report.json`
- `outputs\lanes\football_1x2_global\legacy_parity_report.json`

Readiness por carril:

- `football_1x2_global` puede activar un bridge auditado desde `outputs\latest_polymarket_policy.txt`; se copia a `outputs\lanes\football_1x2_global\policy_bundle.json` con `policy_reoptimized=false`, `thresholds_changed=false` y `policy_transfer_mode=legacy_frozen_policy_bridge`
- ese bridge existe solo para migrar/paritar el carril `1X2` global con el legacy; no permite promocion automatica ni ROI agregado entre carriles
- `football_1x2_global` solo cuenta oportunidades reales cuando un evento tiene mercado `1X2` de tres vias o binarios completos `home/away/draw`; mercados outright tipo ganador de liga quedan fuera del carril
- `run-market-lane` construye candidatos lane-level con book, `top_ask`, `quoted_odds`, `model_prob`, `edge` y `ev`; si falta `outputs\lanes\<lane_id>\model_predictions.csv`, el carril queda bloqueado como `model_probability_missing` y no selecciona picks aunque tenga policy congelada
- `model_predictions.csv` es un contrato por carril, no una policy: columnas minimas `event_slug`, `selection`, `model_prob`; opcionales `market_id`, `probability_source`, `model_variant`
- `model_prediction_template.csv` enumera las filas exactas que debe cubrir el adaptador de modelo del carril; en `football_1x2_global` es por `event_slug + selection`, y en mercados como goles puede ser por `event_slug + market_id + selection`
- `build-market-lane-predictions` rellena ese contrato de forma auditada; primero usa `outputs\lanes\<lane_id>\latest_model.txt` si existe, luego `outputs\latest_niche_model.txt` y finalmente `outputs\latest_model.txt`; tambien acepta `--model-path` para un bundle concreto, y el script `run_multi_market_lane_cycle.ps1` puede pasar ese bundle con `-ModelPath`
- `football_1x2_global` empieza como bridge `legacy-safe`: solo emite probabilidades cuando puede mapear fixture/equipo/liga y tiene soporte historico suficiente; eventos globales sin mapping o sin historico en el bundle quedan bloqueados como `unsupported_league`, `team_mapping_failed`, `history_missing` o `feature_build_failed`
- `football_goals_core` reutiliza las lambdas Poisson del modelo de futbol para calcular `over/under` en lineas parseables y `btts_yes/btts_no`; ya tiene policy nativa separada `football_goals_core_bootstrap_v1`, pero su ROI sigue `not_actionable_until_forward_settled`
- `football_goals_core_bootstrap_v1` es la primera policy propia de goles: `edge_threshold=0.02`, `ev_threshold=0.0`, odds `1.2-6.0`, outcomes `over/under/btts_yes/btts_no`, subtipos `total_goals/btts`, y una apuesta maxima por evento; sirve para acumular muestra forward, no para promocionar ROI
- `model_prediction_report.json` deja trazabilidad de `template_rows`, `predicted_rows`, `prediction_coverage_rate`, `blocker_counts`, `supported_legacy_rows`, `unsupported_global_rows`, `policy_reoptimized=false` y `global_roi_actionable=false`
- `lane_blocker_audit.csv/json` explica cada candidato con liga inferida, equipos parseados, mapping, soporte historico, book, prediccion, policy y accion siguiente; `model_probability_missing` debe quedar siempre convertido en una causa accionable
- `raw_capture_health_report.json` separa `fresh_book`, `stale_book`, `capture_not_attempted` y `no_clob_token`, para saber si el bloqueo real es captura, frescura o inventario
- `forward_ledger.csv` guarda decisiones seleccionadas como `valid_forward_sample` cuando hay picks; si el carril aun no selecciona, conserva inventario `candidate_ready` para diagnostico
- `sample_report.json` expone `sample_status`, `sample_blockers`, `valid_forward_decisions`, `settled_decisions` y `forward_ledger_status_counts`; una muestra no es `sample_ready` hasta cumplir decisiones validas, settled y frescura
- la captura se considera operativamente sana solo cuando `fresh_book_rate >= 0.80`; `selected_candidates > 0` sigue siendo muestra shadow, no promocion
- `football_goals_core_policy_research_report.json` y `football_goals_core_policy_research_rows.csv` preparan datos de research de goles; antes de crear la policy nativa marcan `can_emit_picks=false`, y despues reflejan `policy_ready=true` sin convertir el ROI en accionable
- si existen probabilidades y policy congelada, el carril aplica los thresholds tal cual y fuerza una apuesta maxima por evento; no optimiza thresholds, scopes ni outcomes en esta fase
- `football_goals_core` puede declarar el adaptador Poisson de lambdas como disponible, pero reporta `total_goals` y `btts` por separado y queda bloqueado si falta un subtipo
- tenis, basket, baseball, hockey y cricket siguen `capture_only` hasta tener modelo gratuito fiable y contrato de settlement validado
- `football_1x2_canonical` es `reference_only`; nunca emite picks desde la arquitectura nueva

Ultima medicion de carriles de futbol tras incluir `MEX` y `USA` desde Football-Data (`new/MEX.csv` y `new/USA.csv`) y reentrenar modelos aislados con `2526 + 2425 + 2324`:

- `football_1x2_global`: `prediction_coverage=432/660`, `selected_candidates=67`, `valid_forward_decisions=67`, `fresh_book_rate=1.0`, `settled_decisions=0`, `sample_status=collecting_forward_sample`
- `football_goals_core`: `prediction_coverage=1340/1600`, `selected_candidates=87`, `valid_forward_decisions=87`, `fresh_book_rate=1.0`, `settled_decisions=0`, `sample_status=collecting_forward_sample`
- conclusion operativa: la captura y la policy ya no son el bloqueo principal; falta llegar a `100` decisiones validas por carril y despues a `40` settled por carril antes de analizar ROI

### Pipeline canónica de datos para simulador

El carril `football_sim_data` existe para construir una base historica amplia, reproducible y leakage-free antes de entrenar un motor de simulacion. No emite picks, no calcula ROI y no toca `football_1x2_canonical`, `data\polymarket_shadow.sqlite` ni `data\polymarket_multi_market.sqlite`.

Base aislada:

- `data\football_sim_data.sqlite`

Comandos:

```powershell
predicciones discover-sim-data-sources
predicciones collect-sim-data --source football_data --leagues E0 SP1 D1 I1 F1 N1 P1 MEX USA --seasons 2526 2425 2324
predicciones collect-sim-data --source football_data --profile max_free_v1 --seasons-back 15 --probe-missing
predicciones collect-sim-data --source statsbomb_open_data --profile open_data_all
predicciones collect-sim-data --source clubelo --teams mapped
predicciones normalize-sim-data
predicciones build-sim-features
predicciones export-sim-training-dataset --exclude-market-reference
predicciones train-football-sim --dataset-path outputs\sim_data\simulation_training_dataset.csv
predicciones report-sim-data
```

Artefactos:

- `outputs\sim_data\source_coverage_report.json`
- `outputs\sim_data\source_license_manifest.json`
- `outputs\sim_data\entity_resolution_report.json`
- `outputs\sim_data\leakage_audit_report.json`
- `outputs\sim_data\simulation_feature_manifest.json`
- `outputs\sim_data\simulation_dataset_summary.json`
- `outputs\sim_data\football_data_coverage_matrix.csv`
- `outputs\sim_data\statsbomb_coverage_report.json`
- `outputs\sim_data\clubelo_mapping_report.json`
- `outputs\sim_data\source_column_coverage_report.json`
- `outputs\sim_data\simulation_training_dataset.csv`
- `outputs\sim_data\simulation_training_manifest.json`
- `outputs\sim_data\simulation_training_splits.json`
- `outputs\sim_models\football_sim_poisson_v1_<timestamp>\model_bundle.joblib`
- `outputs\sim_models\football_sim_poisson_v1_<timestamp>\simulation_model_report.json`
- `outputs\latest_football_sim_model.txt`

Fuentes registradas:

- `football_data`: activa por defecto; resultados, estadisticas basicas y cuotas historicas de referencia.
- `statsbomb_open_data`: activa bajo demanda para eventos/alineaciones, cobertura limitada y alta calidad.
- `openfootball`: registrada para calendario/mapping, activacion posterior.
- `clubelo`: activa bajo demanda para ratings historicos de equipos mapeados.
- `worldfootballr`, `soccerdata`, `fbref_understat_quarantine`: cuarentena; no se activan sin auditoria de estabilidad, terminos y rate limits.

Capas:

- `bronze_raw`: payloads originales con `content_hash`, `source_id`, `source_url`, `fetched_at`, `schema_version`, estado de fetch y notas de licencia.
- `silver_entities`: equipos, aliases canonicos y partidos normalizados con IDs estables.
- `gold_sim_features`: features prepartido para simulacion con `as_of_time`, `match_start_time`, `known_before_match` y familias de feature.
- `simulation_training_dataset.csv`: dataset entrenable con targets `home_goals`, `away_goals`, `outcome`, `total_goals`, `btts` y splits temporales `train/dev/locked_holdout`.

Reglas de disciplina:

- las features de `gold_sim_features` solo usan partidos previos al partido evaluado
- alineaciones/eventos reales no entran como feature predictiva salvo que esten marcados como conocidos prepartido
- cuotas historicas quedan como `market_reference`, no como senal principal del primer simulador
- `export-sim-training-dataset` excluye `market_reference` por defecto para no entrenar el primer simulador copiando mercado
- `train-football-sim` entrena `football_sim_poisson_v1` como simulador predictivo puro: genera lambdas, probabilidades `1X2`, `over/under 2.5` y `BTTS`, pero no emite picks ni calcula ROI
- el simulador decide `raw` vs `calibrated` solo con `train/dev`; `locked_holdout` queda como evaluacion final bloqueada
- si una familia no tiene cobertura suficiente, queda bloqueada en `simulation_feature_manifest.json`
- el contrato resultante prepara un futuro `football_sim_poisson_v1` o simulador Monte Carlo sin redisenar la ingesta

Artefactos de merge por deporte:

- `football_sport_context_features.csv`
- `football_sport_merge_report.json`

Reglas de disciplina:

- ningun carril sin modelo, policy, coverage y sample contract listos puede emitir picks
- ningun carril hereda ROI ni promocion de otro
- ningun carril comparte thresholds con otro salvo el bridge explicito `football_1x2_global <- football_1x2_canonical`, que queda congelado y auditado para migracion de la misma decision `1X2`
- `football_1x2_canonical` sigue intacto y conserva sus artefactos actuales
- el merge por deporte solo crea features; no calcula ROI ni promociona estrategias
- no hay portfolio hasta que al menos dos carriles tengan `sample_ready`
- si no se llega a unas `100` decisiones validas por dia, el reporte debe decir si el limite fue inventario, mapping, captura, modelo o policy

### Promotion Report

```powershell
predicciones promote-report
```

Resume si el nicho ha pasado `Stage 1` y `Stage 2` y por que sigue o no bloqueado para automatizacion.

### Entrenamiento final

```powershell
predicciones train-final
```

Guarda el bundle final en `outputs\models\train_final_<timestamp>\model_bundle.joblib`.

### Prediccion de fixtures nuevos

Prepara un CSV con estas columnas:

- `Date`
- `league_code`
- `league_name` opcional
- `season` opcional
- `HomeTeam`
- `AwayTeam`
- `B365H`
- `B365D`
- `B365A`

Ejemplo:

```powershell
predicciones predict --fixtures-file C:\ruta\fixtures.csv
```

El comando usa el ultimo modelo entrenado si no indicas `--model-path`.

### Reporte rapido de un run

```powershell
predicciones report
```

## 3. Comandos opcionales

### Buscar mercados en Polymarket

```powershell
predicciones markets --query "Arsenal Manchester City"
```

### Resumen narrativo de divergencias

```powershell
predicciones claude-report `
  --match "Arsenal vs Manchester City" `
  --bookmaker 0.42 0.27 0.31 `
  --model 0.39 0.25 0.36 `
  --polymarket 0.37 0.00 0.63
```

Si existe `ANTHROPIC_API_KEY`, usa Claude. Si no existe, cae a un reporter heuristico local.

## 4. Principios del backtest

- splits rolling-origin por fecha
- nunca se entrena con partidos del mismo dia que el test
- calibracion temporal fuera de muestra
- politica de apuesta optimizada solo en datos previos al bloque de test
- el modelo principal no usa cuotas como features
- el flujo `backtest-net` distingue entre proxies de cierre y snapshots ejecutables reales
- un `20% ROI real` solo cuenta si el holdout y luego shadow/live usan cuotas capturadas en tiempo de decision
- el flujo `backtest-polymarket-retro` sirve para descartar o ajustar politica rapido, pero no sustituye al `shadow-polymarket`
- el flujo `shadow-polymarket` no usa midpoint ni ultimo trade para simular fills: consume el lado `ask` real del orderbook

## 5. Benchmark heredado

El benchmark previo del sistema reconstruido desde el post se conserva en:

```text
C:\Users\Alberto\Desktop\Predicciones\benchmarks\legacy_v1
```

Los nuevos runs comparan sus metricas con ese punto de referencia si el benchmark existe.
