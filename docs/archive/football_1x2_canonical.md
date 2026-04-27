# football_1x2_canonical Archive

Archived on: 2026-04-27

## Purpose

`football_1x2_canonical` is retained only as a historical reference benchmark for
the original football 1X2 pipeline. It is not an operating lane, not a promotion
candidate, and not a source of policy tuning during the forward validation phase.

## Archived Metrics

Source artifact before purge: `benchmarks/legacy_v1/metrics.json`.

- Accuracy: `0.5343396226415095`
- Log loss: `0.974330597607511`
- Evaluation support: `2650`
- Away precision/recall/F1: `0.5299479166666666` / `0.5068493150684932` / `0.5181413112667091`
- Draw precision/recall/F1: `0.28865979381443296` / `0.04216867469879518` / `0.0735873850197109`
- Home precision/recall/F1: `0.5495798319327732` / `0.8292476754015216` / `0.6610512129380054`
- Macro precision/recall/F1: `0.45606251413795756` / `0.4594218883896033` / `0.41759330307480846`
- Weighted precision/recall/F1: `0.47825329107644804` / `0.5343396226415095` / `0.47054833264374185`

## Archive Decision

The physical benchmark artifacts under `benchmarks/legacy_v1/` were local,
reproducible research artifacts and are intentionally purged from the working
tree. This manifest keeps the benchmark identity and headline metrics available
without letting the old artifact bundle become part of the active workflow.

## Operating Rule

Do not import this benchmark from active modules. Do not tune thresholds,
policies, gates, or lane decisions against it. Active validation remains lane
isolated and forward-sample driven.
