# Artifact Retention

Local artifacts are not part of the clean source base.

## Purged By Default

- `outputs/`
- `benchmarks/`
- `data/*.sqlite*`
- `data/*.db*`
- `data/dataset_*`
- `data/latest_*`
- `__pycache__/`
- `.pytest_cache/`
- `src/predicciones_football.egg-info/`

## Evidence Rule

Before deleting a benchmark or experiment artifact that should remain
auditable, write a manifest under `docs/archive/` or `docs/experiments/` with
the identity, headline metrics, date, decision, and reason it is inactive.

## Current Archive

`football_1x2_canonical` is documented in
`docs/archive/football_1x2_canonical.md` and remains reference-only.
