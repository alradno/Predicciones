# Rejected Experiment: Penaltyblog Challenger

Status: rejected and removed from the active pipeline.

This experiment tested `penaltyblog` as an external `1X2` challenger on the canonical
football benchmark.

## What Was Tested

- variants `pb_dixon_coles` and `pb_bivariate_poisson`
- same universe: `E0 + SP1 + D1`
- same flow: `OOF -> pre_holdout -> locked_holdout`
- same frozen policy
- no odds or Polymarket signals used to train the external model

## Baseline

Baseline run:

- `outputs\runs\backtest_polymarket_retro_20260418_181928`

Frozen comparison:

- champion baseline: `v5/raw`
- `oof_aggregate_roi = -0.1784`
- `pre_holdout_aggregate_roi = -0.0186`
- `locked_holdout_frozen_policy_roi = -0.0379`

## Experimental Run

- `outputs\runs\backtest_polymarket_retro_20260418_190930`

## Result

- `pb_dixon_coles` improved `OOF ROI` to `+0.0649`, but worsened `log_loss`,
  `pre_holdout` (`-0.3601`), and `holdout` (`-0.2925`).
- `pb_bivariate_poisson` was similar: `OOF ROI +0.0501`, `pre_holdout -0.3716`,
  `holdout -0.2925`.
- The champion stayed `v5/raw`, so there was no honest benchmark improvement.

## Decision

- preserve the artifacts as evidence
- remove the integration and dependency from active code
- do not open the `MAPIE` phase, because no `pb_*` variant beat `v5/raw` across the
  required layers
