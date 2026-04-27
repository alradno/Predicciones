# Rejected Experiments: Stability Gate

Status: implemented, audited, and rejected as the current promotion path.

The stability-gate work was useful diagnostically, but it did not produce a robust
promotion candidate. The hard version blocked too much sample. The corrected soft-parent
version improved the mechanics but still failed the required evidence layers.

## Hierarchical stability_gate Before Policy

Status: implemented, audited, and rejected.

What was tested:

- gate applied after `candidate_score_columns` and before `select_candidate_rows`
- training only on OOF rows that already passed the frozen base policy
- crossfit application by `retro_fold_id` in OOF
- forward application to `pre_holdout` and `locked_holdout` using only available OOF
- hierarchical regions:
  - `league_code | selection | odds_band`
  - `selection | odds_band`
  - `selection`
  - `odds_band`
  - `global`
- conservative `gated_prob`, `gated_edge`, and `gated_ev`
- new branches `v6/v6b/v7/v7b + hierarchical_oof_gate`

Baseline:

- `outputs\runs\backtest_polymarket_retro_20260418_211612`
- champion baseline: `v6/raw`
- `oof_aggregate_roi = -0.2130`
- `pre_holdout_aggregate_roi = -0.0379`
- `locked_holdout_frozen_policy_roi = +0.1108`
- `locked_holdout_bets = 37`

Experimental run:

- `outputs\runs\backtest_polymarket_retro_20260418_233516`

Result:

- run champion was `v6/raw + hierarchical_oof_gate`
- gate blocked `76.0%` of candidates and `75.8%` of candidates that would have passed
  the base policy
- `pre_holdout` improved from `-3.9%` to `+82.1%`, but with only `6` bets
- `OOF` worsened slightly from `-21.3%` to `-21.9%`, with only `11` bets
- `locked_holdout` worsened from `+11.1%` to `-40.6%`, dropping from `37` to `16` bets
- `v6b/v7b + hierarchical_oof_gate` blocked `100%` of picks and added no operational
  evidence

Decision:

- preserve the gate artifacts as stability diagnostics
- do not promote the gate or treat it as a benchmark improvement
- the hard global/parental levels overblocked with small OOF evidence
- any future version would need soft parent adjustments and hard blocks only for
  specific regions with enough support

## Soft-Parent Stability Gate

Status: implemented and rejected as a promotable improvement, but accepted as a
methodological correction of the hard gate.

What changed:

- preserved `hierarchical_oof_gate` as historical evidence of the rejected hard filter
- added `hierarchical_soft_parent_gate`
- `global`, `selection`, and `odds_band` can no longer hard-block
- parent levels only apply a soft probability adjustment
- hard blocks are reserved for specific regions with enough support:
  - `league_code | selection | odds_band`
  - `selection | odds_band`

Run:

- `outputs\runs\backtest_polymarket_retro_20260419_020641`

Result:

- final champion returned to ungated `v6/raw`
- `locked_holdout_frozen_policy_roi = +0.1108`
- `oof_aggregate_roi = -0.2130`
- `pre_holdout_aggregate_roi = -0.0379`
- `v6 + hierarchical_soft_parent_gate` improved `OOF` to `-0.1624` and reached
  `pre_holdout_aggregate_roi = +0.0949`
- it still failed gates because `OOF` stayed negative and
  `pre_holdout_positive_window_ratio = 1/3`
- holdout fell to `-0.2911`
- blocking dropped substantially versus the hard gate:
  `blocked_candidate_share = 20.6%` and `blocked_base_policy_pick_share = 3.4%`

Decision:

- the correction fixed mechanical overblocking
- it did not create a robust promotion candidate
- the baseline without the gate remained the disciplined champion
- the key signal is that softer gates help OOF/pre-holdout but not enough; the bottleneck
  remains sample and stability in the bet region
