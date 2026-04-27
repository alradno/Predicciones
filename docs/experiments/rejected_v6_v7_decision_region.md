# Rejected Experiments: v6/v7 And Decision Region

Status: implemented, audited, and rejected as a promotion path.

This file preserves the historical decision-region and v6/v7 experiment notes that used
to live in the README. The common conclusion is that apparent ROI improvements were not
stable across `OOF`, `pre_holdout`, and `locked_holdout`, or they relied on samples too
small to treat as operating evidence.

## v6/v7 Confidence, Backoff, And Decision-Region Scorers

What was tested:

- variants `v6`, `v6b`, `v7`, `v7b`
- long-confidence layer decoupled from short form
- probability backoff toward `v1/raw`
- second-layer decision-region scorers:
  - `heuristic`
  - `dr_logit`
  - `dr_hgb`
- secondary branch training the scorer only on `argmax_all_bets`
- same canonical benchmark: `E0 + SP1 + D1`, `1X2`,
  `OOF -> pre_holdout -> locked_holdout`, frozen policy, and untouched
  `locked_holdout`

Baseline:

- `outputs\runs\backtest_polymarket_retro_20260418_192807`
- champion baseline: `v5/raw`
- `oof_aggregate_roi = -0.1784`
- `pre_holdout_aggregate_roi = -0.0186`
- `locked_holdout_frozen_policy_roi = -0.0379`

Experimental run:

- `outputs\runs\backtest_polymarket_retro_20260418_201858`

Result:

- run champion moved to `v6/raw`, but only with
  `locked_holdout_frozen_policy_roi = +0.1108`, `oof_aggregate_roi = -0.2130`,
  and `pre_holdout_aggregate_roi = -0.0379`
- `v6b` and `v7b` reached `locked_holdout_frozen_policy_roi = +0.4667`, above the
  nominal `45%` target, but were rejected because `oof_aggregate_roi = -0.4388`
  and `pre_holdout_aggregate_roi = -0.0425`
- `v3r` also showed high holdout (`+0.4623`), but with clearly negative `OOF` and
  negative `pre_holdout`
- `dr_logit` and `dr_hgb` did not make `OOF` and `pre_holdout` positive under the
  frozen policy

Decision:

- preserve the full run evidence
- do not count the `45%` target as reached because the improvement appeared only in
  `locked_holdout`
- treat the bottleneck as stability/sample, not missing tuning
- do not reopen this branch as if it had not been tested

## Regional OOF Penalty By Selection And Odds Band

Status: implemented and rejected as an adequate correction.

What was tested:

- regional adjustment trained only with OOF by `selection|odds_band`
- crossfit application by `retro_fold_id` to avoid future leakage
- replacement of `policy_prob`, `policy_edge`, and `policy_ev` with regional adjusted
  versions when available
- evaluation on `v6`, `v6b`, `v7`, `v7b`, and the secondary `argmax_all_bets` branch

Run:

- `outputs\runs\backtest_polymarket_retro_20260418_211612`

Result:

- the benchmark did not materially change versus `20260418_201858`
- champion stayed `v6/raw` with `locked_holdout_frozen_policy_roi = +0.1108`,
  `oof_aggregate_roi = -0.2130`, and `pre_holdout_aggregate_roi = -0.0379`
- branches with `regional_adjustment = oof_region_shrink` were nearly identical to
  their unadjusted equivalents

Decision:

- preserve the run evidence
- reject this branch as a solution to the stability/sample bottleneck
- do not keep adding cosmetic scoring; fix sample and eligible-region definition first

## Long Free History And Obsolete Direct match_id Bug

Status: methodological bug fixed; long history remains valid only after the fix.

What was tested:

- same `E0 + SP1 + D1` universe
- expanded free training history from `4.088` to `9.418` canonical matches
- no changes to leagues, markets, policy, thresholds, scopes, or `T-45m`
- preserved the Polymarket retro benchmark with untouched `locked_holdout`

Bug found:

- the first long-history run (`outputs\runs\backtest_polymarket_retro_20260419_165417`)
  showed `mapped_matches = 1005` but only `734` complete groups, which was not clean
  evidence
- cause: some numeric football-data `match_id` values were not stable when changing
  the season range and linked to Polymarket groups from other seasons
- observed pattern: a 2019 match linked directly by `match_id` to a 2024 market with
  different teams

Correction:

- direct `match_id` mapping now requires plausible teams and a plausible time window
- rescheduled matches are still accepted when teams match and the time displacement is
  plausible
- a unit test rejects obsolete direct IDs with wrong teams

Clean run:

- `outputs\runs\backtest_polymarket_retro_20260419_172523`

Clean result:

- `mapped_matches = 733`
- `unique_groups = 733`
- `duplicated_group_rows = 0`
- champion: `v3r/raw`
- `oof_aggregate_roi = +0.0145`, but `oof_positive_fold_ratio = 0.25`
- `pre_holdout_aggregate_roi = -0.3314`
- `locked_holdout_frozen_policy_roi = -0.0049`
- status: `overfit_rejected`

Decision:

- long history made some variants less negative in OOF and holdout
- it did not solve pre-holdout stability
- no promotion
- the contaminated run is discarded as evidence, and the mapping fix remains a guardrail

## Regional Adjustment On The Actual Bet Region

Status: corrected, audited, and rejected as insufficient.

What was corrected:

- `oof_region_shrink` now learns on the region that would pass the frozen base policy,
  not the whole eligible universe
- the regional adjustment applies after `policy_prob`, `policy_edge`, and `policy_ev`
  are calculated
- missing `won` in the real flow was fixed by inferring victory from
  `selection == actual_outcome`
- `ablation_report.json` and `decision_scorer_ablation.json` save real regional
  training rows and scope

Runs:

- `outputs\runs\backtest_polymarket_retro_20260419_175242`
- `outputs\runs\backtest_polymarket_retro_20260419_182604`

Result:

- hard shrink trained for real (`regional_adjustment_training_scope = base_policy_region`,
  `training_rows ~= 102-106`) but destroyed sample
- `v6 + oof_region_shrink`: `27` OOF bets, `10` pre-holdout bets, `10` holdout bets,
  `oof_aggregate_roi = -0.4388`, `pre_holdout_aggregate_roi = -0.1914`,
  `holdout = -0.0171`
- softer fixed variant `oof_region_soft_shrink` preserved more sample
- `v6 + oof_region_soft_shrink`: `oof_aggregate_roi = -0.0919`,
  `pre_holdout_aggregate_roi = -0.1514`, `holdout = +0.0248`, `holdout_bets = 40`
- `v6b/v7b + oof_region_soft_shrink` improved pre-holdout (`+0.3394`) but failed
  OOF (`-0.2932`) and holdout (`-0.1433`)

Decision:

- hard filters kill sample
- soft filters preserve sample but do not make OOF and pre-holdout positive together
- the bet region remains insufficiently stable under the frozen policy
- next honest work is more reliable forward/historical sample or a different decision
  region, not more similar gates

## Decision-Region Reformulation And Temporal Crossfit

Status: implemented, audited, and rejected as insufficient.

What was corrected:

- added scorers that conservatively recalibrate the probability seen by the frozen policy:
  - `dr_logit_prob`
  - `dr_hgb_prob`
  - `dr_reliability_prob`
- these branches write `decision_adjusted_prob`, `decision_adjusted_edge`, and
  `decision_adjusted_ev`
- the frozen policy only uses those values as hardening; they cannot raise base
  probability or create new bets
- `dr_reliability_prob` uses shrinked empirical rates across broad selection, odds, and
  probability bands, without new libraries
- the second model drops entirely empty numeric columns before training
- decision-region and stability-gate crossfit now trains each OOF fold only on previous
  folds (`past_folds_only`), not future data

Runs:

- `outputs\runs\backtest_polymarket_retro_20260419_191454`
- `outputs\runs\backtest_polymarket_retro_20260419_194530`
- `outputs\runs\backtest_polymarket_retro_20260419_201606`

Most honest final result (`20260419_201606`):

- coverage: `734` complete groups, `733` mapped matches, `2109` candidates, `34` final picks
- champion: `v3r/raw`, `decision_scorer = heuristic`, no gate or regional adjustment
- `locked_holdout_frozen_policy_roi = -0.0049`
- `oof_aggregate_roi = +0.0145`
- `oof_positive_fold_ratio = 0.25`
- `pre_holdout_aggregate_roi = -0.3316`
- status: `overfit_rejected`, `coverage_limited`

Important ablation readings:

- `v3r + dr_logit_prob` improved OOF (`+0.1128`) and holdout (`+0.1594`), but failed
  pre-holdout (`-0.5527`) and left only `18` holdout bets
- `v4 + dr_reliability_prob` reached positive OOF (`+0.1602`) and high holdout
  (`+0.4080`), but only had `5` holdout bets and negative pre-holdout (`-0.4607`)
- `argmax_all_bets + dr_logit_prob` branches that looked positive before temporal
  crossfit were invalidated once OOF used only past folds
- after temporal crossfit, no combination had both `OOF > 0` and `pre_holdout > 0`

Decision:

- the deep decision-region reformulation did not transfer stably
- high ROIs appeared only with samples too small to trust
- no branch promoted
- the next decisive evidence must come from real forward ledgers, not another similar
  offline scorer or gate

## Offline Decision-Region Audit During Forward Capture

Status: diagnostic-only evidence, not promotion.

Historical commands, removed from the clean active CLI:

```powershell
.\scripts\report_forward_capture_status.ps1
.\scripts\report_forward_capture_status.ps1 -Json
.\scripts\run_offline_decision_region_review.ps1
```

Guarantees:

- `report_forward_capture_status.ps1` opened `data\polymarket_shadow.sqlite` read-only
  and did not run active shadow execution
- `run_offline_decision_region_review.ps1` read the latest
  `backtest_polymarket_retro_*` and creates a new `offline_decision_region_review_*`
  run
- no thresholds, scopes, outcomes, policy bundle, or `T-45m` are changed
- `locked_holdout` is not used to train filters or choose hypotheses
- any improvement that fails `OOF` or `pre_holdout` stays rejected even if holdout is high

Initial run:

- `outputs\runs\offline_decision_region_review_20260420_000344`

Initial result:

- `177` picks passed the frozen policy in the analyzed retro run
- `168` picks were flagged as `EV > 0` but unstable region, low confidence, or high odds
- `noise_positive_ev_but_unstable_share = 0.9492`
- the best offline comparison improved OOF and holdout but failed `pre_holdout`, so it
  was `rejected_pre_holdout_negative`
- the stable-ranking reformulation was also `rejected_pre_holdout_negative`

Conclusion: this is evidence that the current problem is noise in the bet region. It is
not a promotion path and does not override the need for settled forward sample.
