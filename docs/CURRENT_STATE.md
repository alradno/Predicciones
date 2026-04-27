# Current State

This is the canonical operating state for the repository.

## Summary

- research freeze is active
- policies are frozen
- ROI is not actionable
- active ROI validation lanes are:
  - `football_1x2_global`
  - `football_goals_core`
- other sports are `capture_only` or paused
- next objective is to accumulate settled forward sample
- ROI 45 is tracked as an honest forward hypothesis, not a promise
- paper/live automation exists as a gated path; live is disabled by default

## Lane Status

| Lane | Mode | Min valid forward decisions | Min settled decisions | Min fresh book rate | Allowed action |
|---|---:|---:|---:|---:|---|
| football_1x2_global | roi_active | 100 | 40 | 0.80 | capture/report/shadow only |
| football_goals_core | roi_active | 100 | 40 | 0.80 | capture/report/shadow only |
| tennis_match_winner | capture_only | — | — | — | capture only |
| basketball_moneyline | capture_only | — | — | — | capture only |
| baseball_moneyline | capture_only | — | — | — | capture only |
| hockey_moneyline | capture_only | — | — | — | capture only |
| cricket_match_winner | capture_only | — | — | — | capture only |

## Operating Meaning

The active job is not to find a new policy. The active job is to keep capture healthy,
run shadow decisions under the frozen policies, and accumulate enough forward evidence
for each active ROI validation lane.

`sample_ready` is reached per lane only when that lane meets all minimums in the table:
valid forward decisions, settled decisions, and fresh book rate. Before then, ROI is
diagnostic-only and must not drive policy, promotion, or capital decisions.

When a lane reaches `sample_ready`, analysis may reopen for that lane. That does not
mean capital promotion, does not allow real Kelly staking, and does not make ROI
portable to any other lane.

## Active Objective

Keep collecting forward sample until the two active ROI validation lanes have enough
settled evidence to analyze honestly:

- `football_1x2_global`
- `football_goals_core`

If a lane is blocked, fix capture, mapping, model contract coverage, settlement coverage,
or book freshness. Do not respond to insufficient sample by tuning the policy.

The execution path is staged as shadow, paper, and then micro-live only after the
lane reaches capital promotion and the ROI 45 hypothesis is supported with forward
evidence. Trading remains blocked without explicit venue, bankroll, jurisdiction,
fresh-book, risk-cap, kill-switch, and reconciliation checks.
