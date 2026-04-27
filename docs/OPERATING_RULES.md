# Operating Rules

These rules define the current operating contract for the repository.

1. Do not optimize policy until sample_ready.
2. Do not promote ROI from history_proxy.
3. Do not aggregate ROI across lanes.
4. Do not use locked_holdout to train gates.
5. Do not emit capital stake unless promotion_status is capital_promotable.
6. Do not use real Kelly staking before capital_promotable.
7. Do not add active sports without model contract and settlement contract.
8. sample_ready means analysis may reopen; it does not mean capital promotion.
9. ROI before sample_ready is diagnostic-only.
10. Research commands are diagnostic-only by default.
11. A 45% ROI is a forward hypothesis, not an optimization target to force.
12. Live trading is disabled unless capital promotion, ROI 45 support, risk caps,
    venue configuration, bankroll, jurisdiction confirmation, kill switch and
    reconciliation checks all pass.
13. No VPN or geographic bypass is allowed.

## Practical Interpretation

Daily operation is capture, report, and shadow validation. Policy changes, threshold
changes, new gates, new active sports, and capital decisions stay frozen unless the
current state is explicitly updated and the relevant lane has enough forward evidence.
