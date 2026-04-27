# Rejected Experiments

Rejected experiments are kept here for auditability. They document ideas that were
implemented or reviewed and then rejected because they did not improve the benchmark
honestly across the required evidence layers.

These notes must not be treated as active strategy. A rejected experiment is historical
evidence, not permission to run that variant in production, tune around it, or promote
ROI from it.

Current archive:

- [Penaltyblog challenger](rejected_penaltyblog.md)
- [v6/v7 and decision-region experiments](rejected_v6_v7_decision_region.md)
- [Stability-gate experiments](rejected_stability_gate.md)

Project rule for rejected experiments:

- if an idea does not improve `OOF`, `pre_holdout`, and `locked_holdout` under the same
  frozen policy, it is rejected
- run artifacts are preserved under `outputs\runs\...` when available
- rejected integrations are not active strategy
