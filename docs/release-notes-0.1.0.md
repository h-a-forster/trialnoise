# trialnoise 0.1.0 release notes

First release, with the `results/v1` study.

- Components of Terminal-Bench 2 trial outcomes, on 108 cells x 89 tasks, with intervals over tasks
  and over tasks and cells, and REML as a check.
- Scaffold-within-model and crossed scaffold terms, and task x cell split by source.
- A run term from replicate jobs in the official leaderboard's job files.
- `trialnoise power`: a calculator by comparison type and pass-rate band, for two targets (tasks
  like these, or these tasks), checked on 39 held-out submissions.
- `trialnoise pairs`: task-clustered and fixed-task tests, and simulated boards.
- Infrastructure, cost and an exploratory integrity screen.
- evalnoise's estimators and planner, vendored at commit 14e5684 with their tests.

## Changes after two independent audits, before any release

Results stay in `results/v1`, because nothing has been released from them.

- Task x cell sources are split inside pass-rate bands. Across bands the near-parity of scaffold and
  model was a pass-rate mix: task x scaffold is 0.0097, not 0.0117, against task x model 0.0159.
- The frontier scaffold share comes with a sweep of its cut, and the report says the cut sets it.
- Stratum, source and MDE intervals now resample tasks and cells; tasks-only intervals stay beside
  them.
- The held-out check predicts each pair at its own trials per task.
- Exception counts now cover the same 60 submissions as the rates beside them.
- The wheel carries `configs/` and `results/v1`, so an installed CLI works and the lowest-dependency
  CI job can pass. Rebuilding commands never write into the wheel.
- A chosen pair on a table with no repeats, on one cell twice, or with no noise no longer crashes.
- Corrupt shards are replaced, dropped connections retried, and unsafe remote names refused.
- A run interval with no upper limit stops the study; an agent that never passes or never fails no
  longer counts as copied.
