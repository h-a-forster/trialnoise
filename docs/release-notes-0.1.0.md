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
