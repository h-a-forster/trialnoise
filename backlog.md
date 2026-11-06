# Backlog

## In progress

## Waiting on me

- Run `uv run trialnoise report` and commit `REPORT.md`, or waive it: the harness refuses report files from subagents, so the file is not committed. The generator and its byte-identical hash are.
- Decide whether trialnoise goes public; it stays private until then.
- Dispatch CI once to prove the lowest-dependency job and the Linux `check` job, or waive it: byte identity has run on Windows only.
- Merge branch `audit-fixes` into `main`: the audit fixes are pushed there, not on the default branch.

## Decided

- 2026-11-06: the planner uses strata of near pairs, with the pooled term for reference only.
- 2026-11-06: the copied cell `final/claude-sonnet-4-5` leaves the primary fit; Forge stays, with a sensitivity row.
- 2026-11-06: text intervals resample tasks and cells; the tasks-only interval stays beside them.
- 2026-11-06: the task x cell split is taken inside pass-rate bands; the frontier share is always quoted with its cut.
- 2026-11-06: the audits' changes stay in `results/v1`, since nothing has been released from it.

## Proposed

- Repeat the run term on jobs a third party reruns, since published jobs may be curated.
- Measure version or provider drift by rerunning one cell months apart.
- Test the integrity screen on a benchmark it was not built on.
- Explain the access violation that stopped `uv run pytest` in 2 of about 20 runs on Python 3.13, once traced to the vendored `simulate`; it did not recur, and the lowest-dependency run had none.
