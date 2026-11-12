# trialnoise

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

How many trials and tasks does it take to believe a 3-point gap between two coding agents?

trialnoise splits the variance of public [Terminal-Bench 2](https://www.tbench.ai) trial outcomes
into task, cell (scaffold x model), task x cell, trial and run, separates scaffold from model, and
turns the result into a calculator. The calculator is calibrated on public multi-vendor data and
checked on submissions it never saw. It extends [evalnoise](https://github.com/h-a-forster/evalnoise),
whose data were Claude-only and whose run term rested on 7 degrees of freedom.

## Findings

All figures are in [results/v1](results/v1), mostly `summary.json` and `planner.json`. Intervals
are 95%. MDEs are at 5% two-sided and 80% power, for near pairs with a pass rate of at least 60%.

- **A 3-point gap:** 17 trials per task to say A beat B on these 89 tasks. To say A beats B on tasks
  like these, no number of trials on 89 tasks is enough: 717 tasks at 5 trials for two scaffolds on
  one model, 1,008 when scaffold and model both differ.
- **More tasks beat more trials.** At today's 89 x 5 the MDE is 8.6 to 10.2 points. Twice the trials
  gives 7.7 to 9.4; 150 tasks at 5 trials gives 6.6 to 7.8, with fewer trials in all.
- **Variance of one trial, 108 cells x 89 tasks, 46,830 trials:** task 31.2%, cell 14.6%, task x
  cell 20.3%, trial 33.9%. With the cell held fixed, ICC(1) is 0.60 (0.55 to 0.65) and ICC(1,5) is
  0.88 (0.86 to 0.90).
- **Scaffold or model depends on where you look.** Across 18 models under at least three scaffolds,
  scaffold-within-model is 5.9% of the variance of cell scores (3.4 to 15.3). Among the 6 models
  whose cells average at least 50% it is 37.2% (23.2 to 100), but the cut sets that figure: 18.2% at
  40% and 47.6% at 60%, because selecting models on score removes model variance. The scaffold SD
  holds at 4.3 to 5.1 points at every cut, while the model SD falls from 10.3 to 4.5.
- **Runs barely shift scores here.** 20 submissions with replicate jobs give a run variance on 72 df
  that estimates below zero, with an upper limit of 0.9 points of run SD. evalnoise had 2.9 points
  (1.6 to 6.4) on 7 df, on a different score and from mm-arena runs that changed the Claude Code
  version; most of it was one setup. Published jobs may be curated, so an unselected rerun could
  vary more.
- **The calculator holds out of sample.** On 252 near pairs among 39 submissions the study never
  saw, observed paired SEs were a median 0.94 of the prediction, and 83% were within 20%.
  Separately, simulation from the fit checks the planner's test, not new data: size 0.046 to 0.051,
  and power at the planned MDE 0.81 to 0.83 for a gain and 0.77 to 0.78 for a loss.
- **Adjacent ranks do not separate, and more trials will not change that.** 0 of 25 adjacent
  best-per-scaffold pairs separate at Holm 0.05 (a set that keeps the copied cell below, to match
  the LessWrong table), and 0 of 107 adjacent cells, whose median gap is 0.45 points. Across all
  5,778 pairs, 56% separate at Holm, none closer than 10.6 points.
- **Lost verdicts are rare but uneven.** 1.4% of trials in 60 complete official submissions ended
  with no verdict, costing a median 0.42 points and up to 9.9.
- **Replicate noise is an integrity signal (exploratory).** One submission's five jobs agree on all
  89 tasks, trial for trial, so it is left out. Four of the six submissions from the three agents
  the organisers named rank in the lowest 9 of 148 for how often trials of one task disagree.

This gap is narrow: prior work already showed that adjacent ranks are noise. The decomposition, the
run term, the scaffold-within-model term and the checked calculator are the new parts.
[docs/DESIGN.md](docs/DESIGN.md) lists the prior art, and what changed from the first plan.

## Use

```sh
uv sync
uv run trialnoise power --comparison same-model --band high --tasks 89 --trials 5 --gap 3
uv run trialnoise power --target fixed --gap 3
uv run trialnoise pairs --top 10
uv run trialnoise pairs --a terminus-2/gpt-5.3-codex --b codex/gpt-5.2
uv run trialnoise components --table your_trials.csv   # columns: cell, task, reward
```

`--comparison` is `same-model`, `same-scaffold` or `neither`. `--band` is the pair's pass rate:
`low` (below 35%), `mid`, `high` (from 60%) or `all`. `--target population` treats tasks as a sample
and `--target fixed` asks about these tasks only. `--run upper` adds the run term's upper limit.

An installed wheel carries `results/v1`, so these commands need no checkout. Commands that rebuild
results write under the working folder.

## Reproduce

The committed tables in `results/v1` are enough to rerun the study.

```sh
uv sync --group dev
uv run trialnoise fetch       # 224 MB, optional: pinned sources, hashes checked
uv run trialnoise extract     # optional: rebuild the two trial tables
uv run trialnoise study       # about 5 minutes
uv run trialnoise validate    # about 2 minutes
uv run trialnoise check       # every output byte-identical
uv run trialnoise report      # write REPORT.md, the full write-up, from results/v1
```

## Limits

Cells are submissions, not a sample. Variance on the probability scale depends on the pass rate.
Task-level MDEs assume new tasks resemble these 89 and tasks are independent. The held-out check
sits almost entirely in the high band. See [docs/DESIGN.md](docs/DESIGN.md).

## Licence

MIT. Sources: `yoonholee/terminalbench-trajectories` and
`harborframework/terminal-bench-2-leaderboard`, both Apache-2.0.
