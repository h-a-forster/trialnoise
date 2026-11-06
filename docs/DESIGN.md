# Design

This file fixes the data handling, model, estimands and intervals, and records where the study
departs from its original specification and why. Results are in [results/v1](../results/v1).

## Question

How much of a public agent leaderboard's score is task, agent, trial and run noise, and how many
trials and tasks does a comparison need? Terminal-Bench 2 (TB2) is the one public benchmark with
trial-level results across many vendors, which is what evalnoise lacked.

## Prior art, and how narrow the gap is

This is a small gap, narrower than trajclaims', because the interval and verdict work is done.

| Work | Has | Does not have |
|---|---|---|
| [LessWrong, 2026-08-05](https://www.lesswrong.com/posts/GiPmLmmbT6DyrwYkH) | Same dataset, task-clustered bootstrap, BH, ICC 0.5 to 0.8, 24 of 25 adjacent pairs noise, scaffold spans | Decomposition, code; its rows include the duplicates; model and scaffold not separated |
| Willison's SQLite loader | 10,947 official trials | Descriptive only |
| [arXiv:2609.17394](https://arxiv.org/abs/2609.17394) | SWE-bench, 254 submissions, no adjacent pair separated among the top thirty, scaffold ranges to 29.8 points | Verdicts only, so no trial term |
| [arXiv:2512.06710](https://arxiv.org/abs/2512.06710) | ICC on GAIA and FRAMES from own runs | Public multi-vendor data; run and interaction terms |
| arXiv:2605.23950 | Position: harness variance can exceed model variance | Public numbers |
| [Anthropic, infrastructure noise](https://www.anthropic.com/engineering/infrastructure-noise) | 6 points on TB2 from VM resources; error rates 5.8% to 2.1% | Public data; run this from their own experiments |
| PyPI evalci, errorbars | CIs, clustered SEs | Components, run terms |
| evalnoise | The estimators and planner used here | Multi-vendor data; its run variance rests on 7 df, all Claude |

New here: the decomposition on public multi-vendor data, the scaffold-within-model term with models
resampled, the run term from replicate jobs, and a calculator calibrated on public data and checked
on submissions it never saw. The conclusion that adjacent ranks do not separate is not new.

## Critique of the specification

Reading the sources before building changed the plan. Each row was checked against the data.

| Planned | Found | Decision |
|---|---|---|
| 109 cells x 89 tasks x 5 trials, parquet rows are trials | 52,104 rows hold 47,152 trials: 4,952 are one trial listed twice, in two scrapes. Some cells looked like 10 trials per task for that reason. Trial counts per task-cell run from 1 to 6, with three cells near 1. | Deduplicate as trajclaims does, keep the full-scrape row, and fail if twins disagree. Estimators take unbalanced counts; the score is the task-equal mean. |
| "Hundreds of df" fixes evalnoise's weakness | True of task x cell and trial, but evalnoise's 7 df were the **run** term, which a job-per-cell design does not identify at all. | Find replicate jobs. See next row. |
| Official `result.json` for 5 to 10 submissions, as a cross-check | A job's `result.json` lists every trial by reward and by exception. All 244 fit in 3 MB, against 121 GB for the repository. 23 submissions have replicate jobs, mostly one trial per task; 20 stay after exclusions. | Fetch them all, per job. This gives a run term on real df, exceptions, and 39 held-out submissions. Trial-level `result.json` files are not needed. |
| Exception info "where present" in the parquet | Not present. | Exceptions come from job files, so infrastructure rates cover official submissions only. A proxy from missing token counts was tested and rejected: those trials are mostly long timeouts, not infrastructure failures. |
| Cost where non-null | `cost_cents` is never null; 0 means unreported. | 0 becomes null. A cell needs 90% of its trials reporting. |
| Task x cell is one term | On the probability scale it is 0.050 pooled but 0.023 to 0.039 for pairs within 10 points, because far-apart cells have different task profiles. A planner on the pooled term overstates the MDE of the pairs people compare. | Strata by comparison (same model, same scaffold, neither) and pair pass-rate band, from each pair's own difference variance. The pooled row stays for reference. |
| One "tasks random" analysis | LessWrong's clustered test asks about tasks like these. A leaderboard's claim is about these tasks. The two need different designs. | Two targets throughout: population (task x cell is noise) and fixed (only trial noise is). |
| Cluster bootstrap over tasks for every component | That holds cells fixed. In simulation it covers the cell term 0.73 of the time. | Report it as asked, and add a pigeonhole bootstrap over tasks and cells (0.92 to 0.95), which the text uses. The nested term resamples tasks and models. |
| 10 models under at least 3 scaffolds | 18 with canonical model names. | Use 18, and add the frontier subset, because the share is dominated by the spread of models otherwise. |
| Scaffold-within-model as "the" scaffold term | It holds scaffold, scaffold x model, and everything else that differs between submissions. | Report what it holds, and fit crossed model and scaffold effects as well. Split task x cell by source using the pair types. |
| 7 df and full df as evalnoise does | In evalnoise those are run df. Task-level df also matter here. | Both: a pilot on 7 df against 88 df for tasks needed, and the run term borrowed on 7 df against the real df. |
| Adjacent pairs as the headline | 108 cells on a 75-point range sit a median 0.45 points apart. No design separates that. | Keep the re-derivation, and report the gap two cells need. |
| All 109 cells are data | One submission's five jobs agree on all 89 tasks, trial for trial (`final`, officially MAYA Claude 4.5 Sonnet). Forge, the top row, was named for reward hacking and its rows are not rescored at the pinned commits. | The copied cell leaves the primary fit (found by rule, `copied_submissions`). Forge stays, with a sensitivity row. Organisers' named submissions are excluded from the run term and held-out set. |
| `install-windows-3-11` | One submission spells a task's folder differently. | Alias it. |

## Data

Sources are pinned in `data.py`: the trajectory dataset's revision and shard hashes, and the
official repository's commit. `trialnoise fetch` downloads them, and `trialnoise extract` writes
`results/v1/trials.parquet`, `official.parquet` and `MANIFEST.json`, with the row counts it expects
asserted. The study reads only the committed tables, so no rerun needs the network.

A trial is `(scaffold, raw model, trial name)`. Model strings map to canonical names through
`models.py`, copied from trajclaims. A cell is `scaffold/model`; the one pair submitted twice keeps
its provider tag, so it stays two cells. Official submissions map to a scaffold and model in
`configs/official.toml`.

## Model

For cell `a`, task `t`, trial `r`, on the probability scale:

    y_tar = mu + u_t + c_a + v_ta + e_tar

`u` is the task, `c` the cell, `v` task x cell, `e` the trial. `E[y] = p` and `y` is Bernoulli, so
the terms sum to about `p(1-p)` and the trial variance is the mean of `p_ta(1-p_ta)`. These are
descriptive components of the probability scale, not a generative claim; the validation uses a
logistic-normal model whose probability-scale components match the fit.

### Estimators

- **Moments.** evalnoise's two-way ANOVA on equally weighted task means (vendored; a test holds the
  matrix version here to it), plus `cell = (MS_cell - MS_inter) / T`, which evalnoise omits because
  it treats arms as fixed. Negative estimates are kept in intervals.
- **REML.** statsmodels MixedLM with crossed task and cell effects on the cell x task means. The
  residual holds task x cell plus trial noise over `k`, so the moment noise term is subtracted.
- **Pair strata.** For cells `a` and `b`, `sigma2_ab = (var(d_t) - noise_a - noise_b) / 2`, with
  each cell's own trial variance. Averaged over near pairs (gap of at most 10 points) by comparison
  and by the pair's mean pass rate: low (< 35%), mid, high (>= 60%).
- **Interaction sources.** Same-model pairs measure task x scaffold plus the three-way term;
  same-scaffold pairs task x model plus the three-way term; pairs sharing neither measure all
  three. The differences identify each, using near pairs inside one pass-rate band so the types are
  comparable: across bands they sit at different pass rates (same-scaffold pairs a mean 36%, same
  model 47%), and task x cell moves with the pass rate. The headline is the mean over the three
  bands.
- **Nested.** One-way unbalanced random-effects ANOVA on (scaffold, model) scores, for models under
  at least three scaffolds. `noise` is `(task x cell, same-model) + trial noise of a mean`, over
  `T`: the sampling variance of one score against others on the same model. The share is
  `scaffold / (scaffold + model)`, with a negative model term counted as zero. The frontier subset
  is models averaging at least 50%. That cut selects on the outcome, which shrinks the model term
  and raises the share, so `frontier_sweep` repeats it at 40% to 60%; the scaffold SD is the part
  that does not depend on the cut.
- **Crossed.** MixedLM on the scores with crossed model and scaffold effects. Its residual, less
  `noise`, bounds model x scaffold plus any one-off submission effect.
- **Run.** Each job is a run. Henderson III within each submission with at least two jobs on 80 or
  more tasks (vendored `RunTerm`), pooled, with the Burdick-Graybill interval. Submissions named by
  the organisers and copied jobs are excluded.

### Intervals

Percentile bootstrap, 10,000 draws with seeds derived from the unit name. Two schemes for the
components, strata, interaction sources and MDEs: tasks, and tasks and cells (Owen's pigeonhole
bootstrap). Each stratum stays the set of pairs chosen on the full table, and a pair counts as often
as its two cells are drawn. Models and tasks for the nested term. Cells are not a random sample, so
the wider scheme is the honest default, not an exact one. ICCs and shares get the same intervals.

### Estimands

- Variance shares; ICC(1) `(task + task_cell) / (task + task_cell + trial)`, the correlation of two
  trials with the cell held fixed (with cells random, `cell` joins the numerator and the total);
  ICC(1,5) by Spearman-Brown; and the same correlation without the task main effect, which is what a
  pair uses. Per-cell one-way ICC(1) to compare with the LessWrong range.
- MDE for two cells, paired on tasks, 5% two-sided at 80% power, from evalnoise's `mde`, under each
  stratum and target, at 5 and 10 trials and 89 and 150 tasks, with intervals from the stratum
  bootstrap. Tasks and trials needed for 3, 5 and 10 points.
- Adjacent pairs: paired t on task means (df 88), Holm and BH. Best-per-scaffold (26 rows, the
  reading of "one agent per scaffold" that gives 25 pairs and 325 in all) and all cells.
- What more trials or tasks change: `simulate_board` redraws the whole board from smoothed
  probabilities, re-ranks and re-tests, so that the selection of neighbours is inside the
  simulation. Smoothing pulls each task mean towards the additive fit by a pseudo-count that makes
  the mean Bernoulli variance equal the fitted trial variance.
- Infrastructure: verdict lost (an exception and no reward, or missing from the job's lists). Its
  effect is `passes / verdicts - passes / n_total`. Timeouts with a verdict are scored as usual, so
  they are reported apart.
- Cost per pass: total cost over passes, on trials with a cost, resampling tasks.

### Validation

Monte Carlo units in `validate.py`, each written atomically and resumable:

| Check | Tests |
|---|---|
| `recovery` | Moment estimates on the study's design against known truth |
| `coverage` | Both bootstrap schemes |
| `planner-*` | Size and power at the planned MDE for the strata and designs in the report |
| `pilot-*` | Plans from a 7-df pilot against 88 df |
| `run-*` | Run term borrowed on 7 df and on the real df, at TB2's upper limit and at evalnoise's SD; includes a run-blind test |
| `fixed-tasks` | Size and power of the fixed-task z-test |

The held-out check is on real data and does not use simulation: strata from the trajectory table
against paired SEs of submissions the dataset never saw, each pair predicted at the harmonic mean
of its trials per task. The `planner-*` checks draw from the fitted components, so they test the
planner's formula, not its fit to new data.

## Report

`report.py` fills `report_text.py` from the committed outputs. `results/v1/report.sha256` is the hash
of the rendered text, so a test can show regeneration is byte-identical from results alone. The prose
states a few facts in words (one copied submission, six named ones, the top cell); `premises` refuses
to render if a new data version breaks them.

`trialnoise check` reruns the study from the committed tables and compares every output. So far that
has run on Windows only: the `check` CI job repeats it on Linux, on manual dispatch. A fast test reruns
the whole study path with next to no bootstrap draws and holds every point estimate to the committed
value within 1e-4.

## Audit findings not acted on as proposed

Two independent audits ran on 2026-11-06. Their other findings are fixed; these were not, or not as
proposed.

| Finding | Decision |
|---|---|
| README quotes tasks-only MDE intervals | Not so: it quotes no MDE interval. The report's MDE table did, and now uses tasks and cells. |
| Simulated-board rates fall from 5 to 50 trials | Not a defect: better ranking makes observed neighbours truly closer. No code change; the report now says the rates sit near Holm's family-wise floor. |
| `report.fields` is covered only by a slow test | Not so: the fast report test calls it. The study path was the real gap. |
| A smoke test on a synthetic table | A fast rerun on the committed tables instead. It also fails when a code change moves a committed number, which a synthetic table cannot. |
| Fix the "best-per-scaffold includes the copied cell" universe | Kept: dropping it loses the post's 26 rows. The report and README now say so. The simulated board uses primary cells, so it has 24 pairs, and the report says that too. |
| Make the report's "four of the six" and similar prose data-driven | Worded prose stays worded; `premises` fails loudly when its facts change. |
| Locate results from the working folder or an environment variable | The wheel carries them, so an installed `power` works anywhere. |
| Retry the parquet download | Left alone: a failed shard leaves no named file, so a rerun repairs it. |

## Not done

- A latent-scale (GLMM) decomposition. Shares on the probability scale do not carry to other pass
  rates, but the planner's question is on that scale, and the strata absorb the dependence.
- Inferring jobs from `started_at` for cells outside the official repository. That field is empty on
  52% of trials, and the 72 run df already bound the term.
- Task metadata (difficulty, category) as covariates.
