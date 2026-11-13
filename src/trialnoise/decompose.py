"""Variance components of a cells x tasks x trials table of pass/fail outcomes.

    y_tar = mu + u_t + c_a + v_ta + e_tar

on the probability scale: task `u`, cell (scaffold x model submission) `c`, task x cell `v`, trial
`e`. The estimators are evalnoise's two-way ANOVA moments on equally weighted task means, plus the
cell term those omit. They run on matrices so that 10,000 bootstrap draws take seconds; a test holds
them to the vendored `estimate`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from trialnoise._evalnoise.components import RunTerm, estimate
from trialnoise._evalnoise.data import DataError

TERMS = ("task", "cell", "task_cell", "trial")
NEAR_GAP = 0.10  # A pair is "near" when its scores differ by at most this.
BANDS = (("low", 0.0, 0.35), ("mid", 0.35, 0.6), ("high", 0.6, 1.01))
COMPARISONS = ("same-model", "same-scaffold", "neither")


@dataclass(frozen=True)
class Layout:
    """Per task-cell sufficient statistics of a fully crossed table (rows cells, columns tasks)."""

    cells: tuple[str, ...]
    tasks: tuple[str, ...]
    mean: np.ndarray
    n: np.ndarray
    ss: np.ndarray

    def take(self, rows: np.ndarray | None = None, cols: np.ndarray | None = None) -> Layout:
        """Sub-table; repeated rows or columns count as distinct units (bootstrap)."""
        r = np.arange(len(self.cells)) if rows is None else rows
        c = np.arange(len(self.tasks)) if cols is None else cols
        grid = np.ix_(r, c)
        return Layout(
            tuple(self.cells[i] for i in r),
            tuple(self.tasks[j] for j in c),
            self.mean[grid],
            self.n[grid],
            self.ss[grid],
        )


def layout(
    trials: pd.DataFrame, cell: str = "cell", task: str = "task", y: str = "reward"
) -> Layout:
    """Pivot a trial table; every cell must have at least one trial on every task."""
    if len(trials) == 0:
        raise DataError("no trials")
    g = trials.groupby([cell, task], sort=True)[y]
    n = g.size().unstack(task)
    if n.isna().any(axis=None):
        missing = int(n.isna().sum().sum())
        raise DataError(f"{missing} task-cell combinations have no trial; the table is not crossed")
    mean = g.mean().unstack(task)
    ss = (g.var(ddof=0) * g.size()).unstack(task)
    return Layout(
        tuple(n.index),
        tuple(n.columns),
        mean.to_numpy(float),
        n.to_numpy(float),
        ss.to_numpy(float),
    )


def moments(table: Layout) -> dict[str, float]:
    """Unbiased moment estimates; negative values are kept, so that intervals stay honest."""
    y, n, ss = table.mean, table.n, table.ss
    a, t = y.shape
    if a < 2 or t < 2:
        raise DataError("need at least two cells and two tasks")
    df_within = float((n - 1).sum())
    if df_within <= 0:
        raise DataError("no task-cell has two trials: trial noise is not identified")
    within = float(ss.sum()) / df_within
    grand = float(y.mean())
    by_cell = y.mean(axis=1)
    by_task = y.mean(axis=0)
    ms_cell = t * float(((by_cell - grand) ** 2).sum()) / (a - 1)
    ms_task = a * float(((by_task - grand) ** 2).sum()) / (t - 1)
    resid = y - by_cell[:, None] - by_task[None, :] + grand
    ms_inter = float((resid * resid).sum()) / ((a - 1) * (t - 1))
    return {
        "task": (ms_task - ms_inter) / a,
        "cell": (ms_cell - ms_inter) / t,
        "task_cell": ms_inter - within * float((1.0 / n).mean()),
        "trial": within,
        "grand": grand,
    }


def statistics(m: dict[str, float], k: int = 5) -> dict[str, float]:
    """Shares and intraclass correlations of one set of components.

    `icc1` is the correlation of two trials of one cell on one task with the cell held fixed, what
    repeating a trial of a chosen cell reproduces; `icc1_k` is the Spearman-Brown reliability of
    the mean of `k` trials; `icc_pair` is the same correlation with the task main effect removed,
    which is what a paired comparison of two cells can use.
    """
    total = sum(m[name] for name in TERMS)
    out = {f"share_{name}": m[name] / total for name in TERMS}
    between = m["task"] + m["task_cell"]
    icc = between / (between + m["trial"])
    out["icc1"] = icc
    out["icc1_k"] = k * icc / (1 + (k - 1) * icc)
    out["icc_pair"] = m["task_cell"] / (m["task_cell"] + m["trial"])
    return out


def per_cell_icc(table: Layout) -> np.ndarray:
    """One-way ANOVA ICC(1) of tasks within each cell, nan where a cell has no repeats."""
    out = np.full(len(table.cells), np.nan)
    t = len(table.tasks)
    for j in range(len(table.cells)):
        y, n, ss = table.mean[j], table.n[j], table.ss[j]
        total = float(n.sum())
        if total - t <= 0:
            continue
        msw = float(ss.sum()) / (total - t)
        grand = float((y * n).sum()) / total
        msb = float((n * (y - grand) ** 2).sum()) / (t - 1)
        n0 = (total - float((n * n).sum()) / total) / (t - 1)
        if msb + (n0 - 1) * msw > 0:
            out[j] = (msb - msw) / (msb + (n0 - 1) * msw)
    return out


def percentile(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    lo, hi = np.quantile(finite, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def bootstrap_draws(
    table: Layout,
    stat: Callable[[Layout], np.ndarray],
    *,
    draws: int,
    seed: int,
    over_cells: bool = False,
) -> np.ndarray:
    """`stat` on resampled tasks, and resampled cells too with `over_cells`.

    Resampling tasks alone holds the cells fixed, so its intervals for the cell term describe task
    sampling only. Resampling both is Owen's pigeonhole bootstrap, the honest interval when the
    cells are treated as a sample.
    """
    rng = np.random.default_rng(seed)
    a, t = table.mean.shape
    out = []
    for _ in range(draws):
        rows = rng.integers(0, a, a) if over_cells else None
        out.append(stat(table.take(rows, rng.integers(0, t, t))))
    return np.asarray(out)


def component_vector(table: Layout, k: int = 5) -> np.ndarray:
    m = moments(table)
    s = statistics(m, k)
    return np.array([*(m[name] for name in TERMS), *(s[f"share_{name}"] for name in TERMS),
                     s["icc1"], s["icc1_k"], s["icc_pair"]])  # fmt: skip


VECTOR_NAMES = (
    *TERMS,
    *(f"share_{name}" for name in TERMS),
    "icc1",
    "icc1_k",
    "icc_pair",
)


def reml_means(table: Layout) -> dict[str, float]:
    """REML on the cell x task means with crossed task and cell effects (statsmodels MixedLM).

    The residual holds task x cell plus the trial noise of a mean; subtracting the moment noise
    term gives the comparable `task_cell`.
    """
    a, t = table.mean.shape
    y = table.mean.ravel()
    rows = np.repeat(np.arange(a), t)
    cols = np.tile(np.arange(t), a)
    design = pd.DataFrame({"y": y, "cell": rows, "task": cols, "group": 1})
    fit = sm.MixedLM.from_formula(
        "y ~ 1",
        groups="group",
        vc_formula={"cell": "0 + C(cell)", "task": "0 + C(task)"},
        data=design,
    ).fit(reml=True, method="lbfgs")
    names = list(fit.model.exog_vc.names)
    vc = dict(zip(names, np.asarray(fit.vcomp, float), strict=True))
    within = moments(table)["trial"]
    return {
        "task": float(vc["task"]),
        "cell": float(vc["cell"]),
        "task_cell": float(fit.scale) - within * float((1.0 / table.n).mean()),
        "residual": float(fit.scale),
        "converged": bool(fit.converged),
    }


def cell_noise(table: Layout) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's trial variance and the noise it adds to a task mean.

    A cell with no repeats borrows the pooled trial variance.
    """
    df = (table.n - 1).sum(axis=1)
    if df.sum() <= 0:
        raise DataError("no task-cell has two trials: trial noise is not identified")
    pooled = float(table.ss.sum()) / float(df.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        within = np.where(df > 0, table.ss.sum(axis=1) / df, pooled)
    return within, within * (1.0 / table.n).mean(axis=1)


def pair_matrices(table: Layout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For all cell pairs: var(d_t), the interaction each pair implies, and mean trial variance.

    `sigma2[a, b] = (var(d_t) - noise_a - noise_b) / 2` is the per-cell task x cell variance the
    pair implies, unbiased for it under the additive noise model.
    """
    s = np.cov(table.mean, ddof=1)
    d = np.diag(s)
    var_d = d[:, None] + d[None, :] - 2 * s
    within, noise = cell_noise(table)
    sigma2 = (var_d - noise[:, None] - noise[None, :]) / 2
    trial = (within[:, None] + within[None, :]) / 2
    return var_d, sigma2, trial


def comparison_masks(scaffolds: list[str], models: list[str]) -> dict[str, np.ndarray]:
    """Upper-triangle masks of cell pairs by what they share."""
    sc = np.asarray(scaffolds)
    mo = np.asarray(models)
    same_s = sc[:, None] == sc[None, :]
    same_m = mo[:, None] == mo[None, :]
    upper = np.triu(np.ones_like(same_s), 1)
    return {
        "same-model": upper & same_m & ~same_s,
        "same-scaffold": upper & same_s & ~same_m,
        "neither": upper & ~same_s & ~same_m,
    }


def near_masks(score: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Comparison x pass-rate band masks of near pairs, plus each comparison across bands."""
    gap = np.abs(score[:, None] - score[None, :])
    level = (score[:, None] + score[None, :]) / 2
    near = gap <= NEAR_GAP
    out: dict[str, np.ndarray] = {}
    for comparison, mask in masks.items():
        out[f"{comparison}/all"] = mask & near
        for band, lo, hi in BANDS:
            out[f"{comparison}/{band}"] = mask & near & (level >= lo) & (level < hi)
    return out


def stratum_vector(table: Layout, strata: dict[str, np.ndarray]) -> np.ndarray:
    """Mean task x cell and trial variance of each stratum (flattened in dict order)."""
    _, sigma2, trial = pair_matrices(table)
    out: list[float] = []
    for mask in strata.values():
        if mask.any():
            out += [float(sigma2[mask].mean()), float(trial[mask].mean())]
        else:
            out += [math.nan, math.nan]
    return np.array(out)


def resampled_strata(
    cells: tuple[str, ...], strata: dict[str, np.ndarray]
) -> Callable[[Layout], np.ndarray]:
    """`stratum_vector` for a table whose rows may be a resample of `cells`.

    Each stratum stays the set of pairs chosen on the full table. A cell drawn twice brings its
    pairs twice; a cell paired with its own copy belongs to no stratum.
    """
    position = {cell: j for j, cell in enumerate(cells)}
    symmetric = {name: mask | mask.T for name, mask in strata.items()}

    def stat(table: Layout) -> np.ndarray:
        rows = [position[cell] for cell in table.cells]
        grid = np.ix_(rows, rows)
        return stratum_vector(table, {k: np.triu(m[grid], 1) for k, m in symmetric.items()})

    return stat


def interaction_structure(s: Mapping[str, float]) -> dict[str, float]:
    """Split task x cell by what the near pairs of each comparison type share.

    With `v_ta = alpha_t,model + beta_t,scaffold + gamma_t,model,scaffold`, same-model pairs measure
    beta + gamma, same-scaffold pairs alpha + gamma, and pairs that share neither measure all three.
    """
    model, scaffold, neither = s["same-model"], s["same-scaffold"], s["neither"]
    return {
        "task_model": neither - model,
        "task_scaffold": neither - scaffold,
        "task_model_scaffold": model + scaffold - neither,
    }


def banded_structure(s: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """`interaction_structure` within each pass-rate band, and its mean over the bands.

    `s` holds a value per `comparison/band`. On the probability scale task x cell moves with the
    pass rate, and the comparison types sit at different pass rates, so their differences are only
    comparable inside a band.
    """
    out = {
        band: interaction_structure({c: s[f"{c}/{band}"] for c in COMPARISONS})
        for band, _, _ in BANDS
    }
    terms = out[BANDS[0][0]]
    out["mean"] = {k: sum(part[k] for part in out.values()) / len(BANDS) for k in terms}
    return out


def scaffold_nesting(
    score: np.ndarray, models: list[str], noise: float, min_scaffolds: int = 3
) -> dict[str, float]:
    """One-way random model on cell scores: model, and scaffold within model.

    `score` holds one value per (model, scaffold) and `noise` is the sampling variance of one
    score around that pair's true value. Only models under at least `min_scaffolds` scaffolds
    enter. Scaffold-within-model holds the scaffold's main effect, its interaction with the model,
    and anything else that differs between the submissions (date, provider route).
    """
    frame = pd.DataFrame({"y": score, "m": models})
    frame = frame[frame.groupby("m")["y"].transform("size") >= min_scaffolds]
    return nested_moments(frame["y"].to_numpy(), frame["m"].to_numpy(), noise)


def nested_moments(y: np.ndarray, groups: np.ndarray, noise: float) -> dict[str, float]:
    labels, code = np.unique(groups, return_inverse=True)
    n = np.bincount(code).astype(float)
    m, total = len(labels), float(len(y))
    if m < 2 or total <= m:
        raise DataError("nested term needs two models and a model under two scaffolds")
    means = np.bincount(code, weights=y) / n
    grand = float(y.mean())
    msw = float(((y - means[code]) ** 2).sum()) / (total - m)
    msb = float((n * (means - grand) ** 2).sum()) / (m - 1)
    n0 = (total - float((n * n).sum()) / total) / (m - 1)
    within_model = msw - noise
    between = (msb - msw) / n0
    # A negative between-model estimate counts as zero, so the share stays in [0, 1].
    denominator = max(within_model, 0.0) + max(between, 0.0)
    return {
        "model": between,
        "scaffold_within_model": within_model,
        "share_scaffold": max(within_model, 0.0) / denominator if denominator > 0 else math.nan,
        "models": float(m),
        "cells": total,
    }


def crossed_cells(score: np.ndarray, scaffolds: list[str], models: list[str]) -> dict[str, float]:
    """REML on the cell scores with crossed model and scaffold effects.

    The residual is model x scaffold plus whatever one submission adds, including its run shock,
    so it bounds the run variance from above on this design's degrees of freedom.
    """
    design = pd.DataFrame({"y": score, "s": scaffolds, "m": models, "group": 1})
    model = sm.MixedLM.from_formula(
        "y ~ 1", groups="group", vc_formula={"s": "0 + C(s)", "m": "0 + C(m)"}, data=design
    )
    with warnings.catch_warnings():
        # statsmodels warns of a boundary at any variance below 0.01, and these are all small.
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit = model.fit(reml=True, method="lbfgs")
    vc = dict(zip(fit.model.exog_vc.names, np.asarray(fit.vcomp, float), strict=True))
    return {
        "scaffold": float(vc["s"]),
        "model": float(vc["m"]),
        "residual": float(fit.scale),
        "converged": bool(fit.converged),
        "df_residual": len(score) - design["s"].nunique() - design["m"].nunique() + 1,
    }


def _above_cancellation(value: float, scale: float) -> float:
    """A moment estimate that is a difference of terms of size ``scale``, with the rounding
    noise of that subtraction set to zero: below 1e-9 of the terms its digits are not data and
    differ between platforms (one submission's two mean squares are equal, giving +-4e-20)."""
    return 0.0 if abs(value) < 1e-9 * scale else value


def run_term(
    official: pd.DataFrame,
    *,
    min_tasks: int = 80,
    skip: tuple[str, ...] = (),
    max_var: float = 0.25,
) -> dict:
    """Pooled run variance from submissions with replicate jobs on (nearly) every task.

    Each job is a run in evalnoise's sense. Job shocks are read inside one submission, where
    scaffold, model and tasks are the same. `max_var` bounds the interval: 0.25 for scores in
    [0, 1], `inf` for unbounded ones.
    """
    keep = official[(official["status"] == "ok") & official["reward"].notna()]
    keep = keep[~keep["submission"].isin(skip)]
    per_submission = []
    terms: list[RunTerm] = []
    for submission, group in keep.groupby("submission", sort=True):
        tasks = group.groupby("job")["task"].nunique()
        jobs = tasks[tasks >= min_tasks].index
        if len(jobs) < 2:
            continue
        g = group[group["job"].isin(jobs)]
        if g["reward"].nunique() < 2:
            continue  # Never a pass, or never a failure: no residual to set the jobs against.
        frame = pd.DataFrame(
            {
                "item": g["task"].to_numpy(),
                "arm": "x",
                "run": g["job"].to_numpy(),
                "repeat": g.groupby(["job", "task"]).cumcount().to_numpy(),
                "cluster": g["task"].to_numpy(),
                "score": g["reward"].to_numpy(float),
            }
        )
        term = estimate(frame).run_term
        if not term.identified() or term.rss == 0:
            raise DataError(f"{submission}: replicate jobs carry no residual variance")
        terms.append(term)
        per_submission.append(
            {
                "submission": str(submission),
                "jobs": len(jobs),
                "df": term.df,
                "pass_rate": float(g["reward"].mean()),
                "ms_run": term.ss / term.df,
                "ms_residual": term.rss / term.df_res,
                "variance": _above_cancellation(term.variance(), term.ss / term.k),
            }
        )
    if not terms:
        raise DataError(f"no submission has two jobs on at least {min_tasks} tasks")
    pooled = RunTerm.pool(terms)
    f = (pooled.ss / pooled.df) / (pooled.rss / pooled.df_res)
    lo, hi = pooled.interval(max_var=max_var)
    if math.isnan(hi):
        raise DataError("replicate jobs agree too closely for any run variance: look for copies")
    return {
        "df": pooled.df,
        "df_residual": pooled.df_res,
        "variance": _above_cancellation(pooled.variance(), pooled.ss / pooled.k),
        "interval": [lo, hi],
        "sd_upper": math.sqrt(hi),
        "f": f,
        "p_above": float(stats.f.sf(f, pooled.df, pooled.df_res)),
        "p_below": float(stats.f.cdf(f, pooled.df, pooled.df_res)),
        "submissions": per_submission,
    }


def split_jobs_false_positives(official: pd.DataFrame, skip: tuple[str, ...]) -> dict[str, float]:
    """A/A check: split one submission's jobs into two arms; a paired test should reject 5%."""
    keep = official[(official["status"] == "ok") & official["reward"].notna()]
    keep = keep[~keep["submission"].isin(skip)]
    splits, rejected = 0, 0
    for _, group in keep.groupby("submission", sort=True):
        wide = group.pivot_table(index="task", columns="job", values="reward", aggfunc="mean")
        jobs = [j for j in wide.columns if wide[j].notna().sum() >= 80]
        if len(jobs) != 5:
            continue
        wide = wide[jobs].dropna()
        values = wide.to_numpy()
        for first in range(4):
            for second in range(first + 1, 5):
                arm = np.zeros(5, dtype=bool)
                arm[[first, second]] = True
                d = values[:, arm].mean(axis=1) - values[:, ~arm].mean(axis=1)
                t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
                rejected += int(2 * stats.t.sf(abs(t), len(d) - 1) < 0.05)
                splits += 1
    return {"splits": splits, "rejected": rejected, "rate": rejected / splits}
