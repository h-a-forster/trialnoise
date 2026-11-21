"""Exact sampling laws of the unbiased pass^k and pass@k estimators.

A task that succeeds with probability p is run n times and succeeds c times. pass^k is the chance
that k fresh trials all succeed, p^k; pass@k that at least one does, 1 - (1 - p)^k. Their unbiased
estimators are C(c,k)/C(n,k) and 1 - C(n-c,k)/C(n,k), for k <= n only: a function of p has an
unbiased estimator from n trials exactly when it is a polynomial of degree at most n.

Everything about one task is an exact `Fraction`. pass@k at p is one minus pass^k at 1 - p with
successes and failures swapped, so the two share variances and worst cases. Trials of a task are
taken as independent with one p; tasks as independent of each other.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from trialnoise._evalnoise.data import DataError

POW, AT = "pass^k", "pass@k"
METRICS = (POW, AT)
KINDS = ("normal", "guaranteed")
MAX_TRIALS = 500
REFERENCE_TRIALS = (4, 5, 8, 16)
REFERENCE_RATES = {"50": Fraction(1, 2), "90": Fraction(9, 10)}
Rate = Fraction | int | float | str


def _check(n: int, k: int, metric: str) -> None:
    if metric not in METRICS:
        raise DataError(f"metric is one of {METRICS}, not {metric!r}")
    if not 1 <= k <= n:
        raise DataError(f"k={k}, n={n}: {metric} has an unbiased estimator for k in 1..n only")


def rate(p: Rate) -> Fraction:
    """p as an exact fraction; a float is read as its decimal text, so 0.8 is 4/5."""
    q = Fraction(repr(p)) if isinstance(p, float) else Fraction(p)
    if not 0 <= q <= 1:
        raise DataError(f"a success probability lies in [0, 1], not {p}")
    return q


def estimate(c: int, n: int, k: int, metric: str = POW) -> Fraction:
    """The unbiased estimator from c successes in n trials."""
    _check(n, k, metric)
    if not 0 <= c <= n:
        raise DataError(f"{c} successes in {n} trials")
    if metric == POW:
        return Fraction(math.comb(c, k), math.comb(n, k))
    return 1 - Fraction(math.comb(n - c, k), math.comb(n, k))


def plugin(c: int, n: int, k: int, metric: str = POW) -> Fraction:
    """The plug-in estimator, (c/n)^k or 1 - (1 - c/n)^k: biased for k >= 2."""
    _check(n, k, metric)
    share = Fraction(c, n)
    return share**k if metric == POW else 1 - (1 - share) ** k


def truth(p: Rate, k: int, metric: str = POW) -> Fraction:
    q = rate(p)
    return q**k if metric == POW else 1 - (1 - q) ** k


def binomial(n: int, p: Rate) -> list[Fraction]:
    q = rate(p)
    return [math.comb(n, c) * q**c * (1 - q) ** (n - c) for c in range(n + 1)]


Estimator = Callable[[int, int, int, str], Fraction]


def law(
    n: int, k: int, p: Rate, metric: str = POW, estimator: Estimator = estimate
) -> dict[Fraction, Fraction]:
    """The exact law of an estimator on one task: value -> probability, in order of value."""
    out: dict[Fraction, Fraction] = {}
    for c, mass in enumerate(binomial(n, p)):
        if mass:
            value = estimator(c, n, k, metric)
            out[value] = out.get(value, Fraction(0)) + mass
    return dict(sorted(out.items()))


def moments(dist: dict[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    """Mean and variance of a law."""
    mean = sum((v * m for v, m in dist.items()), Fraction(0))
    return mean, sum((v * v * m for v, m in dist.items()), Fraction(0)) - mean**2


def variance(n: int, k: int, p: Rate, metric: str = POW) -> Fraction:
    """The estimator's variance on one task, by Hoeffding's formula for a U-statistic.

    Two k-subsets of the n trials that share j trials have covariance q^(2k-j) - q^(2k), and
    C(k,j) C(n-k,k-j) / C(n,k) of the pairs share j.
    """
    _check(n, k, metric)
    q = rate(p) if metric == POW else 1 - rate(p)
    return sum(
        (
            Fraction(math.comb(k, j) * math.comb(n - k, k - j), math.comb(n, k))
            * (q ** (2 * k - j) - q ** (2 * k))
            for j in range(1, k + 1)
        ),
        Fraction(0),
    )


def plugin_bias(n: int, k: int, p: Rate, metric: str = POW) -> Fraction:
    """Expected plug-in estimate minus the truth: above zero for pass^k, below for pass@k."""
    return moments(law(n, k, p, metric, plugin))[0] - truth(p, k, metric)


def variance_estimate(c: int, n: int, k: int, metric: str = POW) -> Fraction:
    """An unbiased estimate of `variance` from the task's own trials, which needs 2k <= n.

    The variance is E[U^2] - p^(2k), and p^(2k) has an unbiased estimator only when 2k <= n. For
    2k > n the variance is a polynomial of degree 2k > n in p, so nothing estimates it unbiasedly.
    """
    _check(n, k, metric)
    if 2 * k > n:
        raise DataError(f"no unbiased variance of {metric} at k={k} from n={n} < 2k trials")
    wins = c if metric == POW else n - c
    return estimate(wins, n, k) ** 2 - estimate(wins, n, 2 * k)


@cache
def worst_variance(n: int, k: int) -> tuple[float, float]:
    """The largest one-task variance over p, and the pass^k p that attains it (pass@k: 1 - p).

    Found numerically (a 2,001-point grid, then a bounded search between its neighbours).
    """
    _check(n, k, POW)
    weights = [
        (2 * k - j, math.comb(k, j) * math.comb(n - k, k - j) / math.comb(n, k))
        for j in range(max(1, 2 * k - n), k + 1)
    ]

    def value(p: Any) -> Any:
        return sum(w * (p**power - p ** (2 * k)) for power, w in weights)

    grid = np.linspace(0.0, 1.0, 2001)
    values = value(grid)
    i = int(np.argmax(values))
    found = optimize.minimize_scalar(
        lambda p: -value(p),
        bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 2000)]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if -found.fun >= values[i]:
        return float(-found.fun), float(found.x)
    return float(values[i]), float(grid[i])


def task_interval(
    c: int, n: int, k: int, metric: str = POW, alpha: float = 0.05
) -> tuple[float, float]:
    """An interval for one task's pass^k or pass@k with coverage of at least 1 - alpha.

    Both are increasing in p, so the Clopper-Pearson limits for p map straight through. Unlike the
    unbiased estimator this needs no k <= n.
    """
    if metric not in METRICS or k < 1 or not 0 <= c <= n or n < 1:
        raise DataError(f"{c} successes in {n} trials at k={k}, {metric}")
    lo = 0.0 if c == 0 else float(stats.beta.ppf(alpha / 2, c, n - c + 1))
    hi = 1.0 if c == n else float(stats.beta.ppf(1 - alpha / 2, c + 1, n - c))
    if metric == POW:
        return lo**k, hi**k
    return 1 - (1 - lo) ** k, 1 - (1 - hi) ** k


def benchmark_law(
    tasks: Sequence[tuple[int, Rate]], k: int, metric: str = POW
) -> dict[Fraction, Fraction]:
    """The exact law of the benchmark estimate, the mean of the tasks' estimates.

    `tasks` holds (trials, success probability) per task; their laws are convolved. Exact
    arithmetic: the support grows with the tasks, so hundreds of tasks take seconds, not more.
    """
    if not tasks:
        raise DataError("no tasks")
    total = {Fraction(0): Fraction(1)}
    for n, p in tasks:
        fresh: dict[Fraction, Fraction] = {}
        for b, y in law(n, k, p, metric).items():
            for a, x in total.items():
                fresh[a + b] = fresh.get(a + b, Fraction(0)) + x * y
        total = fresh
    return {v / len(tasks): m for v, m in sorted(total.items())}


def coverage(dist: dict[Fraction, Fraction], centre: Fraction, half: float) -> Fraction:
    """The probability that an estimate with law `dist` lands within `half` of `centre`."""
    return sum((m for v, m in dist.items() if abs(v - centre) <= half), Fraction(0))


def half_width(variances: Sequence[float], alpha: float = 0.05, kind: str = "normal") -> float:
    """Half the width of an interval for the mean over these tasks, tasks held fixed.

    `variances` are the tasks' one-task variances, or upper bounds on them. `normal` is the usual
    z interval. `guaranteed` holds at any number of tasks: the smaller of Hoeffding's bound and
    Bernstein's with these variances, both fixed before the data are seen.
    """
    if kind not in KINDS:
        raise DataError(f"kind is one of {KINDS}, not {kind!r}")
    t, total = len(variances), float(sum(variances))
    if kind == "normal":
        return float(stats.norm.ppf(1 - alpha / 2)) * math.sqrt(total) / t
    log = math.log(2 / alpha)
    bernstein = log / (3 * t) + math.sqrt(log**2 / (9 * t * t) + 2 * log * total / (t * t))
    return min(bernstein, math.sqrt(log / (2 * t)))


def guaranteed_floor(tasks: int, alpha: float = 0.05) -> float:
    """The guaranteed half-width at zero variance: no number of trials goes below it."""
    return 2 * math.log(2 / alpha) / (3 * tasks)


def trials_needed(
    k: int,
    width: float,
    tasks: int,
    alpha: float = 0.05,
    kind: str = "normal",
    p: Rate | None = None,
    metric: str = POW,
    max_trials: int = MAX_TRIALS,
) -> int | None:
    """Fewest trials per task for an interval no wider than `width` end to end, tasks held fixed.

    At the worst p for every task, or at `p` if every task had that success probability. None if
    `max_trials` do not reach it. A U-statistic's variance never rises with n, so the first n that
    fits is the answer. For tasks like these rather than these tasks, the spread between tasks is
    added and no number of trials removes it.
    """
    if width <= 0 or tasks < 1:
        raise DataError("width and tasks must be positive")
    for n in range(k, max_trials + 1):
        v = worst_variance(n, k)[0] if p is None else float(variance(n, k, p, metric))
        if 2 * half_width([v] * tasks, alpha, kind) <= width:
            return n
    return None


def benchmark(
    counts: Iterable[tuple[int, int]], k: int, metric: str = POW, alpha: float = 0.05
) -> dict[str, Any]:
    """A benchmark's estimate and intervals from (successes, trials) per task.

    Tasks with fewer than k trials cannot be scored and are counted in `dropped`. `se` is the
    spread between tasks over root tasks: right for tasks like these, and too large on average
    for these tasks, since it also counts how tasks differ. `se_fixed` is for these tasks, from
    the unbiased one-task variance estimates, and exists only when every task has 2k <= n.
    `guaranteed` is the fixed-tasks interval of `half_width` at each task's worst-case variance.
    """
    counts = list(counts)
    used = [(c, n) for c, n in counts if n >= k]
    if len(used) < 2:
        raise DataError(f"fewer than two tasks have at least k={k} trials")
    values = [estimate(c, n, k, metric) for c, n in used]
    t = len(used)
    point = float(sum(values, Fraction(0)) / t)
    se = float(np.std([float(v) for v in values], ddof=1)) / math.sqrt(t)
    half = float(stats.t.ppf(1 - alpha / 2, t - 1)) * se
    sure = half_width([worst_variance(n, k)[0] for _, n in used], alpha, "guaranteed")
    fixed = None
    if all(2 * k <= n for _, n in used):
        total = sum((variance_estimate(c, n, k, metric) for c, n in used), Fraction(0))
        fixed = math.sqrt(float(total)) / t
    return {
        "tasks": t,
        "dropped": len(counts) - t,
        "estimate": point,
        "plugin": float(sum((plugin(c, n, k, metric) for c, n in used), Fraction(0)) / t),
        "se": se,
        "half": half,
        "interval": (max(point - half, 0.0), min(point + half, 1.0)),
        "se_fixed": fixed,
        "guaranteed_half": sure,
        "guaranteed": (max(point - sure, 0.0), min(point + sure, 1.0)),
    }


def reference_rows(
    trials: Sequence[int] = REFERENCE_TRIALS, tasks: int = 89, alpha: float = 0.05
) -> list[dict[str, Any]]:
    """One row per (n, k): the pass^k estimator's SD, the plug-in's bias, and benchmark widths."""
    rows = []
    for n in trials:
        for k in range(1, n + 1):
            worst, at = worst_variance(n, k)
            row: dict[str, Any] = {"trials": n, "k": k}
            for name, p in REFERENCE_RATES.items():
                row[f"truth_{name}"] = float(truth(p, k))
                row[f"sd_{name}"] = math.sqrt(variance(n, k, p))
                row[f"plugin_bias_{name}"] = float(plugin_bias(n, k, p))
            row |= {
                "sd_worst": math.sqrt(worst),
                "p_worst": at,
                "half_normal": half_width([worst] * tasks, alpha, "normal"),
                "half_guaranteed": half_width([worst] * tasks, alpha, "guaranteed"),
            }
            rows.append(row)
    return rows


def successes(trials: pd.DataFrame) -> pd.DataFrame:
    """Successes and trials per (cell, task).

    A trial succeeds when its reward is at least 1; a missing reward counts as a failed trial.
    """
    missing = {"cell", "task", "reward"} - set(trials.columns)
    if missing:
        raise DataError(f"missing columns {sorted(missing)}")
    won = trials.assign(won=trials["reward"] >= 1)
    out = won.groupby(["cell", "task"], sort=True)["won"].agg(c="sum", n="size").reset_index()
    return out.astype({"c": int, "n": int})


def cell_rows(trials: pd.DataFrame, max_k: int, alpha: float = 0.05) -> pd.DataFrame:
    """pass^k and pass@k per cell, for k from 1 to the fewest trials the cell ran on any task.

    So every row scores all of the cell's tasks; k stops at `max_k`. `_half` is the half-width of
    the interval for tasks like these, `_sure` the guaranteed one for these tasks (`benchmark`).
    """
    rows = []
    for cell, group in successes(trials).groupby("cell", sort=True):
        counts = list(zip(group["c"], group["n"], strict=True))
        for k in range(1, min(max_k, int(group["n"].min())) + 1):
            row: dict[str, Any] = {"cell": cell, "k": k, "tasks": len(counts)}
            for name, metric in (("pow", POW), ("at", AT)):
                b = benchmark(counts, k, metric, alpha)
                row |= {
                    f"{name}": b["estimate"],
                    f"{name}_plugin": b["plugin"],
                    f"{name}_se": b["se"],
                    f"{name}_lo": b["interval"][0],
                    f"{name}_hi": b["interval"][1],
                    f"{name}_half": b["half"],
                    f"{name}_se_fixed": b["se_fixed"],
                    f"{name}_sure": b["guaranteed_half"],
                }
            rows.append(row)
    return pd.DataFrame(rows)


def study(
    trials: pd.DataFrame, max_k: int = 5, tasks: int = 89
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The cells' pass^k and pass@k, the reference table, and the trials a width needs."""
    rows = cell_rows(trials, max_k)
    by_k = {}
    first = rows[rows["k"] == 1].set_index("cell")["pow"]
    for k, group in rows.groupby("k"):
        bias = group["pow_plugin"] - group["pow"]
        rho = stats.spearmanr(group["pow"], first.loc[group["cell"]]).statistic
        by_k[str(k)] = {
            "cells": len(group),
            "pow_median": float(group["pow"].median()),
            "pow_bias_median": float(bias.median()),
            "pow_bias_max": float(bias.max()),
            "at_bias_median": float((group["at_plugin"] - group["at"]).median()),
            "pow_half_median": float(group["pow_half"].median()),
            "pow_sure_median": float(group["pow_sure"].median()),
            "pow_se_fixed_median": None
            if group["pow_se_fixed"].isna().any()
            else float(group["pow_se_fixed"].median()),
            "spearman_vs_k1": float(rho) if k > 1 else 1.0,
            "plugin_spearman": float(stats.spearmanr(group["pow"], group["pow_plugin"]).statistic),
        }
    needs = [
        {
            "k": k,
            "width": w,
            "tasks": tasks,
            "normal": trials_needed(k, w, tasks),
            "guaranteed": trials_needed(k, w, tasks, kind="guaranteed"),
        }
        for k in (1, 2, 3, 5, 8)
        for w in (0.05, 0.10, 0.20)
    ]
    summary = {
        "max_k": max_k,
        "cells": int(rows["cell"].nunique()),
        "tasks": int(rows["tasks"].max()),
        "by_k": by_k,
        "reference": reference_rows(tasks=tasks),
        "reference_tasks": tasks,
        "needs": needs,
    }
    return rows, summary
