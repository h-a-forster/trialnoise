"""Leaderboard pairs: task-clustered and fixed-task tests, and what more trials or tasks change."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

from trialnoise.decompose import Layout, cell_noise, moments, pair_matrices


def scores(table: Layout) -> np.ndarray:
    """Task-equal pass rate: a task with fewer trials counts the same."""
    return table.mean.mean(axis=1)


def adjust(p: np.ndarray, method: str) -> np.ndarray:
    return multipletests(p, alpha=0.05, method=method)[1]


def two_sided(gap: float, se: float, tail: Callable[[float], float]) -> float:
    """p of a gap; with no noise at all, a gap is certain and no gap is no evidence."""
    return float(2 * tail(abs(gap) / se)) if se > 0 else float(gap == 0)


def fixed_task_se(table: Layout, a: int, b: int) -> float:
    """SE of a score gap when the tasks are the population: only trial noise remains."""
    within, _ = cell_noise(table)

    def variance(j: int) -> np.ndarray:
        n, ss = table.n[j], table.ss[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            s2 = np.where(n > 1, ss / (n - 1), within[j])
        return s2 / n

    return math.sqrt(float((variance(a) + variance(b)).sum())) / len(table.tasks)


def adjacent(table: Layout, order: list[int], universe: str) -> pd.DataFrame:
    """Tests of each neighbouring pair in `order`, Holm and BH across the pairs."""
    t = len(table.tasks)
    score = scores(table)
    var_d, _, _ = pair_matrices(table)
    rows = []
    for rank, (a, b) in enumerate(zip(order[:-1], order[1:], strict=True), start=1):
        gap = float(score[a] - score[b])
        se = math.sqrt(float(var_d[a, b]) / t)
        se_fixed = fixed_task_se(table, a, b)
        rows.append(
            {
                "universe": universe,
                "rank": rank,
                "upper": table.cells[a],
                "lower": table.cells[b],
                "score_upper": float(score[a]),
                "score_lower": float(score[b]),
                "gap": gap,
                "se_task": se,
                "p_task": two_sided(gap, se, lambda x: stats.t.sf(x, t - 1)),
                "se_fixed": se_fixed,
                "p_fixed": two_sided(gap, se_fixed, stats.norm.sf),
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_task_holm"] = adjust(frame["p_task"].to_numpy(), "holm")
    frame["p_task_bh"] = adjust(frame["p_task"].to_numpy(), "fdr_bh")
    frame["p_fixed_holm"] = adjust(frame["p_fixed"].to_numpy(), "holm")
    return frame


def best_per_scaffold(score: np.ndarray, scaffolds: list[str]) -> list[int]:
    """Each scaffold's top cell, best first."""
    seen: set[str] = set()
    out = []
    for j in np.argsort(-score, kind="stable"):
        if scaffolds[j] not in seen:
            seen.add(scaffolds[j])
            out.append(int(j))
    return out


def all_pairs(table: Layout, subset: list[int] | None = None) -> dict[str, float]:
    """Share of all pairs separated by the task-clustered test, by Holm and by BH."""
    keep = np.arange(len(table.cells)) if subset is None else np.asarray(subset)
    t = len(table.tasks)
    score = scores(table)[keep]
    var_d = pair_matrices(table)[0][np.ix_(keep, keep)]
    upper = np.triu_indices(len(keep), 1)
    gap = np.abs(score[:, None] - score[None, :])[upper]
    p = 2 * stats.t.sf(gap / np.sqrt(var_d[upper] / t), t - 1)
    holm, bh = adjust(p, "holm") < 0.05, adjust(p, "fdr_bh") < 0.05
    return {
        "pairs": len(p),
        "holm": int(holm.sum()),
        "bh": int(bh.sum()),
        "share_holm": float(holm.mean()),
        "share_bh": float(bh.mean()),
        "smallest_gap_bh": float(gap[bh].min()) if bh.any() else math.nan,
        "smallest_gap_holm": float(gap[holm].min()) if holm.any() else math.nan,
    }


def smoothed_truth(table: Layout) -> tuple[np.ndarray, dict[str, float]]:
    """Plausible true pass probabilities for each task and cell.

    Raw task means are 0 or 1 too often, which would remove trial noise from a simulation. Each is
    pulled towards the additive fit by a pseudo-count chosen so the mean Bernoulli variance equals
    the fitted trial variance.
    """
    y, n = table.mean, table.n
    fitted = np.clip(y.mean(axis=1, keepdims=True) + y.mean(axis=0, keepdims=True) - y.mean(), 0, 1)
    target = moments(table)["trial"]

    def truth(c: float) -> np.ndarray:
        return (y * n + c * fitted) / (n + c)

    def gap(log_c: float) -> float:
        p = truth(math.exp(log_c))
        return float((p * (1 - p)).mean()) - target

    c = math.exp(optimize.brentq(gap, -8.0, 12.0, xtol=1e-6))
    p = truth(c)
    resid = p - p.mean(axis=1, keepdims=True) - p.mean(axis=0, keepdims=True) + p.mean()
    a, t = p.shape
    interaction = float((resid * resid).sum()) / ((a - 1) * (t - 1))
    return p, {"pseudo_count": c, "task_cell_of_truth": interaction}


def _holm_rejections(p: np.ndarray) -> int:
    """Holm rejections at 0.05. A NaN is never rejected and does not count toward the family."""
    ordered = np.sort(p[~np.isnan(p)])
    m = len(ordered)
    hit = ordered * (m - np.arange(m)) > 0.05
    return int(np.argmax(hit)) if hit.any() else m


def _share_separated(x: np.ndarray, rows: np.ndarray, tasks: int) -> float:
    """Share of neighbouring `rows` of a simulated table that separate at Holm 0.05."""
    d = x[rows[:-1]] - x[rows[1:]]
    se = d.std(axis=1, ddof=1) / math.sqrt(tasks)
    mean = d.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(se > 0, np.abs(mean) / se, 0.0)
    return _holm_rejections(2 * stats.t.sf(stat, tasks - 1)) / len(d)


def simulate_board(
    truth: np.ndarray,
    scaffolds: list[str],
    *,
    trials: int,
    tasks: int,
    reps: int,
    seed: int,
) -> dict[str, float]:
    """Redraw the whole leaderboard, rank it, and test neighbours as the study does.

    With `tasks` other than the fitted number, tasks are drawn with replacement each time. The
    share is the mean over boards of the adjacent pairs separated at Holm 0.05, for all cells and
    for each scaffold's best cell.
    """
    rng = np.random.default_rng(seed)
    t = truth.shape[1]
    codes = pd.factorize(pd.Series(scaffolds))[0]
    shares_all, shares_best = [], []
    for _ in range(reps):
        p = truth if tasks == t else truth[:, rng.integers(0, t, tasks)]
        x = rng.binomial(trials, p) / trials
        order = np.argsort(-x.mean(axis=1), kind="stable")
        shares_all.append(_share_separated(x, order, tasks))
        _, first = np.unique(codes[order], return_index=True)
        shares_best.append(_share_separated(x, order[np.sort(first)], tasks))
    return {
        "trials": trials,
        "tasks": tasks,
        "reps": reps,
        "adjacent_all": float(np.mean(shares_all)),
        "adjacent_all_se": float(np.std(shares_all, ddof=1) / math.sqrt(reps)),
        "adjacent_best": float(np.mean(shares_best)),
        "adjacent_best_se": float(np.std(shares_best, ddof=1) / math.sqrt(reps)),
    }
