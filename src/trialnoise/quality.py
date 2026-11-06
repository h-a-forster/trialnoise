"""Infrastructure errors, cost per solved task, and the replicate-noise integrity screen."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from trialnoise.decompose import percentile

TIMEOUT = "AgentTimeoutError"


def infra_table(official: pd.DataFrame, totals: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Per submission: outcomes lost before a verdict, timeouts, and the score both ways.

    The leaderboard's mean counts a trial with no verdict as zero (`score_zero`); `score_verdict`
    drops those trials. A timeout with a verdict is the task's own limit, scored as usual.
    """
    rows = []
    for submission, group in official.groupby("submission", sort=True):
        n_total = totals[submission]["n_total"]
        verdicts = int(group["reward"].notna().sum())
        passes = int(group["reward"].sum())
        rows.append(
            {
                "submission": submission,
                "scaffold": group["scaffold"].iloc[0],
                "model": group["model"].iloc[0],
                "status": group["status"].iloc[0],
                "n_total": n_total,
                "verdicts": verdicts,
                "no_verdict": n_total - verdicts,
                "no_verdict_rate": (n_total - verdicts) / n_total,
                "timeout_rate": float((group["exception"] == TIMEOUT).sum()) / n_total,
                "score_zero": passes / n_total,
                "score_verdict": passes / verdicts,
            }
        )
    return pd.DataFrame(rows)


def infra_summary(table: pd.DataFrame, official: pd.DataFrame) -> dict[str, Any]:
    """How much lost verdicts move scores, over complete submissions in good standing."""
    full = table[(table["status"] == "ok") & (table["n_total"] >= 400)]
    full = full[full["verdicts"] >= 0.8 * full["n_total"]]
    loss = (full["score_verdict"] - full["score_zero"]).to_numpy()
    score = full["score_zero"].to_numpy()
    variance = float(np.var(score, ddof=1))
    verdict_variance = float(np.var(full["score_verdict"].to_numpy(), ddof=1))
    rho = stats.spearmanr(score, full["score_verdict"].to_numpy()).statistic
    by_type = (
        official[official["submission"].isin(full["submission"])]
        .assign(exception=lambda f: f["exception"].fillna("none"))
        .groupby("exception")
        .agg(
            trials=("reward", "size"),
            verdicts=("reward", "count"),
            passes=("reward", "sum"),
        )
        .sort_values("trials", ascending=False)
    )
    return {
        "submissions": len(full),
        "trials": int(full["n_total"].sum()),
        "no_verdict_rate": float(full["no_verdict"].sum() / full["n_total"].sum()),
        "no_verdict_rate_median": float(full["no_verdict_rate"].median()),
        "no_verdict_rate_max": float(full["no_verdict_rate"].max()),
        "timeout_rate": float(
            (full["timeout_rate"] * full["n_total"]).sum() / full["n_total"].sum()
        ),
        "loss_median": float(np.median(loss)),
        "loss_max": float(loss.max()),
        "loss_sd": float(np.std(loss, ddof=1)),
        "share_of_score_variance": float(np.var(loss, ddof=1) / variance),
        "variance_score_zero": variance,
        "variance_score_verdict": verdict_variance,
        "spearman": float(rho),
        "by_exception": {
            str(k): {key: int(v[key]) for key in ("trials", "verdicts", "passes")}
            for k, v in by_type.iterrows()
        },
    }


def cost_table(
    trials: pd.DataFrame, *, draws: int, seed: int, min_coverage: float = 0.9
) -> pd.DataFrame:
    """Dollars per pass with a task-clustered interval, for cells that report cost.

    A pass costs (sum of trial costs) / (passes), both over the trials with a cost; failed trials
    are paid for. `coverage` is the share of a cell's trials with a cost.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for cell, group in trials.groupby("cell", sort=True):
        known = group[group["cost_cents"].notna()]
        coverage = len(known) / len(group)
        if coverage < min_coverage:
            continue
        by_task = known.groupby("task").agg(cost=("cost_cents", "sum"), passes=("reward", "sum"))
        cost, passes = by_task["cost"].to_numpy() / 100, by_task["passes"].to_numpy(float)
        picks = rng.integers(0, len(by_task), (draws, len(by_task)))
        total_cost, total_pass = cost[picks].sum(axis=1), passes[picks].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = total_cost / total_pass
        ratio = np.where(total_pass > 0, ratio, np.inf)
        lo, hi = np.quantile(ratio, [0.025, 0.975])
        rows.append(
            {
                "cell": cell,
                "trials": len(group),
                "coverage": coverage,
                "pass_rate": float(known["reward"].mean()),
                "cost_per_trial": float(cost.sum() / len(known)),
                "cost_per_pass": float(cost.sum() / passes.sum()) if passes.sum() else math.inf,
                "lo": float(lo),
                "hi": float(hi),
            }
        )
    return pd.DataFrame(rows)


def mixed_task_table(trials: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """How many tasks each submission both passed and failed across its trials.

    Linked cells use the official table, which keeps every job; the rest use the trial table.
    """
    linked = official[official["reward"].notna()]
    a = linked.rename(columns={"submission": "unit"})[["unit", "task", "reward", "status"]]
    rest = trials[trials["submission"].isna()]
    b = pd.DataFrame(
        {"unit": rest["cell"], "task": rest["task"], "reward": rest["reward"], "status": "ok"}
    )
    both = pd.concat([a, b], ignore_index=True)
    g = both.groupby(["unit", "task"])["reward"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= 4]
    g["mixed"] = (g["mean"] > 0) & (g["mean"] < 1)
    out = g.groupby("unit").agg(tasks=("task", "size"), mixed=("mixed", "sum")).reset_index()
    out = out[out["tasks"] >= 80]
    pass_rate = both.groupby("unit")["reward"].mean()
    status = both.drop_duplicates("unit").set_index("unit")["status"]
    out["pass_rate"] = out["unit"].map(pass_rate)
    out["status"] = out["unit"].map(status)
    out["mixed_share"] = out["mixed"] / out["tasks"]
    return out


def integrity_screen(table: pd.DataFrame, threshold: float = -3.0) -> pd.DataFrame:
    """Robust fit of the mixed-task share on pass rate; a large negative residual is flagged.

    Real replicate noise leaves many tasks passed by some trials and failed by others. Almost none
    means trials that do not vary: copied results, or something that fixes the outcome.
    """
    p = table["pass_rate"].to_numpy()
    design = np.column_stack([np.ones_like(p), p, p * p])
    fit = sm.RLM(table["mixed_share"].to_numpy(), design, M=sm.robust.norms.HuberT()).fit()
    resid = table["mixed_share"].to_numpy() - design @ fit.params
    scale = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    out = table.assign(expected=design @ fit.params, z=resid / scale)
    out["flagged"] = out["z"] < threshold
    out["rank"] = out["z"].rank(method="first").astype(int)
    return out.sort_values("z", ignore_index=True)


__all__ = [
    "copied_submissions",
    "cost_table",
    "infra_summary",
    "infra_table",
    "integrity_screen",
    "mixed_task_table",
    "percentile",
]


def copied_submissions(official: pd.DataFrame) -> list[str]:
    """Submissions whose replicate jobs agree on every trial outcome of every task.

    Real agents vary from run to run. Identical outcomes across jobs mean the jobs do not hold
    independent trials, so these jobs cannot estimate noise. An agent that never passes, or never
    fails, repeats itself honestly, so it needs both outcomes to count.
    """
    out = []
    ok = official[official["reward"].notna()]
    for submission, group in ok.groupby("submission", sort=True):
        if group["job"].nunique() < 2:
            continue
        tasks = group.groupby("job")["task"].nunique()
        if (tasks >= 80).sum() < 2:
            continue
        spread = group.groupby("task")["reward"].agg(["min", "max"])
        if (spread["min"] == spread["max"]).all() and spread["max"].nunique() == 2:
            out.append(str(submission))
    return out
