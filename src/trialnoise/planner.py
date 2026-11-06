"""The trials-and-tasks calculator: evalnoise's planner fed with the strata fitted here."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trialnoise._evalnoise.components import Components
from trialnoise._evalnoise.data import DataError
from trialnoise._evalnoise.power import Design, mde, power_at, resolve, se_difference, tasks_needed
from trialnoise.decompose import BANDS, COMPARISONS, NEAR_GAP

ALPHA = 0.05
POWER = 0.8
TARGETS = ("population", "fixed")
REFERENCE_GRID = ((89, 5), (89, 10), (150, 5), (150, 10))
TASK_GRID = (50, 89, 150, 300, 1000)
TRIAL_GRID = (1, 3, 5, 10, 20)
GAPS = (0.03, 0.05, 0.10)
MAX_TRIALS = 500


def components(stratum: dict[str, Any], target: str) -> Components:
    """Planning inputs for one stratum.

    `population`: the tasks are a sample of tasks like these, so task x cell counts as noise.
    `fixed`: the tasks are the population (a statement about this benchmark), so only trial noise
    counts.
    """
    if target not in TARGETS:
        raise DataError(f"target must be one of {TARGETS}")
    interaction = max(stratum["task_cell"], 0.0) if target == "population" else 0.0
    return Components(
        item=0.0, item_arm=interaction, run=0.0, item_run=0.0, within=stratum["trial"]
    )


def resolved(stratum: dict[str, Any], target: str, run_variance: float, run_df: float):
    return resolve(components(stratum, target), run=run_variance, run_df=run_df)


def trials_needed(r: Any, tasks: int, gap: float) -> int | None:
    """Fewest trials per task with an MDE of at most `gap`; None above `MAX_TRIALS`."""
    if mde(r, Design(tasks, 1, MAX_TRIALS), alpha=ALPHA, power=POWER) > gap:
        return None
    lo, hi = 1, MAX_TRIALS
    while lo < hi:
        mid = (lo + hi) // 2
        if mde(r, Design(tasks, 1, mid), alpha=ALPHA, power=POWER) <= gap:
            hi = mid
        else:
            lo = mid + 1
    return hi


def stratum_rows(
    strata: dict[str, Any], run: dict[str, float] | None
) -> tuple[list[dict], list[dict]]:
    """The MDE grid and the trials and tasks a target gap needs, per stratum and run assumption.

    With `run`, each row is repeated at its upper limit; that takes minutes, the rest seconds.
    """
    grid: list[dict] = []
    needs: list[dict] = []
    runs = [("none", 0.0, math.inf)]
    if run is not None:
        runs.append(("upper", run["upper"], run["df"]))
    for name, stratum in strata.items():
        for target in TARGETS:
            for run_label, run_variance, run_df in runs:
                r = resolved(stratum, target, run_variance, run_df)
                for tasks in TASK_GRID:
                    for trials in TRIAL_GRID:
                        d = Design(tasks, 1, trials)
                        grid.append(
                            {
                                "stratum": name,
                                "target": target,
                                "run": run_label,
                                "tasks": tasks,
                                "trials": trials,
                                "se": se_difference(r, d),
                                "mde": mde(r, d, alpha=ALPHA, power=POWER),
                            }
                        )
                for gap in GAPS:
                    needs.append(
                        {
                            "stratum": name,
                            "target": target,
                            "run": run_label,
                            "gap": gap,
                            "trials_at_89_tasks": trials_needed(r, 89, gap),
                            "tasks_at_trials": {
                                str(k): tasks_needed(r, gap, repeats=k, alpha=ALPHA, power=POWER)
                                for k in TRIAL_GRID
                            },
                        }
                    )
    return grid, needs


def heldout_cells(official: pd.DataFrame, skip: tuple[str, ...]) -> pd.DataFrame:
    """Trials of submissions the trajectory dataset never saw, kept to complete ones."""
    frame = official[
        (official["status"] == "ok")
        & official["reward"].notna()
        & ~official["submission"].isin(skip)
    ]
    counts = frame.groupby(["submission", "task"]).size().unstack("task")
    complete = counts[(counts.fillna(0) >= 3).sum(axis=1) >= 85].index
    return frame[frame["submission"].isin(complete)]


def calibrate(frame: pd.DataFrame, strata: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Planner-predicted against observed paired SE on near pairs of held-out submissions.

    Each pair is predicted at its own design: the harmonic mean of its trials per task, which is
    what the trial noise of a task's difference averages to.
    """
    grouped = frame.groupby(["submission", "task"])["reward"]
    means = grouped.mean().unstack("task")
    counts = grouped.size().unstack("task")
    meta = frame.drop_duplicates("submission").set_index("submission")
    names = list(means.index)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = (means.loc[a] - means.loc[b]).dropna()
            d = shared.to_numpy()
            score_a, score_b = float(means.loc[a].mean()), float(means.loc[b].mean())
            if abs(score_a - score_b) > NEAR_GAP or len(d) < 80:
                continue
            ma, mb = meta.loc[a, "model"], meta.loc[b, "model"]
            sa, sb = meta.loc[a, "scaffold"], meta.loc[b, "scaffold"]
            if sa == sb and ma == mb:
                continue
            comparison = "same-scaffold" if sa == sb else "neither"
            if ma == mb and ma != "multiple":
                comparison = "same-model"
            level = (score_a + score_b) / 2
            band = next(name for name, lo, hi in BANDS if lo <= level < hi)
            stratum = strata[f"{comparison}/{band}"]
            per_task = counts.loc[[a, b], shared.index]
            trials = 2 / float((1 / per_task).mean(axis=1).sum())
            predicted = se_difference(
                resolve(components(stratum, "population"), run=0.0), Design(len(d), 1, trials)
            )
            observed = float(d.std(ddof=1)) / math.sqrt(len(d))
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "comparison": comparison,
                    "band": band,
                    "tasks": len(d),
                    "trials": trials,
                    "gap": score_a - score_b,
                    "se_predicted": predicted,
                    "se_observed": observed,
                    "ratio": observed / predicted,
                }
            )
    table = pd.DataFrame(rows)
    ratio = table["ratio"].to_numpy()
    summary: dict[str, Any] = {
        "cells": len(names),
        "pairs": len(table),
        "median_ratio": float(np.median(ratio)),
        "ratio_q05": float(np.quantile(ratio, 0.05)),
        "ratio_q95": float(np.quantile(ratio, 0.95)),
        "within_20pct": float(np.mean(np.abs(ratio - 1) <= 0.2)),
        "by_comparison": {
            c: {
                "pairs": int((table["comparison"] == c).sum()),
                "median_ratio": float(table.loc[table["comparison"] == c, "ratio"].median()),
            }
            for c in COMPARISONS
            if (table["comparison"] == c).any()
        },
        "by_band": {
            b: {
                "pairs": int((table["band"] == b).sum()),
                "median_ratio": float(table.loc[table["band"] == b, "ratio"].median()),
            }
            for b, _, _ in BANDS
            if (table["band"] == b).any()
        },
    }
    return table, summary


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DataError(f"{path} not found; run `trialnoise study`")
    return json.loads(path.read_text(encoding="utf-8"))


def answer(
    planner: dict[str, Any],
    *,
    comparison: str,
    band: str,
    target: str,
    tasks: int,
    trials: int,
    gap: float | None,
    run: str,
) -> dict[str, Any]:
    """One design under one stratum: its SE and MDE, and what `gap` would need."""
    name = f"{comparison}/{band}"
    if name not in planner["strata"]:
        raise DataError(f"unknown stratum {name}; choose from {sorted(planner['strata'])}")
    if run not in ("none", "upper"):
        raise DataError("run must be `none` or `upper`")
    variance = planner["run"]["upper"] if run == "upper" else 0.0
    df = planner["run"]["df"] if run == "upper" else math.inf
    r = resolved(planner["strata"][name], target, variance, df)
    d = Design(tasks, 1, trials)
    out: dict[str, Any] = {
        "stratum": name,
        "target": target,
        "run": run,
        "tasks": tasks,
        "trials": trials,
        "se": se_difference(r, d),
        "mde": mde(r, d, alpha=ALPHA, power=POWER),
    }
    if gap is not None:
        out["gap"] = gap
        out["power_at_gap"] = power_at(r, d, gap, alpha=ALPHA)
        out["tasks_needed_at_these_trials"] = tasks_needed(
            r, gap, repeats=trials, alpha=ALPHA, power=POWER
        )
        out["trials_needed_at_these_tasks"] = trials_needed(r, tasks, gap)
    return out
