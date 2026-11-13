"""Inspect eval logs as a trial table, and the noise and replicate plan they imply.

Inspect (`inspect_ai`) writes one log per task and model. Each sample epoch is a trial: the
sample is the item (named `task/sample id`), the cell is the model unless `cell` says otherwise,
and the log is the run. Logs are read with inspect_ai's own public API, an optional dependency
(the `inspect` extra).

Every trial that does not reach the table is counted under a reason:

- `errored`: the sample raised; with ``errors="zero"`` it scores 0 instead.
- `incomplete`: the sample had not finished when the log was written.
- `unscored`: no value from the chosen scorer, or a NaN.
- `superseded`: a retry of the same log (same Inspect `task_id`) holds the same sample epoch;
  the newest log's copy is kept, and a scored copy beats an errored one.
- `not_crossed`: the item lacks a kept trial in some cell; the estimators need every cell on
  every item.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trialnoise import decompose, planner, study
from trialnoise._evalnoise.data import DataError
from trialnoise._evalnoise.power import Design, mde, power_at, se_difference, tasks_needed

CELLS = ("model", "log")  # or `metadata:KEY`, a key of the metadata given to `eval()`
ERRORS = ("drop", "zero")
REASONS = ("errored", "incomplete", "unscored", "superseded", "not_crossed")
# Inspect's default string values (`value_to_float`), read strictly: anything else is an error.
STRINGS = {"c": 1.0, "p": 0.5, "i": 0.0, "n": 0.0, "yes": 1.0, "no": 0.0, "true": 1.0}
STRINGS["false"] = 0.0


@dataclass
class Reading:
    """The trial table read from a set of logs, and what was left out."""

    trials: pd.DataFrame  # cell, task, run, epoch, reward; crossed
    scorer: str
    scorers: tuple[str, ...]
    logs: int
    statuses: dict[str, int]
    read: int
    excluded: dict[str, int] = field(default_factory=dict)

    @property
    def unit_scale(self) -> bool:
        """Scores in [0, 1] are reported in points (x 100)."""
        r = self.trials["reward"]
        return bool(((r >= 0) & (r <= 1)).all())


def _require() -> Any:
    try:
        import inspect_ai.log as log
    except ImportError as err:
        raise DataError(
            "reading Inspect logs needs inspect_ai: install trialnoise with its `inspect` extra"
        ) from err
    return log


def log_files(paths: Iterable[Path | str]) -> list[str]:
    """Log files named directly, plus every log under a named folder (recursively)."""
    log = _require()
    out: list[str] = []
    for path in map(Path, paths):
        if path.is_dir():
            found = [info.name for info in log.list_eval_logs(str(path), recursive=True)]
            if not found:
                raise DataError(f"{path}: no Inspect logs (.eval or .json) in this folder")
            out += sorted(found)
        elif path.is_file():
            out.append(str(path))
        else:
            raise DataError(f"{path}: no such file or folder")
    return list(dict.fromkeys(out))


def score_value(value: Any, scorer: str) -> float:
    """One Inspect score value as a number; NaN means no score."""
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value) if math.isfinite(value) else math.nan
    if isinstance(value, str):
        key = value.strip().lower()
        if key in STRINGS:
            return STRINGS[key]
        try:
            number = float(key)
        except ValueError:
            number = math.nan
        if math.isfinite(number):
            return number
    raise DataError(f"scorer {scorer}: value {value!r} is not a number, bool or C/P/I/N")


def flat_scores(scores: dict[str, Any] | None) -> dict[str, Any]:
    """Scorer name to value; a scorer with a dict value gives one `name:key` entry per key."""
    out: dict[str, Any] = {}
    for name, score in (scores or {}).items():
        value = score.value
        if isinstance(value, dict):
            out.update({f"{name}:{key}": v for key, v in value.items()})
        else:
            out[name] = value
    return out


def _cell(header: Any, cell: str, path: str) -> str:
    if cell == "model":
        return str(header.eval.model)
    if cell == "log":
        return f"{header.eval.model}@{header.eval.task_id}"
    if cell.startswith("metadata:"):
        key = cell.split(":", 1)[1]
        meta = header.eval.metadata or {}
        if key not in meta:
            raise DataError(f"{Path(path).name}: eval metadata has no key {key!r}")
        return str(meta[key])
    raise DataError(f"cell must be one of {CELLS} or metadata:KEY")


def read_samples(files: Sequence[str], cell: str = "model") -> pd.DataFrame:
    """One row per sample epoch in the logs, scores still raw."""
    log = _require()
    rows = []
    for path in files:
        header = log.read_eval_log(path, header_only=True)
        meta = {
            "cell": _cell(header, cell, path),
            "run": str(header.eval.task_id),
            "created": str(header.eval.created),
            "log": Path(path).name,
            "status": str(header.status),
            "inspect_task": str(header.eval.task),
        }
        for s in log.read_eval_log_sample_summaries(path):
            rows.append(
                {
                    **meta,
                    "task": f"{header.eval.task}/{s.id}",
                    "epoch": int(s.epoch),
                    "errored": s.error is not None,
                    "completed": bool(s.completed),
                    "scores": flat_scores(s.scores),
                }
            )
    if not rows:
        raise DataError("the logs hold no samples")
    return pd.DataFrame(rows)


def read(
    paths: Iterable[Path | str],
    *,
    scorer: str | None = None,
    cell: str = "model",
    errors: str = "drop",
) -> Reading:
    """The crossed trial table of `paths`, with each excluded trial counted under its reason."""
    if errors not in ERRORS:
        raise DataError(f"errors must be one of {ERRORS}")
    files = log_files(paths)
    samples = read_samples(files, cell)
    scorers = tuple(sorted({k for s in samples["scores"] for k in s}))
    if scorer is None:
        if len(scorers) != 1:
            raise DataError(f"choose a scorer with --scorer: the logs have {list(scorers)}")
        scorer = scorers[0]
    elif scorer not in scorers:
        raise DataError(f"no scorer {scorer!r} in the logs; they have {list(scorers)}")
    samples["reward"] = [score_value(s.get(scorer), scorer) for s in samples["scores"]]
    excluded: Counter[str] = Counter()

    # Retries: one copy per sample epoch of a log's task_id, scored copies first, newest log.
    statuses = Counter(samples.drop_duplicates("log")["status"])
    samples["scored"] = samples["completed"] & ~samples["errored"] & samples["reward"].notna()
    samples = samples.sort_values(["scored", "created", "log"], ascending=[False, False, False])
    duplicate = samples.duplicated(["cell", "run", "task", "epoch"])
    excluded["superseded"] = int(duplicate.sum())
    samples = samples[~duplicate]

    incomplete = ~samples["completed"]
    excluded["incomplete"] = int(incomplete.sum())
    samples = samples[~incomplete]
    if errors == "zero":
        samples = samples.assign(reward=samples["reward"].mask(samples["errored"], 0.0))
    else:
        excluded["errored"] = int(samples["errored"].sum())
        samples = samples[~samples["errored"]]
    unscored = samples["reward"].isna()
    excluded["unscored"] = int(unscored.sum())
    samples = samples[~unscored]

    cells = samples["cell"].nunique()
    per_item = samples.groupby("task")["cell"].nunique()
    crossed = samples["task"].isin(per_item[per_item == cells].index)
    excluded["not_crossed"] = int((~crossed).sum())
    samples = samples[crossed]
    if samples.empty:
        raise DataError(f"no trial is left: excluded {dict(excluded)}")

    trials = samples[["cell", "task", "run", "epoch", "reward", "inspect_task"]]
    trials = trials.sort_values(["cell", "task", "run", "epoch"]).reset_index(drop=True)
    return Reading(
        trials,
        scorer,
        scorers,
        len(files),
        dict(sorted(statuses.items())),
        int(len(trials) + sum(excluded.values())),
        {k: excluded[k] for k in REASONS},
    )


def _one_cell(table: decompose.Layout) -> dict[str, float]:
    """Item and trial variance of a single cell (one-way ANOVA moments)."""
    df = float((table.n - 1).sum())
    if df <= 0:
        raise DataError("no item has two trials: trial noise is not identified")
    trial = float(table.ss.sum()) / df
    task = float(np.var(table.mean, ddof=1)) - trial * float((1.0 / table.n).mean())
    return {"task": task, "trial": trial, "grand": float(table.mean.mean())}


def _run(reading: Reading, unit: bool) -> dict[str, Any]:
    """Pooled run variance over cells with two or more logs on every item."""
    t = reading.trials
    covering = t.groupby(["cell", "run"])["task"].nunique() == t["task"].nunique()
    if covering.groupby(level="cell").sum().max() < 2:
        return {"identified": False, "reason": "no cell has two logs covering every item"}
    frame = pd.DataFrame(
        {"submission": t["cell"], "job": t["run"], "task": t["task"], "reward": t["reward"]}
    ).assign(status="ok")
    try:
        out = decompose.run_term(
            frame, min_tasks=t["task"].nunique(), max_var=0.25 if unit else math.inf
        )
    except DataError as err:
        return {"identified": False, "reason": str(err)}
    return {"identified": True, **{k: v for k, v in out.items() if k != "submissions"}}


def analyse(
    reading: Reading,
    *,
    gap: float,
    k: int | None = None,
    bootstrap: int = 2000,
    seed: int = study.SEED,
    run: str = "none",
) -> dict[str, Any]:
    """Variance components of the table and the design `gap` needs, for a paired comparison.

    `gap` is on the score's scale (a fraction for scores in [0, 1]). The plan treats a new pair
    of cells as having this table's pooled task x cell and trial variance; `population` treats
    the items as a sample of items like these, `fixed` asks about these items only.
    """
    if not gap > 0:
        raise DataError("the gap must be positive")
    if run not in ("none", "upper"):
        raise DataError("run must be `none` or `upper`")
    table = decompose.layout(reading.trials)
    tasks = len(table.tasks)
    trials = float(1 / (1.0 / table.n).mean())  # what the noise of an item mean averages to
    k = k or max(1, round(trials))
    unit = reading.unit_scale
    out: dict[str, Any] = {
        "cells": list(table.cells),
        "tasks": tasks,
        "trials": len(reading.trials),
        "trials_per_task": trials,
        "mean": float(table.mean.mean()),
        "bootstrap": {"draws": bootstrap, "seed": seed, "resampled": "items"},
    }
    if len(table.cells) == 1:
        point = _one_cell(table)
        names: tuple[str, ...] = ("task", "trial")
        draws = decompose.bootstrap_draws(
            table, lambda x: np.array([_one_cell(x)[n] for n in names]), draws=bootstrap, seed=seed
        )
        out["components"] = {
            n: {"value": point[n], "interval": list(decompose.percentile(draws[:, j]))}
            for j, n in enumerate(names)
        }
        stratum = {"task_cell": math.nan, "trial": point["trial"]}
        upper = {"task_cell": math.nan, "trial": decompose.percentile(draws[:, 1])[1]}
    else:
        point = decompose.moments(table)
        stats = decompose.statistics(point, k)
        values = {**{n: point[n] for n in decompose.TERMS}, **stats}
        draws = decompose.bootstrap_draws(
            table, lambda x: decompose.component_vector(x, k), draws=bootstrap, seed=seed
        )
        out["components"] = {
            n: {"value": values[n], "interval": list(decompose.percentile(draws[:, j]))}
            for j, n in enumerate(decompose.VECTOR_NAMES)
        }
        stratum = {"task_cell": point["task_cell"], "trial": point["trial"]}
        upper = {
            n: decompose.percentile(draws[:, decompose.VECTOR_NAMES.index(n)])[1]
            for n in ("task_cell", "trial")
        }
    out["run"] = _run(reading, unit)
    if run == "upper" and not out["run"]["identified"]:
        raise DataError(f"--run upper needs a run term: {out['run']['reason']}")
    run_variance = out["run"]["interval"][1] if run == "upper" else 0.0
    run_df = out["run"]["df"] if run == "upper" else math.inf
    out["plan"] = {
        "gap": gap,
        "alpha": planner.ALPHA,
        "power": planner.POWER,
        "run": run,
        "k": k,
        "estimates": {"point": stratum, "upper": upper},
        "rows": [
            plan_row(values, target, gap, tasks, k, run_variance, run_df, basis)
            for basis, values in (("point", stratum), ("upper", upper))
            for target in planner.TARGETS
        ],
    }
    return out


def plan_row(
    stratum: dict[str, float],
    target: str,
    gap: float,
    tasks: int,
    k: int,
    run_variance: float,
    run_df: float,
    basis: str,
) -> dict[str, Any]:
    """One target's design for `gap`; None where it is not identified or not reachable."""
    row: dict[str, Any] = {"basis": basis, "target": target}
    if target == "population" and not math.isfinite(stratum["task_cell"]):
        return {**row, "identified": False, "reason": "task x cell needs two cells"}
    r = planner.resolved(stratum, target, run_variance, run_df)
    d = Design(tasks, 1, k)
    return {
        **row,
        "identified": True,
        "se": se_difference(r, d),
        "mde": mde(r, d, alpha=planner.ALPHA, power=planner.POWER),
        "power_at_gap": power_at(r, d, gap, alpha=planner.ALPHA),
        "trials_needed_at_these_tasks": planner.trials_needed(r, tasks, gap),
        "tasks_needed": {
            str(t): tasks_needed(r, gap, repeats=t, alpha=planner.ALPHA, power=planner.POWER)
            for t in sorted({*planner.TRIAL_GRID, k})
        },
    }


def render(reading: Reading, result: dict[str, Any]) -> str:
    """The decomposition and the plan as text."""
    unit = reading.unit_scale
    scale, units = (100.0, "points") if unit else (1.0, "score units")
    excluded = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in reading.excluded.items() if v)
    statuses = ", ".join(f"{v} {k}" for k, v in reading.statuses.items())
    lines = [
        f"{reading.logs} Inspect logs ({statuses}); scorer {reading.scorer}"
        + (f" of {', '.join(reading.scorers)}" if len(reading.scorers) > 1 else ""),
        f"{reading.read} trials read, {len(reading.trials)} kept"
        + (f"; excluded: {excluded}" if excluded else ""),
        f"{len(result['cells'])} cells x {result['tasks']} items, "
        f"{result['trials_per_task']:.3g} trials per item (harmonic mean), "
        f"mean score {result['mean'] * scale:.4g} {units}",
        "",
        f"Variance of one trial, 95% intervals from {result['bootstrap']['draws']} item "
        f"resamples (seed {result['bootstrap']['seed']}):",
    ]
    for name, c in result["components"].items():
        lo, hi = c["interval"]
        lines.append(f"  {name:16s} {c['value']:10.5f}  [{lo:.5f}, {hi:.5f}]")
    run = result["run"]
    if run["identified"]:
        lo, hi = run["interval"]
        lines.append(
            f"  run              {run['variance']:10.5f}  [{lo:.5f}, {hi:.5f}] on {run['df']} df "
            f"(upper SD {math.sqrt(hi) * scale:.3g} {units})"
        )
    else:
        lines.append(f"  run              not identified: {run['reason']}")
    plan = result["plan"]
    lines += [
        "",
        f"Paired comparison of two cells like these, gap {plan['gap'] * scale:.4g} {units}, "
        f"{plan['alpha']:.0%} two-sided, {plan['power']:.0%} power"
        + (", run term at its upper limit" if plan["run"] == "upper" else "")
        + ":",
    ]
    targets = {"population": "items like these", "fixed": "these items only"}
    bases = {"point": "estimates", "upper": "95% upper limits (conservative)"}
    for row in plan["rows"]:
        head = f"  {targets[row['target']]}, at the {bases[row['basis']]}:"
        if not row["identified"]:
            lines.append(f"{head} not identified ({row['reason']})")
            continue
        needed = row["trials_needed_at_these_tasks"]
        tasks = ", ".join(
            f"{v if v is not None else '-'} at {t}" for t, v in row["tasks_needed"].items()
        )
        lines += [
            head,
            f"    now ({result['tasks']} items x {plan['k']} trials): MDE "
            f"{row['mde'] * scale:.3g} {units}, power at the gap {row['power_at_gap']:.2f}",
            f"    trials per item at {result['tasks']} items: "
            + (str(needed) if needed is not None else f"none up to {planner.MAX_TRIALS}"),
            f"    items needed at k trials per item: {tasks}",
        ]
    return "\n".join(lines)
