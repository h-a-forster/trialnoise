"""Monte Carlo checks of the estimators and the planner, from the fitted components.

Pass/fail data come from a logistic-normal model whose probability-scale components match the fit.
Each check is one unit written atomically to `<units>/<name>.json`, skipped on rerun while its key
(schema, sims, inputs) is unchanged, and pooled into `results/v1/validation.json`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats

from trialnoise._evalnoise.components import Components
from trialnoise._evalnoise.power import Design, _df, mde, resolve
from trialnoise.decompose import (
    TERMS,
    Layout,
    bootstrap_draws,
    component_vector,
    moments,
    percentile,
)
from trialnoise.planner import ALPHA, POWER, components

SCHEMA = 1
SEED = 20261106
PROFILES = {
    "full": {"sims": 4000, "recovery": 300, "coverage": 300, "draws": 300},
    "quick": {"sims": 400, "recovery": 60, "coverage": 60, "draws": 100},
}
NODES = 21


@dataclass(frozen=True)
class Latent:
    """Logit-scale model `eta = mu + s_task z_t + s_cell z_a + s_inter z_ta`."""

    mu: float
    task: float
    cell: float
    inter: float

    def truth(self) -> dict[str, float]:
        """Probability-scale components, by Gauss-Hermite quadrature."""
        z, w = np.polynomial.hermite_e.hermegauss(NODES)
        w = w / w.sum()
        eta = (
            self.mu
            + self.task * z[:, None, None]
            + self.cell * z[None, :, None]
            + self.inter * z[None, None, :]
        )
        p = 1 / (1 + np.exp(-eta))
        www = w[:, None, None] * w[None, :, None] * w[None, None, :]
        mean = float((p * www).sum())
        by_task = (p * www).sum(axis=(1, 2)) / w
        by_cell = (p * www).sum(axis=(0, 2)) / w
        var_task = float((w * (by_task - mean) ** 2).sum())
        var_cell = float((w * (by_cell - mean) ** 2).sum())
        variance = float((p * p * www).sum()) - mean**2
        return {
            "task": var_task,
            "cell": var_cell,
            "task_cell": variance - var_task - var_cell,
            "trial": float((p * (1 - p) * www).sum()),
            "grand": mean,
        }


def fit_latent(target: dict[str, float]) -> Latent:
    """Match pass rate, task, cell and task x cell variance on the probability scale."""

    def residual(x: np.ndarray) -> np.ndarray:
        t = Latent(x[0], math.exp(x[1]), math.exp(x[2]), math.exp(x[3])).truth()
        return np.array(
            [t["grand"] - target["grand"]]
            + [(t[k] - target[k]) / target[k] for k in ("task", "cell", "task_cell")]
        )

    fit = optimize.least_squares(residual, np.array([-0.3, 0.8, 0.5, 0.5]), xtol=1e-10)
    return Latent(fit.x[0], *np.exp(fit.x[1:]).tolist())


def fit_pair(pass_rate: float, task_cell: float, trial: float) -> Latent:
    """Two-arm model: match a stratum's pass rate, interaction and trial variance."""

    def residual(x: np.ndarray) -> np.ndarray:
        t = Latent(x[0], math.exp(x[1]), 1e-6, math.exp(x[2])).truth()
        return np.array(
            [
                t["grand"] - pass_rate,
                (t["task_cell"] - task_cell) / task_cell,
                (t["trial"] - trial) / trial,
            ]
        )

    fit = optimize.least_squares(residual, np.array([0.5, 0.8, 0.5]), xtol=1e-10)
    return Latent(fit.x[0], math.exp(fit.x[1]), 0.0, math.exp(fit.x[2]))


def draw_table(
    latent: Latent, cells: int, tasks: int, trials: int, rng: np.random.Generator
) -> Layout:
    """One balanced table of Binomial(trials) outcomes."""
    eta = (
        latent.mu
        + latent.task * rng.standard_normal((1, tasks))
        + latent.cell * rng.standard_normal((cells, 1))
        + latent.inter * rng.standard_normal((cells, tasks))
    )
    x = rng.binomial(trials, 1 / (1 + np.exp(-eta))).astype(float)
    return Layout(
        tuple(str(j) for j in range(cells)),
        tuple(str(j) for j in range(tasks)),
        x / trials,
        np.full(x.shape, float(trials)),
        x * (1 - x / trials),
    )


def logit_shift(latent: Latent, gap: float, sign: int = 1) -> float:
    """Logit shift of one arm whose mean effect on the pass probability is `sign * gap`."""
    base = latent.truth()["grand"]

    def missed(s: float) -> float:
        moved = Latent(latent.mu + sign * s, latent.task, latent.cell, latent.inter)
        return sign * (moved.truth()["grand"] - base) - gap

    return sign * float(optimize.brentq(missed, 0.0, 8.0, xtol=1e-9))


def run_logit_sd(stratum: dict[str, Any], run_sd: float) -> float:
    """Logit SD whose effect on the mean pass probability has SD `run_sd`.

    Mean dp/deta over tasks is the mean Bernoulli variance, which is the stratum's trial variance.
    """
    return run_sd / float(stratum["trial"])


def pair_differences(
    latent: Latent,
    tasks: int,
    trials: int,
    shift: float,
    sims: int,
    rng: np.random.Generator,
    run_sd: float = 0.0,
) -> np.ndarray:
    """Per-task mean differences (arm 1 - arm 0), (sims, tasks)."""
    u = latent.task * rng.standard_normal((sims, 1, tasks))
    eta = latent.mu + u + latent.inter * rng.standard_normal((sims, 2, tasks))
    eta[:, 1, :] += shift
    if run_sd:
        eta = eta + run_sd * rng.standard_normal((sims, 2, 1))
    x = rng.binomial(trials, 1 / (1 + np.exp(-eta))) / trials
    return x[:, 1, :] - x[:, 0, :]


def _paired_t_reject(d_items: np.ndarray, alpha: float) -> np.ndarray:
    """Two-sided paired t-test over items; ``d_items`` is (sims, tasks)."""
    t = d_items.shape[1]
    mean = d_items.mean(axis=1)
    se = d_items.std(axis=1, ddof=1) / math.sqrt(t)
    tc = stats.t.ppf(1 - alpha / 2, t - 1)
    return np.abs(mean) > tc * se


def _item_differences(
    c: Components, d: Design, effect: float, sims: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-item mean differences (arm 1 - arm 0) for ``sims`` Gaussian experiments."""
    t, r, k = d.tasks, d.runs, d.repeats
    v = rng.normal(0, math.sqrt(c.item_arm), (sims, 2, t))
    w = rng.normal(0, math.sqrt(c.run), (sims, 2, 1, r))
    g = rng.normal(0, math.sqrt(c.item_run), (sims, 2, t, r))
    e = rng.normal(0, math.sqrt(c.within), (sims, 2, t, r, k)).mean(axis=4)
    y = v + (w + g + e).mean(axis=3)  # item means over runs, per arm
    y[:, 1, :] += effect
    return y[:, 1, :] - y[:, 0, :]  # item main effects cancel in a paired design


def mc_interval(hits: int, n: int) -> list[float]:
    """Clopper-Pearson 95% interval for a Monte Carlo rate."""
    lo = stats.beta.ppf(0.025, hits, n - hits + 1) if hits > 0 else 0.0
    hi = stats.beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0
    return [float(lo), float(hi)]


def _rate(hits: int, n: int) -> dict[str, Any]:
    return {"rate": hits / n, "n": n, "interval": mc_interval(hits, n)}


def check_recovery(fit: dict[str, Any], prof: dict[str, int], rng) -> dict[str, Any]:
    """Moment estimates on the study's own design against the known truth."""
    latent = fit_latent(fit["components"])
    truth = latent.truth()
    d = fit["design"]
    sims = prof["recovery"]
    est = np.array(
        [
            [moments(draw_table(latent, d["cells"], d["tasks"], 5, rng))[n] for n in TERMS]
            for _ in range(sims)
        ]
    )
    mean, se = est.mean(axis=0), est.std(axis=0, ddof=1) / math.sqrt(sims)
    return {
        "latent": latent.__dict__,
        "sims": sims,
        "terms": {
            n: {"truth": truth[n], "mean": float(mean[j]), "mc_se": float(se[j])}
            for j, n in enumerate(TERMS)
        },
    }


def check_coverage(fit: dict[str, Any], prof: dict[str, int], rng) -> dict[str, Any]:
    """95% interval coverage of each component: tasks only, and tasks and cells."""
    latent = fit_latent(fit["components"])
    truth = latent.truth()
    d = fit["design"]
    sims, draws = prof["coverage"], prof["draws"]
    hits = {"tasks": np.zeros(len(TERMS)), "tasks_and_cells": np.zeros(len(TERMS))}
    for _ in range(sims):
        table = draw_table(latent, d["cells"], d["tasks"], 5, rng)
        for name, both in (("tasks", False), ("tasks_and_cells", True)):
            seed = int(rng.integers(0, 2**31))
            b = bootstrap_draws(table, component_vector, draws=draws, seed=seed, over_cells=both)
            for j, term in enumerate(TERMS):
                lo, hi = percentile(b[:, j])
                hits[name][j] += lo <= truth[term] <= hi
    return {
        "sims": sims,
        "draws": draws,
        "coverage": {
            name: {term: _rate(int(v[j]), sims) for j, term in enumerate(TERMS)}
            for name, v in hits.items()
        },
    }


def check_planner(
    stratum: dict[str, Any], tasks: int, trials: int, prof: dict[str, int], rng
) -> dict[str, Any]:
    """Size and power of the paired test at the planned MDE, components taken as known."""
    latent = fit_pair(stratum["pass_rate"], stratum["task_cell"], stratum["trial"])
    r = resolve(components(stratum, "population"), run=0.0)
    design = Design(tasks, 1, trials)
    target = mde(r, design, alpha=ALPHA, power=POWER)
    sims = prof["sims"]
    out: dict[str, Any] = {"tasks": tasks, "trials": trials, "mde": target}
    for label, shift in (
        ("size", 0.0),
        ("power_gain", logit_shift(latent, target, 1)),
        ("power_loss", logit_shift(latent, target, -1)),
    ):
        d = pair_differences(latent, tasks, trials, shift, sims, rng)
        out[label] = _rate(int(_paired_t_reject(d, ALPHA).sum()), sims)
    return out


def check_pilot(
    stratum: dict[str, Any], pilot_tasks: int, gap: float, prof: dict[str, int], rng
) -> dict[str, Any]:
    """Plan tasks from a pilot's variance on `pilot_tasks - 1` df, then run the plan."""
    latent = fit_pair(stratum["pass_rate"], stratum["task_cell"], stratum["trial"])
    trials = 5
    sims = prof["sims"]
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
    pilot = pair_differences(latent, pilot_tasks, trials, 0.0, sims, rng)
    variance = pilot.var(axis=1, ddof=1)
    planned = np.maximum(np.ceil(z * z * variance / gap**2).astype(int), 2)
    planned = np.minimum(planned, 5000)
    shift = logit_shift(latent, gap, 1)
    hits, nulls = 0, 0
    for tasks, count in zip(*np.unique(planned, return_counts=True), strict=True):
        d = pair_differences(latent, int(tasks), trials, shift, int(count), rng)
        hits += int(_paired_t_reject(d, ALPHA).sum())
        d = pair_differences(latent, int(tasks), trials, 0.0, int(count), rng)
        nulls += int(_paired_t_reject(d, ALPHA).sum())
    true_variance = float(
        pair_differences(latent, 4000, trials, 0.0, 4, rng).var(axis=1, ddof=1).mean()
    )
    ideal = z * z * true_variance / gap**2
    return {
        "pilot_df": pilot_tasks - 1,
        "gap": gap,
        "planned_tasks": {
            "ideal": ideal,
            "median": float(np.median(planned)),
            "q10": float(np.quantile(planned, 0.1)),
            "q90": float(np.quantile(planned, 0.9)),
            "below_half_ideal": float(np.mean(planned < ideal / 2)),
        },
        "power": _rate(hits, sims),
        "size": _rate(nulls, sims),
    }


def check_run(
    stratum: dict[str, Any],
    run_sd: float,
    run_df: int,
    prof: dict[str, int],
    rng,
) -> dict[str, Any]:
    """One job per arm with a shared shock, run variance borrowed on `run_df` df."""
    tasks, trials = 89, 5
    latent = fit_pair(stratum["pass_rate"], stratum["task_cell"], stratum["trial"])
    variance = run_sd**2
    r = resolve(components(stratum, "population"), run=variance, run_df=float(run_df))
    design = Design(tasks, 1, trials)
    target = mde(r, design, alpha=ALPHA, power=POWER)
    sims = prof["sims"]
    tc_naive = stats.t.ppf(1 - ALPHA / 2, tasks - 1)
    out: dict[str, Any] = {"run_sd": run_sd, "run_df": run_df, "mde": target}
    within = float(stratum["trial"])
    for label, gap in (("size", 0.0), ("power", target)):
        shift = logit_shift(latent, gap, 1) if gap else 0.0
        logit = run_logit_sd(stratum, run_sd)
        d = pair_differences(latent, tasks, trials, shift, sims, rng, run_sd=logit)
        v_items = d.var(axis=1, ddof=1) / tasks
        # The pilot's moment estimate: (MS_run - MS_res) / tasks, on run_df df.
        ms_run = (within + tasks * variance) * rng.chisquare(run_df, sims) / run_df
        v_run = 2 * np.maximum((ms_run - within) / tasks, 0.0)
        total = v_items + v_run
        terms = v_items**2 / (tasks - 1) + v_run**2 / run_df
        dof = total**2 / terms
        reject = np.abs(d.mean(axis=1)) > stats.t.ppf(1 - ALPHA / 2, dof) * np.sqrt(total)
        out[label] = _rate(int(reject.sum()), sims)
        if not gap:
            naive = np.abs(d.mean(axis=1)) > tc_naive * np.sqrt(v_items)
            out["size_ignoring_runs"] = _rate(int(naive.sum()), sims)
    out["planned_df"] = _df(r, design)
    return out


def check_fixed_tasks(stratum: dict[str, Any], prof: dict[str, int], rng) -> dict[str, Any]:
    """The fixed-task z-test: same per-task probabilities in both arms, then a shift."""
    tasks, trials = 89, 5
    latent = fit_pair(stratum["pass_rate"], stratum["task_cell"], stratum["trial"])
    r = resolve(components(stratum, "fixed"), run=0.0)
    target = mde(r, Design(tasks, 1, trials), alpha=ALPHA, power=POWER, method="z")
    sims = prof["sims"]
    out: dict[str, Any] = {"mde": target, "tasks": tasks, "trials": trials}
    zc = stats.norm.ppf(1 - ALPHA / 2)
    shift_size = logit_shift(latent, target, 1)
    for label, shift in (("size", 0.0), ("power", shift_size)):
        s = latent.task * rng.standard_normal((sims, tasks))
        eta = latent.mu + s + latent.inter * rng.standard_normal((sims, tasks))
        p0, p1 = 1 / (1 + np.exp(-eta)), 1 / (1 + np.exp(-(eta + shift)))
        x0, x1 = rng.binomial(trials, p0) / trials, rng.binomial(trials, p1) / trials
        var = (x0 * (1 - x0) + x1 * (1 - x1)) / (trials - 1)
        se = np.sqrt(var.sum(axis=1)) / tasks
        gap = (x1 - x0).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            reject = np.abs(gap) > zc * se
        out[label] = _rate(int(reject.sum()), sims)
    return out


def checks(fit: dict[str, Any], prof: dict[str, int]) -> dict[str, Callable]:
    """All checks by unit name."""
    strata = fit["strata"]
    high, wide = strata["same-model/high"], strata["neither/high"]
    run = fit["run"]
    gap = fit["pilot_gap"]
    out: dict[str, Callable] = {
        "recovery": lambda rng: check_recovery(fit, prof, rng),
        "coverage": lambda rng: check_coverage(fit, prof, rng),
        "planner-same-model-t89-k5": lambda rng: check_planner(high, 89, 5, prof, rng),
        "planner-same-model-t89-k10": lambda rng: check_planner(high, 89, 10, prof, rng),
        "planner-same-model-t150-k5": lambda rng: check_planner(high, 150, 5, prof, rng),
        "planner-neither-t89-k5": lambda rng: check_planner(wide, 89, 5, prof, rng),
        "pilot-7df": lambda rng: check_pilot(high, 8, gap, prof, rng),
        "pilot-88df": lambda rng: check_pilot(high, 89, gap, prof, rng),
        "fixed-tasks": lambda rng: check_fixed_tasks(high, prof, rng),
    }
    for sd_label, sd in (("tb2", run["sd_upper"]), ("evalnoise", 0.029)):
        for df in (7, run["df"]):
            out[f"run-{sd_label}-df{df}"] = lambda rng, sd=sd, df=df: check_run(
                high, sd, df, prof, rng
            )
    return out


def run_validation(
    fit: dict[str, Any],
    units: Path,
    profile: str = "full",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every check not already on disk for these inputs, and pool the results."""
    prof = PROFILES[profile]
    units.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        json.dumps([SCHEMA, prof, fit], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    out: dict[str, Any] = {"profile": profile, "seed": SEED, "checks": {}}
    for name, check in checks(fit, prof).items():
        path = units / f"{key}-{name}.json"
        if path.exists():
            out["checks"][name] = json.loads(path.read_text(encoding="utf-8"))
            continue
        result = check(np.random.default_rng([SEED, zlib.crc32(name.encode())]))
        part = path.with_suffix(".part")
        part.write_text(json.dumps(result), encoding="utf-8")
        os.replace(part, path)
        out["checks"][name] = result
        log(f"check {name} done")
    return out
