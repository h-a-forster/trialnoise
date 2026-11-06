"""The study: components, scaffold terms, run term, planner, pairs, infrastructure, cost."""

from __future__ import annotations

import json
import math
import os
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trialnoise import decompose as dc
from trialnoise import pairs as pr
from trialnoise import planner as pl
from trialnoise import quality as ql
from trialnoise.data import load_tables

SEED = 20261106
DRAWS = 10_000
BOARD_REPS = 1_000
BOARD_GRID = ((5, 89), (10, 89), (20, 89), (50, 89), (5, 150), (10, 150), (5, 300))
FRONTIER = 0.5  # A model is at the frontier when its cells average at least this.


def seed_of(unit: str) -> int:
    return SEED ^ zlib.crc32(unit.encode("utf-8"))


def span_days(submission: dict[str, Any]) -> float:
    """Days between the first and last job start of an official submission."""
    first, last = pd.to_datetime(submission["job_starts"], utc=True, format="ISO8601")
    return (last - first).total_seconds() / 86400


def rounded(x: Any) -> Any:
    """Six significant digits everywhere, so outputs are stable across platforms."""
    if isinstance(x, dict):
        return {str(k): rounded(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [rounded(v) for v in x]
    if isinstance(x, bool | str | int) or x is None:
        return x
    v = float(x)
    if not math.isfinite(v):
        return None
    return float(f"{v:.6g}")


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(rounded(value), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    part = path.with_suffix(".part")
    part.write_text(text, encoding="utf-8", newline="\n")
    os.replace(part, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    part = path.with_suffix(".part")
    frame.to_csv(part, index=False, float_format="%.6g", lineterminator="\n")
    os.replace(part, path)


def component_rows(
    scope: str, values: dict[str, float], interval: str = "", cis: dict | None = None
) -> list[dict[str, Any]]:
    rows = []
    for term, value in values.items():
        lo, hi = (cis or {}).get(term, (math.nan, math.nan))
        rows.append(
            {"scope": scope, "term": term, "estimate": value, "interval": interval,
             "lo": lo, "hi": hi}
        )  # fmt: skip
    return rows


def interval_map(draws: np.ndarray, names: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    return {name: dc.percentile(draws[:, j]) for j, name in enumerate(names)}


def fit_components(table: dc.Layout) -> tuple[dict[str, float], list[dict], dict[str, Any]]:
    m = dc.moments(table)
    stats = dc.statistics(m)
    values = {**{t: m[t] for t in dc.TERMS}, **stats}
    tasks_ci = interval_map(
        dc.bootstrap_draws(table, dc.component_vector, draws=DRAWS, seed=seed_of("tasks")),
        dc.VECTOR_NAMES,
    )
    both_ci = interval_map(
        dc.bootstrap_draws(
            table, dc.component_vector, draws=DRAWS, seed=seed_of("both"), over_cells=True
        ),
        dc.VECTOR_NAMES,
    )
    rows = component_rows("primary", values, "tasks", tasks_ci)
    rows += component_rows("primary", values, "tasks+cells", both_ci)
    icc = dc.per_cell_icc(table)
    summary = {
        "grand": m["grand"],
        "values": values,
        "tasks": tasks_ci,
        "tasks_and_cells": both_ci,
        "per_cell_icc": {
            "median": float(np.nanmedian(icc)),
            "q10": float(np.nanquantile(icc, 0.1)),
            "q90": float(np.nanquantile(icc, 0.9)),
            "min": float(np.nanmin(icc)),
            "max": float(np.nanmax(icc)),
        },
        "df": {
            "task": len(table.tasks) - 1,
            "cell": len(table.cells) - 1,
            "task_cell": (len(table.tasks) - 1) * (len(table.cells) - 1),
            "trial": int((table.n - 1).sum()),
        },
    }
    return values, rows, summary


def fit_scaffold(
    table: dc.Layout,
    meta: pd.DataFrame,
    same_model_interaction: float,
) -> tuple[list[dict], dict[str, Any]]:
    """Nested and crossed scaffold terms on (scaffold, model) pairs."""
    cells = pd.DataFrame({"cell": table.cells, "score": pr.scores(table)}).merge(meta, on="cell")
    cells["pair"] = cells["scaffold"] + "/" + cells["model"]
    pairs = cells.groupby("pair", sort=False).agg(
        score=("score", "mean"), scaffold=("scaffold", "first"), model=("model", "first")
    )
    code = pd.factorize(cells["pair"])[0]
    counts = np.bincount(code).astype(float)
    model_of_pair = pairs["model"].to_numpy()
    t = len(table.tasks)
    noise = (same_model_interaction + float(dc.cell_noise(table)[1].mean())) / t
    noise_fixed = float(dc.cell_noise(table)[1].mean()) / t

    def select(minimum: int, frontier: bool) -> np.ndarray:
        size = pairs.groupby("model")["score"].transform("size").to_numpy()
        mean = pairs.groupby("model")["score"].transform("mean").to_numpy()
        keep = size >= minimum
        return keep & (mean >= FRONTIER) if frontier else keep

    rows: list[dict] = []
    out: dict[str, Any] = {"noise": noise, "noise_fixed_tasks": noise_fixed}
    for name, keep in (
        ("all-models", select(3, False)),
        ("frontier-models", select(3, True)),
    ):
        y = pairs["score"].to_numpy()[keep]
        groups = model_of_pair[keep]
        point = dc.nested_moments(y, groups, noise)
        labels = sorted(set(groups))
        rng_seed = seed_of(f"nested-{name}")
        rng = np.random.default_rng(rng_seed)
        draws: dict[str, list[dict[str, float]]] = {"tasks": [], "tasks+models": []}
        for _ in range(DRAWS):
            cols = rng.integers(0, t, t)
            score = table.mean[:, cols].mean(axis=1)
            pair_score = (np.bincount(code, weights=score) / counts)[keep]
            draws["tasks"].append(dc.nested_moments(pair_score, groups, noise))
            picks = rng.integers(0, len(labels), len(labels))
            ys, gs = [], []
            for position, m in enumerate(picks):
                mask = groups == labels[m]
                ys.append(pair_score[mask])
                gs.append(np.full(int(mask.sum()), position))
            draws["tasks+models"].append(
                dc.nested_moments(np.concatenate(ys), np.concatenate(gs), noise)
            )
        terms = ("model", "scaffold_within_model", "share_scaffold")
        cis = {}
        for label, values in draws.items():
            arr = np.array([[d[k] for k in terms] for d in values])
            cis[label] = interval_map(arr, terms)
            rows += component_rows(
                f"nested/{name}", {k: point[k] for k in terms}, label, cis[label]
            )
        out[name] = {**point, "intervals": cis}
        spans = (
            pairs[keep]
            .groupby("model")["score"]
            .agg(scaffolds="size", low="min", high="max")
            .assign(span=lambda f: f["high"] - f["low"])
            .sort_values("span", ascending=False)
        )
        out[name]["spans"] = {
            m: {k: float(v) for k, v in row.items()} for m, row in spans.iterrows()
        }
    crossed = dc.crossed_cells(
        pairs["score"].to_numpy(), list(pairs["scaffold"]), list(pairs["model"])
    )
    crossed["noise"] = noise
    crossed["interaction_upper"] = max(crossed["residual"] - noise, 0.0)
    out["crossed"] = crossed
    rows += component_rows(
        "crossed", {k: crossed[k] for k in ("scaffold", "model", "residual")}, "REML"
    )
    return rows, out


def fit_strata(table: dc.Layout, meta: pd.DataFrame) -> tuple[dict, list[dict], dict]:
    frame = meta.set_index("cell").loc[list(table.cells)]
    masks = dc.comparison_masks(list(frame["scaffold"]), list(frame["model"]))
    score = pr.scores(table)
    strata = dc.near_masks(score, masks)
    names = list(strata)
    point = dc.stratum_vector(table, strata)
    draws = dc.bootstrap_draws(
        table, lambda x: dc.stratum_vector(x, strata), draws=DRAWS, seed=seed_of("strata")
    )
    level = (score[:, None] + score[None, :]) / 2
    out: dict[str, Any] = {}
    rows: list[dict] = []
    for j, name in enumerate(names):
        if not strata[name].any():
            continue
        task_cell, trial = point[2 * j], point[2 * j + 1]
        out[name] = {
            "pairs": int(strata[name].sum()),
            "pass_rate": float(level[strata[name]].mean()),
            "task_cell": task_cell,
            "task_cell_interval": dc.percentile(draws[:, 2 * j]),
            "trial": trial,
            "trial_interval": dc.percentile(draws[:, 2 * j + 1]),
        }
        rows += component_rows(
            f"stratum/{name}",
            {"task_cell": task_cell, "trial": trial},
            "tasks",
            {
                "task_cell": dc.percentile(draws[:, 2 * j]),
                "trial": dc.percentile(draws[:, 2 * j + 1]),
            },
        )
    # Each comparison's near pairs across bands split task x cell by source.
    index = {name: j for j, name in enumerate(names)}
    part = {c: f"{c}/all" for c in dc.COMPARISONS}
    values = dc.interaction_structure({c: point[2 * index[part[c]]] for c in dc.COMPARISONS})
    structure_draws = [
        dc.interaction_structure({c: row[2 * index[part[c]]] for c in dc.COMPARISONS})
        for row in draws
    ]
    terms = tuple(values)
    arr = np.array([[d[k] for k in terms] for d in structure_draws])
    cis = interval_map(arr, terms)
    rows += component_rows("interaction", values, "tasks", cis)
    structure = {"values": values, "intervals": cis}
    return out, rows, {"structure": structure, "draws": draws, "names": names}


def reference_mde(
    strata: dict[str, Any], fit: dict[str, Any], names: tuple[str, ...]
) -> dict[str, Any]:
    """MDE with an interval at the spec's four designs, from the stratum bootstrap draws."""
    index = {name: j for j, name in enumerate(fit["names"])}
    out: dict[str, Any] = {}
    for name in names:
        s = strata[name]
        r = pl.resolved(s, "population", 0.0, math.inf)
        for tasks, trials in pl.REFERENCE_GRID:
            d = pl.Design(tasks, 1, trials)
            point = pl.mde(r, d, alpha=pl.ALPHA, power=pl.POWER)
            j = index[name]
            variance = (
                2 * np.maximum(fit["draws"][:, 2 * j], 0) + 2 * fit["draws"][:, 2 * j + 1] / trials
            )
            scale = np.sqrt(variance / (2 * s["task_cell"] + 2 * s["trial"] / trials))
            lo, hi = dc.percentile(point * scale)
            out[f"{name}|t{tasks}|k{trials}"] = {"mde": point, "interval": [lo, hi]}
    return out


def fit_pairs(
    all_table: dc.Layout, table: dc.Layout, meta: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    frame = meta.set_index("cell")
    scaffolds_all = list(frame.loc[list(all_table.cells), "scaffold"])
    scaffolds = list(frame.loc[list(table.cells), "scaffold"])
    best = pr.best_per_scaffold(pr.scores(all_table), scaffolds_all)
    order = [int(j) for j in np.argsort(-pr.scores(table), kind="stable")]
    a = pr.adjacent(all_table, best, "scaffold-best")
    b = pr.adjacent(table, order, "all-cells")
    # What doubling the trials does to each adjacent SE, the gap held still.
    var_d = dc.pair_matrices(table)[0]
    noise = dc.cell_noise(table)[1]
    ratio = []
    for up, low in zip(order[:-1], order[1:], strict=True):
        removed = (noise[up] + noise[low]) / 2
        ratio.append(math.sqrt(max(var_d[up, low] - removed, 0) / var_d[up, low]))
    b["se_ratio_k10"] = ratio
    a["se_ratio_k10"] = math.nan

    def tally(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "pairs": len(frame),
            "median_gap": float(frame["gap"].median()),
            "median_se_task": float(frame["se_task"].median()),
            "median_se_fixed": float(frame["se_fixed"].median()),
            "raw_05": int((frame["p_task"] < 0.05).sum()),
            "bh_05": int((frame["p_task_bh"] < 0.05).sum()),
            "holm_05": int((frame["p_task_holm"] < 0.05).sum()),
            "fixed_raw_05": int((frame["p_fixed"] < 0.05).sum()),
            "fixed_holm_05": int((frame["p_fixed_holm"] < 0.05).sum()),
        }

    truth, truth_meta = pr.smoothed_truth(table)
    board = [
        pr.simulate_board(
            truth, scaffolds, trials=k, tasks=n, reps=BOARD_REPS, seed=seed_of(f"board-{k}-{n}")
        )
        for k, n in BOARD_GRID
    ]
    summary = {
        "scaffold_best": {
            **tally(a),
            "all_pairs": pr.all_pairs(all_table, best),
            "cells": [all_table.cells[j] for j in best],
            "top_is_final": bool(all_table.cells[best[0]] == "final/claude-sonnet-4-5"),
        },
        "all_cells": {
            **tally(b),
            "all_pairs": pr.all_pairs(table),
            "median_se_ratio_k10": float(np.median(ratio)),
            "top": [table.cells[j] for j in order[:10]],
        },
        "truth": {**truth_meta, "fitted_task_cell": dc.moments(table)["task_cell"]},
        "board": board,
    }
    return pd.concat([a, b], ignore_index=True), summary


def sensitivity(
    trials: pd.DataFrame, official: pd.DataFrame, copied_cells: list[str]
) -> tuple[list[dict], dict[str, Any]]:
    """Refit under each defensible change of the trial set."""
    verdicts = set(official.loc[official["reward"].notna(), "trial"])
    primary = trials[~trials["cell"].isin(copied_cells)]
    with_verdict = primary[primary["submission"].isna() | primary["trial"].isin(verdicts)]
    # A cell with one trial per task can lose a task with its verdict, which uncrosses the table.
    crossed = with_verdict.groupby("cell")["task"].transform("nunique") == primary["task"].nunique()
    counts = primary.groupby(["cell", "task"]).size().unstack("task").fillna(0)
    full_cells = counts.index[(counts >= 4).mean(axis=1) >= 0.9]
    variants = {
        "all-cells": trials,
        "without-reward-hacked-cell": primary[primary["cell"] != "forge/gemini-3.1-pro"],
        "cells-with-repeats": primary[primary["cell"].isin(full_cells)],
        "verdicts-only": with_verdict[crossed],
    }
    out: dict[str, Any] = {}
    rows: list[dict] = []
    for name, frame in variants.items():
        table = dc.layout(frame)
        m = dc.moments(table)
        s = dc.statistics(m)
        out[name] = {
            "cells": len(table.cells),
            "trials": len(frame),
            **{t: m[t] for t in dc.TERMS},
            "icc1": s["icc1"],
        }
        rows += component_rows(f"sensitivity/{name}", {t: m[t] for t in dc.TERMS})
    return rows, out


def validation_fit(results: Path) -> dict[str, Any]:
    """What the Monte Carlo checks read: the committed fit, rounded as committed."""
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    planner = json.loads((results / "planner.json").read_text(encoding="utf-8"))
    values = summary["components"]["values"]
    return {
        "components": {**{t: values[t] for t in dc.TERMS}, "grand": summary["components"]["grand"]},
        "design": {k: summary["design"][k] for k in ("cells", "tasks")},
        "strata": {k: v for k, v in planner["strata"].items() if k != "pooled/all"},
        "run": planner["run"],
        "pilot_gap": 0.10,
    }


def run_study(results: Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    trials, official, manifest = load_tables(results)
    meta = trials.groupby("cell", as_index=False)[["scaffold", "model"]].first()

    mixed = ql.mixed_task_table(trials, official)
    copied = ql.copied_submissions(official)
    copied_cells = sorted(
        trials.loc[trials["submission"].isin(copied), "cell"].dropna().unique().tolist()
    )
    primary = trials[~trials["cell"].isin(copied_cells)]
    all_table = dc.layout(trials)
    table = dc.layout(primary)
    log(f"table: {len(table.cells)} cells x {len(table.tasks)} tasks, {len(primary)} trials")

    values, rows, components = fit_components(table)
    log("components done")
    reml = dc.reml_means(table)
    rows += component_rows(
        "reml", {k: reml[k] for k in ("task", "cell", "task_cell", "residual")}, "REML"
    )
    sens_rows, sens = sensitivity(trials, official, copied_cells)
    rows += sens_rows

    strata, strata_rows, strata_fit = fit_strata(table, meta)
    rows += strata_rows
    strata["pooled/all"] = {
        "pairs": len(table.cells) * (len(table.cells) - 1) // 2,
        "pass_rate": components["grand"],
        "task_cell": values["task_cell"],
        "task_cell_interval": components["tasks"]["task_cell"],
        "trial": values["trial"],
        "trial_interval": components["tasks"]["trial"],
    }
    log("strata done")
    scaffold_rows, scaffold = fit_scaffold(table, meta, strata["same-model/all"]["task_cell"])
    rows += scaffold_rows
    log("scaffold terms done")

    run = dc.run_term(official, skip=tuple(copied))
    aa = dc.split_jobs_false_positives(official, tuple(copied))
    run["median_span_days"] = float(
        np.median([span_days(manifest["submissions"][r["submission"]]) for r in run["submissions"]])
    )
    rows += component_rows(
        "run",
        {"run": run["variance"]},
        "Burdick-Graybill",
        {"run": tuple(run["interval"])},
    )
    run_plan = {
        "variance": max(run["variance"], 0.0),
        "upper": run["interval"][1],
        "sd_upper": run["sd_upper"],
        "df": run["df"],
    }
    grid, needs = pl.stratum_rows(strata, run_plan)
    reference = reference_mde(
        strata, strata_fit, ("same-model/high", "same-scaffold/high", "neither/high", "neither/mid")
    )
    held = pl.heldout_cells(official[~official["trial"].isin(trials["trial"])], tuple(copied))
    heldout_table, heldout = pl.calibrate(held, strata)
    log("planner done")

    pair_table, pair_summary = fit_pairs(all_table, table, meta)
    log("pairs done")

    infra = ql.infra_table(official, manifest["submissions"])
    infra_summary = ql.infra_summary(infra, official)
    cost = ql.cost_table(trials, draws=DRAWS, seed=seed_of("cost"))
    screen = ql.integrity_screen(mixed)

    score = pd.Series(pr.scores(all_table), index=list(all_table.cells), name="score")
    icc = pd.Series(dc.per_cell_icc(all_table), index=list(all_table.cells), name="icc1")
    within = pd.Series(dc.cell_noise(all_table)[0], index=list(all_table.cells), name="trial_var")
    n_trials = trials.groupby("cell").size().rename("trials")
    cells = (
        meta.set_index("cell")
        .join([n_trials, score, icc, within])
        .join(trials.groupby("cell")["reward"].mean().rename("score_by_rows"))
        .join(trials.groupby("cell")["submission"].first())
        .sort_values(["score", "trials"], ascending=False)
        .reset_index()
    )
    cells["primary"] = ~cells["cell"].isin(copied_cells)

    twins = [c for c in table.cells if c.startswith("gemini-cli/gemini-3-flash@")]
    twin = {}
    if len(twins) == 2:
        one, two = (table.cells.index(c) for c in twins)
        s = pr.scores(table)
        var_d = dc.pair_matrices(table)[0]
        twin = {
            "cells": twins,
            "gap": float(abs(s[one] - s[two])),
            "se": math.sqrt(float(var_d[one, two]) / len(table.tasks)),
        }

    named = screen[screen["status"] != "ok"]
    others = screen[screen["status"] == "ok"]
    summary = {
        "data": {**manifest["counts"], "copied_submissions": copied, "copied_cells": copied_cells},
        "design": {
            "cells": len(table.cells),
            "tasks": len(table.tasks),
            "trials": len(primary),
            "scaffolds": int(primary["scaffold"].nunique()),
            "models": int(primary["model"].nunique()),
            "trials_per_task_cell": {
                str(k): int(v)
                for k, v in pd.Series(table.n.ravel().astype(int)).value_counts().items()
            },
        },
        "components": components,
        "reml": reml,
        "sensitivity": sens,
        "strata": strata,
        "interaction": strata_fit["structure"],
        "scaffold": scaffold,
        "twin": twin,
        "run": run,
        "split_jobs": aa,
        "reference_mde": reference,
        "heldout": heldout,
        "pairs": pair_summary,
        "infra": infra_summary,
        "cost": {
            "cells": len(cost),
            "median_cost_per_trial": float(cost["cost_per_trial"].median()),
            "median_cost_per_pass": float(cost["cost_per_pass"].median()),
            "cheapest": cost.nsmallest(5, "cost_per_pass").to_dict("records"),
            "dearest": cost.nlargest(3, "cost_per_pass").to_dict("records"),
            "cheapest_above_half": cost[cost["pass_rate"] >= 0.5]
            .nsmallest(5, "cost_per_pass")
            .to_dict("records"),
        },
        "integrity": {
            "units": len(screen),
            "flagged": screen.loc[screen["flagged"], "unit"].tolist(),
            "named_ranks": {row["unit"]: int(row["rank"]) for _, row in named.iterrows()},
            "auc_named_low": float(np.mean([(others["z"] > z).mean() for z in named["z"]])),
        },
    }
    planner = {
        "alpha": pl.ALPHA,
        "power": pl.POWER,
        "note": "Near pairs differ by at most 10 points; bands are by the pair's mean pass rate.",
        "strata": strata,
        "run": run_plan,
        "grid": grid,
        "needs": needs,
    }
    write_csv(results / "components.csv", pd.DataFrame(rows))
    write_csv(results / "pairs.csv", pair_table)
    write_csv(results / "cells.csv", cells)
    write_csv(results / "infra.csv", infra)
    write_csv(results / "cost.csv", cost)
    write_csv(results / "integrity.csv", screen)
    write_csv(results / "heldout.csv", heldout_table)
    write_json(results / "planner.json", planner)
    write_json(results / "summary.json", summary)
    return summary
