"""REPORT.md, generated from results/v1 so that every number traces to a committed file."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from trialnoise.decompose import TERMS
from trialnoise.report_text import TEMPLATE

LABEL = {"task": "Task", "cell": "Cell", "task_cell": "Task x cell", "trial": "Trial"}
STAMP = "report.sha256"


def pts(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def sd(variance: float, digits: int = 1) -> str:
    return f"{100 * math.sqrt(max(variance, 0.0)):.{digits}f}"


def ci(pair: Any, digits: int = 1) -> str:
    return f"{pts(pair[0], digits)} to {pts(pair[1], digits)}"


def ci3(pair: Any) -> str:
    return f"{pair[0]:.3f} to {pair[1]:.3f}"


def rate(check: dict[str, Any]) -> str:
    return f"{check['rate']:.3f}"


def table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def need(planner: dict, stratum: str, target: str, gap: float) -> dict:
    for row in planner["needs"]:
        if (row["stratum"], row["target"], row["run"], row["gap"]) == (
            stratum,
            target,
            "none",
            gap,
        ):
            return row
    raise KeyError((stratum, target, gap))


def grid(planner: dict, stratum: str, target: str, tasks: int, trials: int, run="none") -> dict:
    key = (stratum, target, run, tasks, trials)
    for row in planner["grid"]:
        if (row["stratum"], row["target"], row["run"], row["tasks"], row["trials"]) == key:
            return row
    raise KeyError(key)


def tables(s: dict, p: dict, v: dict) -> dict[str, str]:
    """Every Markdown table of the report."""
    c = s["components"]
    values, both, tasks = c["values"], c["tasks_and_cells"], c["tasks"]
    strata, ref = s["strata"], s["reference_mde"]
    best, every = s["pairs"]["scaffold_best"], s["pairs"]["all_cells"]
    nest_all, nest_top = s["scaffold"]["all-models"], s["scaffold"]["frontier-models"]
    out: dict[str, str] = {}
    out["components"] = table(
        ["Term", "Variance", "SD, points", "Share", "95% CI, tasks and cells",
         "95% CI, tasks only"],
        [
            [LABEL[t], f"{values[t]:.4f}", sd(values[t]), f"{pts(values['share_' + t])}%",
             ci3(both[t]), ci3(tasks[t])]
            for t in TERMS
        ],
    )  # fmt: skip

    def mde_cell(name: str, n: int, k: int) -> str:
        row = ref[f"{name}|t{n}|k{k}"]
        return f"{pts(row['mde'])} ({ci(row['interval'])})"

    names = ("same-model/high", "same-scaffold/high", "neither/high", "neither/mid")
    out["mde"] = table(
        ["Near pairs", "Pairs", "Task x cell", "Trial", "89 x 5", "89 x 10", "150 x 5", "150 x 10"],
        [
            [n, strata[n]["pairs"], f"{strata[n]['task_cell']:.4f}", f"{strata[n]['trial']:.4f}",
             mde_cell(n, 89, 5), mde_cell(n, 89, 10), mde_cell(n, 150, 5), mde_cell(n, 150, 10)]
            for n in names
        ],
    )  # fmt: skip
    rows = []
    for n in (*names, "pooled/all"):
        for gap in (0.03, 0.05):
            at = need(p, n, "population", gap)["tasks_at_trials"]
            fixed = need(p, n, "fixed", gap)["trials_at_89_tasks"]
            rows.append([n, pts(gap, 0), at["5"], at["10"], at["20"], fixed])
    out["needs"] = table(
        ["Near pairs", "Gap, points", "Tasks at 5 trials", "at 10", "at 20", "Trials on these 89"],
        rows,
    )
    out["board"] = table(
        ["Trials", "Tasks", "All 108 cells", "Best per scaffold"],
        [
            [b["trials"], b["tasks"], f"{100 * b['adjacent_all']:.2f}%",
             f"{100 * b['adjacent_best']:.2f}%"]
            for b in s["pairs"]["board"]
        ],
    )  # fmt: skip
    spans = sorted(nest_all["spans"].items(), key=lambda kv: -kv[1]["span"])[:8]
    out["spans"] = table(
        ["Model", "Scaffolds", "Lowest", "Highest", "Span"],
        [[m, int(r["scaffolds"]), pts(r["low"]), pts(r["high"]), pts(r["span"])] for m, r in spans],
    )
    out["near"] = table(
        ["Near pairs", "Pairs", "Task x cell", "95% CI"],
        [
            [name, strata[f"{name}/all"]["pairs"], f"{strata[f'{name}/all']['task_cell']:.4f}",
             ci3(strata[f"{name}/all"]["task_cell_interval"])]
            for name in ("same-model", "same-scaffold", "neither")
        ],
    )  # fmt: skip

    def nest_row(label: str, n: dict) -> list[Any]:
        share = n["intervals"]
        return [
            label, f"{n['cells']:.0f}", sd(n["model"]), sd(n["scaffold_within_model"]),
            f"{pts(n['share_scaffold'])}%", ci(share["tasks+models"]["share_scaffold"]),
            ci(share["tasks"]["share_scaffold"]),
        ]  # fmt: skip

    out["nested"] = table(
        ["Models", "Cells", "Model SD", "Scaffold-within-model SD", "Scaffold share",
         "95% CI, tasks and models", "95% CI, tasks only"],
        [nest_row(f"All {nest_all['models']:.0f}", nest_all),
         nest_row(f"{nest_top['models']:.0f} frontier", nest_top)],
    )  # fmt: skip
    out["pairs"] = table(
        ["Universe", "Adjacent pairs", "Median gap", "Median SE, tasks", "Separated: raw", "BH",
         "Holm", "Fixed-task Holm"],
        [
            [name, x["pairs"], pts(x["median_gap"], 2), pts(x["median_se_task"], 2), x["raw_05"],
             x["bh_05"], x["holm_05"], x["fixed_holm_05"]]
            for name, x in (("Best per scaffold", best), ("All cells", every))
        ],
    )  # fmt: skip
    cover = v["coverage"]["coverage"]
    out["coverage"] = table(
        ["Interval", *[LABEL[t] for t in TERMS]],
        [
            ["Tasks only", *[rate(cover["tasks"][t]) for t in TERMS]],
            ["Tasks and cells", *[rate(cover["tasks_and_cells"][t]) for t in TERMS]],
        ],
    )
    recovery = v["recovery"]["terms"]
    out["recovery"] = table(
        ["Term", "Truth", "Mean estimate", "MC SE"],
        [
            [LABEL[t], f"{recovery[t]['truth']:.4f}", f"{recovery[t]['mean']:.4f}",
             f"{recovery[t]['mc_se']:.4f}"]
            for t in TERMS
        ],
    )  # fmt: skip
    planner_keys = (
        "planner-same-model-t89-k5",
        "planner-same-model-t89-k10",
        "planner-same-model-t150-k5",
        "planner-neither-t89-k5",
    )
    out["planner"] = table(
        ["Design", "MDE", "Size", "Power, gain", "Power, loss"],
        [
            [k.removeprefix("planner-"), pts(v[k]["mde"]), rate(v[k]["size"]),
             rate(v[k]["power_gain"]), rate(v[k]["power_loss"])]
            for k in planner_keys
        ],
    )  # fmt: skip
    out["run"] = table(
        ["Run SD, points", "Run df", "MDE", "Size", "Power", "Size, run-blind test"],
        [
            [pts(v[k]["run_sd"]), v[k]["run_df"], pts(v[k]["mde"]), rate(v[k]["size"]),
             rate(v[k]["power"]), rate(v[k]["size_ignoring_runs"])]
            for k in sorted(k for k in v if k.startswith("run-"))
        ],
    )  # fmt: skip
    return out


def fields(s: dict, p: dict, v: dict, pairs: pd.DataFrame, integrity: pd.DataFrame) -> dict:
    """Every number the prose uses, by name."""
    c = s["components"]
    values, both = c["values"], c["tasks_and_cells"]
    run, held, infra, cost = s["run"], s["heldout"], s["infra"], s["cost"]
    best, every = s["pairs"]["scaffold_best"], s["pairs"]["all_cells"]
    nest_all, nest_top = s["scaffold"]["all-models"], s["scaffold"]["frontier-models"]
    crossed, structure, strata = s["scaffold"]["crossed"], s["interaction"], s["strata"]
    ref, board = s["reference_mde"], {(b["trials"], b["tasks"]): b for b in s["pairs"]["board"]}
    d, design = s["data"], s["design"]
    named = sorted(s["integrity"]["named_ranks"].items(), key=lambda kv: kv[1])
    flagged = integrity[integrity["flagged"]]
    same = strata["same-model/high"]
    hold = held["by_comparison"]
    cover = v["coverage"]["coverage"]
    p7, p88, fixed = v["pilot-7df"], v["pilot-88df"], v["fixed-tasks"]
    head = v["planner-same-model-t89-k5"]
    sens = s["sensitivity"].values()
    timeouts = infra["by_exception"]["AgentTimeoutError"]
    top = pairs[pairs["universe"] == "all-cells"].iloc[0]
    trials_fixed_3 = need(p, "neither/high", "fixed", 0.03)["trials_at_89_tasks"]
    trials_fixed_5 = need(p, "neither/high", "fixed", 0.05)["trials_at_89_tasks"]
    cheap, dear = cost["cheapest_above_half"][0], cost["dearest"][0]
    counts = design["trials_per_task_cell"]
    per_cell = c["per_cell_icc"]
    out = {
        "trials_fixed_3": trials_fixed_3,
        "tasks_same_3": need(p, "same-model/high", "population", 0.03)["tasks_at_trials"]["5"],
        "tasks_neither_3": need(p, "neither/high", "population", 0.03)["tasks_at_trials"]["5"],
        "ten_low": need(p, "same-model/high", "population", 0.10)["trials_at_89_tasks"],
        "ten_high": need(p, "neither/high", "population", 0.10)["trials_at_89_tasks"],
        "trial_share_same": pts((same["trial"] / 5) / (same["task_cell"] + same["trial"] / 5), 0),
        "share_task": pts(values["share_task"]),
        "share_cell": pts(values["share_cell"]),
        "share_task_cell": pts(values["share_task_cell"]),
        "share_trial": pts(values["share_trial"]),
        "cells": design["cells"],
        "tasks": design["tasks"],
        "trials": f"{design['trials']:,}",
        "icc1": f"{values['icc1']:.2f}",
        "icc1_long": f"{values['icc1']:.3f}",
        "icc1_ci": ci3(both["icc1"]),
        "icc5": f"{values['icc1_k']:.2f}",
        "icc5_ci": ci3(both["icc1_k"]),
        "icc_pair": f"{values['icc_pair']:.2f}",
        "icc_pair_ci": ci3(both["icc_pair"]),
        "icc_median": f"{per_cell['median']:.2f}",
        "icc_q10": f"{per_cell['q10']:.2f}",
        "icc_q90": f"{per_cell['q90']:.2f}",
        "icc_min": f"{per_cell['min']:.2f}",
        "nest_models": f"{nest_all['models']:.0f}",
        "nest_share": pts(nest_all["share_scaffold"]),
        "nest_ci": ci(nest_all["intervals"]["tasks+models"]["share_scaffold"]),
        "top_models": f"{nest_top['models']:.0f}",
        "top_share": pts(nest_top["share_scaffold"]),
        "top_ci": ci(nest_top["intervals"]["tasks+models"]["share_scaffold"]),
        "top_scaffold_sd": sd(nest_top["scaffold_within_model"]),
        "top_model_sd": sd(nest_top["model"]),
        "frontier": ", ".join(sorted(nest_top["spans"])),
        "task_scaffold": f"{structure['values']['task_scaffold']:.4f}",
        "task_scaffold_ci": ci3(structure["intervals"]["task_scaffold"]),
        "task_model": f"{structure['values']['task_model']:.4f}",
        "task_model_ci": ci3(structure["intervals"]["task_model"]),
        "task_both": f"{structure['values']['task_model_scaffold']:.4f}",
        "task_both_ci": ci3(structure["intervals"]["task_model_scaffold"]),
        "run_subs": len(run["submissions"]),
        "run_df": run["df"],
        "run_span": f"{run['median_span_days']:.0f}",
        "run_sd_upper": pts(run["sd_upper"]),
        "run_var": f"{run['variance']:.5f}",
        "run_upper": f"{run['interval'][1]:.6f}",
        "run_f": f"{run['f']:.2f}",
        "run_p_below": f"{run['p_below']:.2f}",
        "aa_rejected": s["split_jobs"]["rejected"],
        "aa_splits": s["split_jobs"]["splits"],
        "aa_rate": pts(s["split_jobs"]["rate"]),
        "held_pairs": held["pairs"],
        "held_cells": held["cells"],
        "held_median": f"{held['median_ratio']:.2f}",
        "held_q05": f"{held['ratio_q05']:.2f}",
        "held_q95": f"{held['ratio_q95']:.2f}",
        "held_within": pts(held["within_20pct"], 0),
        "held_same": f"{hold['same-model']['median_ratio']:.2f}",
        "held_same_pairs": hold["same-model"]["pairs"],
        "held_neither": f"{hold['neither']['median_ratio']:.2f}",
        "held_neither_pairs": hold["neither"]["pairs"],
        "held_high_pairs": held["by_band"]["high"]["pairs"],
        "sim_size": f"{head['size']['rate']:.3f}",
        "sim_gain": f"{head['power_gain']['rate']:.2f}",
        "sim_loss": f"{head['power_loss']['rate']:.2f}",
        "sim_n": head["size"]["n"],
        "best_pairs": best["pairs"],
        "best_holm": best["holm_05"],
        "best_bh": best["bh_05"],
        "best_raw": best["raw_05"],
        "best_fixed_holm": best["fixed_holm_05"],
        "best_noise": best["pairs"] - best["holm_05"],
        "best_unseparated_bh": pts(1 - best["all_pairs"]["share_bh"], 0),
        "every_pairs": every["pairs"],
        "every_holm": every["holm_05"],
        "every_gap": pts(every["median_gap"], 2),
        "every_cells": every["pairs"] + 1,
        "every_share_holm": pts(every["all_pairs"]["share_holm"], 0),
        "every_all_pairs": f"{every['all_pairs']['pairs']:,}",
        "gap_holm": pts(every["all_pairs"]["smallest_gap_holm"]),
        "gap_bh": pts(every["all_pairs"]["smallest_gap_bh"]),
        "se_ratio": f"{every['median_se_ratio_k10']:.2f}",
        "board_5": f"{100 * board[(5, 89)]['adjacent_best']:.1f}",
        "board_10": f"{100 * board[(10, 89)]['adjacent_best']:.1f}",
        "board_150": f"{100 * board[(5, 150)]['adjacent_best']:.1f}",
        "board_300": f"{100 * board[(5, 300)]['adjacent_best']:.1f}",
        "truth_inter": f"{s['pairs']['truth']['task_cell_of_truth']:.4f}",
        "truth_fitted": f"{s['pairs']['truth']['fitted_task_cell']:.4f}",
        "top_upper": top["upper"],
        "top_lower": top["lower"],
        "top_gap": pts(top["gap"]),
        "top_se": pts(top["se_task"]),
        "nv_rate": pts(infra["no_verdict_rate"]),
        "nv_subs": infra["submissions"],
        "nv_trials": f"{infra['trials']:,}",
        "nv_median": pts(infra["no_verdict_rate_median"]),
        "nv_max": pts(infra["no_verdict_rate_max"]),
        "nv_share": pts(infra["share_of_score_variance"], 2),
        "nv_loss": pts(infra["loss_median"], 2),
        "nv_loss_max": pts(infra["loss_max"]),
        "nv_loss_sd": pts(infra["loss_sd"], 2),
        "nv_rho": f"{infra['spearman']:.3f}",
        "timeout_rate": pts(infra["timeout_rate"]),
        "timeout_verdicts": f"{timeouts['verdicts']:,}",
        "timeout_trials": f"{timeouts['trials']:,}",
        "timeout_passes": timeouts["passes"],
        "copied": d["copied_submissions"][0],
        "copied_cell": d["copied_cells"][0],
        "named_low": named[3][1],
        "named_ranks": ", ".join(str(rank) for _, rank in named),
        "auc": f"{s['integrity']['auc_named_low']:.2f}",
        "units": s["integrity"]["units"],
        "flagged_n": len(flagged),
        "flagged": ", ".join(f"`{u}`" for u in flagged["unit"]),
        "forge_score": pts(top["score_upper"]),
        "forge_rank": s["integrity"]["named_ranks"]["Forge__Gemini-3.1-Pro-Preview"],
        "rows": f"{d['rows']:,}",
        "dups": f"{d['duplicate_rows']:,}",
        "all_trials": f"{d['trials']:,}",
        "all_cells": d["cells"],
        "official_trials": f"{d['official_trials']:,}",
        "official_verdicts": f"{d['official_with_verdict']:,}",
        "linked_trials": f"{d['linked_trials']:,}",
        "linked_cells": d["linked_cells"],
        "reps": ", ".join(
            f"{n:,} with {k}" for k, n in sorted(counts.items(), key=lambda kv: int(kv[0]))
        ),
        "reml_task": f"{s['reml']['task']:.4f}",
        "reml_cell": f"{s['reml']['cell']:.4f}",
        "reml_inter": f"{s['reml']['task_cell']:.4f}",
        "df_task": c["df"]["task"],
        "df_cell": c["df"]["cell"],
        "df_inter": f"{c['df']['task_cell']:,}",
        "df_trial": f"{c['df']['trial']:,}",
        "cover_cell": f"{cover['tasks']['cell']['rate']:.2f}",
        "sens_low": f"{min(x['task_cell'] for x in sens):.4f}",
        "sens_high": f"{max(x['task_cell'] for x in sens):.4f}",
        "sens_icc_low": f"{min(x['icc1'] for x in sens):.3f}",
        "sens_icc_high": f"{max(x['icc1'] for x in sens):.3f}",
        "pooled_inter": f"{values['task_cell']:.4f}",
        "pooled_mde": pts(grid(p, "pooled/all", "population", 89, 5)["mde"]),
        "same_mde": pts(grid(p, "same-model/high", "population", 89, 5)["mde"]),
        "same_mde_upper": pts(grid(p, "same-model/high", "population", 89, 5, "upper")["mde"]),
        "same_limit": pts(grid(p, "same-model/high", "population", 89, 20)["mde"]),
        "crossed_scaffold_sd": sd(crossed["scaffold"]),
        "crossed_model_sd": sd(crossed["model"]),
        "crossed_inter_sd": sd(crossed["interaction_upper"]),
        "crossed_df": crossed["df_residual"],
        "twin_a": s["twin"]["cells"][0],
        "twin_b": s["twin"]["cells"][1],
        "twin_gap": pts(s["twin"]["gap"]),
        "twin_se": pts(s["twin"]["se"]),
        "cost_cells": cost["cells"],
        "cost_trial": f"{cost['median_cost_per_trial']:.2f}",
        "cost_pass": f"{cost['median_cost_per_pass']:.2f}",
        "cost_five": f"{cost['median_cost_per_trial'] * trials_fixed_5 * 89:,.0f}",
        "trials_fixed_5": trials_fixed_5,
        "cheap": cheap["cell"],
        "cheap_cost": f"{cheap['cost_per_pass']:.2f}",
        "cheap_lo": f"{cheap['lo']:.2f}",
        "cheap_hi": f"{cheap['hi']:.2f}",
        "dear": dear["cell"],
        "dear_cost": f"{dear['cost_per_pass']:.2f}",
        "dear_lo": f"{dear['lo']:.2f}",
        "dear_hi": f"{dear['hi']:.2f}",
        "rec_sims": v["recovery"]["sims"],
        "cov_sims": v["coverage"]["sims"],
        "cov_draws": v["coverage"]["draws"],
        "implied_trial": f"{v['recovery']['terms']['trial']['truth']:.4f}",
        "fitted_trial": f"{values['trial']:.4f}",
        "p7_median": f"{p7['planned_tasks']['median']:.0f}",
        "p7_ideal": f"{p7['planned_tasks']['ideal']:.0f}",
        "p7_q10": f"{p7['planned_tasks']['q10']:.0f}",
        "p7_q90": f"{p7['planned_tasks']['q90']:.0f}",
        "p7_below": pts(p7["planned_tasks"]["below_half_ideal"], 0),
        "p7_power": rate(p7["power"]),
        "p88_q10": f"{p88['planned_tasks']['q10']:.0f}",
        "p88_q90": f"{p88['planned_tasks']['q90']:.0f}",
        "p88_power": rate(p88["power"]),
        "blind_size": rate(v[f"run-evalnoise-df{run['df']}"]["size_ignoring_runs"]),
        "aware_size_7": rate(v["run-evalnoise-df7"]["size"]),
        "fixed_size": rate(fixed["size"]),
        "fixed_power": rate(fixed["power"]),
        "fixed_mde": pts(fixed["mde"]),
    }
    for name, stratum in (("same", "same-model/high"), ("neither", "neither/high")):
        for label, key in (("895", "t89|k5"), ("8910", "t89|k10"), ("1505", "t150|k5")):
            out[f"mde_{name}_{label}"] = pts(ref[f"{stratum}|{key}"]["mde"])
    return out


def render(results: Path) -> str:
    def read(name: str) -> dict:
        return json.loads((results / name).read_text(encoding="utf-8"))

    s, p = read("summary.json"), read("planner.json")
    v = read("validation.json")["checks"]
    pairs = pd.read_csv(results / "pairs.csv")
    integrity = pd.read_csv(results / "integrity.csv")
    values = fields(s, p, v, pairs, integrity)
    values.update({f"table_{k}": t for k, t in tables(s, p, v).items()})
    return TEMPLATE.format(**values)


def stamp(results: Path) -> None:
    """Record the hash of the rendered report, so regeneration is checkable from results alone."""
    digest = hashlib.sha256(render(results).encode("utf-8")).hexdigest()
    (results / STAMP).write_text(digest + "\n", encoding="utf-8", newline="\n")


def write(results: Path, target: Path) -> None:
    target.write_text(render(results), encoding="utf-8", newline="\n")
    stamp(results)
