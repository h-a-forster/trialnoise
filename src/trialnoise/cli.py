"""Command line: `trialnoise components|power|pairs`, and the steps that rebuild the study."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from trialnoise import __version__, data, decompose, pairs, planner, report, study, validate
from trialnoise._evalnoise.data import DataError

RESULTS = data.SHIPPED / "results" / "v1"  # What the read-only commands answer from.
OUTPUT = data.WORK / "results" / "v1"  # What the rebuilding commands write; never the wheel's copy.
RAW = data.WORK / "data" / "raw"
OFFICIAL = data.WORK / "data" / "official"
UNITS = data.WORK / "data" / "validate"
TABLES = ("trials.parquet", "official.parquet", "MANIFEST.json")
OUTPUTS = (
    "components.csv", "cells.csv", "cost.csv", "heldout.csv", "infra.csv", "integrity.csv",
    "pairs.csv", "planner.json", "summary.json",
)  # fmt: skip


def _read_table(path: Path) -> pd.DataFrame:
    """A user's trial table: csv or parquet with cell, task and reward columns."""
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = {"cell", "task", "reward"} - set(frame.columns)
    if missing:
        raise DataError(f"{path.name}: missing columns {sorted(missing)}")
    return frame


def cmd_components(args: argparse.Namespace) -> int:
    trials = _read_table(args.table)
    table = decompose.layout(trials)
    point = decompose.moments(table)
    stats = decompose.statistics(point, args.trials)
    draws = decompose.bootstrap_draws(
        table,
        lambda x: decompose.component_vector(x, args.trials),
        draws=args.bootstrap,
        seed=args.seed,
    )
    values = {**{t: point[t] for t in decompose.TERMS}, **stats}
    print(f"{len(table.cells)} cells x {len(table.tasks)} tasks, {len(trials)} trials")
    for j, name in enumerate(decompose.VECTOR_NAMES):
        lo, hi = decompose.percentile(draws[:, j])
        print(f"{name:18s} {values[name]:9.5f}  [{lo:.5f}, {hi:.5f}]")
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    fit = planner.load(args.planner)
    out = planner.answer(
        fit,
        comparison=args.comparison,
        band=args.band,
        target=args.target,
        tasks=args.tasks,
        trials=args.trials,
        gap=None if args.gap is None else args.gap / 100,
        run=args.run,
    )
    print(json.dumps(out, indent=1))
    return 0


def cmd_pairs(args: argparse.Namespace) -> int:
    if args.a or args.b:
        if not (args.a and args.b):
            raise DataError("give both --a and --b")
        table = decompose.layout(_read_table(args.table))
        for name in (args.a, args.b):
            if name not in table.cells:
                raise DataError(f"unknown cell {name}; see results/v1/cells.csv")
        if args.a == args.b:
            raise DataError("--a and --b name the same cell")
        order = [table.cells.index(args.a), table.cells.index(args.b)]
        row = pairs.adjacent(table, order, "chosen").iloc[0]
        print(
            row.drop(["universe", "rank", "p_task_holm", "p_task_bh", "p_fixed_holm"]).to_string()
        )
        return 0
    frame = pd.read_csv(args.results / "pairs.csv")
    frame = frame[frame["universe"] == args.universe].head(args.top)
    shown = ["rank", "upper", "lower", "gap", "se_task", "p_task_holm", "se_fixed", "p_fixed_holm"]
    print(frame[shown].to_string(index=False))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    data.fetch_parquet(args.raw)
    data.fetch_official(args.official)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    manifest = data.write_tables(args.raw, args.official, args.results)
    print(json.dumps(manifest["counts"], indent=1))
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    study.run_study(args.results)
    if (args.results / "validation.json").exists():
        report.stamp(args.results)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    out = validate.run_validation(study.validation_fit(args.results), args.units, args.profile)
    if args.profile == "full":
        study.write_json(args.results / "validation.json", out)
        report.stamp(args.results)
    else:
        print(json.dumps(study.rounded(out), indent=1))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report.write(args.results, data.WORK / "REPORT.md")
    if data.SHIPPED.name != "_bundled":  # a checkout; a wheel has no README to keep in step
        report.sync_readme(args.results, data.WORK / "README.md")
    return 0


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_check(args: argparse.Namespace) -> int:
    """Rerun the study from the committed tables; every output must match byte for byte."""
    with tempfile.TemporaryDirectory() as folder:
        fresh = Path(folder)
        for name in TABLES:
            shutil.copy(args.results / name, fresh / name)
        study.run_study(fresh, log=lambda _: None)
        changed = [n for n in OUTPUTS if _digest(fresh / n) != _digest(args.results / n)]
    rendered = hashlib.sha256(report.render(args.results).encode("utf-8")).hexdigest()
    if rendered != (args.results / report.STAMP).read_text(encoding="utf-8").strip():
        changed.append("the rendered report")
    if changed:
        print(f"differs: {', '.join(changed)}")
        return 1
    print(f"{len(OUTPUTS)} outputs and the rendered report identical")
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="trialnoise", description=__doc__)
    top.add_argument("--version", action="version", version=__version__)
    sub = top.add_subparsers(dest="command", required=True)

    def add(name: str, func, helptext: str, results: Path = RESULTS) -> argparse.ArgumentParser:  # type: ignore[no-untyped-def]
        p = sub.add_parser(name, help=helptext)
        p.set_defaults(func=func)
        p.add_argument("--results", type=Path, default=results)
        return p

    p = add("components", cmd_components, "variance components of a trial table")
    p.add_argument("--table", type=Path, default=RESULTS / "trials.parquet")
    p.add_argument("--trials", type=int, default=5, help="k of ICC(1,k)")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=study.SEED)

    p = add("power", cmd_power, "MDE of a design, and what a gap needs")
    p.add_argument("--planner", type=Path, default=RESULTS / "planner.json")
    p.add_argument("--comparison", choices=decompose.COMPARISONS, default="neither")
    p.add_argument("--band", choices=("low", "mid", "high", "all"), default="high")
    p.add_argument("--target", choices=planner.TARGETS, default="population")
    p.add_argument("--tasks", type=int, default=89)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--gap", type=float, help="gap to detect, in points")
    p.add_argument("--run", choices=("none", "upper"), default="none")

    p = add("pairs", cmd_pairs, "adjacent pairs, or a test of two named cells")
    p.add_argument("--table", type=Path, default=RESULTS / "trials.parquet")
    p.add_argument("--universe", choices=("all-cells", "scaffold-best"), default="all-cells")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--a", help="first cell of a chosen pair")
    p.add_argument("--b", help="second cell of a chosen pair")

    p = add("fetch", cmd_fetch, "download the pinned sources")
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--official", type=Path, default=OFFICIAL)

    p = add("extract", cmd_extract, "build the committed trial tables", OUTPUT)
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--official", type=Path, default=OFFICIAL)

    add("study", cmd_study, "write every output under results/v1", OUTPUT)

    p = add("validate", cmd_validate, "Monte Carlo checks; resumes from data/validate", OUTPUT)
    p.add_argument("--units", type=Path, default=UNITS)
    p.add_argument("--profile", choices=tuple(validate.PROFILES), default="full")

    add("report", cmd_report, "write REPORT.md from results/v1", OUTPUT)
    add("check", cmd_check, "rerun the study and compare every output byte for byte")
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (DataError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
