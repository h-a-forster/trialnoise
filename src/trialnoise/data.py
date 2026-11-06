"""Fetch the pinned sources and build the two trial tables the study reads.

`trials.parquet` is the trajectory dataset reduced to one row per unique trial. `official.parquet`
is every per-trial outcome listed in the official leaderboard's job-level `result.json` files.
Both are committed, so a study rerun needs neither download.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trialnoise._evalnoise.data import DataError
from trialnoise.models import MODELS

DATASET = "yoonholee/terminalbench-trajectories"
REVISION = "04e8940f5b6736a7ce8d22224fe2f2af74163ed2"
SHARDS = {
    "train-00000-of-00002.parquet": (
        "e95bc8dc9f3e1b2f3c6c383d0a060064205d87e93e9ac9541847d1c511effdfa"
    ),
    "train-00001-of-00002.parquet": (
        "00d61614054bedcbadda9974cef30fe1e07d1701e7126edfa210f99840ae99b2"
    ),
}
OFFICIAL = "harborframework/terminal-bench-2-leaderboard"
OFFICIAL_REVISION = "572b2614be2c0cb2527e14f5b1e4026f1072e6c1"
OFFICIAL_PREFIX = "submissions/terminal-bench/2.0"
COLUMNS = [
    "task_name",
    "agent",
    "model",
    "reward",
    "duration_seconds",
    "cost_cents",
    "trial_name",
    "trial_id",
]
# Dirac's submission spells this task's folder differently.
TASK_ALIASES = {"install-windows-3-11": "install-windows-3.11"}
EXPECTED = {"rows": 52_104, "trials": 47_152, "tasks": 89, "cells": 109}
PACKAGE = Path(__file__).resolve().parent


def shipped_root(package: Path = PACKAGE) -> Path:
    """Where `configs/` and `results/` are: the checkout, or the copy a wheel carries."""
    checkout = package.parents[1]
    return checkout if (checkout / "configs" / "official.toml").exists() else package / "_bundled"


SHIPPED = shipped_root()
# Generated files go to the checkout, or to the working folder of an installed package.
WORK = Path() if SHIPPED.name == "_bundled" else SHIPPED
CONFIG = SHIPPED / "configs" / "official.toml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(url: str, attempts: int = 5) -> bytes | None:
    """GET a fixed huggingface.co URL; None for 404; waits out rate limits."""
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "trialnoise"})  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return response.read()
        except (urllib.error.URLError, TimeoutError) as err:
            code = getattr(err, "code", None)
            if code == 404:
                return None
            if attempt == attempts - 1:
                raise
            time.sleep(60 if code == 429 else 5 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_parquet(raw_dir: Path, log: Callable[[str], None] = print) -> None:
    """Download any shard that is missing or fails its hash; only a good file gets its name."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in SHARDS.items():
        path = raw_dir / name
        if not path.exists() or sha256_file(path) != expected:
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{name}"
            log(f"downloading {url}")
            part = path.with_suffix(".part")
            urllib.request.urlretrieve(url, part)  # noqa: S310 - fixed https URL
            actual = sha256_file(part)
            if actual != expected:
                part.unlink()
                raise DataError(f"{name}: sha256 {actual}, expected {expected}")
            os.replace(part, path)
        log(f"ok {name}")


def safe_name(name: str) -> str:
    """A name from a remote listing, refused unless it is one plain path component."""
    if name in ("", ".", "..") or PureWindowsPath(name).name != name:
        raise DataError(f"unsafe name in the remote listing: {name!r}")
    return name


def fetch_official(out: Path, log: Callable[[str], None] = print) -> None:
    """Download each job's `result.json`: one per job, about 3 MB in all, so resumable per file.

    A job file lists every trial by reward and by exception, so no per-trial file is needed.
    """
    out.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{OFFICIAL}/tree/{OFFICIAL_REVISION}"
    files = f"https://huggingface.co/datasets/{OFFICIAL}/resolve/{OFFICIAL_REVISION}"

    def folders(path: str) -> list[str]:
        raw = _get(f"{api}/{urllib.parse.quote(path)}?limit=1000")
        if raw is None:
            raise DataError(f"listing failed: {path}")
        entries = json.loads(raw)
        names = [e["path"].rsplit("/", 1)[1] for e in entries if e["type"] == "directory"]
        return [safe_name(name) for name in names]

    for submission in folders(OFFICIAL_PREFIX):
        target = out / submission
        target.mkdir(exist_ok=True)
        jobs = folders(f"{OFFICIAL_PREFIX}/{submission}")
        for job in jobs:
            path = target / f"{job}.json"
            if path.exists():
                continue
            quoted = urllib.parse.quote(f"{OFFICIAL_PREFIX}/{submission}/{job}/result.json")
            body = _get(f"{files}/{quoted}")
            if body is None:
                log(f"no job result: {submission}/{job}")
                continue
            part = path.with_suffix(".part")
            part.write_bytes(body)
            os.replace(part, path)
        log(f"ok {submission}: {len(jobs)} jobs")


def load_config(path: Path = CONFIG) -> dict[str, dict[str, str]]:
    return tomllib.loads(path.read_text(encoding="utf-8"))["submissions"]


def task_of(trial: str) -> str:
    task = trial.rsplit("__", 1)[0]
    return TASK_ALIASES.get(task, task)


def parse_official(official_dir: Path) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Every trial a job file names, and each submission's job count and designed trial count.

    A trial with no reward ran into an exception before a verdict; the leaderboard's mean counts
    it as zero (the job's `metrics` value equals passes / `n_total_trials`).
    """
    config = load_config()
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, int]] = {}
    found = sorted(p.name for p in official_dir.iterdir() if p.is_dir())
    unknown = sorted(set(found) - set(config))
    if unknown:
        raise DataError(f"submissions missing from configs/official.toml: {unknown}")
    for submission in found:
        meta = config[submission]
        jobs = sorted(p for p in (official_dir / submission).glob("*.json"))
        stats_jobs = 0
        n_total = 0
        starts: list[str] = []
        for path in jobs:
            body = json.loads(path.read_text(encoding="utf-8"))
            if "stats" not in body:
                continue
            stats_jobs += 1
            n_total += int(body.get("n_total_trials") or 0)
            if body.get("started_at"):
                starts.append(body["started_at"])
            for evaluation in body["stats"].get("evals", {}).values():
                outcome: dict[str, dict[str, Any]] = {}
                for value, names in (
                    (evaluation.get("reward_stats") or {}).get("reward", {}).items()
                ):
                    for name in names:
                        outcome.setdefault(name, {})["reward"] = float(value)
                for exception, names in (evaluation.get("exception_stats") or {}).items():
                    for name in names:
                        outcome.setdefault(name, {})["exception"] = exception
                for name, fields in outcome.items():
                    rows.append(
                        {
                            "submission": submission,
                            "scaffold": meta["scaffold"],
                            "model": meta["model"],
                            "status": meta["status"],
                            "job": path.stem,
                            "task": task_of(name),
                            "trial": name,
                            "reward": fields.get("reward", np.nan),
                            "exception": fields.get("exception"),
                        }
                    )
        if stats_jobs:
            totals[submission] = {
                "jobs": stats_jobs,
                "n_total": n_total,
                "job_starts": [min(starts), max(starts)] if starts else [],
            }
    frame = pd.DataFrame(rows).sort_values(["submission", "job", "trial"], ignore_index=True)
    values = set(frame["reward"].dropna().unique())
    if not values <= {0.0, 1.0}:
        raise DataError(f"non-binary official rewards: {sorted(values)}")
    if frame["trial"].duplicated().any():
        raise DataError("a trial name appears in two official jobs")
    return frame, totals


def read_parquet_trials(raw_dir: Path) -> tuple[pd.DataFrame, int]:
    """One row per unique trial, keeping the full-scrape row of each twin."""
    missing = [name for name in SHARDS if not (raw_dir / name).exists()]
    if missing:
        raise DataError(f"missing shards in {raw_dir}: {missing}; run `trialnoise fetch`")
    frames = [pq.read_table(raw_dir / name, columns=COLUMNS).to_pandas() for name in sorted(SHARDS)]
    rows = pd.concat(frames, ignore_index=True)
    key = ["agent", "model", "trial_name"]
    if rows.groupby(key, sort=False)["reward"].nunique().max() > 1:
        raise DataError("listed twins of one trial disagree on reward")
    rows = rows.assign(full=rows["trial_id"].astype(bool))
    kept = rows.sort_values("full", ascending=False, kind="stable").drop_duplicates(key)
    unknown = sorted(set(kept["model"]) - set(MODELS))
    if unknown:
        raise DataError(f"unknown model strings: {unknown}")
    return kept, len(rows)


def cell_names(scaffold: pd.Series, canonical: pd.Series, raw: pd.Series) -> pd.Series:
    """`scaffold/model`; two submissions of one pair keep their raw provider tag apart."""
    base = scaffold + "/" + canonical
    pairs = pd.DataFrame({"base": base, "raw": raw}).drop_duplicates()
    twins = pairs.loc[pairs.duplicated("base", keep=False), "base"].unique()
    tag = raw.str.rsplit("@", n=1).str[1]
    return base.where(~base.isin(twins), base + "@" + tag)


def build_tables(
    raw_dir: Path, official_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """The trial table linked to official job, exception and verdict, with a manifest."""
    kept, n_rows = read_parquet_trials(raw_dir)
    official, totals = parse_official(official_dir)
    known = set(kept["task_name"])
    unknown_task = ~official["task"].isin(known)
    dropped_tasks = sorted(official.loc[unknown_task, "task"].unique())
    official = official[~unknown_task].reset_index(drop=True)

    trials = pd.DataFrame(
        {
            "scaffold": kept["agent"],
            "model": kept["model"].map(MODELS),
            "model_raw": kept["model"],
            "task": kept["task_name"],
            "trial": kept["trial_name"],
            "reward": kept["reward"].astype(int),
            "duration_s": kept["duration_seconds"].round(3),
            # The dataset writes 0 where a scaffold reported no cost.
            "cost_cents": kept["cost_cents"].where(kept["cost_cents"] > 0).round(4),
        }
    )
    trials.insert(2, "cell", cell_names(trials["scaffold"], trials["model"], trials["model_raw"]))
    link = official[["trial", "submission", "job", "reward", "exception"]].rename(
        columns={"reward": "official_reward"}
    )
    trials = trials.merge(link, on="trial", how="left")
    both = trials["official_reward"].notna()
    if (trials.loc[both, "reward"] != trials.loc[both, "official_reward"]).any():
        raise DataError("a linked trial has different rewards in the two sources")
    cells_per_submission = trials.dropna(subset=["submission"]).groupby("submission")["cell"]
    if cells_per_submission.nunique().max() > 1:
        raise DataError("one official submission links to two cells")
    trials = trials.drop(columns="official_reward")
    trials = trials.sort_values(["cell", "task", "trial"], ignore_index=True)

    counts = {
        "rows": n_rows,
        "trials": len(trials),
        "tasks": int(trials["task"].nunique()),
        "cells": int(trials["cell"].nunique()),
    }
    if counts != EXPECTED:
        raise DataError(f"expected {EXPECTED}, found {counts}")
    manifest = {
        "dataset": {"repo": DATASET, "revision": REVISION, "sha256": SHARDS},
        "official": {
            "repo": OFFICIAL,
            "revision": OFFICIAL_REVISION,
            "files": "job-level result.json only",
        },
        "counts": {
            **counts,
            "duplicate_rows": n_rows - len(trials),
            "official_trials": len(official),
            "official_with_verdict": int(official["reward"].notna().sum()),
            "linked_trials": int(trials["submission"].notna().sum()),
            "linked_cells": int(trials.dropna(subset=["submission"])["cell"].nunique()),
            "official_tasks_dropped": dropped_tasks,
        },
        "submissions": totals,
    }
    return trials, official, manifest


def write_tables(raw_dir: Path, official_dir: Path, out: Path) -> dict[str, Any]:
    trials, official, manifest = build_tables(raw_dir, official_dir)
    out.mkdir(parents=True, exist_ok=True)
    trials.to_parquet(out / "trials.parquet", compression="zstd", index=False)
    official.to_parquet(out / "official.parquet", compression="zstd", index=False)
    manifest_text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    (out / "MANIFEST.json").write_text(manifest_text, encoding="utf-8", newline="\n")
    return manifest


def load_tables(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = [out / name for name in ("trials.parquet", "official.parquet", "MANIFEST.json")]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise DataError(f"missing {missing} in {out}; run `trialnoise extract`")
    manifest = json.loads(paths[2].read_text(encoding="utf-8"))
    return pd.read_parquet(paths[0]), pd.read_parquet(paths[1]), manifest
