"""Per-trial result tables: reading, column mapping and validation.

A trial table has one row per trial (one attempt of one arm on one item) with columns:

- ``item``: task or benchmark item id (required);
- ``arm``: the configuration being compared, e.g. a model or prompt (required);
- ``run``: the run the trial belongs to (a batch launched together; defaults to one run);
- ``repeat``: repeat index within a run (optional, informational);
- ``cluster``: resampling unit for the bootstrap (defaults to ``item``); use it when items
  are nested in larger units, e.g. several seeds of one game;
- ``score``: the numeric outcome (required, finite).

Run labels are scoped to the arm: run ``r1`` of arm A and run ``r1`` of arm B are different
runs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("item", "arm", "score")
OPTIONAL = ("run", "repeat", "cluster")
COLUMNS = ("item", "arm", "run", "repeat", "cluster", "score")


class DataError(ValueError):
    """The input table is malformed or cannot support the requested analysis."""


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or JSON Lines file into a DataFrame without interpreting columns."""
    p = Path(path)
    if not p.is_file():
        raise DataError(f"{p}: no such file")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        with p.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"{p}:{lineno}: invalid JSON ({exc.msg})") from None
                if not isinstance(row, dict):
                    raise DataError(f"{p}:{lineno}: expected a JSON object per line")
                rows.append(row)
        return pd.DataFrame(rows)
    raise DataError(f"{p}: unsupported file type {suffix!r} (use .csv or .jsonl)")


def to_trials(
    frame: pd.DataFrame,
    columns: Mapping[str, str] | None = None,
    *,
    default_arm: str | None = None,
) -> pd.DataFrame:
    """Map source columns to the trial schema and validate.

    ``columns`` maps schema names (``item``, ``arm``, ``run``, ``repeat``, ``cluster``,
    ``score``) to column names in ``frame``; unmapped names are looked up as-is.
    ``default_arm`` labels every row when the table has no arm column.
    """
    mapping = dict(columns or {})
    unknown = set(mapping) - set(COLUMNS)
    if unknown:
        raise DataError(f"unknown schema column(s): {', '.join(sorted(unknown))}")
    out: dict[str, pd.Series] = {}
    for name in COLUMNS:
        src = mapping.get(name, name)
        if src in frame.columns:
            out[name] = frame[src]
        elif name in mapping:
            raise DataError(f"column {src!r} (for {name}) not found; have {list(frame.columns)}")
    if "arm" not in out:
        if default_arm is None:
            raise DataError("no arm column; pass an arm column or a default arm label")
        out["arm"] = pd.Series([default_arm] * len(frame), index=frame.index)
    for name in REQUIRED:
        if name not in out:
            raise DataError(f"missing required column {name!r}; have {list(frame.columns)}")
    df = pd.DataFrame(out)
    if "run" not in df:
        df["run"] = "run-1"
    if "cluster" not in df:
        df["cluster"] = df["item"]
    if "repeat" not in df:
        df["repeat"] = df.groupby(["arm", "run", "item"]).cumcount()
    return validate(df)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy of a trial table or raise :class:`DataError`."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"missing column(s): {', '.join(missing)}")
    if len(df) == 0:
        raise DataError("no trials")
    out = df.loc[:, list(COLUMNS)].copy()
    for name in ("item", "arm", "run", "cluster"):
        col = out[name]
        if col.isna().any():
            raise DataError(f"{name}: {int(col.isna().sum())} missing value(s)")
        out[name] = col.astype(str)
        if (out[name].str.strip() == "").any():
            raise DataError(f"{name}: empty label(s)")
    scores = pd.to_numeric(out["score"], errors="coerce").astype(float)
    bad = ~np.isfinite(scores.to_numpy())
    if bad.any():
        first = out.loc[bad, "score"].iloc[0]
        raise DataError(
            f"score: {int(bad.sum())} non-numeric or non-finite value(s), e.g. {first!r}"
        )
    out["score"] = scores.astype(float)
    out["repeat"] = pd.to_numeric(out["repeat"], errors="coerce").fillna(-1).astype(int)
    # A repeat label names one trial: the same label twice is a table read twice, and would
    # pass as two repeats that agree exactly. -1: no label.
    labelled = out[out["repeat"] >= 0]
    twice = labelled.duplicated(["arm", "run", "item", "repeat"])
    if twice.any():
        raise DataError(f"{int(twice.sum())} row(s) repeat the same arm, run, item and repeat")
    # An item must sit in exactly one cluster, or resampling clusters would split it.
    per_item = out.groupby("item")["cluster"].nunique()
    if (per_item > 1).any():
        item = per_item[per_item > 1].index[0]
        raise DataError(f"item {item!r} belongs to more than one cluster")
    return out.reset_index(drop=True)


def load_trials(
    path: str | Path,
    columns: Mapping[str, str] | None = None,
    *,
    default_arm: str | None = None,
) -> pd.DataFrame:
    """Read a CSV/JSONL trial file and validate it (see :func:`to_trials`)."""
    return to_trials(read_table(path), columns, default_arm=default_arm)


def describe(df: pd.DataFrame) -> dict[str, object]:
    """Counts that determine which variance components a table can identify."""
    cells = df.groupby(["arm", "run", "item"]).size()
    runs = df.groupby("arm")["run"].nunique()
    return {
        "trials": int(len(df)),
        "items": int(df["item"].nunique()),
        "clusters": int(df["cluster"].nunique()),
        "arms": int(df["arm"].nunique()),
        "runs_per_arm": {str(k): int(v) for k, v in runs.items()},
        "max_repeats_per_run": int(cells.max()),
        "cells_with_repeats": int((cells >= 2).sum()),
    }
