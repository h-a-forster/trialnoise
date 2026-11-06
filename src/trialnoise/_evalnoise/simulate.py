"""Synthetic trial tables drawn from the variance-components model (for validation)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trialnoise._evalnoise.components import Components


def simulate(
    c: Components,
    *,
    tasks: int,
    arms: int = 2,
    runs: int = 1,
    repeats: int = 1,
    effect: float = 0.0,
    mean: float = 0.0,
    paired: bool = True,
    binary: bool = False,
    seed: int | np.random.Generator = 0,
) -> pd.DataFrame:
    """Draw one table. Arm ``j`` has mean ``mean + j * effect``.

    With ``binary=True`` the components are on the logit scale and each trial is a
    Bernoulli draw; ``within`` is then ignored (the Bernoulli draw is the trial noise) and
    ``effect`` shifts the logit. ``paired=False`` gives each arm its own items.
    """
    for name in ("item", "item_arm", "run", "item_run", "within"):
        v = getattr(c, name)
        if not (math.isfinite(v) and v >= 0):
            raise ValueError(f"{name} must be a finite non-negative variance, got {v}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    sd = {k: math.sqrt(getattr(c, k)) for k in ("item", "item_arm", "run", "item_run", "within")}
    n_items = tasks if paired else tasks * arms
    u = rng.normal(0.0, sd["item"], n_items)
    a_idx, i_idx, r_idx, k_idx = np.meshgrid(
        np.arange(arms), np.arange(tasks), np.arange(runs), np.arange(repeats), indexing="ij"
    )
    a_idx, i_idx, r_idx, k_idx = (x.ravel() for x in (a_idx, i_idx, r_idx, k_idx))
    item = i_idx if paired else a_idx * tasks + i_idx
    v = rng.normal(0.0, sd["item_arm"], (arms, n_items))
    w = rng.normal(0.0, sd["run"], (arms, runs))
    g = rng.normal(0.0, sd["item_run"], (arms, n_items, runs))
    eta = mean + a_idx * effect + u[item] + v[a_idx, item] + w[a_idx, r_idx] + g[a_idx, item, r_idx]
    if binary:
        y = (rng.random(len(eta)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    else:
        y = eta + rng.normal(0.0, sd["within"], len(eta))
    return pd.DataFrame(
        {
            "item": [f"t{j}" for j in item],
            "arm": [f"arm{j}" for j in a_idx],
            "run": [f"r{j}" for j in r_idx],
            "repeat": k_idx,
            "cluster": [f"t{j}" for j in item],
            "score": y,
        }
    )
