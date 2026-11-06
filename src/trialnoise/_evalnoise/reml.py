"""REML cross-check of the moment estimates.

All random effects are crossed, so the marginal covariance is dense:
``V = s2 I + sum_k s_k Z_k Z_k'``. The restricted log-likelihood is maximised over the
variances with L-BFGS-B and analytic gradients. This is O(n^3) per evaluation and meant for
the few thousand trials of an agent eval, not for large benchmarks. The test suite checks
it against statsmodels' MixedLM on small tables.

Only components the layout identifies are fitted, and the result follows the same
absorbed-component convention as :func:`trialnoise._evalnoise.components.estimate`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import linalg, optimize

from trialnoise._evalnoise.components import NAN, Components
from trialnoise._evalnoise.data import DataError

MAX_TRIALS = 6000


def _joined(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pair two label columns (object strings on pandas 2, where np.char.add refuses them)."""
    return (pd.Series(a) + "\x1f" + pd.Series(b)).to_numpy()


def reml(df: pd.DataFrame, arms: Sequence[str] | None = None) -> Components:
    """REML point estimates (not bootstrapped)."""
    if arms is not None:
        df = df[df["arm"].isin(list(arms))]
    n = len(df)
    if n == 0:
        raise DataError("no trials")
    if n > MAX_TRIALS:
        raise DataError(f"{n} trials: the dense REML cross-check is limited to {MAX_TRIALS}")
    y = df["score"].to_numpy(float)
    arm = df["arm"].astype(str).to_numpy()
    item = df["item"].astype(str).to_numpy()
    run = _joined(arm, df["run"].astype(str).to_numpy())
    n_arms = len(np.unique(arm))
    runs_per_arm = pd.Series(run).groupby(arm).nunique()
    has_runs = bool((runs_per_arm >= 2).any())
    cell = _joined(run, item)
    has_reps = len(np.unique(cell)) < n

    if n_arms == 1 and not (has_runs or has_reps):
        # One trial per item: item variance and trial noise are one number.
        return Components(
            item=float(np.var(y, ddof=1)), absorbed={"item_run": "item", "within": "item"}
        )
    factors = {"item": item}
    if n_arms > 1 and (has_runs or has_reps):
        factors["item_arm"] = _joined(item, arm)
    if has_runs:
        factors["run"] = run
    if has_runs and has_reps:
        factors["item_run"] = _joined(item, run)
    names = list(factors)
    codes = [pd.factorize(factors[k])[0] for k in names]
    zz = [c[:, None] == c[None, :] for c in codes]
    x = pd.get_dummies(arm).to_numpy(float) if n_arms > 1 else np.ones((n, 1))

    fit = _maximise(y, x, codes, zz)
    est = dict(zip(names, fit[:-1], strict=True))
    resid = float(fit[-1])

    item_v = est.get("item", NAN)
    item_arm = est.get("item_arm", NAN)
    run_v = est.get("run", NAN)
    absorbed: dict[str, str] = {}
    target = "item_arm" if n_arms > 1 else "item"
    if has_runs and has_reps:
        item_run, within = est["item_run"], resid
    elif has_runs:
        item_run, within = resid, NAN
        absorbed["within"] = "item_run"
    elif has_reps:
        item_run, within = NAN, resid
        absorbed["item_run"] = target
    else:
        item_run, within = NAN, NAN
        absorbed["item_run"] = target
        absorbed["within"] = target
        item_arm = resid
    return Components(
        item=item_v,
        item_arm=item_arm,
        run=run_v,
        item_run=item_run,
        within=within,
        absorbed=absorbed,
    )


def _maximise(
    y: np.ndarray, x: np.ndarray, codes: list[np.ndarray], zz: list[np.ndarray]
) -> np.ndarray:
    n = len(y)
    scale = float(np.var(y)) or 1.0
    yz = y / np.sqrt(scale)
    eye = np.eye(n)

    def parts(theta: np.ndarray) -> tuple[float, np.ndarray]:
        v = theta[-1] * eye
        for t, z in zip(theta[:-1], zz, strict=True):
            v += t * z
        c = linalg.cho_factor(v, lower=True)
        vi = linalg.cho_solve(c, eye)
        vix = vi @ x
        xvx = x.T @ vix
        cx = linalg.cho_factor(xvx, lower=True)
        p = vi - vix @ linalg.cho_solve(cx, vix.T)
        py = p @ yz
        logdet = 2.0 * np.sum(np.log(np.diag(c[0]))) + 2.0 * np.sum(np.log(np.diag(cx[0])))
        nll = 0.5 * (logdet + float(yz @ py))
        grad = np.empty(len(theta))
        for k, (code, z) in enumerate(zip(codes, zz, strict=True)):
            s = np.bincount(code, weights=py)
            grad[k] = 0.5 * (float(np.sum(p[z])) - float(s @ s))
        grad[-1] = 0.5 * (float(np.trace(p)) - float(py @ py))
        return nll, grad

    k = len(zz)
    start = np.full(k + 1, 1.0 / (k + 1))
    bounds = [(0.0, None)] * k + [(1e-8, None)]
    res = optimize.minimize(
        parts, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500}
    )
    if not res.success:
        raise DataError(f"REML did not converge: {res.message}")
    return res.x * scale
