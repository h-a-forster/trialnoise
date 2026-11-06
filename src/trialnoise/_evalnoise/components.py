"""Method-of-moments estimates of variance components for crossed eval designs.

Model for the score of arm ``a`` on item ``i`` in run ``r`` (repeat ``k``)::

    y = mu_a + u_i + v_ai + w_ar + g_air + e_airk

with independent zero-mean effects: item ``u`` (variance ``item``), item x arm ``v``
(``item_arm``), run ``w`` (``run``; a shock shared by every trial of one run of one arm),
item x run ``g`` (``item_run``) and trial noise ``e`` (``within``). See docs/DESIGN.md.

Which components a table identifies depends on its layout:

- ``within`` needs cells (arm, run, item) with two or more trials;
- ``run`` and ``item_run + within`` need an arm with two or more runs;
- ``item_arm`` needs two or more arms. Without a residual estimate it absorbs the residual,
  and with a single run per arm it absorbs ``item_run`` (flagged in ``absorbed``).

The estimators are unbiased moment estimators (Henderson method III for runs, two-way
ANOVA on run-adjusted item means for items) and are cheap enough to bootstrap.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from trialnoise._evalnoise.data import DataError

NAN = float("nan")


@dataclass(frozen=True)
class Components:
    """Variance components in squared score units. ``nan`` means not identified."""

    item: float = NAN
    item_arm: float = NAN
    run: float = NAN
    item_run: float = NAN
    within: float = NAN
    # Components folded into another one because the layout cannot separate them,
    # e.g. {"item_run": "item_arm"}.
    absorbed: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Components:
        known = set(cls.__dataclass_fields__)
        extra = set(data) - known
        if extra:
            raise ValueError(f"unknown component(s): {', '.join(sorted(extra))}")
        values: dict[str, Any] = {}
        for k, v in data.items():
            if k == "absorbed":
                values[k] = dict(v)  # type: ignore[call-overload]
            else:  # None: not identified (how result files store nan)
                values[k] = NAN if v is None else float(v)  # type: ignore[arg-type]
        return cls(**values)

    def clipped(self) -> Components:
        """Negative estimates set to zero (the usual reporting convention)."""

        def clip(v: float) -> float:
            return max(v, 0.0) if math.isfinite(v) else v

        return Components(
            item=clip(self.item),
            item_arm=clip(self.item_arm),
            run=clip(self.run),
            item_run=clip(self.item_run),
            within=clip(self.within),
            absorbed=dict(self.absorbed),
        )


@dataclass
class RunTerm:
    """Pooled Henderson-III quantities for the run component (arms with >= 2 runs)."""

    ss: float = 0.0  # run sum of squares adjusted for items
    df: int = 0  # its degrees of freedom
    k: float = 0.0  # coefficient of the run variance in E[ss]
    rss: float = 0.0  # residual sum of squares of the item + run fit
    df_res: int = 0
    n_harmonic: float = NAN  # harmonic mean trials per cell in those arms
    # Residuals of the item + run fit on cell means and their item codes (one arm only;
    # ``pool`` leaves them empty).
    residuals: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    residual_items: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64), repr=False, compare=False
    )

    @classmethod
    def pool(cls, terms: Iterable[RunTerm]) -> RunTerm:
        """Sum the quantities of several arms (``n_harmonic`` is left unset)."""
        out = cls()
        for t in terms:
            out.ss += t.ss
            out.df += t.df
            out.k += t.k
            out.rss += t.rss
            out.df_res += t.df_res
        return out

    def identified(self) -> bool:
        return self.df >= 1 and self.df_res >= 1 and self.k > 0

    def variance(self) -> float:
        """Moment estimate of the run variance; nan when not identified."""
        if not self.identified():
            return NAN
        return (self.ss - self.df * self.rss / self.df_res) / self.k

    def interval(self, level: float = 0.95, max_var: float = math.inf) -> tuple[float, float]:
        """Burdick-Graybill modified large-sample (MLS) interval for the run variance.

        The estimate is ``c (ms_run - ms_res)`` with ``c = df / k``; the MLS interval for a
        difference of mean squares accounts for the uncertainty of both (it reduces to the
        chi-square interval ``ss / E[ms] ~ chi2(df) / df`` when the residual df are large),
        and is exact in the limit for balanced normal data. ``max_var`` caps both ends, e.g.
        at ``(range / 2) ** 2`` for scores bounded to a range. When even the upper end falls
        below zero (the runs agree more closely than the residual allows), no run variance
        fits and the interval is degenerate: (nan, nan), not a zero-width [0, 0].
        """
        if not self.identified():
            return (NAN, NAN)
        if not max_var > 0:
            raise ValueError("max_var must be positive")
        a = (1.0 - level) / 2
        n1, n2 = self.df, self.df_res
        m1, m2 = self.ss / n1, self.rss / n2
        c = n1 / self.k
        # F quantiles with infinite denominator df are chi2 / df; "u" leaves a to the right.
        f1u, f1l = stats.chi2.ppf(1 - a, n1) / n1, stats.chi2.ppf(a, n1) / n1
        f2u, f2l = stats.chi2.ppf(1 - a, n2) / n2, stats.chi2.ppf(a, n2) / n2
        fu, fl = stats.f.ppf(1 - a, n1, n2), stats.f.ppf(a, n1, n2)
        g1, h1 = 1 - 1 / f1u, 1 / f1l - 1
        g2, h2 = 1 - 1 / f2u, 1 / f2l - 1
        g12 = ((fu - 1) ** 2 - g1**2 * fu**2 - h2**2) / fu
        h12 = ((1 - fl) ** 2 - h1**2 * fl**2 - g2**2) / fl
        v_lo = g1**2 * m1**2 + h2**2 * m2**2 + g12 * m1 * m2
        v_hi = h1**2 * m1**2 + g2**2 * m2**2 + h12 * m1 * m2
        est = c * (m1 - m2)
        lo = est - c * math.sqrt(max(v_lo, 0.0))
        hi = est + c * math.sqrt(max(v_hi, 0.0))
        if hi <= 0:
            return (NAN, NAN)
        return (min(max(lo, 0.0), max_var), min(hi, max_var))


@dataclass
class Estimate:
    components: Components
    arms: list[str]
    items: int
    clusters: int
    runs: dict[str, int]
    run_term: RunTerm
    arm_means: dict[str, float]
    per_arm: dict[str, dict[str, float]]
    pairs: dict[str, dict[str, float]]
    # Run term of each arm with two or more runs; ``run_term`` pools them.
    run_terms: dict[str, RunTerm] = field(default_factory=dict)


class _Coded:
    """Integer-coded trial table with per-cell sufficient statistics."""

    def __init__(self, df: pd.DataFrame, arms: Sequence[str] | None = None) -> None:
        if arms is not None:
            missing = sorted(set(arms) - set(df["arm"]))
            if missing:
                raise DataError(f"arm(s) not in data: {', '.join(missing)}")
            df = df[df["arm"].isin(list(arms))]
        if len(df) == 0:
            raise DataError("no trials")
        self.arm_names = list(arms) if arms is not None else sorted(df["arm"].unique())
        arm_index = {a: j for j, a in enumerate(self.arm_names)}
        self.a = df["arm"].map(arm_index).to_numpy()
        self.item_codes, self.item_names = pd.factorize(df["item"], sort=True)
        runs = df["arm"] + "\x1f" + df["run"]
        self.run_codes, run_names = pd.factorize(runs, sort=True)
        self.run_arm = np.array([arm_index[n.split("\x1f")[0]] for n in run_names])
        self.cluster_codes, _ = pd.factorize(df["cluster"], sort=True)
        self.y = df["score"].to_numpy(dtype=float)
        self.n_items = len(self.item_names)
        self.n_runs = len(run_names)
        self.n_clusters = int(self.cluster_codes.max()) + 1
        # The data layer guarantees one cluster per item.
        self.item_cluster = np.empty(self.n_items, dtype=np.int64)
        self.item_cluster[self.item_codes] = self.cluster_codes


def _cells(a: np.ndarray, i: np.ndarray, r: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """Unique (run, item) cells with arm, n, sum and within sum of squares."""
    key = r.astype(np.int64) * (int(i.max()) + 1) + i
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    n = np.diff(np.r_[starts, len(k_sorted)])
    ys = y[order]
    s = np.add.reduceat(ys, starts)
    dev = ys - np.repeat(s / n, n)  # two-pass: sum(y^2) - s^2/n cancels on large scores
    ss = np.add.reduceat(dev * dev, starts)
    first = order[starts]
    return a[first], i[first], r[first], n, s / n, np.maximum(ss, 0.0)


def _run_fit(
    cell_item: np.ndarray, cell_run: np.ndarray, m: np.ndarray
) -> tuple[float, int, float, float, int, dict[int, float], np.ndarray]:
    """Henderson III for one arm: item + run fixed-effects fit on cell means.

    Returns (ss_run, df_run, k, rss_full, df_res, run_effects, residuals) where run effects
    are centred to mean zero over runs and residuals are those of the full fit, per cell.
    """
    items, i_idx = np.unique(cell_item, return_inverse=True)
    runs, r_idx = np.unique(cell_run, return_inverse=True)
    n_i = np.bincount(i_idx).astype(float)
    item_mean = np.bincount(i_idx, weights=m) / n_i
    dev = m - item_mean[i_idx]
    rss_item = float(dev @ dev)
    n_r = len(runs)
    # Reduced normal equations for run effects after absorbing items.
    q = np.bincount(r_idx, weights=dev, minlength=n_r)
    c = np.zeros((n_r, n_r))
    np.add.at(c, (r_idx, r_idx), 1.0)
    # Subtract sum_i z_ij z_il / n_i: cells of the same item in runs j and l.
    for item in range(len(items)):
        rr = r_idx[i_idx == item]
        if len(rr):
            c[np.ix_(rr, rr)] -= 1.0 / n_i[item]
    # C is singular by construction (its rows sum to zero); rounding in the per-item
    # subtraction leaves tiny eigenvalues that the default tolerances count as rank.
    tol = 1e-9 * max(float(np.abs(c).max()), 1.0)
    rank = int(np.sum(np.linalg.eigvalsh(c) > tol)) if n_r > 1 else 0
    c_pinv = np.linalg.pinv(c, rcond=1e-9, hermitian=True)
    w = c_pinv @ q
    ss_run = float(q @ w)
    k = float(np.trace(c))
    rss_full = max(rss_item - ss_run, 0.0)
    df_res = len(m) - len(items) - rank
    effects = {int(runs[j]): float(w[j] - w.mean()) for j in range(n_r)}
    # Full-fit residuals: item-centred cell means minus the item-centred run effects.
    zw = w[r_idx]
    resid = dev - (zw - (np.bincount(i_idx, weights=zw) / n_i)[i_idx])
    return ss_run, rank, k, rss_full, df_res, effects, resid


def estimate(
    df: pd.DataFrame,
    arms: Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] = (),
) -> Estimate:
    """Moment estimates of the variance components of a validated trial table."""
    coded = _Coded(df, arms)
    return _estimate_coded(coded, coded.item_codes, list(pairs), coded.item_cluster)


def _estimate_coded(
    coded: _Coded,
    item_codes: np.ndarray,
    pairs: list[tuple[str, str]],
    item_cluster: np.ndarray | None = None,
) -> Estimate:
    a_all, r_all, y = coded.a, coded.run_codes, coded.y
    n_arms = len(coded.arm_names)
    ca, ci, cr, cn, cm, css = _cells(a_all, item_codes, r_all, y)

    # Within-cell variance.
    dfw = float(np.sum(cn - 1))
    within = float(np.sum(css) / dfw) if dfw > 0 else NAN

    # Run term, pooled over arms with >= 2 runs.
    run_terms: dict[str, RunTerm] = {}
    run_effect: dict[int, float] = {}
    inv_n: list[float] = []
    for arm in range(n_arms):
        sel = ca == arm
        if len(np.unique(cr[sel])) < 2:
            continue
        ss, dfr, k, rss, dfe, eff, resid = _run_fit(ci[sel], cr[sel], cm[sel])
        n_h = 1.0 / float(np.mean(1.0 / cn[sel]))
        run_terms[coded.arm_names[arm]] = RunTerm(ss, dfr, k, rss, dfe, n_h, resid, ci[sel])
        run_effect.update(eff)
        inv_n.extend((1.0 / cn[sel]).tolist())
    rt = RunTerm.pool(run_terms.values())
    absorbed: dict[str, str] = {}
    if rt.identified():
        s2_cell = rt.rss / rt.df_res
        run = rt.variance()
        rt.n_harmonic = 1.0 / float(np.mean(inv_n))
        if math.isfinite(within):
            item_run = s2_cell - within / rt.n_harmonic
        else:
            # Every cell in the replicated arms holds one trial: s2_cell is item_run + within.
            item_run = s2_cell
            absorbed["within"] = "item_run"
    else:
        run = NAN
        item_run = NAN
        target = "item_arm" if n_arms > 1 else "item"
        absorbed["item_run"] = target
        if not math.isfinite(within):
            absorbed["within"] = target

    def cell_noise(n: np.ndarray) -> np.ndarray:
        """Variance of a cell mean around its item x arm x run expectation."""
        out = np.zeros(len(n))
        if math.isfinite(item_run):
            out += item_run
        if math.isfinite(within):
            out += within / n
        return out

    # Run-adjusted cell means, then item x arm means and their noise variances.
    adj = cm - np.array([run_effect.get(int(r), 0.0) for r in cr])
    noise = cell_noise(cn.astype(float))
    key = ca.astype(np.int64) * coded.n_items + ci
    uk, inv = np.unique(key, return_inverse=True)
    runs_per = np.bincount(inv).astype(float)
    ybar = np.bincount(inv, weights=adj) / runs_per
    vbar = np.bincount(inv, weights=noise) / runs_per**2
    ua = uk // coded.n_items
    ui = uk % coded.n_items
    table = np.full((n_arms, coded.n_items), np.nan)
    vtab = np.full((n_arms, coded.n_items), np.nan)
    table[ua, ui] = ybar
    vtab[ua, ui] = vbar
    present = np.all(np.isfinite(table), axis=0)
    n_common = int(present.sum())
    if n_common < 2:
        raise DataError("fewer than 2 items are shared by all arms")
    yt = table[:, present]
    vt = vtab[:, present]

    item_arm = NAN
    if n_arms == 1:
        item = float(np.var(yt[0], ddof=1) - vt.mean())
    else:
        grand = yt.mean()
        col = yt.mean(axis=0)
        row = yt.mean(axis=1)
        ms_item = n_arms * float(np.sum((col - grand) ** 2)) / (n_common - 1)
        resid = yt - row[:, None] - col[None, :] + grand
        ms_inter = float(np.sum(resid**2)) / ((n_arms - 1) * (n_common - 1))
        item_arm = ms_inter - float(vt.mean())
        item = (ms_item - ms_inter) / n_arms

    comps = Components(
        item=item, item_arm=item_arm, run=run, item_run=item_run, within=within, absorbed=absorbed
    )

    arm_means = {}
    per_arm: dict[str, dict[str, float]] = {}
    for j, name in enumerate(coded.arm_names):
        sel = a_all == j
        arm_means[name] = float(y[sel].mean())
        row_vals = table[j, np.isfinite(table[j])]
        row_noise = vtab[j, np.isfinite(table[j])]
        cells_j = ca == j
        dfw_j = float(np.sum(cn[cells_j] - 1))
        per_arm[name] = {
            "mean": arm_means[name],
            "trials": int(sel.sum()),
            "runs": int(len(np.unique(cr[cells_j]))),
            "item_plus_item_arm": float(np.var(row_vals, ddof=1) - row_noise.mean())
            if len(row_vals) > 1
            else NAN,
            "within": float(np.sum(css[cells_j]) / dfw_j) if dfw_j > 0 else NAN,
        }

    pair_out: dict[str, dict[str, float]] = {}
    index = {n: j for j, n in enumerate(coded.arm_names)}
    for left, right in pairs:
        if left not in index or right not in index:
            raise DataError(f"pair {left} - {right}: arm not in data")
        ja, jb = index[left], index[right]
        both = np.isfinite(table[ja]) & np.isfinite(table[jb])
        d = table[ja, both] - table[jb, both]
        nd = len(d)
        if nd < 2:
            raise DataError(f"pair {left} - {right}: fewer than 2 shared items")
        noise_d = float(np.mean(vtab[ja, both] + vtab[jb, both]))
        var_d = float(np.var(d, ddof=1))
        pair_out[f"{left} - {right}"] = {
            "items": nd,
            "items_differing": int(np.count_nonzero(np.abs(d) > 1e-9 * (1.0 + np.abs(d).max()))),
            "difference": float(d.mean()),
            "se_items": math.sqrt(var_d / nd),
            "var_d": var_d,
            # Per-arm item x arm variance implied by this pair (half the difference variance).
            "item_arm": (var_d - noise_d) / 2.0,
        }
        if item_cluster is not None:
            # Cluster-robust SE of the mean difference (equals se_items with one item per
            # cluster): items in one cluster may share more than the model's item effect.
            _, g = np.unique(item_cluster[both], return_inverse=True)
            n_g = int(g.max()) + 1
            s = np.bincount(g, weights=d - d.mean())
            pair_out[f"{left} - {right}"]["clusters"] = n_g
            pair_out[f"{left} - {right}"]["se_clusters"] = (
                math.sqrt(n_g / (n_g - 1) * float(s @ s)) / nd if n_g > 1 else NAN
            )

    runs = {name: int(len(np.unique(cr[ca == j]))) for j, name in enumerate(coded.arm_names)}
    return Estimate(
        components=comps,
        arms=list(coded.arm_names),
        items=n_common,
        clusters=coded.n_clusters,
        runs=runs,
        run_term=rt,
        arm_means=arm_means,
        per_arm=per_arm,
        pairs=pair_out,
        run_terms=run_terms,
    )


@dataclass
class Bootstrap:
    """Percentile intervals from resampling clusters (items, or groups of items)."""

    draws: int
    level: float
    components: dict[str, tuple[float, float]]
    pairs: dict[str, dict[str, tuple[float, float]]]
    samples: dict[str, np.ndarray] = field(repr=False, default_factory=dict)


def bootstrap(
    df: pd.DataFrame,
    arms: Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] = (),
    *,
    draws: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> Bootstrap:
    """Resample clusters with replacement and re-estimate; duplicated items stay distinct.

    Runs are not resampled (there are rarely more than a handful), so the intervals for
    ``run`` reflect item sampling only; use :meth:`RunTerm.interval` for the run variance.
    """
    if draws < 10:
        raise ValueError("draws must be at least 10")
    coded = _Coded(df, arms)
    pairs = list(pairs)
    rng = np.random.default_rng(seed)
    rows_by_cluster = pd.Series(np.arange(len(coded.y))).groupby(coded.cluster_codes).indices
    cluster_rows = [rows_by_cluster[c] for c in range(coded.n_clusters)]
    names = ("item", "item_arm", "run", "item_run", "within")
    out: dict[str, list[float]] = {n: [] for n in names}
    pair_keys = [f"{left} - {right}" for left, right in pairs]
    pair_out: dict[str, dict[str, list[float]]] = {
        k: {"difference": [], "item_arm": [], "se_items": []} for k in pair_keys
    }
    failures = 0
    for _ in range(draws):
        pick = rng.integers(0, coded.n_clusters, coded.n_clusters)
        rows = np.concatenate([cluster_rows[c] for c in pick])
        copy = np.concatenate([np.full(len(cluster_rows[c]), d) for d, c in enumerate(pick)])
        sub = _subset(coded, rows)
        items = coded.item_codes[rows].astype(np.int64) + copy * coded.n_items
        _, items = np.unique(items, return_inverse=True)
        sub.n_items = int(items.max()) + 1
        try:
            est = _estimate_coded(sub, items, pairs)
        except DataError:
            failures += 1
            continue
        c = est.components
        for n in names:
            out[n].append(getattr(c, n))
        for k in pair_keys:
            for f in ("difference", "item_arm", "se_items"):
                pair_out[k][f].append(est.pairs[k][f])
    if failures > draws // 10:
        raise DataError(f"{failures} of {draws} bootstrap draws could not be estimated")
    alpha = 1.0 - level

    def pct(v: list[float]) -> tuple[float, float]:
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return (NAN, NAN)
        lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
        return (float(lo), float(hi))

    return Bootstrap(
        draws=draws - failures,
        level=level,
        components={n: pct(out[n]) for n in names},
        pairs={k: {f: pct(v) for f, v in d.items()} for k, d in pair_out.items()},
        samples={n: np.asarray(out[n]) for n in names},
    )


def _subset(coded: _Coded, rows: np.ndarray) -> _Coded:
    sub = object.__new__(_Coded)
    sub.arm_names = coded.arm_names
    sub.a = coded.a[rows]
    sub.item_codes = coded.item_codes[rows]
    sub.item_names = coded.item_names
    sub.run_codes = coded.run_codes[rows]
    sub.run_arm = coded.run_arm
    sub.cluster_codes = coded.cluster_codes[rows]
    sub.y = coded.y[rows]
    sub.n_items = coded.n_items
    sub.n_runs = coded.n_runs
    sub.n_clusters = coded.n_clusters
    sub.item_cluster = coded.item_cluster
    return sub
