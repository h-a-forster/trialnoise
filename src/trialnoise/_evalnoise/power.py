"""Minimum detectable effects and required tasks, repeats and runs for two-arm comparisons.

For two arms with ``T`` items each, ``R`` runs per arm and ``K`` repeats per item per run,
the variance of the difference in arm means under the model in
:mod:`trialnoise._evalnoise.components` is::

    paired   (same items in both arms):
        2 * (item_arm + item_run / R + within / (R K)) / T  +  2 * run / R
    unpaired (independent item samples per arm):
        2 * (item + item_arm + item_run / R + within / (R K)) / T  +  2 * run / R

The run term does not shrink with more items or repeats: only more runs per arm reduce it.
A component that the data could not separate is handled conservatively (see
:func:`resolve`), so the reported MDE is never smaller than the model implies.

Two optional adjustments (see :func:`resolve`): a design effect multiplies the item-level
term (items that share a cluster carry less information than independent ones), and for
pass/fail scores the shifted arm's half of the variance grows with ``p (1 - p)`` when its
pass rate moves towards 50%, so a loss near a 100% pass rate (or a gain near 0%) needs a
larger effect than the base-rate variance suggests; the MDE is the larger of gain and loss.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from scipy import integrate, optimize, stats

from trialnoise._evalnoise.components import Components


@dataclass(frozen=True)
class Design:
    tasks: int
    runs: int = 1
    repeats: int = 1
    paired: bool = True

    def __post_init__(self) -> None:
        if self.tasks < 2:
            raise ValueError("tasks must be at least 2")
        if self.runs < 1 or self.repeats < 1:
            raise ValueError("runs and repeats must be at least 1")

    @property
    def trials_per_arm(self) -> int:
        return self.tasks * self.runs * self.repeats


@dataclass(frozen=True)
class Resolved:
    """Finite, non-negative components ready for planning, plus notes on assumptions."""

    item: float
    item_arm: float
    run: float
    item_run: float
    within: float
    notes: tuple[str, ...]
    # Degrees of freedom of the run-variance estimate available besides the experiment's
    # own replicate runs (e.g. from a pilot); inf = known exactly, 0 = none.
    run_df: float = math.inf
    # Multiplies the item-level variance term (clustered items); 1 = independent items.
    design_effect: float = 1.0
    # Pass/fail scores: base pass rate (0 to 1) and the score of a pass (e.g. 100 for pp).
    pass_rate: float | None = None
    pass_score: float = 1.0


def resolve(
    c: Components,
    *,
    run: float | None = None,
    paired: bool = True,
    run_df: float = math.inf,
    design_effect: float = 1.0,
    pass_rate: float | None = None,
    pass_score: float = 1.0,
) -> Resolved:
    """Turn estimated components into planning inputs.

    ``run`` overrides the run variance (required when the data had one run per arm).
    ``run_df`` is the degrees of freedom of the run-variance estimate the analysis can
    borrow (a pilot's pooled runs minus one); the analysis pools it with the experiment's
    own ``2 (R - 1)``. The default treats the run variance as known exactly.
    ``design_effect`` (at least 1) multiplies the item-level term, e.g. the squared ratio
    of cluster-robust to item-level SEs; it is taken as constant across designs, and the
    item-level df stay ``T - 1``. ``pass_rate`` (strictly between 0 and 1) marks pass/fail
    scores worth ``pass_score`` per pass: the components are taken as estimated at that
    rate, and the arm shifted by an effect has its half of the variance scaled by
    ``max(1, p' (1 - p') / (p (1 - p)))`` at its own rate ``p'``. The scale is not allowed
    below 1: a shift on the logit scale moves items by different amounts, which adds
    variance that offsets the Bernoulli variance's fall (in trial runs of the ``binary-*``
    validation checks, scaling it down made the plan optimistic at about 60% pass).
    Unidentified residual parts are set to zero only when they are already folded into a
    component that does not shrink with repeats or runs, which errs towards a larger MDE.
    """
    c = c.clipped()
    notes: list[str] = []
    if run is not None:
        if not math.isfinite(run) or run < 0:
            raise ValueError("run variance must be a non-negative number")
        run_v = float(run)
        if math.isfinite(c.run):
            notes.append(f"run variance set to {run_v:.4g} (estimated {c.run:.4g})")
        else:
            notes.append(f"run variance set to {run_v:.4g} (not identified by the data)")
    elif math.isfinite(c.run):
        run_v = c.run
    else:
        raise ValueError(
            "the run variance is not identified (one run per arm); supply it, or pass 0 "
            "to assume runs do not differ"
        )
    item_arm = c.item_arm
    if paired and not math.isfinite(item_arm):
        raise ValueError("paired planning needs the item x arm variance (data with two arms)")
    item_arm = item_arm if math.isfinite(item_arm) else 0.0
    if not paired and not math.isfinite(c.item):
        raise ValueError("unpaired planning needs the item variance")
    item = c.item if math.isfinite(c.item) else 0.0
    for name in ("item_run", "within"):
        if not math.isfinite(getattr(c, name)) and name not in c.absorbed:
            raise ValueError(
                f"{name} is missing and not folded into another component; give it "
                "(0 if it is known to be absent)"
            )
    item_run = c.item_run if math.isfinite(c.item_run) else 0.0
    within = c.within if math.isfinite(c.within) else 0.0
    if c.absorbed.get("within") == "item_run":
        notes.append(
            "within-run noise is not separated from item x run noise: repeats inside a run "
            "are assumed not to help (conservative)"
        )
    if "item_run" in c.absorbed and c.absorbed["item_run"] != "item_run":
        notes.append(
            f"item x run noise is folded into {c.absorbed['item_run']}: extra runs are assumed "
            "not to reduce it (conservative)"
        )
    if c.absorbed.get("within") in ("item_arm", "item"):
        notes.append(
            "no repeats or replicate runs: trial noise is folded into the item terms, so "
            "repeats are assumed not to help (conservative)"
        )
    if not run_df >= 0:
        raise ValueError("run_df must be non-negative (inf = run variance known)")
    if not (math.isfinite(design_effect) and design_effect >= 1):
        raise ValueError("design_effect must be a finite number of at least 1")
    if pass_rate is not None and not 0 < pass_rate < 1:
        raise ValueError("pass_rate must be strictly between 0 and 1")
    if not (math.isfinite(pass_score) and pass_score > 0):
        raise ValueError("pass_score must be positive")
    if pass_rate is None and pass_score != 1.0:
        raise ValueError("pass_score applies only with pass_rate")
    # Scores of 0 or pass_score vary by at most (pass_score / 2)^2. Without this check, pp
    # variances with the default pass_score would skip the scaling: every shift would land
    # on a bound, where the variance scale is 1.
    if pass_rate is not None and item + item_arm + run_v + item_run + within > pass_score**2:
        raise ValueError(
            f"these variances are too large for pass/fail scores worth {pass_score:g}: set "
            "pass_score to the score of a pass (e.g. 100 for percentage points)"
        )
    if design_effect > 1:
        notes.append(f"item-level variance multiplied by a design effect of {design_effect:.3g}")
    if pass_rate is not None:
        notes.append(
            f"pass/fail scores at a {pass_rate:.1%} pass rate: the shifted arm's variance is "
            "taken at its own rate where that is larger, and the MDE is the larger of gain "
            "and loss"
        )
    return Resolved(
        item,
        item_arm,
        run_v,
        item_run,
        within,
        tuple(notes),
        float(run_df),
        float(design_effect),
        pass_rate,
        float(pass_score),
    )


def variance_terms(r: Resolved, d: Design, effect: float = 0.0) -> tuple[float, float]:
    """(item-level term, run term) of the variance of the estimated difference when one arm
    is shifted by ``effect`` (which matters only for pass/fail scores)."""
    per_item = r.item_arm + r.item_run / d.runs + r.within / (d.runs * d.repeats)
    if not d.paired:
        per_item += r.item
    v_items, v_run = 2.0 * per_item * r.design_effect / d.tasks, 2.0 * r.run / d.runs
    if r.pass_rate is not None and effect != 0:
        p = r.pass_rate
        shifted = min(max(p + effect / r.pass_score, 0.0), 1.0)
        f = (1.0 + max(shifted * (1.0 - shifted) / (p * (1.0 - p)), 1.0)) / 2.0
        v_items, v_run = v_items * f, v_run * f
    return v_items, v_run


def se_difference(r: Resolved, d: Design, effect: float = 0.0) -> float:
    v_items, v_run = variance_terms(r, d, effect)
    return math.sqrt(v_items + v_run)


def _df(r: Resolved, d: Design, effect: float = 0.0) -> float:
    """Satterthwaite degrees of freedom of the variance estimate the analysis would use."""
    v_items, v_run = variance_terms(r, d, effect)
    df_items = (d.tasks - 1) if d.paired else 2 * (d.tasks - 1)
    terms = [(v_items, df_items)]
    if v_run > 0:
        df_run = 2 * (d.runs - 1) + r.run_df
        if df_run == 0:
            raise ValueError(
                "one run per arm and no other estimate of the run variance: a test cannot "
                "estimate the run term; add runs or supply run_df from a pilot"
            )
        terms.append((v_run, df_run))
    denom = sum(v * v / df for v, df in terms)
    total = v_items + v_run
    return math.inf if denom == 0 else total * total / denom


def power_at(
    r: Resolved, d: Design, effect: float, *, alpha: float = 0.05, method: str = "t"
) -> float:
    """Two-sided power to detect ``effect`` (score units) at level ``alpha``."""
    _check_level(alpha, "alpha")
    if method not in ("t", "z"):
        raise ValueError("method must be 't' or 'z'")
    se = se_difference(r, d, effect)
    if se == 0:
        return 1.0 if effect != 0 else alpha
    ncp = abs(effect) / se
    df = _df(r, d, effect) if method == "t" else math.inf
    if math.isinf(df):
        zc = stats.norm.ppf(1 - alpha / 2)
        return float(stats.norm.sf(zc - ncp) + stats.norm.cdf(-zc - ncp))
    tc = stats.t.ppf(1 - alpha / 2, df)
    return _t_power(float(tc), df, ncp)


def _t_power(tc: float, df: float, ncp: float) -> float:
    """P(|T| > tc) for noncentral t; scipy's nct returns nan in parts of the parameter space."""
    p = float(stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))
    if math.isfinite(p):
        return min(max(p, 0.0), 1.0)
    if df > 1e4:
        return float(stats.norm.sf(tc - ncp) + stats.norm.cdf(-tc - ncp))
    return _t_power_quad(tc, df, ncp)


def _t_power_quad(tc: float, df: float, ncp: float) -> float:
    """T = (Z + ncp) / sqrt(X / df), X ~ chi2(df): integrate the normal tails over X."""

    def f(x: float) -> float:
        s = tc * math.sqrt(x / df)
        return float(stats.chi2.pdf(x, df) * (stats.norm.sf(s - ncp) + stats.norm.cdf(-s - ncp)))

    val, _ = integrate.quad(f, 0.0, math.inf, limit=200)
    return min(max(float(val), 0.0), 1.0)


def mde(
    r: Resolved,
    d: Design,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    method: str = "t",
    direction: int = 0,
) -> float:
    """Smallest absolute effect detected with probability ``power`` at level ``alpha``.

    For pass/fail scores (``r.pass_rate`` set) the gain (``direction=1``) and the loss
    (``-1``) differ; the default (0) is the larger, so a two-sided plan holds either way.
    """
    if direction not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0 or 1")
    _check_level(alpha, "alpha")
    _check_level(power, "power")
    if power <= alpha:
        raise ValueError("power must exceed alpha (the rejection rate of a zero effect)")
    se = se_difference(r, d)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    if se == 0:
        return 0.0
    if method not in ("t", "z"):
        raise ValueError("method must be 't' or 'z'")
    if method == "z" and r.pass_rate is None:
        return float(z * se)

    def solve(sign: float) -> float:
        def gap(e: float) -> float:
            return power_at(r, d, sign * e, alpha=alpha, method=method) - power

        hi = z * se
        while gap(hi) < 0:
            hi *= 2.0
        return float(optimize.brentq(gap, 0.0, hi, xtol=1e-12))

    if r.pass_rate is None:
        return solve(1.0)
    return solve(float(direction)) if direction else max(solve(1.0), solve(-1.0))


def tasks_needed(
    r: Resolved,
    target: float,
    *,
    runs: int = 1,
    repeats: int = 1,
    paired: bool = True,
    alpha: float = 0.05,
    power: float = 0.8,
    method: str = "t",
    max_tasks: int = 1_000_000,
) -> int | None:
    """Fewest tasks per arm whose MDE is at most ``target``; ``None`` if unreachable.

    Unreachable means the run term alone exceeds the budget: add runs, not tasks.
    """
    if target <= 0:
        raise ValueError("target effect must be positive")

    def ok(t: int) -> bool:
        return (
            mde(r, Design(t, runs, repeats, paired), alpha=alpha, power=power, method=method)
            <= target
        )

    if not ok(max_tasks):
        return None
    lo, hi = 2, 2
    while not ok(hi):
        lo, hi = hi, min(hi * 2, max_tasks)
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return hi


def naive_false_positive_rate(r: Resolved, d: Design, *, alpha: float = 0.05) -> float:
    """False-positive rate of a test that ignores run effects (e.g. paired over items).

    Such a test estimates only the item-level term of the variance, so a run-to-run shift
    reads as an arm difference. The arms have equal mean scores but keep their item x arm
    effects; for literally identical arms pass ``item_arm = 0``. Equals ``alpha`` when the
    run variance is zero. The test is a t-test over items (``T - 1`` df paired, ``2 (T - 1)``
    unpaired), whose statistic is ``sqrt((v_items + v_run) / v_items)`` times a central t.
    """
    v_items, v_run = variance_terms(r, d)
    if v_items + v_run == 0:
        return alpha
    df = (d.tasks - 1) if d.paired else 2 * (d.tasks - 1)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(2 * stats.t.sf(tc * math.sqrt(v_items / (v_items + v_run)), df))


def optimal_repeats(r: Resolved, *, runs: int = 1, task_cost: float = 0.0) -> float:
    """Repeats per item per run that minimise variance for a fixed budget.

    ``task_cost`` is the extra cost of adding a task, in units of one trial (0 when tasks
    already exist and only trials cost money). With ``task_cost = 0`` the answer is 1: a
    new task always buys more precision than another repeat of an old one.
    """
    if task_cost < 0:
        raise ValueError("task_cost must be non-negative")
    a = r.item_arm + r.item_run / runs
    b = r.within / runs
    if task_cost == 0 or b == 0:
        return 1.0
    if a == 0:
        return math.inf
    return max(1.0, math.sqrt(b * task_cost / (a * runs)))


def plan(
    r: Resolved,
    tasks: Iterable[int],
    runs: Iterable[int] = (1,),
    repeats: Iterable[int] = (1,),
    *,
    paired: bool = True,
    alpha: float = 0.05,
    power: float = 0.8,
    method: str = "t",
) -> list[dict[str, float | int]]:
    """MDE, standard error and naive false-positive rate over a grid of designs."""
    rows: list[dict[str, float | int]] = []
    for nr in runs:
        for k in repeats:
            for t in tasks:
                d = Design(t, nr, k, paired)
                rows.append(
                    {
                        "tasks": t,
                        "runs": nr,
                        "repeats": k,
                        "trials_per_arm": d.trials_per_arm,
                        "se": se_difference(r, d),
                        "mde": mde(r, d, alpha=alpha, power=power, method=method),
                        "naive_fpr": naive_false_positive_rate(r, d, alpha=alpha),
                    }
                )
    return rows


def _check_level(x: float, name: str) -> None:
    if not 0 < x < 1:
        raise ValueError(f"{name} must be between 0 and 1")
