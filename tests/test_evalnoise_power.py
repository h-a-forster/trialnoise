import math

import numpy as np
import pytest
from scipy import stats

from trialnoise._evalnoise.components import Components, estimate
from trialnoise._evalnoise.power import (
    Design,
    _t_power_quad,
    mde,
    naive_false_positive_rate,
    optimal_repeats,
    plan,
    power_at,
    resolve,
    se_difference,
    tasks_needed,
)
from trialnoise._evalnoise.simulate import simulate
from trialnoise.validate import _item_differences, _paired_t_reject

C = Components(item=400.0, item_arm=60.0, run=20.0, item_run=100.0, within=300.0)


@pytest.mark.parametrize(
    "design",
    [Design(30, 1, 1), Design(30, 2, 3), Design(15, 4, 2), Design(40, 1, 3, paired=False)],
)
def test_variance_formula_matches_simulation(design):
    r = resolve(C)
    diffs = []
    for s in range(1500):
        df = simulate(
            C,
            tasks=design.tasks,
            runs=design.runs,
            repeats=design.repeats,
            paired=design.paired,
            seed=s,
        )
        m = df.groupby("arm")["score"].mean()
        diffs.append(m["arm1"] - m["arm0"])
    emp = np.var(diffs, ddof=1)
    theory = se_difference(r, design) ** 2
    # var of a sample variance from n normal draws: 2 sigma^4 / (n - 1)
    assert abs(emp - theory) < 4 * theory * math.sqrt(2 / (len(diffs) - 1))


def test_z_mde_formula_and_power_round_trip():
    r = resolve(C)
    d = Design(50, 2, 2)
    se = se_difference(r, d)
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    assert mde(r, d, method="z") == pytest.approx(z * se)
    m = mde(r, d)
    assert m > mde(r, d, method="z")  # t quantiles are wider
    assert power_at(r, d, m) == pytest.approx(0.8, abs=1e-6)
    assert power_at(r, d, 0.0) == pytest.approx(0.05, abs=1e-9)


def test_t_power_equals_noncentral_t_for_paired_items():
    r = resolve(Components(item=1.0, item_arm=2.0, run=0.0, item_run=0.0, within=3.0))
    d = Design(12, 1, 2)
    se = se_difference(r, d)
    tc = stats.t.ppf(0.975, 11)
    expect = stats.nct.sf(tc, 11, 1.7 / se) + stats.nct.cdf(-tc, 11, 1.7 / se)
    assert power_at(r, d, 1.7) == pytest.approx(expect, rel=1e-9)


def test_power_where_scipy_nct_returns_nan(monkeypatch):
    # Regression: some scipy releases return nan from nct.cdf here (e.g. at
    # (-2.7764, 4.0001, 10.86)), which made the MDE root-finder fail. Force the nan so the
    # fallback is exercised whatever scipy is installed.
    monkeypatch.setattr(stats.nct, "cdf", lambda *a, **k: math.nan)
    r = resolve(Components(item=3000, item_arm=44.0, run=22.1, item_run=227.6, within=0.0))
    p = power_at(r, Design(1_000_000, 3, 1), 41.69)
    assert math.isfinite(p) and p > 0.999
    assert math.isfinite(mde(r, Design(1_000_000, 3, 1)))


@pytest.mark.parametrize("df, ncp", [(3.0, 2.0), (10.0, 1.5), (4.0, 6.0), (2.5, 0.0)])
def test_quadrature_matches_nct(df, ncp):
    tc = float(stats.t.ppf(0.975, df))
    expect = stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp)
    assert _t_power_quad(tc, df, ncp) == pytest.approx(expect, abs=1e-7)


def test_tasks_needed_is_minimal_and_unreachable_is_none():
    r = resolve(C)
    t = tasks_needed(r, 12.0, runs=4, repeats=3)
    assert t is not None
    assert mde(r, Design(t, 4, 3)) <= 12.0 < mde(r, Design(t - 1, 4, 3))
    # One run: the run term alone gives an MDE above 12 for any number of tasks.
    assert mde(r, Design(10**6, 1, 1)) > 12.0
    assert tasks_needed(r, 12.0, runs=1) is None


def test_design_effect_scales_the_tasks_needed():
    # No run term, z method: tasks needed scale exactly with the item-level variance.
    c = Components(item=1.0, item_arm=60.0, run=0.0, item_run=0.0, within=300.0)
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    for deff in (1.0, 1.5):
        exact = 2 * 360.0 * deff * z**2 / 5.0**2
        assert tasks_needed(resolve(c, design_effect=deff), 5.0, method="z") == math.ceil(exact)
    with pytest.raises(ValueError, match="design_effect"):
        resolve(c, design_effect=0.8)


def test_pass_fail_mde_uses_the_variance_at_the_shifted_rate():
    # 90% pass in pp: a loss raises the variance (towards 50%), so it is harder to detect
    # than a gain, which keeps the base-rate variance; the MDE is the loss.
    c = Components(item=400.0, item_arm=40.0, run=0.0, item_run=0.0, within=500.0)
    d = Design(50, 1, 3)
    r = resolve(c, pass_rate=0.9, pass_score=100.0)
    m = mde(r, d)
    assert m > mde(resolve(c), d)
    assert power_at(r, d, -m) == pytest.approx(0.8, abs=1e-6)
    assert mde(r, d, direction=1) == pytest.approx(mde(resolve(c), d))
    assert power_at(r, d, m) > 0.85
    # At 50% every shift moves away from 50%: the base-rate variance (and MDE) stand.
    assert mde(resolve(c, pass_rate=0.5, pass_score=100.0), d) == pytest.approx(mde(resolve(c), d))
    with pytest.raises(ValueError, match="pass_rate"):
        resolve(c, pass_rate=1.0)
    # pp variances need pass_score=100: with 1, every shift lands on a bound and the
    # variance scale never applies.
    with pytest.raises(ValueError, match="pass_score"):
        resolve(c, pass_rate=0.9)
    with pytest.raises(ValueError, match="pass_score"):
        resolve(c, pass_score=100.0)


def test_mde_needs_power_above_alpha():
    r = resolve(C)
    with pytest.raises(ValueError, match="power must exceed alpha"):
        mde(r, Design(20), alpha=0.9, power=0.5)


def test_naive_false_positive_rate():
    no_run = resolve(Components(item=1.0, item_arm=1.0, run=0.0, item_run=0.5, within=1.0))
    assert naive_false_positive_rate(no_run, Design(20)) == pytest.approx(0.05)
    r = resolve(C)
    assert naive_false_positive_rate(r, Design(20)) > naive_false_positive_rate(r, Design(20, 5))


def test_resolve_requires_run_and_flags_assumptions():
    single_run = estimate(simulate(Components(1, 1, 0, 0, 1), tasks=10, repeats=2, seed=0))
    with pytest.raises(ValueError, match="not identified"):
        resolve(single_run.components)
    r = resolve(single_run.components, run=0.0)
    assert r.run == 0.0
    assert any("folded into item_arm" in n for n in r.notes)
    with pytest.raises(ValueError, match="non-negative"):
        resolve(single_run.components, run=-1.0)
    one_arm = Components(item=2.0, within=1.0, run=0.0, absorbed={"item_run": "item"})
    with pytest.raises(ValueError, match="item x arm"):
        resolve(one_arm)
    assert resolve(one_arm, paired=False).item == 2.0


def test_optimal_repeats():
    r = resolve(C)
    assert optimal_repeats(r) == 1.0
    k = optimal_repeats(r, task_cost=20.0)
    a, b = C.item_arm + C.item_run, C.within
    assert k == pytest.approx(math.sqrt(b * 20.0 / a))
    a2, b2 = C.item_arm + C.item_run / 2, C.within / 2
    assert optimal_repeats(r, runs=2, task_cost=20.0) == pytest.approx(
        math.sqrt(b2 * 20.0 / (a2 * 2))
    )


def test_plan_grid_shape():
    rows = plan(resolve(C), [10, 20], [1, 2], [1, 3])
    assert len(rows) == 8
    assert rows[0]["trials_per_arm"] == 10


def test_design_validation():
    with pytest.raises(ValueError):
        Design(1)
    with pytest.raises(ValueError):
        Design(10, 0)
    with pytest.raises(ValueError):
        mde(resolve(C), Design(10), power=1.5)


def test_realised_power_at_planned_mde():
    # Gaussian data, no run effect, analysed by the paired t-test over items.
    comps = Components(item=400.0, item_arm=60.0, run=0.0, item_run=0.0, within=300.0)
    d = Design(20, 1, 3)
    target = mde(resolve(comps), d)
    rng = np.random.default_rng(0)
    sims = 6000
    hit = _paired_t_reject(_item_differences(comps, d, target, sims, rng), 0.05).mean()
    null = _paired_t_reject(_item_differences(comps, d, 0.0, sims, rng), 0.05).mean()
    mc = math.sqrt(0.8 * 0.2 / sims)
    assert abs(hit - 0.8) < 4 * mc
    assert abs(null - 0.05) < 4 * math.sqrt(0.05 * 0.95 / sims)


def test_naive_rate_matches_simulation():
    comps = Components(item=400.0, item_arm=60.0, run=20.0, item_run=100.0, within=0.0)
    # Few tasks, where a normal critical value would overstate the rate by about 0.03.
    d = Design(10, 1, 1)
    rng = np.random.default_rng(1)
    sims = 20000
    rate = _paired_t_reject(_item_differences(comps, d, 0.0, sims, rng), 0.05).mean()
    nominal = naive_false_positive_rate(resolve(comps), d)
    assert nominal > 0.1
    assert abs(rate - nominal) < 4 * math.sqrt(nominal * (1 - nominal) / sims)


def test_mde_falls_with_runs_whatever_the_run_df():
    # Regression: one run treated the run variance as known while two or more runs
    # estimated it with 2 (R - 1) df, so adding a run could raise the MDE.
    for run_df in (math.inf, 4.0, 0.0):
        r = resolve(C, run_df=run_df)
        runs = [2, 3, 5, 10] if run_df == 0 else [1, 2, 3, 5, 10]
        mdes = [mde(r, Design(500, k)) for k in runs]
        assert mdes == sorted(mdes, reverse=True), (run_df, mdes)
    with pytest.raises(ValueError, match="cannot estimate the run term"):
        mde(resolve(C, run_df=0.0), Design(50, 1))
    with pytest.raises(ValueError, match="run_df"):
        resolve(C, run_df=-1.0)
