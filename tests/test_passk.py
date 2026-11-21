import itertools
import math
from fractions import Fraction

import pandas as pd
import pytest

from trialnoise import passk as pk
from trialnoise._evalnoise.data import DataError

RATES = [Fraction(0), Fraction(1, 7), Fraction(1, 2), Fraction(4, 5), Fraction(1)]
DESIGNS = [(n, k) for n in range(1, 8) for k in range(1, n + 1)]


def brute_force(n, k, p, metric):
    """Every one of the 2^n outcome sequences; the estimator as the share of k-subsets that pass.

    Nothing here uses a binomial coefficient, so it does not share the module's formulas.
    """
    subsets = list(itertools.combinations(range(n), k))
    law: dict[Fraction, Fraction] = {}
    for outcome in itertools.product((0, 1), repeat=n):
        mass = math.prod(p if x else 1 - p for x in outcome)
        judge = all if metric == pk.POW else any
        value = Fraction(sum(judge(outcome[i] for i in s) for s in subsets), len(subsets))
        law[value] = law.get(value, Fraction(0)) + mass
    return {v: m for v, m in sorted(law.items()) if m}


@pytest.mark.parametrize("metric", pk.METRICS)
def test_law_matches_enumeration_of_every_outcome_sequence(metric):
    for n, k in DESIGNS:
        for p in RATES:
            assert pk.law(n, k, p, metric) == brute_force(n, k, p, metric), (n, k, p)


@pytest.mark.parametrize("metric", pk.METRICS)
def test_estimator_is_unbiased_and_the_closed_form_variance_is_the_law_s(metric):
    for n, k in DESIGNS:
        for p in RATES:
            mean, var = pk.moments(pk.law(n, k, p, metric))
            assert mean == pk.truth(p, k, metric)
            assert var == pk.variance(n, k, p, metric)
            assert sum(pk.law(n, k, p, metric).values()) == 1


def test_first_order_cases_are_the_textbook_ones():
    p = Fraction(3, 10)
    assert pk.variance(9, 1, p) == p * (1 - p) / 9
    assert pk.variance(4, 4, p) == p**4 - p**8  # one Bernoulli: did all four pass
    assert pk.estimate(3, 5, 2) == Fraction(3, 10)
    assert pk.estimate(3, 5, 2, pk.AT) == Fraction(9, 10)
    assert pk.truth(p, 3, pk.AT) == 1 - Fraction(7, 10) ** 3


def test_pass_at_k_is_pass_hat_k_of_the_failures():
    for n, k in DESIGNS:
        for c in range(n + 1):
            assert pk.estimate(c, n, k, pk.AT) == 1 - pk.estimate(n - c, n, k)
        assert pk.variance(n, k, Fraction(1, 7), pk.AT) == pk.variance(n, k, Fraction(6, 7))


def test_plugin_overstates_pass_hat_k_and_understates_pass_at_k():
    for n, k in DESIGNS:
        for c in range(n + 1):  # pointwise, so in expectation too
            assert pk.plugin(c, n, k) >= pk.estimate(c, n, k)
            assert pk.plugin(c, n, k, pk.AT) <= pk.estimate(c, n, k, pk.AT)
        for p in RATES:
            strict = k >= 2 and 0 < p < 1
            up, down = pk.plugin_bias(n, k, p), pk.plugin_bias(n, k, p, pk.AT)
            assert (up > 0 and down < 0) if strict else (up == 0 and down == 0), (n, k, p)
    # (c/5)^2 at p = 1/2: E[C^2]/25 = (5/4 + 25/4)/25 = 3/10, against 1/4.
    assert pk.plugin_bias(5, 2, Fraction(1, 2)) == Fraction(1, 20)


def test_no_unbiased_estimator_beyond_n_trials():
    with pytest.raises(DataError, match="1..n only"):
        pk.estimate(3, 4, 5)
    with pytest.raises(DataError):
        pk.law(4, 0, Fraction(1, 2))
    with pytest.raises(DataError):
        pk.rate(1.2)
    assert pk.rate(0.8) == Fraction(4, 5)


@pytest.mark.parametrize("metric", pk.METRICS)
def test_variance_estimate_is_unbiased_and_never_negative(metric):
    for n, k in [(n, k) for n, k in DESIGNS if 2 * k <= n]:
        assert all(pk.variance_estimate(c, n, k, metric) >= 0 for c in range(n + 1))
        for p in RATES:
            expected = sum(
                m * pk.variance_estimate(c, n, k, metric) for c, m in enumerate(pk.binomial(n, p))
            )
            assert expected == pk.variance(n, k, p, metric)
    with pytest.raises(DataError, match="2k"):
        pk.variance_estimate(2, 5, 3)


def test_worst_variance_bounds_the_variance_at_every_rate():
    for n, k in [(4, 1), (5, 2), (5, 5), (8, 3), (16, 8), (16, 16)]:
        worst, at = pk.worst_variance(n, k)
        grid = [Fraction(i, 200) for i in range(201)]
        top = max(pk.variance(n, k, p) for p in grid)
        assert worst >= float(top) - 1e-15
        assert worst == pytest.approx(float(pk.variance(n, k, Fraction(at))), rel=1e-9)
        assert worst <= float(top) * 1.01  # and is not a loose bound
    assert pk.worst_variance(10, 1) == (pytest.approx(1 / 40), pytest.approx(0.5, abs=1e-6))
    # k = n is one Bernoulli(p^n): a quarter at most.
    assert pk.worst_variance(5, 5)[0] == pytest.approx(0.25)


def test_worst_variance_never_rises_with_more_trials():
    for k in (1, 2, 3, 5):
        values = [pk.worst_variance(n, k)[0] for n in range(k, 40)]
        assert all(a >= b for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize("metric", pk.METRICS)
def test_task_interval_covers_at_least_95_percent_exactly(metric):
    for n in (1, 4, 5, 8, 16):
        for k in (1, 2, n, n + 3):  # the interval, unlike the estimator, allows k > n
            intervals = [pk.task_interval(c, n, k, metric) for c in range(n + 1)]
            for i in range(0, 101):
                p = Fraction(i, 100)
                target = float(pk.truth(p, k, metric))
                covered = sum(
                    m
                    for (lo, hi), m in zip(intervals, pk.binomial(n, p), strict=True)
                    if lo - 1e-12 <= target <= hi + 1e-12
                )
                assert covered >= Fraction(95, 100), (n, k, p)
    # Not vacuous: all five passing still leaves pass^5 anywhere above 2.5%.
    assert pk.task_interval(5, 5, 5) == (pytest.approx(0.025), 1.0)


def joint_law(tasks, k, metric):
    """The benchmark law from every joint vector of success counts, without convolution."""
    law: dict[Fraction, Fraction] = {}
    masses = [pk.binomial(n, p) for n, p in tasks]
    for counts in itertools.product(*(range(n + 1) for n, _ in tasks)):
        mass = math.prod(m[c] for m, c in zip(masses, counts, strict=True))
        value = sum(
            pk.estimate(c, n, k, metric) for c, (n, _) in zip(counts, tasks, strict=True)
        ) / len(tasks)
        law[value] = law.get(value, Fraction(0)) + mass
    return {v: m for v, m in sorted(law.items()) if m}


@pytest.mark.parametrize("metric", pk.METRICS)
def test_benchmark_law_is_the_joint_enumeration(metric):
    tasks = [(3, Fraction(1, 3)), (4, Fraction(1, 2)), (5, Fraction(9, 10)), (3, Fraction(1))]
    for k in (1, 2, 3):
        dist = pk.benchmark_law(tasks, k, metric)
        assert dist == joint_law(tasks, k, metric)
        mean, var = pk.moments(dist)
        assert mean == sum(pk.truth(p, k, metric) for _, p in tasks) / 4
        assert var == sum(pk.variance(n, k, p, metric) for n, p in tasks) / 16


def test_guaranteed_and_normal_half_widths_cover_as_stated():
    """Exact coverage from the convolved law, at the worst-case p and at an easy one."""
    for n, k, p in [(5, 1, Fraction(1, 2)), (5, 3, Fraction(4, 5)), (5, 5, Fraction(87, 100))]:
        tasks = [(n, p)] * 30
        dist = pk.benchmark_law(tasks, k)
        worst = [pk.worst_variance(n, k)[0]] * 30
        sure = pk.half_width(worst, kind="guaranteed")
        assert pk.coverage(dist, pk.truth(p, k), sure) >= Fraction(95, 100)
        # The z interval is an approximation: near 95% at 30 tasks, not guaranteed.
        assert pk.coverage(dist, pk.truth(p, k), pk.half_width(worst)) >= Fraction(93, 100)
        assert sure > pk.half_width(worst)
    # Hoeffding caps the guaranteed width whatever the variances.
    assert pk.half_width([0.25] * 10, kind="guaranteed") == pytest.approx(
        math.sqrt(math.log(40) / 20)
    )
    assert pk.half_width([0.05] * 89) == pytest.approx(1.959964 * math.sqrt(0.05 / 89))


def test_trials_needed_is_the_first_n_that_fits():
    # k = 1 at the worst p: 2 * 1.96 * sqrt(1 / (4 n * 89)) <= 0.05 from n = 18.
    assert pk.trials_needed(1, 0.05, 89) == 18
    for k, width, kind in [(3, 0.10, "normal"), (2, 0.20, "guaranteed"), (5, 0.15, "normal")]:
        n = pk.trials_needed(k, width, 89, kind=kind)
        assert n is not None and n > k

        def full(m, k=k, kind=kind):
            return 2 * pk.half_width([pk.worst_variance(m, k)[0]] * 89, kind=kind)

        assert full(n) <= width < full(n - 1)
    assert pk.trials_needed(3, 0.01, 10, max_trials=50) is None
    # The guaranteed width cannot fall below Bernstein's floor, however many trials.
    floor = pk.guaranteed_floor(89)
    assert pk.half_width([0.0] * 89, kind="guaranteed") == pytest.approx(floor)
    assert 2 * floor > 0.05 and pk.trials_needed(1, 0.05, 89, kind="guaranteed") is None
    assert pk.trials_needed(1, 3 * floor, 89, kind="guaranteed") is not None  # about 36
    easy = pk.trials_needed(3, 0.10, 89, p=Fraction(99, 100))
    assert easy is not None and easy < pk.trials_needed(3, 0.10, 89)
    with pytest.raises(DataError):
        pk.trials_needed(3, 0.0, 89)


def test_benchmark_scores_a_table_and_counts_what_it_drops():
    counts = [(5, 5), (0, 5), (3, 5), (2, 2), (4, 6)]
    b = pk.benchmark(counts, 3)
    assert (b["tasks"], b["dropped"]) == (4, 1)  # the task with two trials has no pass^3
    assert b["estimate"] == pytest.approx((1 + 0 + 1 / 10 + 4 / 20) / 4)
    assert b["plugin"] == pytest.approx((1 + 0 + 0.6**3 + (4 / 6) ** 3) / 4)
    assert b["se_fixed"] is None  # 2k > n on some task
    one = pk.benchmark(counts, 1)
    assert one["dropped"] == 0 and one["estimate"] == pytest.approx(one["plugin"])
    # k = 1, these tasks: sum of c (n - c) / (n^2 (n - 1)) over tasks, the usual p q / (n - 1).
    assert one["se_fixed"] == pytest.approx(math.sqrt(6 / 100 + 8 / 180) / 5)
    assert one["interval"][0] <= one["estimate"] <= one["interval"][1]
    with pytest.raises(DataError, match="fewer than two tasks"):
        pk.benchmark(counts, 6)


def test_cell_rows_from_a_trial_table():
    trials = pd.DataFrame(
        {
            "cell": ["a"] * 8 + ["b"] * 8,
            "task": (["t1"] * 4 + ["t2"] * 4) * 2,
            "reward": [1, 1, 1, 1, 1, 0, 0, 0] + [0, 0, 0, 0, 1, 1, 0, 1],
        }
    )
    rows = pk.cell_rows(trials, 9).set_index(["cell", "k"])
    assert sorted(rows.index.get_level_values("k").unique()) == [1, 2, 3, 4]  # k stops at min n
    assert rows.loc[("a", 1), "pow"] == pytest.approx(5 / 8)
    assert rows.loc[("a", 4), "pow"] == pytest.approx(0.5)
    assert rows.loc[("a", 4), "at"] == pytest.approx(1.0)
    assert rows.loc[("b", 4), "pow"] == pytest.approx(0.0)
    assert rows.loc[("b", 4), "pow_plugin"] == pytest.approx(0.75**4 / 2)
    with pytest.raises(DataError, match="missing columns"):
        pk.successes(trials.drop(columns="reward"))


def test_passk_command_scores_a_table_and_refuses_an_unreachable_k(tmp_path, capsys):
    from trialnoise import cli

    table = tmp_path / "t.csv"
    pd.DataFrame(
        {"cell": ["a"] * 6, "task": ["x", "x", "x", "y", "y", "y"], "reward": [1, 1, 1, 1, 0, 1]}
    ).to_csv(table, index=False)
    assert cli.main(["passk", "--table", str(table), "--k", "3", "--only"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[1].split()[:4] == ["a", "3", "2", "0.500"]
    assert cli.main(["passk", "--table", str(table), "--k", "4", "--only"]) == 2
    assert "no cell has 4 trials" in capsys.readouterr().err


def test_reference_rows_cover_every_k_of_the_four_designs():
    rows = pk.reference_rows()
    assert [(r["trials"], r["k"]) for r in rows] == [
        (n, k) for n in (4, 5, 8, 16) for k in range(1, n + 1)
    ]
    for r in rows:
        assert r["sd_50"] <= r["sd_worst"] + 1e-12 and r["sd_90"] <= r["sd_worst"] + 1e-12
        assert (r["plugin_bias_50"] > 0) == (r["k"] >= 2)
        assert r["half_normal"] < r["half_guaranteed"]
