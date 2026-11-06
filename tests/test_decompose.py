import math

import numpy as np
import pandas as pd
import pytest

from trialnoise import decompose as dc
from trialnoise._evalnoise.components import estimate
from trialnoise._evalnoise.data import DataError
from trialnoise.validate import Latent, draw_table


def _gaussian_table(seed: int, a=30, t=60, k=4, cell=2.0, task=3.0, inter=1.0, trial=4.0):
    rng = np.random.default_rng(seed)
    y = (
        rng.normal(0, math.sqrt(task), (1, t, 1))
        + rng.normal(0, math.sqrt(cell), (a, 1, 1))
        + rng.normal(0, math.sqrt(inter), (a, t, 1))
        + rng.normal(0, math.sqrt(trial), (a, t, k))
    )
    return dc.Layout(
        tuple(f"c{j}" for j in range(a)),
        tuple(f"t{j}" for j in range(t)),
        y.mean(axis=2),
        np.full((a, t), float(k)),
        ((y - y.mean(axis=2, keepdims=True)) ** 2).sum(axis=2),
    )


def _trials(seed: int) -> pd.DataFrame:
    """Unbalanced pass/fail trials over three cells and six tasks."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(3):
        for t in range(6):
            for _ in range(int(rng.integers(1, 7))):
                rows.append(
                    {"cell": f"c{a}", "task": f"t{t}", "reward": int(rng.random() < 0.2 + 0.1 * t)}
                )
    return pd.DataFrame(rows)


def test_moments_equal_the_vendored_estimator_on_unbalanced_data():
    trials = _trials(1)
    m = dc.moments(dc.layout(trials))
    table = pd.DataFrame(
        {
            "item": trials["task"],
            "arm": trials["cell"],
            "run": "r",
            "repeat": trials.groupby(["cell", "task"]).cumcount(),
            "cluster": trials["task"],
            "score": trials["reward"].astype(float),
        }
    )
    c = estimate(table).components
    assert m["task"] == pytest.approx(c.item)
    assert m["task_cell"] == pytest.approx(c.item_arm)
    assert m["trial"] == pytest.approx(c.within)


def test_moments_recover_known_components():
    runs = np.array([[dc.moments(_gaussian_table(s))[n] for n in dc.TERMS] for s in range(300)])
    mean, se = runs.mean(axis=0), runs.std(axis=0, ddof=1) / math.sqrt(len(runs))
    truth = np.array([3.0, 2.0, 1.0, 4.0])
    assert np.all(np.abs(mean - truth) < 4 * se)


def test_moments_recover_probability_scale_components_of_binary_data():
    latent = Latent(-0.5, 2.5, 1.5, 1.5)
    truth = latent.truth()
    rng = np.random.default_rng(5)
    runs = np.array(
        [[dc.moments(draw_table(latent, 40, 60, 5, rng))[n] for n in dc.TERMS] for _ in range(200)]
    )
    mean, se = runs.mean(axis=0), runs.std(axis=0, ddof=1) / math.sqrt(len(runs))
    assert np.all(np.abs(mean - np.array([truth[n] for n in dc.TERMS])) < 4 * se)


def test_layout_refuses_a_table_that_is_not_crossed():
    trials = pd.DataFrame({"cell": ["a", "a", "b"], "task": ["t", "u", "t"], "reward": [1, 0, 1]})
    with pytest.raises(DataError, match="not crossed"):
        dc.layout(trials)


def test_icc_uses_the_task_terms_within_a_cell():
    s = dc.statistics({"task": 3.0, "cell": 5.0, "task_cell": 1.0, "trial": 4.0}, 5)
    assert s["icc1"] == pytest.approx(0.5)
    assert s["icc1_k"] == pytest.approx(5 / 6)
    assert s["icc_pair"] == pytest.approx(0.2)


def test_interaction_structure_recovers_its_three_sources():
    rng = np.random.default_rng(3)
    models, scaffolds = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    models, scaffolds = models.ravel(), scaffolds.ravel()
    t = 3000
    alpha = rng.normal(0, math.sqrt(0.3), (8, t))[models]
    beta = rng.normal(0, math.sqrt(0.2), (8, t))[scaffolds]
    gamma = rng.normal(0, math.sqrt(0.1), (64, t))
    k = 5
    y = alpha + beta + gamma + rng.normal(0, 1, (64, t, k)).mean(axis=2)
    table = dc.Layout(
        tuple(str(j) for j in range(64)),
        tuple(str(j) for j in range(t)),
        y,
        np.full(y.shape, float(k)),
        np.full(y.shape, float(k - 1)),
    )
    masks = dc.comparison_masks(list(scaffolds.astype(str)), list(models.astype(str)))
    _, sigma2, _ = dc.pair_matrices(table)
    parts = dc.interaction_structure({c: float(sigma2[m].mean()) for c, m in masks.items()})
    assert parts["task_model"] == pytest.approx(0.3, abs=0.02)
    assert parts["task_scaffold"] == pytest.approx(0.2, abs=0.02)
    assert parts["task_model_scaffold"] == pytest.approx(0.1, abs=0.02)


def test_nested_moments_recover_model_and_scaffold_terms():
    rng = np.random.default_rng(4)
    out = []
    for _ in range(400):
        n = rng.integers(3, 8, 12)
        groups = np.repeat(np.arange(12), n)
        y = rng.normal(0, math.sqrt(0.04), 12)[groups] + rng.normal(0, math.sqrt(0.01), n.sum())
        y = y + rng.normal(0, math.sqrt(0.002), n.sum())
        out.append(dc.nested_moments(y, groups, 0.002))
    assert np.mean([o["model"] for o in out]) == pytest.approx(0.04, rel=0.1)
    assert np.mean([o["scaffold_within_model"] for o in out]) == pytest.approx(0.01, rel=0.05)


def test_scaffold_share_stays_a_share_when_the_model_term_estimates_negative():
    # Group means closer than chance: the between-model estimate is negative.
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.1, 0.9])
    out = dc.nested_moments(y, np.array([0, 0, 1, 1, 2, 2]), 0.0)
    assert out["model"] < 0
    assert out["share_scaffold"] == 1.0


def _jobs(submission: str, shock: float, seed: int, copies: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = rng.random(89)
    first = None
    for j in range(5):
        p = np.clip(base + rng.normal(0, shock), 0, 1)
        y = (rng.random(89) < p).astype(float)
        if copies:
            first = y if first is None else first
            y = first
        for t in range(89):
            rows.append(
                {"submission": submission, "status": "ok", "job": f"j{j}", "task": f"t{t}",
                 "reward": y[t]}
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_run_term_finds_a_job_shock_and_none_where_there_is_none():
    with_shock = pd.concat([_jobs(f"s{j}", 0.1, j) for j in range(8)])
    without = pd.concat([_jobs(f"s{j}", 0.0, j + 50) for j in range(8)])
    a = dc.run_term(with_shock)
    b = dc.run_term(without)
    assert a["df"] == 32
    assert a["variance"] == pytest.approx(0.01, abs=0.005)
    assert a["p_above"] < 0.01
    assert b["interval"][0] == 0.0
    assert b["variance"] < a["interval"][0]


def test_run_term_refuses_copied_jobs_unless_they_are_skipped():
    frame = pd.concat([_jobs("real", 0.0, 1), _jobs("copy", 0.0, 2, copies=True)])
    with pytest.raises(DataError, match="copy"):
        dc.run_term(frame)
    assert dc.run_term(frame, skip=("copy",))["df"] == 4


def test_a_a_splits_reject_at_about_alpha_without_a_shock():
    frame = pd.concat([_jobs(f"s{j}", 0.0, j) for j in range(40)])
    out = dc.split_jobs_false_positives(frame, ())
    assert out["splits"] == 400
    assert 0.02 < out["rate"] < 0.09


def test_reml_agrees_with_moments_on_a_balanced_table():
    table = _gaussian_table(9, a=20, t=30)
    m, r = dc.moments(table), dc.reml_means(table)
    for term in ("task", "cell", "task_cell"):
        assert r[term] == pytest.approx(m[term], rel=0.01)


@pytest.mark.slow
def test_task_and_cell_bootstrap_covers_every_component():
    latent = Latent(-0.9, 3.0, 2.0, 1.8)
    truth = latent.truth()
    rng = np.random.default_rng(11)
    sims, hits = 150, np.zeros(len(dc.TERMS))
    for s in range(sims):
        table = draw_table(latent, 40, 40, 5, rng)
        draws = dc.bootstrap_draws(table, dc.component_vector, draws=200, seed=s, over_cells=True)
        for j, term in enumerate(dc.TERMS):
            lo, hi = dc.percentile(draws[:, j])
            hits[j] += lo <= truth[term] <= hi
    coverage = hits / sims
    # 95% nominal; 3 Monte Carlo SEs of 0.018 below it, on a design smaller than the study's.
    assert np.all(coverage >= 0.87)
