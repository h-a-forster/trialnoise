import math

import numpy as np
import pytest

from trialnoise import pairs as pr
from trialnoise.decompose import Layout, moments
from trialnoise.validate import Latent, draw_table


def _table(rows, trials=5):
    """Cells as rows of passes per task, out of `trials`."""
    x = np.asarray(rows, dtype=float)
    return Layout(
        tuple(f"c{j}" for j in range(x.shape[0])),
        tuple(f"t{j}" for j in range(x.shape[1])),
        x / trials,
        np.full(x.shape, float(trials)),
        x * (1 - x / trials),
    )


def test_scores_weigh_tasks_equally_whatever_their_trial_counts():
    table = Layout(
        ("a",), ("t", "u"), np.array([[1.0, 0.0]]), np.array([[1.0, 9.0]]), np.zeros((1, 2))
    )
    assert pr.scores(table)[0] == 0.5


def test_adjacent_tests_match_a_paired_t_test_and_adjust_over_pairs():
    rng = np.random.default_rng(1)
    table = draw_table(Latent(0.0, 2.0, 1.0, 1.0), 6, 40, 5, rng)
    order = [int(j) for j in np.argsort(-pr.scores(table))]
    frame = pr.adjacent(table, order, "u")
    d = table.mean[order[0]] - table.mean[order[1]]
    assert frame["gap"].iloc[0] == pytest.approx(d.mean())
    assert frame["se_task"].iloc[0] == pytest.approx(d.std(ddof=1) / math.sqrt(40))
    assert (frame["p_task_holm"] >= frame["p_task"]).all()
    assert (frame["p_task_holm"] >= frame["p_task_bh"]).all()


def test_a_pair_with_no_noise_at_all_gets_a_p_value_not_a_crash():
    same = _table([[2, 3, 1, 4], [2, 3, 1, 4]])
    row = pr.adjacent(same, [0, 1], "u").iloc[0]
    assert (row["p_task"], row["p_fixed"]) == (1.0, 1.0)
    apart = _table([[5, 5, 5, 5], [0, 0, 0, 0]])
    row = pr.adjacent(apart, [0, 1], "u").iloc[0]
    assert (row["p_task"], row["p_fixed"]) == (0.0, 0.0)


def test_fixed_task_se_is_the_trial_noise_of_the_gap():
    """Two arms with one known p per task: the SE should match the spread of simulated gaps."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0.1, 0.9, 89)
    gaps, ses = [], []
    for _ in range(3000):
        x = rng.binomial(5, p, (2, 89))
        table = _table(x)
        gaps.append(pr.scores(table)[0] - pr.scores(table)[1])
        ses.append(pr.fixed_task_se(table, 0, 1))
    assert np.mean(ses) == pytest.approx(np.std(gaps, ddof=1), rel=0.03)


def test_best_per_scaffold_keeps_each_scaffolds_top_cell_in_rank_order():
    score = np.array([0.5, 0.9, 0.7, 0.8])
    assert pr.best_per_scaffold(score, ["a", "b", "a", "b"]) == [1, 2]


def test_smoothed_truth_reproduces_the_fitted_trial_variance():
    rng = np.random.default_rng(3)
    table = draw_table(Latent(-0.5, 2.5, 1.5, 1.5), 30, 40, 5, rng)
    truth, meta = pr.smoothed_truth(table)
    assert float((truth * (1 - truth)).mean()) == pytest.approx(moments(table)["trial"], rel=1e-4)
    assert meta["pseudo_count"] > 0


def test_holm_rejections_leave_nan_out_of_the_family():
    nan = math.nan
    assert pr._holm_rejections(np.array([0.02, 0.03])) == 2
    assert pr._holm_rejections(np.array([0.02, nan, 0.03])) == 2
    assert pr._holm_rejections(np.array([nan, 0.04, 0.5])) == 0
    assert pr._holm_rejections(np.array([nan, nan])) == 0


def test_simulated_board_is_seeded_and_separates_cells_that_differ_by_design():
    truth = np.array([[0.95] * 60, [0.5] * 60, [0.05] * 60])
    one = pr.simulate_board(truth, ["a", "b", "c"], trials=5, tasks=60, reps=20, seed=1)
    two = pr.simulate_board(truth, ["a", "b", "c"], trials=5, tasks=60, reps=20, seed=1)
    assert one == two
    assert one["adjacent_all"] == 1.0


def test_simulated_board_resamples_tasks_for_a_longer_benchmark():
    truth = np.array([[0.9] * 30 + [0.1] * 30, [0.1] * 30 + [0.9] * 30])
    out = pr.simulate_board(truth, ["a", "b"], trials=5, tasks=200, reps=50, seed=2)
    assert out["tasks"] == 200
    assert out["adjacent_all"] < 0.2
