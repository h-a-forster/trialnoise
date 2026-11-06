import json

import numpy as np
import pandas as pd
import pytest

from trialnoise import cli
from trialnoise import planner as pl
from trialnoise._evalnoise.data import DataError
from trialnoise._evalnoise.power import Design, mde, resolve, se_difference

STRATUM = {"task_cell": 0.03, "trial": 0.08, "pass_rate": 0.65}
FIT = {"strata": {"neither/high": STRATUM}, "run": {"upper": 8e-5, "df": 72}}


def test_fixed_target_counts_trial_noise_only():
    r = pl.resolved(STRATUM, "fixed", 0.0, float("inf"))
    assert se_difference(r, Design(89, 1, 5)) == pytest.approx((2 * 0.08 / 5 / 89) ** 0.5)


def test_population_target_cannot_reach_a_gap_below_its_task_floor():
    r = pl.resolved(STRATUM, "population", 0.0, float("inf"))
    floor = (2 * 0.03 / 89) ** 0.5 * 2.8
    assert pl.trials_needed(r, 89, floor * 0.9) is None
    assert pl.trials_needed(r, 89, floor * 1.2) is not None


def test_trials_needed_is_the_smallest_design_that_meets_the_gap():
    r = pl.resolved(STRATUM, "fixed", 0.0, float("inf"))
    k = pl.trials_needed(r, 89, 0.03)
    assert mde(r, Design(89, 1, k)) <= 0.03 < mde(r, Design(89, 1, k - 1))


def test_answer_reports_a_design_and_what_a_gap_needs():
    out = pl.answer(
        FIT, comparison="neither", band="high", target="fixed", tasks=89, trials=5, gap=0.05,
        run="none",
    )  # fmt: skip
    assert out["mde"] > 0.05
    assert out["power_at_gap"] < 0.8
    assert out["trials_needed_at_these_tasks"] > 5


def test_a_run_allowance_never_shrinks_the_mde():
    args = dict(
        comparison="neither", band="high", target="population", tasks=89, trials=5, gap=None
    )
    assert pl.answer(FIT, run="upper", **args)["mde"] > pl.answer(FIT, run="none", **args)["mde"]


def test_answer_refuses_what_it_cannot_plan():
    args = dict(band="high", target="fixed", tasks=89, trials=5, gap=None, run="none")
    with pytest.raises(DataError, match="unknown stratum"):
        pl.answer(FIT, comparison="same-model", **args)
    with pytest.raises(DataError, match="target"):
        pl.answer(FIT, comparison="neither", **{**args, "target": "both"})


def _heldout(seed: int, cells: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    task = rng.normal(0, 0.25, 89)
    rows = []
    for c in range(cells):
        p = np.clip(0.7 + task + rng.normal(0, 0.03**0.5, 89), 0.02, 0.98)
        for t in range(89):
            for _ in range(5):
                rows.append(
                    {"submission": f"s{c}", "scaffold": f"s{c}", "model": f"m{c}", "task": f"t{t}",
                     "reward": float(rng.random() < p[t])}
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_calibration_ratio_is_near_one_when_the_stratum_is_right():
    strata = {
        f"neither/{band}": {"task_cell": 0.03, "trial": 0.15, "pass_rate": 0.7}
        for band in ("low", "mid", "high")
    }
    _, summary = pl.calibrate(_heldout(1), strata)
    assert summary["pairs"] > 20
    assert summary["median_ratio"] == pytest.approx(1.0, abs=0.15)


def test_committed_planner_holds_the_spec_designs():
    fit = json.loads((cli.RESULTS / "planner.json").read_text(encoding="utf-8"))
    r = resolve(pl.components(fit["strata"]["same-model/high"], "population"), run=0.0)
    row = next(
        g for g in fit["grid"]
        if (g["stratum"], g["target"], g["run"], g["tasks"], g["trials"])
        == ("same-model/high", "population", "none", 89, 5)
    )  # fmt: skip
    assert row["mde"] == pytest.approx(mde(r, Design(89, 1, 5)), rel=1e-4)
