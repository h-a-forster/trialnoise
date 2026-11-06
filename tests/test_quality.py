import numpy as np
import pandas as pd
import pytest

from trialnoise import quality as ql
from trialnoise.study import sensitivity


def _official(rows):
    return pd.DataFrame(
        rows, columns=["submission", "scaffold", "model", "status", "job", "task", "trial",
                       "reward", "exception"]
    )  # fmt: skip


def test_infra_counts_a_trial_without_a_verdict_as_the_leaderboard_does():
    frame = _official(
        [
            ("s", "a", "m", "ok", "j", "t1", "1", 1.0, None),
            ("s", "a", "m", "ok", "j", "t2", "2", 0.0, "AgentTimeoutError"),
            ("s", "a", "m", "ok", "j", "t3", "3", np.nan, "DaytonaError"),
        ]
    )
    row = ql.infra_table(frame, {"s": {"n_total": 4}}).iloc[0]
    assert row["no_verdict"] == 2  # One named, one the job file lost entirely.
    assert row["score_zero"] == 0.25
    assert row["score_verdict"] == 0.5
    assert row["timeout_rate"] == 0.25


def _replicates(identical: bool):
    rng = np.random.default_rng(1)
    rows = []
    first = rng.integers(0, 2, 89).astype(float)
    for j in range(5):
        y = first if identical else rng.integers(0, 2, 89).astype(float)
        for t in range(89):
            rows.append(("s", "a", "m", "ok", f"j{j}", f"t{t}", f"{j}-{t}", y[t], None))
    return _official(rows)


def test_copied_submissions_finds_jobs_that_never_disagree():
    assert ql.copied_submissions(_replicates(True)) == ["s"]
    assert ql.copied_submissions(_replicates(False)) == []


def test_cost_treats_an_unreported_cost_as_missing_and_pays_for_failures():
    trials = pd.DataFrame(
        {
            "cell": ["c"] * 11,
            "task": [f"t{j}" for j in range(10)] + ["t0"],
            "reward": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "cost_cents": [100.0] * 10 + [np.nan],
        }
    )
    row = ql.cost_table(trials, draws=200, seed=1).iloc[0]
    assert row["cost_per_pass"] == 2.0
    assert row["coverage"] == pytest.approx(10 / 11)
    assert row["lo"] <= 2.0 <= row["hi"]


def test_cost_skips_a_cell_that_barely_reports_it():
    trials = pd.DataFrame(
        {
            "cell": ["c"] * 4,
            "task": ["t"] * 4,
            "reward": [1] * 4,
            "cost_cents": [1.0] + [np.nan] * 3,
        }
    )
    assert ql.cost_table(trials, draws=10, seed=1).empty


def test_integrity_screen_flags_a_cell_whose_trials_never_disagree():
    rng = np.random.default_rng(2)
    pass_rate = rng.uniform(0.1, 0.9, 80)
    share = 0.3 + 0.2 * pass_rate * (1 - pass_rate) + rng.normal(0, 0.02, 80)
    frame = pd.DataFrame(
        {"unit": [f"u{j}" for j in range(80)], "pass_rate": pass_rate, "mixed_share": share}
    )
    frame.loc[0, ["pass_rate", "mixed_share"]] = [0.5, 0.0]
    out = ql.integrity_screen(frame)
    assert out["unit"].iloc[0] == "u0"
    assert out["flagged"].iloc[0]


def test_sensitivity_survives_a_one_trial_cell_losing_its_only_verdict():
    rows = []
    for cell, n in (("a", 4), ("b", 4), ("thin", 1)):
        for t in ("t1", "t2", "t3"):
            for r in range(n):
                rows.append(
                    {"cell": cell, "task": t, "trial": f"{cell}-{t}-{r}", "reward": r % 2,
                     "submission": "s" if cell == "thin" else None}
                )  # fmt: skip
    trials = pd.DataFrame(rows)
    verdicts = trials[trials["trial"] != "thin-t2-0"]
    official = pd.DataFrame({"trial": verdicts["trial"], "reward": 1.0})
    _, out = sensitivity(trials, official, [])
    assert out["verdicts-only"]["cells"] == 2
