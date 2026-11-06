import hashlib
import json
import re

import numpy as np
import pytest

from trialnoise import cli, report
from trialnoise.validate import (
    Latent,
    fit_latent,
    fit_pair,
    pair_differences,
    run_logit_sd,
)

RESULTS = cli.RESULTS


def test_report_regenerates_byte_for_byte_from_the_committed_results():
    text = report.render(RESULTS)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert digest == (RESULTS / report.STAMP).read_text(encoding="utf-8").strip()


def test_report_is_written_with_unix_newlines(tmp_path):
    target = tmp_path / "out.md"
    copy = tmp_path / "results"
    copy.mkdir()
    for path in RESULTS.iterdir():
        if path.suffix in (".json", ".csv"):
            (copy / path.name).write_bytes(path.read_bytes())
    report.write(copy, target)
    assert target.read_bytes() == report.render(copy).encode("utf-8")
    assert b"\r" not in target.read_bytes()


def test_rendered_report_leaves_no_field_unfilled():
    assert not re.search(r"\{[a-z_0-9]+\}", report.render(RESULTS))


def test_committed_outputs_carry_no_machine_details():
    pattern = re.compile(r"[A-Za-z]:[\\/]|/home/|/Users/")
    for path in RESULTS.iterdir():
        if path.suffix in (".json", ".csv"):
            assert not pattern.search(path.read_text(encoding="utf-8")), path.name


def test_power_command_answers_from_the_committed_planner(capsys):
    code = cli.main(["power", "--comparison", "same-model", "--band", "high", "--gap", "3"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["trials_needed_at_these_tasks"] is None or out["mde"] > 0.03
    assert out["tasks_needed_at_these_trials"] > 89


def test_pairs_command_tests_two_named_cells(capsys):
    code = cli.main(["pairs", "--a", "terminus-2/gpt-5.3-codex", "--b", "terminus-2/gpt-oss-20b"])
    assert code == 0
    assert "p_task" in capsys.readouterr().out


def test_a_bad_cell_name_fails_loudly(capsys):
    assert cli.main(["pairs", "--a", "nope", "--b", "terminus-2/gpt-5.3-codex"]) == 2
    assert "unknown cell" in capsys.readouterr().err


def test_latent_fit_matches_its_targets():
    target = {"task": 0.08, "cell": 0.03, "task_cell": 0.05, "grand": 0.42}
    fitted = fit_latent(target).truth()
    for key, value in target.items():
        assert fitted[key] == pytest.approx(value, rel=1e-4)


def test_run_shock_has_its_stated_size_on_the_probability_scale():
    """The logit SD is scaled by the mean Bernoulli variance, not by p(1-p) at the mean."""
    stratum = {"pass_rate": 0.65, "task_cell": 0.025, "trial": 0.08}
    latent = fit_pair(0.65, 0.025, 0.08)
    run_sd = 0.03
    rng = np.random.default_rng(1)
    with_run = pair_differences(
        latent, 89, 5, 0.0, 20000, rng, run_sd=run_logit_sd(stratum, run_sd)
    )
    without = pair_differences(latent, 89, 5, 0.0, 20000, rng)
    added = with_run.mean(axis=1).var() - without.mean(axis=1).var()
    assert added == pytest.approx(2 * run_sd**2, rel=0.15)


def test_a_bernoulli_mean_needs_no_trial_noise_above_its_own_variance():
    truth = Latent(0.0, 2.0, 1.0, 1.0).truth()
    total = sum(truth[k] for k in ("task", "cell", "task_cell", "trial"))
    assert total == pytest.approx(truth["grand"] * (1 - truth["grand"]))


@pytest.mark.slow
def test_study_reruns_byte_for_byte():
    assert cli.main(["check"]) == 0
