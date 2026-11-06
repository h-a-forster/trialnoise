import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from trialnoise import cli, report, study, validate
from trialnoise._evalnoise.data import DataError
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


def test_a_missing_results_folder_fails_with_a_message(tmp_path, capsys):
    assert cli.main(["pairs", "--results", str(tmp_path)]) == 2
    assert "pairs.csv" in capsys.readouterr().err


def test_only_rebuilding_commands_default_to_the_writable_results(tmp_path, monkeypatch):
    shipped, writable = tmp_path / "wheel", tmp_path / "work"
    monkeypatch.setattr(cli, "RESULTS", shipped)
    monkeypatch.setattr(cli, "OUTPUT", writable)
    for command in ("extract", "study", "validate", "report"):
        assert cli.parser().parse_args([command]).results == writable
    for command in ("components", "power", "pairs", "check"):
        assert cli.parser().parse_args([command]).results == shipped


def test_a_table_without_repeats_refuses_a_chosen_pair_instead_of_crashing(tmp_path, capsys):
    rows = [{"cell": c, "task": f"t{t}", "reward": t % 2} for c in "ab" for t in range(10)]
    table = tmp_path / "one.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    assert cli.main(["pairs", "--table", str(table), "--a", "a", "--b", "b"]) == 2
    assert "not identified" in capsys.readouterr().err


def test_a_cell_cannot_be_compared_with_itself(capsys):
    name = "terminus-2/gpt-5.3-codex"
    assert cli.main(["pairs", "--a", name, "--b", name]) == 2
    assert "same cell" in capsys.readouterr().err


def test_the_prose_refuses_data_it_was_not_written_for(tmp_path):
    copy = tmp_path / "results"
    copy.mkdir()
    for path in RESULTS.iterdir():
        if path.suffix in (".json", ".csv"):
            (copy / path.name).write_bytes(path.read_bytes())
    summary = json.loads((copy / "summary.json").read_text(encoding="utf-8"))
    summary["data"]["copied_submissions"] = []
    (copy / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(DataError, match="one copied submission"):
        report.render(copy)


# What the bootstrap and the simulated boards set; the rest does not depend on their size.
BOOTSTRAPPED = re.compile(r"interval|^tasks$|^tasks_and_cells$|^board$|^lo$|^hi$")
PLANNED = ("same-model/high", "neither/high", "pooled/all")


def _flat(value, path=""):
    if isinstance(value, dict):
        items = [(k, v) for k, v in value.items() if not BOOTSTRAPPED.search(k)]
    elif isinstance(value, list):
        items = list(enumerate(value))
    else:
        return {path: value}
    return {key: leaf for k, v in items for key, leaf in _flat(v, f"{path}/{k}").items()}


def _planned(rows):
    return [r for r in rows if r["stratum"] in PLANNED and r["run"] == "none"]


def test_study_code_reproduces_the_committed_point_estimates(tmp_path, monkeypatch):
    """The whole study path on the committed tables, kept fast and run without `--runslow`.

    Next to no bootstrap draws, and the planner's rows for three strata without the run allowance,
    which is what takes the study minutes. Within 1e-4, so that optimiser noise across platforms
    does not fail it; `trialnoise check` is the byte-identical test.
    """
    stratum_rows = study.pl.stratum_rows
    monkeypatch.setattr(study, "DRAWS", 10)
    monkeypatch.setattr(study, "BOARD_REPS", 2)
    monkeypatch.setattr(
        study.pl,
        "stratum_rows",
        lambda strata, run: stratum_rows({k: strata[k] for k in PLANNED}, None),
    )
    for name in cli.TABLES:
        (tmp_path / name).write_bytes((RESULTS / name).read_bytes())
    study.run_study(tmp_path, log=lambda _: None)
    for name in ("cells.csv", "heldout.csv", "infra.csv", "integrity.csv", "pairs.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / name), pd.read_csv(RESULTS / name), rtol=1e-4, atol=1e-9
        )
    for name in ("summary.json", "planner.json"):
        fresh = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        committed = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        if name == "planner.json":
            fresh |= {k: _planned(fresh[k]) for k in ("grid", "needs")}
            committed |= {k: _planned(committed[k]) for k in ("grid", "needs")}
        fresh, committed = _flat(fresh), _flat(committed)
        assert fresh.keys() == committed.keys()
        floats = {k for k, v in committed.items() if type(v) is float}
        assert {k: fresh[k] for k in floats} == pytest.approx(
            {k: committed[k] for k in floats}, rel=1e-4, abs=1e-9
        )
        assert {k: v for k, v in fresh.items() if k not in floats} == {
            k: v for k, v in committed.items() if k not in floats
        }


def test_validation_skips_finished_units_until_its_inputs_change(tmp_path, monkeypatch):
    calls = []
    unit = {"one": lambda rng: calls.append(1) or {"rate": 0.5}}
    monkeypatch.setattr(validate, "checks", lambda fit, prof: unit)
    for fit, expected in (({"a": 1}, 1), ({"a": 1}, 1), ({"a": 2}, 2)):
        out = validate.run_validation(fit, tmp_path, "quick", log=lambda _: None)
        assert out["checks"]["one"] == {"rate": 0.5}
        assert len(calls) == expected


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
