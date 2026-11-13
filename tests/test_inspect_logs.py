"""The Inspect log reader, on small logs written by inspect_ai itself with its mock model."""

import json
import math
import os
import random

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import Task, eval, eval_retry, task  # noqa: E402
from inspect_ai.dataset import Sample  # noqa: E402
from inspect_ai.scorer import Score, mean, scorer  # noqa: E402
from inspect_ai.solver import solver  # noqa: E402

from trialnoise import cli, decompose  # noqa: E402
from trialnoise import inspect_logs as il  # noqa: E402
from trialnoise._evalnoise.data import DataError  # noqa: E402

ITEMS = 6
EPOCHS = 3
BROKEN = {"x": (3, 2)}  # cell x's sample 3 raises on epoch 2
FIXED = "TRIALNOISE_TEST_FIXED"  # an env var: eval_retry reloads this module from its file


def outcome(model: str, item: int, epoch: int, salt: str) -> float:
    """The pass/fail outcome the `coin` scorer gives: a model, item and run effect plus noise."""
    rng = random.Random(f"{model}|{item}|{epoch}|{salt}")
    p = 0.2 + 0.1 * item + (0.15 if model.endswith("x") else 0.0) + (0.1 if salt == "b" else 0.0)
    return float(rng.random() < min(p, 0.95))


def model_name(state) -> str:  # type: ignore[no-untyped-def]
    return str(state.model).split("/")[-1]


@scorer(metrics=[mean()])
def coin(salt: str = "a"):  # type: ignore[no-untyped-def]
    async def score(state, target):  # type: ignore[no-untyped-def]
        value = outcome(model_name(state), int(state.sample_id), state.epoch, salt)
        return Score(value="C" if value else "I")

    return score


@scorer(metrics=[mean()])
def grade():  # type: ignore[no-untyped-def]
    """A 0-10 grade, NaN on item 6's first epoch, and a dict-valued twin."""

    async def score(state, target):  # type: ignore[no-untyped-def]
        item = int(state.sample_id)
        if item == 6 and state.epoch == 1:
            return Score(value=math.nan)
        return Score(value=float(item + state.epoch))

    return score


@scorer(metrics=[mean()])
def parts():  # type: ignore[no-untyped-def]
    async def score(state, target):  # type: ignore[no-untyped-def]
        return Score(value={"exact": int(state.epoch == 1), "fuzzy": 0.5})

    return score


@solver
def flaky():  # type: ignore[no-untyped-def]
    async def solve(state, generate):  # type: ignore[no-untyped-def]
        bad = BROKEN.get(model_name(state))
        if not os.environ.get(FIXED) and bad == (int(state.sample_id), state.epoch):
            raise RuntimeError("tool crashed")
        return await generate(state)

    return solve


def dataset(items: int = ITEMS) -> list[Sample]:
    return [Sample(input=f"q{i}", target="C", id=i) for i in range(1, items + 1)]


@task
def noisy(salt: str = "a", items: int = ITEMS):  # type: ignore[no-untyped-def]
    return Task(
        dataset=dataset(items),
        solver=flaky(),
        scorer=[coin(salt), grade(), parts()],
        epochs=EPOCHS,
    )


@pytest.fixture(scope="module")
def logs(tmp_path_factory):  # type: ignore[no-untyped-def]
    """Folders of logs: two models, a replicate run of each, a retried log, a partial model."""
    root = tmp_path_factory.mktemp("inspect")
    quiet = {"display": "none", "fail_on_error": False}
    eval(noisy("a"), model=["mockllm/x", "mockllm/y"], log_dir=str(root / "main"), **quiet)
    eval(noisy("b"), model=["mockllm/x", "mockllm/y"], log_dir=str(root / "main"),
         log_format="json", **quiet)  # fmt: skip
    eval(noisy("a", items=3), model="mockllm/z", log_dir=str(root / "partial"), **quiet)
    [first] = eval(noisy("a"), model="mockllm/x", log_dir=str(root / "retry"), **quiet)
    os.environ[FIXED] = "1"
    try:
        eval_retry(first, log_dir=str(root / "retry"), display="none")
    finally:
        del os.environ[FIXED]
    return root


def expected(models=("x", "y"), salts=("a", "b")) -> dict[tuple[str, str, int], float]:
    out = {}
    for m in models:
        for salt in salts:
            for i in range(1, ITEMS + 1):
                for e in range(1, EPOCHS + 1):
                    if BROKEN.get(m) != (i, e):
                        out[(f"mockllm/{m}", f"noisy/{i}", e, salt)] = outcome(m, i, e, salt)
    return out


def test_samples_epochs_models_and_runs_become_the_trial_table(logs):
    r = il.read([logs / "main"], scorer="coin")
    assert set(r.scorers) == {"coin", "grade", "parts:exact", "parts:fuzzy"}
    assert r.logs == 4 and r.read == 2 * 2 * ITEMS * EPOCHS
    assert r.excluded == {
        "errored": 2, "incomplete": 0, "unscored": 0, "superseded": 0, "not_crossed": 0,
    }  # fmt: skip
    assert r.trials["run"].nunique() == 4 and r.unit_scale
    # Each run is one log; tell the two runs of a model apart by their outcomes.
    want = expected()
    got = {}
    for (cell, _), g in r.trials.groupby(["cell", "run"]):
        triples = zip(g["task"], g["epoch"], g["reward"], strict=True)
        rows = {(cell, t, e): y for t, e, y in triples}
        salt = next(s for s in "ab" if all(want.get((*k, s)) == y for k, y in rows.items()))
        got.update({(*k, salt): y for k, y in rows.items()})
    assert got == want


def test_errored_samples_are_dropped_or_scored_zero_never_silently(logs):
    zero = il.read([logs / "main"], scorer="coin", errors="zero")
    assert zero.excluded["errored"] == 0 and len(zero.trials) == zero.read
    broken = zero.trials[(zero.trials["task"] == "noisy/3") & (zero.trials["epoch"] == 2)]
    assert (broken[broken["cell"] == "mockllm/x"]["reward"] == 0).all()


def test_a_retried_log_supersedes_the_attempt_it_retries(logs):
    r = il.read([logs / "retry"], scorer="coin")
    # The retry copies 17 finished sample epochs and reruns the errored one: all 18 supersede.
    assert r.excluded["superseded"] == ITEMS * EPOCHS and r.excluded["errored"] == 0
    assert len(r.trials) == ITEMS * EPOCHS and r.trials["run"].nunique() == 1


def test_items_missing_from_a_cell_leave_the_table_and_are_counted(logs):
    r = il.read([logs / "main", logs / "partial"], scorer="coin")
    assert r.excluded["not_crossed"] == (ITEMS - 3) * EPOCHS * 4  # items 4 to 6 of four logs
    assert sorted(r.trials["task"].unique()) == ["noisy/1", "noisy/2", "noisy/3"]


def test_non_binary_dict_and_missing_scores(logs):
    grade = il.read([logs / "main"], scorer="grade")
    assert grade.excluded["unscored"] == 4  # item 6, epoch 1, in four logs
    assert not grade.unit_scale and grade.trials["reward"].max() == ITEMS + EPOCHS
    exact = il.read([logs / "main"], scorer="parts:exact")
    assert set(exact.trials["reward"]) == {0.0, 1.0}
    with pytest.raises(DataError, match="choose a scorer"):
        il.read([logs / "main"])
    with pytest.raises(DataError, match="no scorer 'parts'"):
        il.read([logs / "main"], scorer="parts")


@pytest.mark.parametrize(
    ("value", "number"),
    [("C", 1.0), ("P", 0.5), ("I", 0.0), ("N", 0.0), (True, 1.0), ("yes", 1.0), ("0.25", 0.25),
     (3, 3.0), (None, math.nan), (math.inf, math.nan)],
)  # fmt: skip
def test_score_values_follow_inspects_defaults(value, number):
    got = il.score_value(value, "s")
    assert got == number or (math.isnan(got) and math.isnan(number))


@pytest.mark.parametrize("value", ["maybe", [1, 0]])
def test_values_that_are_not_scores_fail_loudly(value):
    with pytest.raises(DataError, match="not a number"):
        il.score_value(value, "s")


def test_the_plan_uses_the_pooled_terms_of_the_logs(logs):
    r = il.read([logs / "main"], scorer="coin")
    out = il.analyse(r, gap=0.1, bootstrap=200, seed=1)
    m = decompose.moments(decompose.layout(r.trials))
    assert out["components"]["trial"]["value"] == pytest.approx(m["trial"])
    assert out["run"]["identified"] and out["run"]["df"] == 2  # one per model
    fixed = next(
        row for row in out["plan"]["rows"] if row["target"] == "fixed" and row["basis"] == "point"
    )
    k = out["plan"]["k"]
    assert k == 6  # two runs of three epochs
    assert fixed["se"] == pytest.approx(math.sqrt(2 * m["trial"] / k / ITEMS))
    needed = fixed["trials_needed_at_these_tasks"]
    assert needed is None or needed >= 1


def test_one_cell_plans_the_fixed_target_only(logs):
    r = il.read([logs / "partial"], scorer="coin")
    out = il.analyse(r, gap=0.2, bootstrap=100, seed=1)
    assert set(out["components"]) == {"task", "trial"}
    population = [row for row in out["plan"]["rows"] if row["target"] == "population"]
    assert not any(row["identified"] for row in population)
    assert all(row["identified"] for row in out["plan"]["rows"] if row["target"] == "fixed")


def test_cli_prints_the_decomposition_and_plan(logs, tmp_path, capsys):
    table = tmp_path / "trials.csv"
    argv = ["inspect", str(logs / "main"), "--scorer", "coin", "--gap", "10", "--bootstrap", "100"]
    assert cli.main([*argv, "--write-table", str(table)]) == 0
    text = capsys.readouterr().out
    assert "2 errored" in text and "task_cell" in text and "items needed" in text
    assert table.exists()
    assert cli.main([*argv, "--json", "--run", "upper"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["plan"]["run"] == "upper" and out["plan"]["gap"] == pytest.approx(0.1)
    assert cli.main(["components", "--table", str(table), "--bootstrap", "50"]) == 0


def test_cli_refuses_missing_logs_and_unknown_cells(logs, tmp_path, capsys):
    assert cli.main(["inspect", str(tmp_path / "none"), "--gap", "5"]) == 2
    assert "no such file" in capsys.readouterr().err
    argv = ["inspect", str(logs / "main"), "--scorer", "coin", "--gap", "5"]
    assert cli.main([*argv, "--cell", "metadata:arm"]) == 2
    assert "no key 'arm'" in capsys.readouterr().err
