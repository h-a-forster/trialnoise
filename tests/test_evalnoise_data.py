import pandas as pd
import pytest

from trialnoise._evalnoise.data import DataError, describe, load_trials, to_trials


def test_csv_with_column_mapping_and_defaults(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("task,model,ok\na,m1,1\na,m1,0\nb,m1,1\nb,m2,0\n", encoding="utf-8")
    df = load_trials(p, {"item": "task", "arm": "model", "score": "ok"})
    assert list(df.columns) == ["item", "arm", "run", "repeat", "cluster", "score"]
    assert (df["run"] == "run-1").all()
    assert (df["cluster"] == df["item"]).all()
    assert df["repeat"].tolist() == [0, 1, 0, 0]
    assert describe(df)["max_repeats_per_run"] == 2


def test_jsonl_and_default_arm(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"item": "a", "score": 0.5}\n\n{"item": "b", "score": 1}\n', encoding="utf-8")
    df = load_trials(p, default_arm="x")
    assert df["arm"].tolist() == ["x", "x"]
    with pytest.raises(DataError, match="no arm column"):
        load_trials(p)


@pytest.mark.parametrize(
    "rows, message",
    [
        ({"item": ["a"], "arm": ["x"], "score": ["oops"]}, "non-numeric"),
        ({"item": ["a"], "arm": ["x"], "score": [float("inf")]}, "non-finite"),
        ({"item": [""], "arm": ["x"], "score": [1.0]}, "empty label"),
        ({"item": ["a"], "arm": ["x"]}, "missing required column 'score'"),
        (
            {"item": ["a", "a"], "arm": ["x", "x"], "cluster": ["c1", "c2"], "score": [1, 0]},
            "more than one cluster",
        ),
        (
            {"item": ["a", "a"], "arm": ["x", "x"], "repeat": [0, 0], "score": [1, 0]},
            "same arm, run, item and repeat",
        ),
    ],
)
def test_validation_fails_loudly(rows, message):
    with pytest.raises(DataError, match=message):
        to_trials(pd.DataFrame(rows))


def test_unknown_mapping_and_missing_source_column():
    frame = pd.DataFrame({"item": ["a"], "arm": ["x"], "score": [1.0]})
    with pytest.raises(DataError, match="unknown schema"):
        to_trials(frame, {"tasks": "item"})
    with pytest.raises(DataError, match="not found"):
        to_trials(frame, {"run": "batch"})


def test_unsupported_file(tmp_path):
    p = tmp_path / "r.parquet"
    p.write_bytes(b"x")
    with pytest.raises(DataError, match="unsupported"):
        load_trials(p)


def test_unlabelled_repeats_are_not_duplicates():
    rows = {"item": ["a", "a"], "arm": ["x", "x"], "repeat": ["", ""], "score": [1, 0]}
    assert len(to_trials(pd.DataFrame(rows))) == 2
