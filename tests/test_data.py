import hashlib
import json
import urllib.error

import pandas as pd
import pytest

from trialnoise import data
from trialnoise._evalnoise.data import DataError


def _job(path, rewards, exceptions=None, total=None):
    body = {
        "n_total_trials": total if total is not None else sum(map(len, rewards.values())),
        "stats": {
            "evals": {
                "x__terminal-bench": {
                    "reward_stats": {"reward": rewards},
                    "exception_stats": exceptions or {},
                }
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def test_parse_official_reads_rewards_exceptions_and_trials_without_a_verdict(tmp_path):
    _job(
        tmp_path / "Mux__GPT-5.2" / "job.json",
        {"1.0": ["task-a__1"], "0.0": ["task-b__2"]},
        {"AgentTimeoutError": ["task-b__2"], "DaytonaError": ["task-c__3"]},
        total=3,
    )
    frame, totals = data.parse_official(tmp_path)
    assert totals == {"Mux__GPT-5.2": {"jobs": 1, "n_total": 3, "job_starts": []}}
    row = frame.set_index("trial")
    assert row.loc["task-a__1", "reward"] == 1.0
    assert row.loc["task-b__2", "exception"] == "AgentTimeoutError"
    assert pd.isna(row.loc["task-c__3", "reward"])
    assert row.loc["task-c__3", "exception"] == "DaytonaError"
    assert row.loc["task-a__1", "scaffold"] == "mux"


def test_parse_official_maps_the_misspelt_task_and_skips_jobs_without_lists(tmp_path):
    _job(tmp_path / "Mux__GPT-5.2" / "a.json", {"1.0": ["install-windows-3-11__1"]})
    (tmp_path / "Mux__GPT-5.2" / "b.json").write_text(json.dumps({"id": "x"}), encoding="utf-8")
    frame, totals = data.parse_official(tmp_path)
    assert frame["task"].tolist() == ["install-windows-3.11"]
    assert totals["Mux__GPT-5.2"]["jobs"] == 1


def test_parse_official_refuses_an_unmapped_submission_and_graded_rewards(tmp_path):
    _job(tmp_path / "Brand-New__Model" / "a.json", {"1.0": ["t__1"]})
    with pytest.raises(DataError, match="configs/official.toml"):
        data.parse_official(tmp_path)
    (tmp_path / "Brand-New__Model" / "a.json").unlink()
    (tmp_path / "Brand-New__Model").rmdir()
    _job(tmp_path / "Mux__GPT-5.2" / "a.json", {"0.5": ["t__1"]})
    with pytest.raises(DataError, match="non-binary"):
        data.parse_official(tmp_path)


def _shards(folder, rows):
    names = sorted(data.SHARDS)
    columns = {c: [] for c in data.COLUMNS}
    for row in rows:
        for c in data.COLUMNS:
            columns[c].append(row.get(c, ""))
    pd.DataFrame(columns).to_parquet(folder / names[0], index=False)
    pd.DataFrame({c: pd.Series(columns[c][:0], dtype=pd.DataFrame(columns)[c].dtype)
                  for c in columns}).to_parquet(folder / names[1], index=False)  # fmt: skip


def _row(trial, reward, trial_id=""):
    return {
        "task_name": "t",
        "agent": "a",
        "model": "gpt-5@openai",
        "reward": reward,
        "duration_seconds": 1.0,
        "cost_cents": 0.0,
        "trial_name": trial,
        "trial_id": trial_id,
    }


def test_twins_collapse_to_the_full_scrape_row(tmp_path):
    _shards(tmp_path, [_row("x", 1), _row("x", 1, "id"), _row("y", 0)])
    kept, rows = data.read_parquet_trials(tmp_path)
    assert rows == 3
    assert len(kept) == 2
    assert kept.set_index("trial_name").loc["x", "trial_id"] == "id"


def test_twins_that_disagree_on_reward_stop_the_run(tmp_path):
    _shards(tmp_path, [_row("x", 1), _row("x", 0, "id")])
    with pytest.raises(DataError, match="disagree"):
        data.read_parquet_trials(tmp_path)


def test_an_unknown_model_string_stops_the_run(tmp_path):
    _shards(tmp_path, [{**_row("x", 1), "model": "new-model@acme"}])
    with pytest.raises(DataError, match="unknown model"):
        data.read_parquet_trials(tmp_path)


def test_only_a_real_twin_keeps_its_provider_tag():
    names = data.cell_names(
        pd.Series(["g", "g", "h"]),
        pd.Series(["m", "m", "m"]),
        pd.Series(["m-raw@Google", "m-raw@google", "m-raw@Google"]),
    )
    assert names.tolist() == ["g/m@Google", "g/m@google", "h/m"]


def test_every_official_folder_in_the_config_has_a_known_status():
    config = data.load_config()
    assert len(config) == 75
    allowed = {"ok", "cheating", "reward-hacking", "no-job-stats"}
    assert {row["status"] for row in config.values()} <= allowed


def test_a_wheel_finds_its_own_copy_of_the_shipped_files(tmp_path):
    checkout = tmp_path / "repo"
    (checkout / "src" / "trialnoise").mkdir(parents=True)
    (checkout / "configs").mkdir()
    (checkout / "configs" / "official.toml").touch()
    assert data.shipped_root(checkout / "src" / "trialnoise") == checkout
    site = tmp_path / "venv" / "Lib" / "site-packages" / "trialnoise"
    site.mkdir(parents=True)
    assert data.shipped_root(site) == site / "_bundled"


def test_linked_trials_must_agree_on_the_reward(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "EXPECTED", {"rows": 1, "trials": 1, "tasks": 1, "cells": 1})
    (tmp_path / "raw").mkdir()
    _shards(tmp_path / "raw", [_row("t__1", 1)])
    _job(tmp_path / "official" / "Mux__GPT-5.2" / "job.json", {"1.0": ["t__1"]})
    trials, _, manifest = data.build_tables(tmp_path / "raw", tmp_path / "official")
    assert trials["job"].tolist() == ["job"]
    assert manifest["counts"]["linked_trials"] == 1
    _job(tmp_path / "official" / "Mux__GPT-5.2" / "job.json", {"0.0": ["t__1"]})
    with pytest.raises(DataError, match="different rewards"):
        data.build_tables(tmp_path / "raw", tmp_path / "official")


def test_fetch_replaces_a_corrupt_shard_and_never_names_a_bad_download(tmp_path, monkeypatch):
    good = b"good"
    monkeypatch.setattr(data, "SHARDS", {"s.parquet": hashlib.sha256(good).hexdigest()})
    served = [b"bad", good]
    monkeypatch.setattr(
        data.urllib.request, "urlretrieve", lambda _, part: part.write_bytes(served.pop(0))
    )
    (tmp_path / "s.parquet").write_bytes(b"truncated")
    with pytest.raises(DataError, match="sha256"):
        data.fetch_parquet(tmp_path, log=lambda _: None)
    assert [p.name for p in tmp_path.iterdir()] == ["s.parquet"]
    assert (tmp_path / "s.parquet").read_bytes() == b"truncated"
    data.fetch_parquet(tmp_path, log=lambda _: None)
    assert (tmp_path / "s.parquet").read_bytes() == good
    data.fetch_parquet(tmp_path, log=lambda _: None)  # Nothing left to serve: no new download.


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return b"body"


def test_get_retries_a_dropped_connection(monkeypatch):
    outcomes = [urllib.error.URLError("reset"), TimeoutError("slow"), _Response()]

    def urlopen(*_, **__):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    assert data._get("https://huggingface.co/x") == b"body"


@pytest.mark.parametrize("name", ["..", "a\\..\\..\\b", "C:evil", ""])
def test_fetch_official_refuses_a_listed_name_that_leaves_its_folder(tmp_path, monkeypatch, name):
    listing = json.dumps([{"type": "directory", "path": f"submissions/{name}"}]).encode()
    monkeypatch.setattr(data, "_get", lambda _: listing)
    out = tmp_path / "official"
    with pytest.raises(DataError, match="unsafe name"):
        data.fetch_official(out, log=lambda _: None)
    assert list(out.iterdir()) == []
