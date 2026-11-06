import json

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
