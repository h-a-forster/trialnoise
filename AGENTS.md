# AGENTS.md

Guide for coding agents (and humans) working on trialnoise. Read `docs/DESIGN.md` before changing
any statistics or data handling: it fixes them and records why.

## What this is

A study and small package. It decomposes the variance of public Terminal-Bench 2 trial outcomes
(task, cell, task x cell, trial, run), separates scaffold from model, and calibrates a calculator
for how many trials and tasks a comparison needs.

## Conventions

- uv + hatchling, `src/` layout, Python >= 3.11 (local dev 3.13), typed code, `ruff check` +
  `ruff format` (line length 100), pytest. `[tool.uv] exclude-newer` keeps the lock on releases at
  least two weeks old; move it deliberately.
- Runtime dependencies: numpy, scipy, pandas, pyarrow, statsmodels (MixedLM is part of the study).
- The wheel force-includes `configs/` and `results/v1` as `trialnoise/_bundled`. `data.SHIPPED` finds
  them in a checkout or a wheel; `data.WORK` is where generated files go, never the wheel's copy.
- `src/trialnoise/_evalnoise/` is vendored: see `VENDORED.md`. Change evalnoise, then copy. Its
  tests are `tests/test_evalnoise_*.py`.
- The study reads only `results/v1/trials.parquet` and `official.parquet`; `fetch` and `extract`
  are the only network and raw-data steps. No model calls.
- Every stochastic function takes a seed; units derive theirs from the unit name. Outputs are
  rounded to six significant digits, so they regenerate byte for byte.
- Validate before writing; raise `DataError`. Count every excluded trial. `extract` asserts the
  expected row counts.
- Long work: `validate` writes one atomic file per check under `data/validate/`, keyed by a hash of
  its inputs and `SCHEMA`, and skips finished ones. Bump `SCHEMA` in `validate.py` when a check's
  results would change.
- Every number in the report and the README's findings comes from a field in `report.py`; prose
  lives in `report_text.py`. `trialnoise report` rewrites both; commit them and the regenerated
  `results/v1/report.sha256` with any change to either.
- `data/` and `tmp/` are gitignored. Never commit raw shards or job files.
- Commits and PRs carry no generated-by trailers. CI runs on pushes to main, pull requests and
  manual dispatch, the Linux `check` job (the byte-identical rerun) included.
- Release hygiene: no personal names, local user paths or machine details in committed files.
- Statistical honesty: report sizes, seeds and intervals; say when a number is a bound; cells are
  submissions, not a sample; the integrity screen is exploratory and accuses nobody.

## Architecture map

| Path | Role |
|---|---|
| `src/trialnoise/data.py` | pinned fetch, job-file parsing, the two trial tables |
| `src/trialnoise/models.py` | raw model string to canonical name (from trajclaims) |
| `src/trialnoise/decompose.py` | components, ICC, strata, nested and crossed terms, run term |
| `src/trialnoise/pairs.py` | adjacent and all-pairs tests, simulated boards |
| `src/trialnoise/planner.py` | the calculator over evalnoise's planner; held-out calibration |
| `src/trialnoise/quality.py` | infrastructure, cost, integrity screen |
| `src/trialnoise/validate.py` | Monte Carlo checks |
| `src/trialnoise/study.py` | runs everything and writes the outputs |
| `src/trialnoise/report.py`, `report_text.py` | the report, from the outputs |
| `configs/official.toml` | official submissions: scaffold, model, standing |
| `results/v1/` | committed tables and outputs |

## How to test

```sh
uv sync --group dev
uv run pytest                 # offline
uv run pytest --runslow       # plus coverage and the byte-identical study rerun, about 10 minutes
uv run ruff check .
uv run ruff format --check .
uv lock --check
```

## Reproduce

```sh
uv run trialnoise fetch       # 224 MB, hashes checked
uv run trialnoise extract     # the two trial tables
uv run trialnoise study       # about 5 minutes
uv run trialnoise validate    # about 2 minutes
uv run trialnoise check       # every output byte-identical
uv run trialnoise report      # write REPORT.md
```

Open work lives in `backlog.md`.
