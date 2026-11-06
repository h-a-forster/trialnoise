# Contributing

Read [docs/DESIGN.md](docs/DESIGN.md) first: it fixes the data handling, model and estimands.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git.

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests must run offline, write only under pytest's `tmp_path`, use fixed seeds and pass on Windows.
Statistical assertions state their tolerance in Monte Carlo standard errors. Mark tests that take
more than a few seconds `@pytest.mark.slow`, except the fast study rerun, which is what ties CI to
the committed results. A bug fix comes with a test that fails without it.

## Code

- Validate inputs before writing, and fail with a clear `DataError` rather than dropping data.
- Do not edit `src/trialnoise/_evalnoise/`; change evalnoise and copy it (see `VENDORED.md`).
- A change that moves a result regenerates `results/v1` and `report.sha256` in the same commit, with
  a new `results/v<n>` if the data or a frozen definition changes.
- Put change notes in `docs/release-notes-<version>.md`.

## Data

Never commit raw downloads or job files (`data/`). The committed tables are derived from
Apache-2.0 sources, with pinned revisions in `MANIFEST.json`.

Security issues: see [SECURITY.md](SECURITY.md).
