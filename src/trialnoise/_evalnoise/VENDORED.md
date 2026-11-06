# Vendored evalnoise

`components.py`, `data.py`, `power.py`, `reml.py` and `simulate.py` are copied from
[evalnoise](https://github.com/h-a-forster/evalnoise) at commit
`14e5684e4e33557471f9a8d50cfeb1647aff4950`, with only the import paths rewritten and re-sorted. Their tests
are `tests/test_evalnoise_*.py`. Do not edit these files here: change evalnoise and copy again.
trialnoise's own extensions (the cell component, strata, nested terms) live in
`trialnoise/decompose.py`, and a test holds them to these estimators.
