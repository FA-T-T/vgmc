# Development

Install `requirements-dev.lock` and the package in editable mode. The source package uses typed numerical interfaces, Google-style docstrings and 79-column Black formatting. English comments describe the purpose of numerical operations. Changes to numerical behavior include an independent expected result or a failure case.

```sh
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
black --check src tests examples
ruff check src tests examples
mypy src
pytest -q tests --doctest-modules src/vgmc
```

Tests cover analytic allocations, an independent conic formulation, covariance and bitstring identities, sparse assembly, checkpoint recovery, task commits and PennyLane execution. The molecular notebooks use registered input files. New scientific runs use new output directories and retain their effective configuration, task manifest and random-stream identity.

`src/vgmc/` contains the package. `tests/` contains focused regression checks. `examples/` contains the framework example. `docs/` contains interface and workflow documentation.
