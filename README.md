# VGMC v1.0

VGMC constructs qubit-wise commuting measurement libraries and optimizes their estimator coefficients and shot allocations. 
The Python package provides a framework-neutral design interface and a PennyLane adapter.

## Installation

Use Python 3.12 or 3.13.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
jupyter lab notebooks
```

The numerical module installs with `python -m pip install .`. Optional dependency groups are `experiments`, `notebooks`, `pennylane`, `tequila` and `dev`. The complete experiment and development environments use the corresponding lock files. ICS uses `requirements-tequila.lock`; the framework example uses `requirements-pennylane.lock`.

## Measurement design

```python
import numpy as np
from vgmc import PauliHamiltonian, SolverConfig, design_measurements

h = PauliHamiltonian(("X", "Z"), [1.0, 2.0], constant=0.5)
design = design_measurements(h, np.eye(2),
                             config=SolverConfig(ridge=0.0))
plan = design.allocate_shots(300)
energy = plan.estimate([np.zeros((n, 1), dtype=int)
                        for n in plan.counts])
```

A covariance provider can supply blocks on demand. `StateCovariance` supplies blocks from a pure state. `max_added` enables bounded VGMC group construction. 

The [API guide](docs/api.md) describes shapes, units, status fields and input/output examples. 

The [PennyLane example](examples/pennylane_measurement.py) converts an operator, designs measurements, executes product-basis tapes and reconstructs energy.

## Hamiltonians

The input files contain nonidentity Pauli terms, coefficients, reference and CISD states, covariance blocks and initial group libraries. Source files, commit, SHA-256 hashes and identity coefficients are recorded in [data/hamiltonians.json](data/hamiltonians.json). Method references are listed in [docs/sources.md](docs/sources.md).

## Experiments

| Notebook | Purpose | Output |
| --- | --- | --- |
| [E1](notebooks/E1.ipynb) | Compare reference-state variance under exact and CISD design covariance. | Two variance tables. |
| [E2](notebooks/E2.ipynb) | Estimate finite-shot energy RMSE for frozen ICS, SOCP and VGMC designs. | RMSE tables and figures. |
| [E3](notebooks/E3.ipynb) | Vary response updates and covariance during group growth. | Trajectories and endpoint tables. |

Each notebook contains purpose and design, imports and settings, execution, data collection and plots. `MODE = "analysis_only"` processes supplied records. `MODE = "full"` runs the selected tasks. `MODE = "render"` loads saved analysis. Run directories use explicit `fresh`, `resume`, `reuse` or `auto` modes.

```sh
python -m vgmc reproduce E1 --output outputs/E1/report
python -m vgmc plan E1 --molecules H2 --covariances exact --methods SI SOCP VGMC --output outputs/E1/run-001
python -m vgmc run E1 --molecules H2 --covariances exact --methods SI SOCP VGMC --output outputs/E1/run-001 --allow-compute
```
