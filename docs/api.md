# Python API

The package uses a declared Hamiltonian term order and a declared wire order. Numerical inputs become float64 arrays; state vectors become complex128 arrays. Coefficients and estimates use the caller's energy units. Pauli covariance and shot fractions are dimensionless. Variance coefficients use squared energy units. The molecular datasets use Hartree.

## Hamiltonian, covariance and design

```python
import numpy as np
from vgmc import PauliHamiltonian, SolverConfig, design_measurements

h = PauliHamiltonian(("X", "Z"), [1.0, 2.0],
                     constant=0.5, wire_order=("q",))
d = design_measurements(h, np.eye(2),
                        config=SolverConfig(ridge=0.0))
plan = d.allocate_shots(300)
print(d.groups)                  # ((1,), (0,))
print(plan.counts.tolist())       # [200, 100]
print(plan.estimate([np.zeros((200, 1), dtype=int),
                     np.ones((100, 1), dtype=int)]))  # 1.5
```

`PauliHamiltonian(words, coefficients, constant=0, wire_order=None)` combines unique nonidentity Pauli words `(n_terms,)`, their real energy coefficients `(n_terms,)`, an identity offset and unique hashable wire labels. Character zero corresponds to the first declared wire and the most significant state-vector bit. The default wire labels are `range(n_qubits)`. Constant Hamiltonians use an empty term list and can declare idle wires explicitly.

`design_measurements(hamiltonian, covariance, *, groups=None, config=None, expansion=None, max_added=0, initial_a=None)` returns `MeasurementDesign`. Its default library completes each SI basis with all visible terms. An explicit library supplies QWC groups of distinct term indices. Groups can overlap and jointly cover every term. `max_added=0` optimizes that library; a positive value enables VGMC growth. `ExpansionConfig.solver` supplies numerical settings during growth. The entry point returns a design after its numerical endpoint checks pass.

`MeasurementDesign` stores copied read-only arrays. Its `groups[g]`, `coefficients[g]` and `allocation[g]` describe one linear estimator block. Local coefficient vectors have shape `(len(groups[g]),)`; allocation has shape `(n_groups,)`. Coefficient blocks recover the Hamiltonian coefficients when summed at shared term indices. Allocations lie on the nonnegative simplex. A zero fraction requires a zero coefficient block. `bases` returns complete XYZ strings, using Z on unused wires. `status` and `gap` retain the fixed-library solver endpoint. `to_record()` returns a JSON-compatible representation when wire labels are JSON-compatible values.

`MeasurementDesign.allocate_shots(budget)` returns `ShotPlan`. Each nonzero block receives at least one shot; largest-remainder rounding distributes the remaining budget. Its `counts` array has shape `(n_groups,)` and integer dtype. `total_shots` reports the actual total. An exactly constant estimator requires zero quantum shots.

`ShotPlan.estimate(samples)` consumes one binary array `(counts[g], n_qubits)` per group, in the declared wire order. A zero-shot group accepts `None` or an empty two-axis array. The estimator applies term parities within each joint bitstring, averages each group's score and adds the identity offset. `ShotPlan.variance(covariance)` evaluates the corresponding finite-count variance in squared energy units.

## Covariance providers

`Covariance(matrix)` stores a copied symmetric matrix `(n_terms, n_terms)`. Paired NaNs mark unavailable entries. Every requested block is finite. `block(group, ridge=0)` returns a new float64 array `(len(group), len(group))` in the requested order, with `ridge * I` added. `fingerprint()` identifies the frozen input. The selected design blocks are positive definite after regularization.

```python
from vgmc import CallableCovariance, StateCovariance

# A callback supplies unregularized blocks in the requested term order.
provider = CallableCovariance(2, lambda g: np.eye(len(g)),
                              identity="identity-covariance-v1")
print(provider.block([1, 0], ridge=0.1).tolist())
# [[1.1, 0.0], [0.0, 1.1]]

# The state uses the Hamiltonian's declared wire order.
state_provider = StateCovariance(h, [1.0, 0.0])
print(state_provider.block([0, 1]).tolist())
# [[1.0, 0.0], [0.0, 0.0]]
```

`CovarianceProvider` specifies `n_terms`, `block(group, ridge)` and `fingerprint()`. A provider supplies consistent principal blocks in a common term order. A callback's explicit identity includes its scientific inputs and implementation. Concurrent candidate workers require a thread-safe callback. `StateCovariance` builds real Gram blocks from centered Pauli actions and uses a bounded vector cache. Its `cache_bytes` controls retained vector storage; requested blocks and work arrays use additional memory. A changed state, covariance model or Hamiltonian produces a new design input.

## Fixed-library solver

```python
from vgmc import solve

result = solve([[0], [1]], [1.0, 2.0], np.eye(2),
               config=SolverConfig(ridge=0.0), initial_a=[1.0, 2.0])
print(result["status"], result["physical_V"])
# converged 9.0
print(result["x"], result["a"])
# [[1.0], [2.0]] [0.3333333333333333, 0.6666666666666666]
```

`solve(groups, c, covariance, *, config=None, ridge=None, relative_gap=None, initial_a=None, threads=None, max_updates=None, checkpoint=None)` accepts real coefficients `(n_terms,)`, a covariance matrix or provider, and a covering index library. Its individual keyword options override `SolverConfig`. `initial_a` supplies copied nonnegative weights `(n_groups,)`, normalized and floored internally. A progress callback receives an audited result after configured accepted updates. Callback exceptions propagate.

| Result field | Meaning |
| --- | --- |
| `groups`, `x`, `a` | Library, local energy coefficient blocks and continuous shot fractions. |
| `mu` | Response vector `(n_terms,)` used for precision scoring. |
| `upper_bound`, `lower_bound` | Numerical model bounds with the design ridge, in squared energy units. |
| `gap` | `upper_bound / lower_bound - 1`; zero for the zero target. |
| `physical_V` | Reference-covariance variance coefficient with zero ridge. |
| `conservation_error` | Maximum absolute Hamiltonian coefficient residual. |
| `iterations`, `evaluations` | Accepted allocation updates and objective calls. |
| `status` | `converged`, `update_limit`, `evaluation_limit`, `numerical_stagnation` or `numerical_failure`. |
| `preparation` | Backend, occupied precision entries, assembly size and preparation time. |
| `numerical_history` | Per-stage optimizer messages and gap summaries. |

`converged` records the requested numerical gap and coefficient-recovery checks. Other statuses retain the last audited estimator. `design_measurements` raises `RuntimeError` for an unresolved endpoint. Invalid shapes, nonfinite values, complex coefficients with nonzero imaginary parts and fractional term indices raise `ValueError`. A nonpositive-definite selected design block raises `numpy.linalg.LinAlgError`. Zero targets return zero variance and zero gap.

`SolverConfig` records ridge, gap, allocation floor, residual refinements, conservation tolerance, callback interval, update/evaluation budgets, boundary search limits, stagnation count and BLAS threads. All fields are described in its class docstring. Unknown configuration fields are rejected. `max_updates=0` audits the initial allocation; `None` removes that explicit limit. Default update and evaluation limits remain finite.

## Candidate construction

`expand(words, c, covariance, groups, settings, ...)` performs group growth and reoptimization. `ExpansionConfig` records its solver settings, admission margin, batch policy, initial mass and finite-pool stopping tolerance. The returned record contains `groups`, `solution`, `trajectory`, `status` and, when scanned, `pool_audit`. Trajectory variance uses physical covariance. Pool stopping refers to the current library and the candidates generated from the enumerated seeds. QWC growth uses a greedy heuristic.

```python
from vgmc import Covariance, ExpansionConfig, expand

record = expand(["X", "Z"], np.array([1., 2.]), Covariance(np.eye(2)),
                [[0], [1]], ExpansionConfig(), max_added=0)
print(record["status"], record["trajectory"][0]["added"])
# added_budget_reached 0
```

`schur_increments(group, options, mu, covariance, ridge)` returns score increments `(len(options),)` after one current-group factorization. The denominator is a best linear prediction residual variance. The scalar `schur` interface computes one extension. `score` evaluates a group's complete precision score. `grow` uses full Schur increments or diagonal utility, then scores the completed visible support using full covariance. Equal scores use input-index order.

## PennyLane

```python
import pennylane as qml
from vgmc.adapters.pennylane import from_pennylane, make_tapes, to_pennylane

operator = 2 * qml.X("a") + 3 * qml.I("a")
h_pl = from_pennylane(operator, wire_order=("a",))
d_pl = design_measurements(h_pl, StateCovariance(h_pl, [1., 0.]))
tapes, postprocess = make_tapes([], d_pl.allocate_shots(1000))
device = qml.device("default.qubit", wires=["a"], seed=12)
energy = postprocess(qml.execute(tapes, device, diff_method=None))
```

`from_pennylane(operator, wire_order=None)` converts a concrete real Pauli operator, combines duplicate terms and stores the identity separately. Explicit wire order can include idle wires. `to_pennylane(hamiltonian)` returns the corresponding operator; constant operators declare at least one framework wire. `make_tapes(operations, plan)` returns finite-shot `QuantumScript` objects and an energy postprocessor. X readout applies Hadamard. Y readout applies S-dagger followed by Hadamard. Samples carry explicit shot and wire axes at the VGMC boundary.

The caller owns the device, state preparation and parameter updates. A `MeasurementDesign` contains frozen classical coefficients and allocations. Recompute the design when changing its covariance input. The adapter uses concrete coefficients and finite-shot tape execution. The executable [two-wire example](../examples/pennylane_measurement.py) reports sampled energy, analytic energy, predicted standard error and actual shots. The adapter targets PennyLane 0.45.1; its primary API references are listed in [sources.md](sources.md).

## Reproduction helpers

`vgmc.grouping` provides compatibility, basis completion, visible support, SI and stable LF. `vgmc.baselines.ogm` provides construction, SLSQP probability fitting, partial derandomization and conditional pooled variance. `vgmc.baselines.tequila.ics` calls the pinned external implementation. `vgmc.baselines.socp.solve` returns independently checked primal/dual evidence, including a `verified` flag.

`vgmc.sampling.integer_counts` performs copied, support-preserving integer allocation. `basis_distribution` returns joint probabilities; `score_outcomes` returns corresponding energy scores. `sample` retains replicate errors and RNG checkpoints, the configured RMSE reference threshold and the predicate `rmse_ci95_below_threshold`. `summarize` returns RMSE, empirical bias and a 95% percentile bootstrap interval.

`vgmc.data.load_input` verifies registered checksums and validates molecular axes, states and libraries. `load_design` verifies and validates frozen E2 estimators. `vgmc.experiments` exposes E1/E2/E3 collectors and runners. Workflow functions and CLI stages are described in [workflow.md](workflow.md). Public numerical docstrings contain runnable examples; the development check executes them with pytest.
