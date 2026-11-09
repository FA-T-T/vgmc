# Code design

## Mathematical

`PauliHamiltonian` declares the real coefficients of Pauli operators `P_j`, an identity coefficient and the wire tensor order. A measurement group `G_alpha` selects terms through `S_alpha`. Its physical covariance is `Sigma_alpha`; its design block is `Sigma_alpha + ridge * I`. 

The fixed-library precision matrix is

```text
W_alpha = S_alpha.T @ inverse(Sigma_alpha + ridge * I) @ S_alpha
K(a)    = sum_alpha a_alpha * W_alpha
mu      = inverse(K(a)) @ c
x_alpha = a_alpha * inverse(Sigma_alpha + ridge * I) @ S_alpha @ mu
```

The model starts with nonnegative allocations that sum to one and requires exact coefficient recovery, `sum_alpha S_alpha.T @ x_alpha = c`. Numeric updates use a positive allocation floor, while the recovery step repairs the final numerical coefficient residual in a strongly sampled occurrence. From there, the model upper bound evaluates the recovered feasible coefficients, the dual lower bound uses the maximum group precision score with an explicit numerical safety margin, and the physical variance evaluates the recovered coefficients with a zero-ridge reference covariance. Finite-shot plans use integer counts directly.

For candidate expansion, candidate growth scores an added term through a Schur complement, where the denominator is the best linear prediction residual variance. A batch of extensions shares the current-group factorization, and completion adds every Hamiltonian term visible in the resulting product basis. Finally, QWC generation is a deterministic greedy heuristic, and a finite-pool stopping result covers the current groups and the generated seed candidates.

## Modules and dependency

| Module | Responsibility |
| --- | --- |
| `measurement.py` | Hamiltonian identities, validated estimator blocks and integer shot plans. |
| `covariance.py`, `state.py` | Dense, callback and state-based covariance providers. |
| `api.py` | Framework-neutral measurement-design entry point. |
| `solver.py` | Fixed-library validation, allocation stages, budgets and endpoint status. |
| `_core.py`, `_linear.py`, `_boundary.py` | Occupied-entry precision assembly, refined dense/sparse solves and simplex updates. |
| `grouping.py`, `candidates.py`, `certificates.py`, `expansion.py` | QWC support, Schur growth, score checks and finite-pool construction. |
| `adapters/pennylane.py` | Optional Pauli conversion, product-basis tapes and energy reconstruction. |
| `baselines/` | OGM/LF/SI reproduction support, external ICS and independent SOCP reference. |
| `sampling.py` | Joint-outcome moments, finite-shot replicates, integer allocation and bootstrap summaries. |
| `config.py`, `_validation.py`, `_records.py` | Immutable configuration, numeric boundary checks and result types. |
| `data.py`, `artifacts.py`, `runtime.py` | Registered inputs, atomic committed artifacts and run lifecycle. |
| `experiments.py`, `workflow.py`, `plots.py` | E1/E2/E3 task execution, saved analysis and rendering. |

The core imports NumPy, SciPy, mpmath and threadpoolctl. Experiment tables and plots use optional pandas and Matplotlib. SOCP uses optional CVXPY/Clarabel. The PennyLane and Tequila packages enter at their adapter boundaries. The numerical layer returns domain results; the workflow owns task paths and lifecycle. Notebooks import the installed package.

Here is a more logically connected English version that keeps the code-documentation tone:

## Numerical and Data

The solver records actual accepted updates, objective calls, and stage messages, and its explicit endpoints distinguish requested numerical convergence from update, evaluation, and stagnation limits. Observer callbacks receive copied records, and callback failures propagate; thread counts, numerical precision, and code/dependency identities are included in the run binding. Covariance is supplied through two interchangeable interfaces: `StateCovariance` computes real centered-Pauli Gram blocks and retains vectors within a byte-limited cache, providing an integration route without a stored global covariance, while `CallableCovariance` supplies the same interface for externally measured or modeled blocks, with the callback declaring a stable input/implementation identity and consistent block semantics.

In integration, a framework adapter converts its operator to `PauliHamiltonian`, supplies covariance in that term order, calls `design_measurements`, creates an integer `ShotPlan`, executes joint product-basis measurements, and passes ordered bitstrings to `ShotPlan.estimate`. The caller owns state preparation and device execution, and parameter changes that alter design covariance create a new design input; the identity contribution is retained throughout conversion and postprocessing. PennyLane tapes contain state-preparation operations followed by explicit X/Y basis rotations and joint samples, with wire labels and tensor order preserved, while the energy postprocessor uses the frozen coefficients and actual counts and the numerical kernel executes on the CPU.

For data and recovery, raw molecular and saved-result files retain their original values and registered hashes; per-task commits are independently checksummed, completion commits a verified manifest, and checkpoint restoration uses a compatible valid generation and records fallback recovery. Details are in [docs/workflow.md](docs/workflow.md), and primary references plus pinned upstream identities are listed in [docs/sources.md](docs/sources.md).
