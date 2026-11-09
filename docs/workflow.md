# Experiment workflow

The repository provides the E1, E2 and E3 notebooks. The numerical module accepts domain arrays and immutable configuration. `workflow.py` plans runs, `runtime.py` manages committed tasks, `artifacts.py` publishes verified files, and `plots.py` reads saved analysis through the workflow renderer.

## Configuration

E1 uses `E1Config`, E2 uses `SamplingConfig`, and E3 uses `E3Config`. Nested numerical configuration uses `ExpansionConfig` and `SolverConfig`. Defaults are dataclass fields. Experiment JSON supplies declared values. An explicit validated configuration object and explicit selection arguments supply caller overrides. Unknown fields, invalid ranges and inconsistent nested settings raise an error before computation.

`prepare_run(experiment, root, output, ...)` verifies selected registered inputs and returns an unopened `RunStore` plus the effective configuration. Its plan creates no files. Each identity records effective values, field sources, field effects, selections, task set, actual input hashes, code hashes, dependency versions, Python, machine and numerical precision. Scientific fields select datasets, designs and random streams. Numerical fields control solver accuracy and admission. Runtime fields control checkpoint and chunk intervals. Bootstrap fields control analysis. Figure style belongs to presentation. Run identity conservatively includes every effective field.

## Lifecycle

| Mode | Directory state | Action |
| --- | --- | --- |
| `fresh` | New explicit path. | Create identity and execute tasks. |
| `resume` | Matching incomplete run. | Skip verified commits and continue compatible checkpoints. |
| `reuse` | Matching complete run. | Verify and read committed task files. |
| `auto` | Absent, incomplete or complete. | Select fresh, resume or reuse after identity checks. |

CLI computation requires `--allow-compute`. The explicit Python runners authorize their requested computation. `run_experiment` requires `allow_compute=True` for a computing mode. The notebooks' `full` mode supplies it. Analysis and rendering read recorded inputs; missing records raise an error.

One coordinator owns each run directory on local POSIX storage. A filesystem lock rejects a second writer. Task commits contain schema, scientific binding, task identity, payload and SHA-256. Identical repeated submissions retain the first commit; conflicting submissions raise an error. Completion publishes the expected-task manifest after every task verifies. Status files record running, interrupted, failed, incomplete or completed state. `verify` checks the manifest and every referenced task.

Checkpoints retain a validated current and previous generation. Writes use a temporary file, file flush, file `fsync`, atomic replacement and directory `fsync`. An unreadable or checksum-invalid current generation falls back to the previous valid generation. A valid checkpoint with a different scientific binding raises an identity error. Path traversal and symlink components are rejected at artifact boundaries.

VGMC checkpoints retain the last verified group library, estimator and trajectory. An interrupted candidate scan repeats from that endpoint. Inner solver checkpoints provide a compatible allocation warm start. E2 checkpoints retain group cursor, accumulated replicate estimates, analytic moments and every budget's RNG state. Random-reference checkpoints retain the last fixed-count endpoint and RNG state. Accepted optimization iterations and task commits remain separate units.

## Commands

Run from the repository root after installation:

```sh
python -m vgmc plan E1 --molecules H2 --covariances exact --methods SI SOCP VGMC --output outputs/E1/run-001
python -m vgmc run E1 --molecules H2 --covariances exact --methods SI SOCP VGMC --output outputs/E1/run-001 --allow-compute
python -m vgmc status E1 --output outputs/E1/run-001
python -m vgmc verify E1 --output outputs/E1/run-001
python -m vgmc run E1 --molecules H2 --covariances exact --methods SI SOCP VGMC --output outputs/E1/run-001 --mode reuse
python -m vgmc analyze E1 --run outputs/E1/run-001 --output outputs/E1/report
python -m vgmc render E1 --analysis outputs/E1/report/analysis --output outputs/E1/report
```

The same `run` command with `--mode resume --allow-compute` continues an incomplete matching run. `reproduce` processes supplied data and renders it. `analyze` recalculates E2 bootstrap summaries from supplied replicate errors, or reads committed computed-task metrics. Its analysis JSON stores source hashes and complete-precision numeric rows. `render` verifies this analysis and exports tables and figures.

## Numerical execution

The CPU solver selects dense or sparse precision solves from problem size and occupied entries. Sparse assembly indexes only occupied matrix entries. Local covariance factorizations serve coefficient recovery and candidate scoring. Residual refinement uses the platform's NumPy `longdouble`; its mantissa length enters run identity. Recomputed endpoints retain their measured numerical status and gap.

The supplied E1 records retain manuscript table precision. E2 records retain 300 replicate errors per design and budget. E3 records retain the plotted trajectories and five Random seeds. `DATA_MANIFEST.json` records file hashes. Hamiltonian source files, transforms and identity coefficients are recorded in `data/hamiltonians.json`.
