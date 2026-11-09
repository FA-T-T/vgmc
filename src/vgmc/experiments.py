"""Verified data collection and task execution for E1, E2 and E3."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .artifacts import (
    CheckpointStore,
    checkpoint_lock,
    computation_identity,
    identity,
)
from .baselines import partition_design, ogm, socp
from .baselines.tequila import ics
from ._records import SolverResult
from .config import E1Config, E3Config, ExpansionConfig, SamplingConfig
from .covariance import variance
from .data import (
    ExperimentInput,
    load_design,
    load_input,
    read_json,
    verify_file,
)
from .expansion import expand
from .grouping import compatible, largest_first, sorted_insertion, visible
from .runtime import RunMode, RunStore
from .sampling import sample, stream_id, summarize
from .solver import solve

LOGGER = logging.getLogger(__name__)
METHODS = ["LF", "OGM", "SI", "ICS", "SOCP", "VGMC"]


def collect_e1(root: Path) -> pd.DataFrame:
    """Read the verified E1 table at the stored table precision.

    Args:
        root: Reproduction repository containing ``DATA_MANIFEST.json``.

    Returns:
        One row per molecule, covariance source and method. ``V_Ha2`` is the
        variance coefficient in squared Hartree, and ``precision`` records
        the stored table precision.
    """
    return pd.read_csv(verify_file(root, "data/E1/results.csv"))


def collect_e2(root: Path) -> pd.DataFrame:
    """Recalculate E2 RMSE and intervals from verified replicate errors.

    Args:
        root: Reproduction repository root.

    Returns:
        One row per frozen design and shot budget, with RMSE in Hartree and
        two-element percentile bootstrap intervals in ``rmse_ci95``.

    Raises:
        ValueError: Data checksums, replicate counts or budget axes differ.
    """
    cfg = SamplingConfig.from_mapping(read_json(root / "configs/E2.json"))
    records = read_json(verify_file(root, "data/E2/samples.json.gz"))
    streams: dict[tuple[str, str, str], np.random.Generator] = {}
    budget_axes: dict[tuple[str, str, str], list[int]] = {}
    rows = []
    for record in records:
        key = (record["molecule"], record["covariance"], record["method"])
        if len(record["errors"]) != cfg.replicates:
            raise ValueError("Saved E2 replicate axis differs")
        budget_axes.setdefault(key, []).append(record["budget"])
        if key not in streams:
            stream = stream_id(*key)
            streams[key] = np.random.default_rng(
                np.random.SeedSequence([cfg.bootstrap_seed, stream])
            )
        metrics = summarize(
            np.asarray(record["errors"]),
            streams[key],
            cfg.bootstrap_replicates,
        )
        row = {
            k: v
            for k, v in record.items()
            if k not in ("errors", "rmse", "rmse_ci95", "empirical_bias")
        }
        rows.append(
            row
            | metrics
            | dict(
                accuracy_threshold=cfg.epsilon,
                rmse_ci95_below_threshold=metrics["rmse_ci95"][1]
                <= cfg.epsilon,
            )
        )
    if any(tuple(axis) != cfg.budgets for axis in budget_axes.values()):
        raise ValueError("Saved E2 budget axis differs")
    return pd.DataFrame(rows)


def collect_e3(root: Path) -> pd.DataFrame:
    """Read verified E3 growth trajectories and five-seed Random references.

    Args:
        root: Reproduction repository root.

    Returns:
        DataFrame with molecule, method, seed, added count, physical variance,
        initial variance, gain percentage and active-group percentage.
    """
    return pd.read_csv(verify_file(root, "data/E3/plotted_data.csv"))


def _compute_e1(
    root: Path,
    store: RunStore,
    molecules: list[str],
    covariances: list[str],
    methods: list[str],
    workers: int,
    config: E1Config,
) -> pd.DataFrame:
    """Construct missing E1 task designs and commit their evaluated rows."""
    rows = []
    for molecule in molecules:
        for mode in covariances:
            data = load_input(root, molecule, mode)
            words, c = data["words"], data["coefficients"]
            covariance = data["design_covariance"]
            groups = data["library"]["groups"]
            for method in methods:
                task = f"{molecule}_{mode}_{method}"
                saved = store.read_task(task)
                if saved is not None:
                    rows.append(saved["row"])
                    continue
                LOGGER.info("E1 %s", task)
                design: dict[str, Any]
                if method in ("LF", "SI"):
                    partition = (
                        largest_first(words)
                        if method == "LF"
                        else sorted_insertion(words, c)
                    )
                    design = partition_design(partition, c, covariance)
                elif method == "OGM":
                    design = ogm.fit(
                        words, c, ogm.OGMConfig(shots=config.ogm_shots)
                    )
                elif method == "ICS":
                    design = ics(
                        words, c, covariance, iterations=config.ics_iterations
                    )
                elif method == "SOCP":
                    design = socp.solve(groups, c, covariance)
                    if not design["verified"]:
                        raise ArithmeticError(
                            "SOCP result fails its gap check"
                        )
                elif method == "VGMC":
                    result = expand(
                        words,
                        c,
                        covariance,
                        groups,
                        config.expansion,
                        initial_a=data["initial_a"],
                        workers=workers,
                        checkpoint=store.checkpoint_path(task),
                        mode="auto",
                        certify_admission=False,
                    )
                    if result["status"] not in (
                        "candidate_pool_tolerance_reached",
                        "zero_target",
                    ):
                        raise ArithmeticError(
                            f"E1 expansion: {result['status']}"
                        )
                    design = dict(
                        result["solution"],
                        groups=result["groups"],
                        outer_status=result["status"],
                    )
                else:
                    raise ValueError("Unknown E1 method")
                if method == "OGM":
                    metrics = ogm.evaluate_counts(
                        words,
                        c,
                        design["bases"],
                        design["schedule"]["counts"],
                        data["reference_covariance"],
                        data["exact_means"],
                    )
                    value, bias = metrics["V_effective"], metrics["bias"]
                else:
                    value = variance(
                        design["groups"],
                        design["x"],
                        design["a"],
                        data["reference_covariance"],
                    )
                    bias = 0.0
                row = dict(
                    molecule=molecule,
                    covariance=mode,
                    method=method,
                    V_Ha2=value,
                    bias_Ha=bias,
                    precision="float64",
                    status=design.get(
                        "outer_status",
                        design.get(
                            "termination",
                            design.get(
                                "solver_status",
                                (
                                    "fixed_updates_completed"
                                    if method == "ICS"
                                    else "completed"
                                ),
                            ),
                        ),
                    ),
                    gap=design.get("gap"),
                )
                store.commit_task(task, dict(row=row, design=design))
                rows.append(row)
    return pd.DataFrame(rows)


def _compute_e2(
    root: Path,
    store: RunStore,
    molecules: list[str],
    covariances: list[str],
    methods: list[str],
    config: SamplingConfig,
) -> pd.DataFrame:
    """Resume missing E2 task streams from their committed group
    checkpoints.
    """
    records = []
    for molecule in molecules:
        for mode in covariances:
            data = load_input(root, molecule, mode)
            for method in methods:
                task = f"{molecule}_{mode}_{method}"
                saved = store.read_task(task)
                if saved is None:
                    design = load_design(
                        root, molecule, mode, method, data=data
                    )
                    path = store.checkpoint_path(task)
                    exists = CheckpointStore(path, "probe").exists()
                    rows = sample(
                        data,
                        design,
                        config,
                        molecule=molecule,
                        covariance=mode,
                        method=method,
                        checkpoint=path,
                        resume=exists,
                    )
                    saved = dict(rows=rows)
                    store.commit_task(task, saved)
                records.extend(saved["rows"])
    return pd.DataFrame(
        [{k: v for k, v in row.items() if k != "errors"} for row in records]
    )


@checkpoint_lock
def random_trajectory(
    data: ExperimentInput,
    initial: Mapping[str, Any],
    targets: list[int],
    seed: int,
    settings: Mapping[str, Any] | ExpansionConfig,
    maximum_attempts: int = 1000000,
    *,
    checkpoint: Path | None = None,
    resume: bool = False,
    progress: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Grow random-permutation QWC groups at a fixed addition grid.

    Args:
        data: Molecular inputs and the common initial group library.
        initial: Verified common initial coefficient/allocation solution.
        targets: Increasing added-group counts beginning at zero.
        seed: Random-permutation stream seed.
        settings: Effective expansion and solver options.
        maximum_attempts: Maximum number of random group proposals.
        checkpoint: Optional committed target-endpoint checkpoint.
        resume: Restore a compatible checkpoint instead of starting fresh.
        progress: Callback after each committed target index.

    Returns:
        Physical-variance and activity rows in the recorded target order.
        Checkpoints retain the RNG and last verified target endpoint.
    """
    from ._validation import integer

    cfg = (
        settings
        if isinstance(settings, ExpansionConfig)
        else ExpansionConfig.from_mapping(settings)
    )
    integer(seed, "seed")
    integer(maximum_attempts, "maximum_attempts", 1)
    targets = [integer(k, "target") for k in targets]
    if not targets or targets[0] != 0 or sorted(set(targets)) != targets:
        raise ValueError("Random targets must start at zero and increase")
    words, c = data["words"], data["coefficients"]
    covariance = data["design_covariance"]
    groups = [g.copy() for g in data["library"]["groups"]]
    base = len(groups)
    rng = np.random.default_rng(seed)
    attempts = cursor = 0
    solution = dict(initial)
    v0 = variance(groups, initial["x"], initial["a"], covariance)
    rows: list[dict[str, Any]] = []
    binding = identity(
        dict(
            code=computation_identity(),
            covariance=covariance.fingerprint(),
            words=words,
            c=c,
            initial={
                k: initial[k]
                for k in (
                    "groups",
                    "x",
                    "a",
                    "mu",
                    "upper_bound",
                    "lower_bound",
                    "gap",
                    "status",
                )
            },
            targets=targets,
            seed=seed,
            config=cfg.to_dict(),
            attempts=maximum_attempts,
        )
    )
    store = CheckpointStore(checkpoint, binding) if checkpoint else None
    if resume:
        if store is None:
            raise ValueError("Resume requires a Random checkpoint")
        saved = store.load().payload
        groups, solution, rows = (
            saved["groups"],
            saved["solution"],
            saved["rows"],
        )
        attempts, cursor = saved["attempts"], saved["cursor"]
        rng.bit_generator.state = saved["rng"]
    elif store and store.exists():
        raise FileExistsError("Random checkpoint exists; select resume")
    seen = {tuple(sorted(group)) for group in groups}
    for target_index in range(cursor, len(targets)):
        target = targets[target_index]
        previous = len(groups)
        while len(groups) - base < target and attempts < maximum_attempts:
            attempts += 1
            basis = "I" * len(words[0])
            for i in rng.permutation(len(words)):
                if compatible(basis, words[i]):
                    basis = "".join(
                        b if b != "I" else a
                        for a, b in zip(basis, words[i], strict=True)
                    )
            group = visible(words, basis.replace("I", "Z"))
            if tuple(group) not in seen:
                groups.append(group)
                seen.add(tuple(group))
        if len(groups) > previous:
            count = len(groups) - previous
            a = np.r_[
                np.asarray(solution["a"]) * (1 - cfg.new_mass),
                np.full(count, cfg.new_mass / count),
            ]
            solution = dict(
                solve(
                    groups,
                    c,
                    covariance,
                    config=cfg.solver.with_gap(cfg.terminal_gap),
                    initial_a=a,
                )
            )
            if solution["status"] != "converged":
                raise ArithmeticError(
                    "Random fixed-count optimization unresolved"
                )
        added = len(groups) - base
        value = variance(groups, solution["x"], solution["a"], covariance)
        active = int(
            np.count_nonzero(
                np.asarray(solution["a"])[base:] > cfg.active_threshold
            )
        )
        rows.append(
            dict(
                added=added,
                V=value,
                V0=v0,
                gain_pct=100 * (1 - value / v0) if v0 else 0.0,
                active_added=active,
                active_pct=100 * active / added if added else None,
                seed=seed,
                status=(
                    "count_grid_reached"
                    if added == target
                    else "proposal_budget_reached"
                ),
            )
        )
        if store:
            store.save(
                dict(
                    groups=groups,
                    solution=solution,
                    rows=rows,
                    attempts=attempts,
                    cursor=target_index + 1,
                    rng=rng.bit_generator.state,
                )
            )
        if progress:
            progress(target_index + 1)
        if added < target:
            break
    return rows


def _compute_e3(
    root: Path,
    store: RunStore,
    molecules: list[str],
    arms: list[str],
    workers: int,
    random_reference: bool,
    config: E3Config,
) -> pd.DataFrame:
    """Run missing E3 arms and Random reference tasks from shared inputs."""
    cfg = config.numerics
    labels = {"A": "VGMC", "B": "Frozen", "C": "Diag", "D": "Frozen+Diag"}
    rows = []
    for molecule in molecules:
        data = load_input(root, molecule)
        groups, covariance = (
            data["library"]["groups"],
            data["design_covariance"],
        )
        initial_path = store.path / "initial" / f"{molecule}.json"
        binding = identity(dict(parent=store.binding, molecule=molecule))
        initial_store = CheckpointStore(initial_path, binding)
        if initial_store.exists():
            initial = initial_store.load().payload
        else:
            result = solve(
                groups,
                data["coefficients"],
                covariance,
                config=cfg.solver.with_gap(cfg.terminal_gap),
                initial_a=data["initial_a"],
            )
            if result["status"] != "converged":
                raise ArithmeticError("Common initial optimization unresolved")
            initial = dict(result)
            initial_store.save(initial)
        limits = config.molecules[molecule]
        for arm in arms:
            task = f"{molecule}_{arm}"
            saved = store.read_task(task)
            if saved is None:
                expanded = expand(
                    data["words"],
                    data["coefficients"],
                    covariance,
                    groups,
                    cfg,
                    initial_solution=cast(SolverResult, initial),
                    arm=arm,
                    batch_size=limits.batch,
                    max_added=limits.budget,
                    targets=list(limits.targets),
                    workers=workers,
                    checkpoint=store.checkpoint_path(task),
                    mode="auto",
                )
                if expanded["status"] not in (
                    "added_budget_reached",
                    "candidate_pool_tolerance_reached",
                    "zero_target",
                ):
                    raise ArithmeticError(
                        f"E3 expansion: {expanded['status']}"
                    )
                saved = dict(
                    rows=[
                        dict(
                            row,
                            molecule=molecule,
                            method=labels[arm],
                            seed=None,
                            status=expanded["status"],
                        )
                        for row in expanded["trajectory"]
                    ]
                )
                store.commit_task(task, saved)
            rows.extend(saved["rows"])
        if random_reference:
            for seed in config.random_seeds:
                task = f"{molecule}_Random_{seed}"
                saved = store.read_task(task)
                if saved is None:
                    path = store.checkpoint_path(task)
                    trajectory = random_trajectory(
                        data,
                        initial,
                        list(limits.targets),
                        seed,
                        cfg,
                        checkpoint=path,
                        resume=CheckpointStore(path, "probe").exists(),
                    )
                    saved = dict(
                        rows=[
                            dict(row, molecule=molecule, method="Random")
                            for row in trajectory
                        ]
                    )
                    store.commit_task(task, saved)
                rows.extend(saved["rows"])
    return pd.DataFrame(rows)


def run_e1(
    root: Path,
    output: Path,
    *,
    molecules: list[str],
    covariances: list[str],
    methods: list[str] | None = None,
    workers: int = 1,
    mode: RunMode = "fresh",
    config: E1Config | None = None,
) -> pd.DataFrame:
    """Run E1 through the shared lifecycle and return variance-coefficient
    rows.

    Args:
        root: Verified reproduction repository root.
        output: Explicit run directory. Fresh mode requires a new directory.
        molecules: Selected registered datasets.
        covariances: Selected ``exact`` and/or ``cisd`` design sources.
        methods: Selected LF, OGM, SI, ICS, SOCP and/or VGMC methods.
        workers: Explicit candidate worker count.
        mode: Fresh, verified reuse, compatible resume or automatic selection.
        config: Optional validated E1 overrides.

    Returns:
        DataFrame with one row per declared design task.
    """
    from .workflow import run_experiment

    return run_experiment(
        "E1",
        root,
        output,
        molecules=molecules,
        covariances=covariances,
        methods=methods,
        workers=workers,
        mode=mode,
        allow_compute=True,
        config=config,
    )


def run_e2(
    root: Path,
    output: Path,
    *,
    molecules: list[str],
    covariances: list[str],
    methods: list[str] | None = None,
    settings: Mapping[str, Any] | SamplingConfig | None = None,
    mode: RunMode = "fresh",
) -> pd.DataFrame:
    """Run E2 sampling through the shared lifecycle and return RMSE rows.

    Args:
        root: Verified reproduction repository root.
        output: Explicit run directory.
        molecules: Selected molecular datasets.
        covariances: Frozen design covariance sources.
        methods: ICS, SOCP and/or VGMC.
        settings: Validated sampling settings or canonical JSON overrides.
        mode: Fresh, verified reuse, compatible resume or automatic selection.

    Returns:
        RMSE/bias rows. Replicate errors are retained in task commits.
    """
    from .workflow import run_experiment

    return run_experiment(
        "E2",
        root,
        output,
        molecules=molecules,
        covariances=covariances,
        methods=methods,
        mode=mode,
        allow_compute=True,
        config=(
            SamplingConfig.from_mapping(settings)
            if isinstance(settings, Mapping)
            else settings
        ),
    )


def run_e3(
    root: Path,
    output: Path,
    *,
    molecules: list[str],
    arms: list[str] | None = None,
    workers: int = 1,
    random_reference: bool = True,
    config: Mapping[str, Any] | E3Config | None = None,
    mode: RunMode = "fresh",
) -> pd.DataFrame:
    """Run E3 through the shared lifecycle and return growth-trajectory rows.

    Args:
        root: Verified reproduction repository root.
        output: Explicit run directory.
        molecules: Selected molecular datasets.
        arms: Selected A/B/C/D growth arms.
        workers: Candidate worker count, recorded in scientific identity.
        random_reference: Include the configured fixed-count Random seeds.
        config: Validated E3 settings or canonical JSON overrides.
        mode: Fresh, verified reuse, compatible resume or automatic selection.

    Returns:
        Physical-variance, group-count and activity trajectory rows.
    """
    from .workflow import run_experiment

    return run_experiment(
        "E3",
        root,
        output,
        molecules=molecules,
        arms=arms,
        workers=workers,
        random_reference=random_reference,
        mode=mode,
        allow_compute=True,
        config=(
            E3Config.from_mapping(config)
            if isinstance(config, Mapping)
            else config
        ),
    )
