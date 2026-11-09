"""Shared experiment planning, execution, analysis and rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import file_hash, identity, read_json, write_json
from .config import E1Config, E3Config, SamplingConfig
from .data import verify_file
from .runtime import RunMode, RunStore

ExperimentConfig = E1Config | SamplingConfig | E3Config


def prepare_run(
    experiment: str,
    root: Path,
    output: Path,
    *,
    molecules: Sequence[str] | None = None,
    covariances: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    arms: Sequence[str] | None = None,
    workers: int = 1,
    random_reference: bool = True,
    mode: RunMode = "fresh",
    allow_compute: bool = False,
    config: ExperimentConfig | None = None,
) -> tuple[RunStore, ExperimentConfig]:
    """Validate experiment inputs and plan a run without writing files.

    Args:
        experiment: ``E1``, ``E2`` or ``E3``.
        root: Repository with the registered input manifest.
        output: Explicit run directory.
        molecules: Dataset selection; the configured set is the default.
        covariances: E1/E2 design sources, ``exact`` and/or ``cisd``.
        methods: E1/E2 method selection.
        arms: E3 growth arms, selected from A, B, C and D.
        workers: Positive candidate-worker count.
        random_reference: Include every configured E3 Random seed.
        mode: Run lifecycle mode.
        allow_compute: Authorize computation when entering the returned store.
        config: Validated experiment configuration; defaults to repository
        JSON.

    Returns:
        An unopened run store and the effective immutable configuration.

    Raises:
        ValueError: Selections, configuration or registered input hashes
        differ.
    """
    from ._validation import integer

    classes: dict[
        str, type[E1Config] | type[SamplingConfig] | type[E3Config]
    ] = {"E1": E1Config, "E2": SamplingConfig, "E3": E3Config}
    if experiment not in classes:
        raise ValueError("Experiment must be E1, E2 or E3")
    integer(workers, "workers", 1)
    if type(random_reference) is not bool:
        raise ValueError("random_reference must be boolean")
    cls: type[E1Config] | type[SamplingConfig] | type[E3Config] = classes[
        experiment
    ]
    configured = read_json(root / f"configs/{experiment}.json")
    cfg: ExperimentConfig = config or cls.from_mapping(configured)
    if not isinstance(cfg, cls):
        raise ValueError("Configuration type differs from the experiment")

    def select(
        values: Sequence[str] | None, allowed: Sequence[str]
    ) -> list[str]:
        """Canonicalize a nonempty unique selection in protocol order."""
        chosen = list(allowed if values is None else values)
        if (
            not chosen
            or len(set(chosen)) != len(chosen)
            or set(chosen) - set(allowed)
        ):
            raise ValueError(
                "Selection must contain distinct configured entries"
            )
        return [value for value in allowed if value in chosen]

    known_molecules = ["H2", "LiH", "BeH2", "H2O", "NH3"]
    mols = select(
        molecules,
        (
            list(cfg.molecules)
            if isinstance(cfg, (E1Config, E3Config))
            else known_molecules
        ),
    )
    covs = select(
        covariances,
        (
            list(cfg.covariances)
            if isinstance(cfg, E1Config)
            else ["exact", "cisd"]
        ),
    )
    meths = select(
        methods,
        (
            list(cfg.methods)
            if isinstance(cfg, E1Config)
            else ["ICS", "SOCP", "VGMC"]
        ),
    )
    selected_arms = select(arms, ["A", "B", "C", "D"])
    tasks = (
        (
            [f"{m}_{a}" for m in mols for a in selected_arms]
            + (
                [f"{m}_Random_{s}" for m in mols for s in cfg.random_seeds]
                if random_reference and isinstance(cfg, E3Config)
                else []
            )
        )
        if experiment == "E3"
        else [f"{m}_{c}_{k}" for m in mols for c in covs for k in meths]
    )
    paths = [
        f"data/inputs/{m}/{name}"
        for m in mols
        for name in (
            "state.npz",
            "covariance.npz",
            "library.json",
            "warm.json",
        )
    ]
    if experiment == "E2":
        paths += [
            f"data/designs/{m}_{c}_{k}.json.gz"
            for m in mols
            for c in covs
            for k in meths
        ]
    inputs = {path: file_hash(verify_file(root, path)) for path in paths}
    fields: dict[str, Any] = {}

    def describe(values: Mapping[str, Any], prefix: str = "") -> None:
        """Record each effective field and its source and execution effect."""
        for key, value in values.items():
            path = prefix + key
            if isinstance(value, Mapping):
                describe(value, path + ".")
            else:
                source = (
                    "explicit configuration object"
                    if config is not None
                    else f"configs/{experiment}.json"
                )
                if config is None:
                    cursor: Any = configured
                    present = True
                    for component in path.split("."):
                        if (
                            not isinstance(cursor, Mapping)
                            or component not in cursor
                        ):
                            present = False
                            break
                        cursor = cursor[component]
                    if not present:
                        source = "dataclass default"
                effect = (
                    "analysis"
                    if key.startswith("bootstrap")
                    else (
                        "runtime"
                        if key
                        in (
                            "checkpoint_interval",
                            "checkpoint_groups",
                            "replicate_chunk",
                        )
                        else (
                            "numerics"
                            if "solver." in path or "numerics." in path
                            else "science"
                        )
                    )
                )
                fields[path] = dict(value=value, source=source, effect=effect)

    describe(cfg.to_dict())
    scientific = dict(
        experiment=experiment,
        config=cfg.to_dict(),
        configuration_fields=fields,
        inputs=inputs,
        molecules=mols,
        covariances=covs,
        methods=meths,
        arms=selected_arms,
        workers=workers,
        random_reference=random_reference,
    )
    return (
        RunStore(
            output, scientific, tasks, mode=mode, allow_compute=allow_compute
        ),
        cfg,
    )


def _task_frame(store: RunStore, experiment: str) -> pd.DataFrame:
    """Read a complete task table without invoking experiment code."""
    rows: list[dict[str, Any]] = []
    for task in store.tasks:
        payload = store.read_task(task)
        if payload is None:
            raise ValueError("A committed task is missing")
        rows.extend(
            [payload["row"]] if experiment == "E1" else payload["rows"]
        )
    return pd.DataFrame(
        [{k: v for k, v in row.items() if k != "errors"} for row in rows]
    )


def run_experiment(
    experiment: str,
    root: Path,
    output: Path,
    **options: Any,
) -> pd.DataFrame:
    """Execute or reuse one planned experiment and return committed rows.

    Args:
        experiment: E1, E2 or E3.
        root: Verified repository root.
        output: Explicit run directory.
        **options: Keyword arguments documented by :func:`prepare_run`.

    Returns:
        Verified committed task rows. Reuse performs no experiment work.

    Raises:
        PermissionError: A computing mode lacks ``allow_compute=True``.
        ArithmeticError: A required numerical endpoint remains unresolved.
    """
    store, cfg = prepare_run(experiment, root, output, **options)
    with store:
        if store.decision is not None and store.decision.compute:
            from .experiments import _compute_e1, _compute_e2, _compute_e3

            s = store.scientific
            if isinstance(cfg, E1Config):
                _compute_e1(
                    root,
                    store,
                    s["molecules"],
                    s["covariances"],
                    s["methods"],
                    s["workers"],
                    cfg,
                )
            elif isinstance(cfg, SamplingConfig):
                _compute_e2(
                    root,
                    store,
                    s["molecules"],
                    s["covariances"],
                    s["methods"],
                    cfg,
                )
            else:
                _compute_e3(
                    root,
                    store,
                    s["molecules"],
                    s["arms"],
                    s["workers"],
                    s["random_reference"],
                    cfg,
                )
            store.complete()
        return _task_frame(store, experiment)


def analyze(
    experiment: str,
    root: Path,
    output: Path,
    *,
    run: Path | None = None,
    molecules: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Write a checksummed analysis table from saved data or a completed run.

    Args:
        experiment: E1, E2 or E3.
        root: Verified reproduction repository.
        output: Analysis directory, separate from immutable run files.
        run: Completed compatible run; defaults to the supplied saved records.
        molecules: Optional dataset subset for the saved analysis.

    Returns:
        Analysis rows. E2 intervals use the recorded bootstrap protocol.

    Raises:
        FileNotFoundError: Required saved records or run manifest are absent.
        ValueError: Input or run integrity checks fail.
    """
    from .experiments import collect_e1, collect_e2, collect_e3

    collectors = {"E1": collect_e1, "E2": collect_e2, "E3": collect_e3}
    if experiment not in collectors:
        raise ValueError("Experiment must be E1, E2 or E3")
    if run is None:
        frame = collectors[experiment](root)
        relative = {
            "E1": "data/E1/results.csv",
            "E2": "data/E2/samples.json.gz",
            "E3": "data/E3/plotted_data.csv",
        }[experiment]
        source: dict[str, Any] = dict(
            kind="saved", input=file_hash(verify_file(root, relative))
        )
        if experiment == "E2":
            source["configuration"] = identity(
                SamplingConfig.from_mapping(
                    read_json(root / "configs/E2.json")
                ).to_dict()
            )
    else:
        frozen = read_json(run / "identity.json")
        s = frozen["scientific"]
        if s["experiment"] != experiment:
            raise ValueError("Run experiment differs")
        store = RunStore(run, s, frozen["tasks"], mode="reuse")
        store.plan()
        frame = _task_frame(store, experiment)
        source = dict(
            kind="run",
            binding=store.binding,
            manifest=file_hash(run / "manifest.json"),
        )
    if molecules is not None:
        if not molecules or set(molecules) - set(frame.molecule):
            raise ValueError("Analysis molecule selection differs")
        frame = frame[frame.molecule.isin(molecules)]
        source["molecules"] = list(molecules)
    # Object conversion retains full float precision and encodes missing seeds.
    rows = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    payload = dict(schema=1, experiment=experiment, source=source, rows=rows)
    write_json(
        output / f"{experiment}-analysis.json",
        dict(payload, sha256=identity(payload)),
    )
    return frame


def load_analysis(experiment: str, analysis: Path) -> pd.DataFrame:
    """Read rows from an analysis with a matching experiment and checksum.

    Args:
        experiment: E1, E2 or E3.
        analysis: Directory containing the analysis JSON.

    Returns:
        DataFrame in committed row order.

    Raises:
        FileNotFoundError: The selected analysis is absent.
        ValueError: Experiment identity or checksum differs.
    """
    record = read_json(analysis / f"{experiment}-analysis.json")
    payload = {k: v for k, v in record.items() if k != "sha256"}
    if (
        record.get("sha256") != identity(payload)
        or record.get("experiment") != experiment
    ):
        raise ValueError("Analysis identity or checksum differs")
    return pd.DataFrame(record["rows"])


def render(experiment: str, analysis: Path, output: Path) -> list[Path]:
    """Render an existing verified analysis into CSV tables and figures.

    Args:
        experiment: E1, E2 or E3.
        analysis: Directory containing the committed analysis JSON.
        output: Destination for CSV tables and PDF/PNG figures.

    Returns:
        Paths of generated artifacts.

    Raises:
        FileNotFoundError: The analysis file is absent.
        ValueError: Analysis identity, checksum or experiment differs.
    """
    from .plots import plot_e2, plot_e3, save_figure, summarize_e3, table_e1

    frame = load_analysis(experiment, analysis)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    molecules = [
        m
        for m in ("H2", "LiH", "BeH2", "H2O", "NH3")
        if m in set(frame.molecule)
    ]
    if experiment == "E1":
        for covariance in frame.covariance.unique():
            path = output / f"E1-{covariance}.csv"
            table_e1(frame, covariance).to_csv(path)
            paths.append(path)
    else:
        path = output / f"{experiment}-table.csv"
        (frame if experiment == "E2" else summarize_e3(frame)).to_csv(
            path, index=False
        )
        paths.append(path)
        import matplotlib.pyplot as plt

        if experiment == "E2":
            for covariance in frame.covariance.unique():
                for zoom in (False, True):
                    figure = plot_e2(
                        frame,
                        covariance=covariance,
                        zoom=zoom,
                        molecules=molecules,
                    )
                    name = f"E2-{covariance}-{'zoom' if zoom else 'full'}"
                    save_figure(figure, output, name)
                    plt.close(figure)
                    paths.extend(
                        output / f"{name}.{s}" for s in ("pdf", "png")
                    )
        elif experiment == "E3":
            figure = plot_e3(frame, molecules=molecules)
            save_figure(figure, output, "E3")
            plt.close(figure)
            paths.extend(output / f"E3.{s}" for s in ("pdf", "png"))
        else:
            raise ValueError("Experiment must be E1, E2 or E3")
    return paths
