"""Repository-input integrity and computation-free analysis stages."""

from pathlib import Path

import numpy as np
import pytest

from vgmc import SamplingConfig
from vgmc.data import load_design, load_input
from vgmc.sampling import sample
from vgmc.workflow import analyze, prepare_run, render, run_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_sampling_interruption_restores_identical_errors(
    tmp_path: Path,
) -> None:
    data = load_input(ROOT, "H2")
    design = load_design(ROOT, "H2", "exact", "VGMC", data=data)
    cfg = SamplingConfig(
        budgets=(64, 128),
        replicates=24,
        replicate_chunk=7,
        checkpoint_groups=1,
        bootstrap_replicates=50,
    )
    path = tmp_path / "sample.json"

    def interrupt(cursor: int, total: int) -> None:
        if cursor == 1:
            raise KeyboardInterrupt()

    kwargs = dict(molecule="H2", covariance="exact", method="VGMC")
    with pytest.raises(KeyboardInterrupt):
        sample(
            data, design, cfg, checkpoint=path, progress=interrupt, **kwargs
        )
    resumed = sample(data, design, cfg, checkpoint=path, resume=True, **kwargs)
    reference = sample(data, design, cfg, **kwargs)
    assert resumed == reference
    assert [r["actual_shots"] for r in resumed] == [64, 128]


def test_plan_and_missing_render_never_start_computation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run"
    store, _ = prepare_run(
        "E1",
        ROOT,
        path,
        molecules=["H2"],
        covariances=["exact"],
        methods=["SI"],
    )
    assert store.plan().action == "fresh"
    assert not path.exists()
    with pytest.raises(PermissionError):
        run_experiment(
            "E1",
            ROOT,
            path,
            molecules=["H2"],
            covariances=["exact"],
            methods=["SI"],
        )
    assert not path.exists()
    with pytest.raises(FileNotFoundError):
        render("E1", tmp_path / "absent", tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


def test_small_run_commits_and_reuses_the_same_results(tmp_path: Path) -> None:
    path = tmp_path / "run"
    kwargs = dict(
        molecules=["H2"], covariances=["exact"], methods=["LF", "SI", "SOCP"]
    )
    fresh = run_experiment("E1", ROOT, path, allow_compute=True, **kwargs)
    reused = run_experiment("E1", ROOT, path, mode="reuse", **kwargs)
    assert np.allclose(fresh.V_Ha2, reused.V_Ha2)
    frame = analyze("E1", ROOT, tmp_path / "analysis", run=path)
    assert len(frame) == 3
    assert render("E1", tmp_path / "analysis", tmp_path / "figures")
