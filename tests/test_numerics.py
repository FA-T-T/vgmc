"""Independent numerical checks for fixed libraries and candidate growth."""

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from vgmc import Covariance, ExpansionConfig, SolverConfig, solve
from vgmc.baselines.socp import solve as conic_solve
from vgmc.candidates import schur_increments
from vgmc.config import SamplingConfig
from vgmc.expansion import expand


def test_partition_has_analytic_allocation_and_closed_zero_boundary() -> None:
    result = solve(
        [[0], [1], [2]],
        [1.0, -2.0, 0.0],
        np.diag([4.0, 9.0, 1.0]),
        config=SolverConfig(ridge=0.0, relative_gap=1e-6),
    )
    assert result["status"] == "converged"
    assert result["physical_V"] == pytest.approx(64.0, rel=1e-6)
    assert np.allclose(result["a"], [0.25, 0.75, 0.0], atol=1e-5)
    assert result["conservation_error"] < 1e-12


def test_overlapping_library_matches_independent_conic_formulation() -> None:
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(4, 4))
    covariance = matrix @ matrix.T + np.eye(4)
    groups = [[0, 1], [1, 2], [0, 2, 3], [3]]
    c = np.array([0.7, -0.4, 1.2, -0.3])
    result = solve(
        groups, c, covariance, config=SolverConfig(relative_gap=1e-6)
    )
    conic = conic_solve(groups, c, covariance)
    assert result["status"] == "converged"
    assert conic["verified"]
    assert result["lower_bound"] <= conic["upper_bound"] * (1 + 1e-8)
    assert result["upper_bound"] == pytest.approx(
        conic["upper_bound"], rel=2e-6
    )
    recovered = np.zeros(4)
    for group, x in zip(groups, result["x"], strict=True):
        np.add.at(recovered, group, x)
    assert np.allclose(recovered, c, atol=1e-12)


def test_schur_batch_matches_direct_augmented_inverse() -> None:
    rng = np.random.default_rng(23)
    m = rng.normal(size=(9, 9))
    matrix = m @ m.T + np.eye(9)
    response = rng.normal(size=9)
    group, options, ridge = [0, 3, 6], [1, 2, 4, 5], 1e-8
    actual = schur_increments(
        group, options, response, Covariance(matrix), ridge
    )
    block = matrix[np.ix_(group, group)] + ridge * np.eye(3)
    base = response[group] @ np.linalg.solve(block, response[group])
    expected = []
    for term in options:
        trial = group + [term]
        block = matrix[np.ix_(trial, trial)] + ridge * np.eye(4)
        expected.append(
            response[trial] @ np.linalg.solve(block, response[trial]) - base
        )
    assert np.allclose(actual, expected, atol=1e-12)


def test_sparse_preparation_and_audit_avoid_global_dense_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vgmc import CallableCovariance

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Sparse solve requested a global dense matrix")

    monkeypatch.setattr(csr_matrix, "toarray", forbidden)
    n = 1024
    provider = CallableCovariance(
        n, lambda group: np.eye(len(group)), "identity-v1"
    )
    result = solve(
        [[i] for i in range(n)],
        np.ones(n),
        provider,
        config=SolverConfig(ridge=0.0),
    )
    assert result["status"] == "converged"
    assert result["physical_V"] == pytest.approx(float(n * n))
    assert result["preparation"]["backend"] == "sparse"
    assert result["preparation"]["assembly_rows"] == n


def test_sparse_gradient_optimization_matches_partition_solution() -> None:
    from vgmc import CallableCovariance

    n = 256
    c = np.linspace(1.0, 2.0, n)
    provider = CallableCovariance(
        n, lambda group: np.eye(len(group)), "identity-gradient-v1"
    )
    result = solve(
        [[i] for i in range(n)],
        c,
        provider,
        config=SolverConfig(ridge=0.0, relative_gap=1e-5),
    )
    assert result["status"] == "converged"
    assert result["preparation"]["backend"] == "sparse"
    assert result["iterations"] > 0 and result["evaluations"] > 0
    assert result["physical_V"] == pytest.approx(float(c.sum() ** 2), rel=1e-5)
    assert np.allclose(result["a"], c / c.sum(), atol=1e-6)


def test_zero_target_and_accepted_update_budget() -> None:
    zero = solve([[0], [1]], [0.0, 0.0], np.eye(2))
    assert zero["status"] == "converged"
    assert zero["gap"] == zero["physical_V"] == 0
    assert conic_solve([[0]], [0.0], [[1.0]])["a"] == [1.0]
    limited = solve(
        [[0], [1]], [1.0, 9.0], np.eye(2), config=SolverConfig(max_updates=0)
    )
    assert limited["status"] == "update_limit"
    assert limited["iterations"] == limited["evaluations"] == 0


def test_expansion_resume_and_read_only_reuse(tmp_path: Path) -> None:
    words = ["XI", "IZ", "XX", "ZZ"]
    c = np.array([0.7, 0.5, -0.3, 0.2])
    covariance = Covariance(np.eye(4))
    groups = [[i] for i in range(4)]
    cfg = ExpansionConfig()
    path = tmp_path / "growth.json"

    def interrupt(state: object) -> None:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        expand(
            words,
            c,
            covariance,
            groups,
            cfg,
            max_added=2,
            checkpoint=path,
            progress=interrupt,
        )
    resumed = expand(
        words,
        c,
        covariance,
        groups,
        cfg,
        max_added=2,
        checkpoint=path,
        mode="resume",
    )
    reference = expand(words, c, covariance, groups, cfg, max_added=2)
    assert resumed["groups"] == reference["groups"]
    assert np.allclose(resumed["solution"]["a"], reference["solution"]["a"])
    before = path.read_bytes()
    reused = expand(
        words,
        c,
        covariance,
        groups,
        cfg,
        max_added=2,
        checkpoint=path,
        mode="reuse",
    )
    assert path.read_bytes() == before
    assert reused["status"] in (
        "added_budget_reached",
        "candidate_pool_tolerance_reached",
    )
    with pytest.raises(ValueError):
        expand(
            words,
            c,
            covariance,
            groups,
            cfg,
            max_added=2,
            checkpoint=path,
            mode="reuse",
            initial_a=[1, 2, 3, 4],
        )


def test_config_rejects_unknown_fields_and_invalid_limits() -> None:
    with pytest.raises(ValueError):
        SolverConfig.from_mapping({"max_update": 5})
    with pytest.raises(ValueError):
        SolverConfig(max_evaluations=0)
    with pytest.raises(ValueError):
        SamplingConfig(budgets=(20, 10))
    with pytest.raises(ValueError):
        ExpansionConfig(batch_minimum=3, batch_maximum=2)


def test_evaluation_budget_and_callback_failure_are_observable() -> None:
    limited = solve(
        [[0], [1]],
        [1.0, 9.0],
        np.eye(2),
        config=SolverConfig(max_evaluations=1),
    )
    assert limited["status"] == "evaluation_limit"
    assert limited["evaluations"] == 1

    def fail(record: object) -> None:
        raise ArithmeticError("callback storage failure")

    with pytest.raises(ArithmeticError, match="callback storage failure"):
        solve(
            [[0], [1]],
            [1.0, 9.0],
            np.eye(2),
            config=SolverConfig(checkpoint_interval=1),
            checkpoint=fail,
        )


def test_callback_records_do_not_mutate_the_solver() -> None:
    from typing import Any

    def mutate(record: dict[str, Any]) -> None:
        record["groups"][0][0] = 99
        record["a"][0] = -1

    result = solve(
        [[0], [1]],
        [1.0, 9.0],
        np.eye(2),
        config=SolverConfig(checkpoint_interval=1),
        checkpoint=mutate,
    )
    assert result["status"] == "converged"
    assert result["groups"] == [[0], [1]]
    assert min(result["a"]) >= 0
