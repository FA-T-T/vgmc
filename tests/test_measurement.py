"""Backend-neutral estimator contracts and bitstring conventions."""

import numpy as np
import pytest

from vgmc import (
    MeasurementDesign,
    PauliHamiltonian,
    SolverConfig,
    StateCovariance,
    design_measurements,
)


def test_constant_estimator_uses_zero_shots() -> None:
    h = PauliHamiltonian((), [], constant=3.5, wire_order=("idle",))
    d = design_measurements(h, np.empty((0, 0)))
    plan = d.allocate_shots(100)
    assert plan.total_shots == 0
    assert plan.estimate([]) == 3.5


def test_joint_parity_and_constant_offset() -> None:
    h = PauliHamiltonian(("ZI", "IZ", "ZZ"), [1, 2, 3], constant=0.5)
    d = MeasurementDesign(h, ((0, 1, 2),), ([1, 2, 3],), [1])
    bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    assert d.allocate_shots(4).estimate([bits]) == 0.5


def test_shared_term_coefficients_and_readonly_arrays() -> None:
    h = PauliHamiltonian(("ZI", "IZ"), [1, 2])
    source = np.array([0.5, 2.0])
    d = MeasurementDesign(h, ((0, 1), (0,)), (source, [0.5]), [0.5, 0.5])
    source[0] = 4
    assert d.coefficients[0][0] == 0.5
    assert not d.allocation.flags.writeable
    with pytest.raises(ValueError):
        d.allocate_shots(2).estimate([[[0, 0]], [[0.1, 0]]])


def test_non_qwc_library_and_wrong_recovery_are_rejected() -> None:
    h = PauliHamiltonian(("X", "Z"), [1, 2])
    with pytest.raises(ValueError):
        MeasurementDesign(h, ((0, 1),), ([1, 2],), [1])
    with pytest.raises(ValueError):
        MeasurementDesign(h, ((0,), (1,)), ([1], [1]), [0.5, 0.5])


def test_state_covariance_matches_independent_pauli_matrices() -> None:
    rng = np.random.default_rng(4)
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state /= np.linalg.norm(state)
    h = PauliHamiltonian(("XI", "IY", "ZZ"), [1, 2, 3])
    provider = StateCovariance(h, state, cache_bytes=0)
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    paulis = [np.kron(matrices[w[0]], matrices[w[1]]) for w in h.words]
    means = np.array([np.vdot(state, p @ state).real for p in paulis])
    expected = np.array(
        [[np.vdot(state, p @ q @ state).real for q in paulis] for p in paulis]
    ) - np.outer(means, means)
    np.testing.assert_allclose(
        provider.block([0, 1, 2]), expected, atol=1e-14, rtol=0
    )
    assert len(provider._cache) == 0


def test_nonzero_target_api_result_is_a_valid_estimator() -> None:
    h = PauliHamiltonian(("X", "Z"), [1, 2])
    d = design_measurements(h, np.eye(2), config=SolverConfig(ridge=0))
    assert d.status == "converged"
    assert d.allocate_shots(30).counts.sum() == 30
