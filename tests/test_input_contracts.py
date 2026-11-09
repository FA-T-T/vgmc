"""Public input contracts for covariance and allocation calculations."""

import numpy as np
import pytest

from vgmc.covariance import Covariance, quadratic, variance
from vgmc.sampling import integer_counts, score_outcomes
from vgmc.solver import solve


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_quadratic_rejects_nonfinite_vectors(bad: float) -> None:
    with pytest.raises(ValueError):
        quadratic(np.array([bad]), np.ones((1, 1)))


def test_variance_rejects_nan_allocation() -> None:
    with pytest.raises(ValueError):
        variance([[0]], [[1.0]], [np.nan], Covariance(np.eye(1)))


def test_integer_covariance_accepts_real_ridge() -> None:
    covariance = Covariance(np.array([[1, 0], [0, 1]]))
    np.testing.assert_allclose(covariance.block([0], 0.25), [[1.25]])


def test_solver_rejects_fractional_indices() -> None:
    with pytest.raises((ValueError, TypeError)):
        solve([[0.25], [1]], np.ones(2), np.eye(2), max_updates=2)


def test_solver_rejects_complex_coefficients() -> None:
    with pytest.raises((ValueError, TypeError)):
        solve([[0]], np.array([1.0 + 0.1j]), np.eye(1))


def test_integer_counts_preserves_caller_allocation() -> None:
    allocation = np.array([2.0, 1.0])
    previous = allocation.copy()
    integer_counts(allocation, [[1.0], [1.0]], 11)
    np.testing.assert_array_equal(allocation, previous)


def test_covariance_content_identity_is_frozen() -> None:
    raw = np.eye(2)
    provider = Covariance(raw)
    binding = provider.fingerprint()
    raw[0, 0] = 9
    block = provider.block([0])
    block[0, 0] = 4
    assert provider.block([0])[0, 0] == 1
    assert provider.fingerprint() == binding
    with pytest.raises(ValueError):
        provider.matrix.flags.writeable = True


def test_joint_outcome_score_requires_one_product_basis() -> None:
    with pytest.raises(ValueError):
        score_outcomes(["X", "Z"], [0, 1], [1.0, 1.0])
