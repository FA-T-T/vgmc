"""Covariance providers and variance evaluation for Pauli readouts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, index_group, integer, positive, real_array


@runtime_checkable
class CovarianceProvider(Protocol):
    """Provide dimensionless Pauli covariance in Hamiltonian index order."""

    @property
    def n_terms(self) -> int:
        """Number of Pauli terms in the provider's index space."""
        ...

    def block(self, group: Sequence[int], ridge: float = 0.0) -> FloatArray:
        """Return a finite symmetric principal block plus ``ridge * I``."""
        ...

    def fingerprint(self) -> str:
        """Return the SHA-256 identity of the effective covariance input."""
        ...


class Covariance:
    """Store an immutable dense covariance matrix.

    Args:
        matrix: Symmetric real array of shape ``(n_terms, n_terms)``. Pauli
            readouts are dimensionless. Paired NaNs identify unavailable
            entries. Each requested block must be finite.

    Attributes:
        matrix: Read-only float64 copy of the covariance matrix.
        n_terms: Number of Pauli terms.

    Examples:
        >>> cov = Covariance([[1, 0], [0, 1]])
        >>> cov.block([0], ridge=0.25).tolist()
        [[1.25]]
    """

    def __init__(self, matrix: ArrayLike) -> None:
        raw = np.asarray(matrix)
        if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
            raise ValueError("Covariance must be a square matrix")
        if raw.dtype.kind not in "iufc":
            raise ValueError("Covariance must have real numeric entries")
        if np.iscomplexobj(raw):
            if np.any(raw.imag != 0):
                raise ValueError("Covariance must be real")
            raw = raw.real
        value = np.array(raw, dtype=np.float64, order="C", copy=True)
        if np.any(np.isinf(value)) or not np.array_equal(
            value, value.T, equal_nan=True
        ):
            raise ValueError("Covariance must be symmetric")
        value.flags.writeable = False
        # Immutable byte storage preserves the cached covariance identity.
        self._matrix = np.frombuffer(
            value.tobytes(), dtype=np.float64
        ).reshape(value.shape)
        self._fingerprint = hashlib.sha256(value.tobytes()).hexdigest()

    @property
    def matrix(self) -> FloatArray:
        """Immutable float64 covariance matrix in Hamiltonian term order."""
        return self._matrix

    @property
    def n_terms(self) -> int:
        """Number of rows in the frozen covariance input."""
        return len(self._matrix)

    def block(self, group: Sequence[int], ridge: float = 0.0) -> FloatArray:
        """Read a covariance block and add diagonal design regularization.

        Args:
            group: Distinct Pauli indices. Their order defines both axes.
            ridge: Nonnegative regularization in covariance units.

        Returns:
            New float64 array of shape ``(len(group), len(group))``.

        Raises:
            ValueError: Indices, ridge or requested entries are invalid.
        """
        indices = index_group(group, self.n_terms, empty=True)
        ridge = positive(ridge, "ridge", zero=True)
        result = self.matrix[np.ix_(indices, indices)].copy()
        if not np.all(np.isfinite(result)):
            raise ValueError("Requested covariance entries are unavailable")
        if indices:
            result.flat[:: len(indices) + 1] += ridge
        return result

    def fingerprint(self) -> str:
        """Return the immutable matrix content identity."""
        return self._fingerprint


class CallableCovariance:
    """Adapt a block callback to the covariance interface.

    Args:
        n_terms: Number of Pauli terms.
        callback: Function accepting a tuple of term indices and returning
            their unregularized symmetric covariance block.
        identity: Stable identity of the callback's scientific inputs and
            implementation. It enters checkpoint compatibility checks.

    Examples:
        >>> provider = CallableCovariance(2, lambda g: np.eye(len(g)), "v1")
        >>> provider.block([1], 0.5).tolist()
        [[1.5]]
    """

    def __init__(
        self,
        n_terms: int,
        callback: Callable[[tuple[int, ...]], ArrayLike],
        identity: str,
    ) -> None:
        self.n_terms = integer(n_terms, "n_terms")
        if (
            not callable(callback)
            or not isinstance(identity, str)
            or not identity
        ):
            raise ValueError("A callback and scientific identity are required")
        self._callback = callback
        self._identity = hashlib.sha256(
            f"{n_terms}:{identity}".encode()
        ).hexdigest()

    def block(self, group: Sequence[int], ridge: float = 0.0) -> FloatArray:
        """Validate and regularize a callback block in the requested order."""
        indices = index_group(group, self.n_terms, empty=True)
        ridge = positive(ridge, "ridge", zero=True)
        value = real_array(self._callback(tuple(indices)), "block", 2)
        if value.shape != (len(indices), len(indices)) or not np.array_equal(
            value, value.T
        ):
            raise ValueError("Callback covariance axes must be symmetric")
        if indices:
            value.flat[:: len(indices) + 1] += ridge
        return value

    def fingerprint(self) -> str:
        """Return the callback input and implementation identity."""
        return self._identity


def as_covariance(
    value: ArrayLike | CovarianceProvider, n_terms: int
) -> CovarianceProvider:
    """Resolve a provider and check its Hamiltonian index-space size."""
    provider = (
        value if isinstance(value, CovarianceProvider) else Covariance(value)
    )
    if provider.n_terms != n_terms:
        raise ValueError("Covariance and coefficient dimensions differ")
    return provider


def quadratic(vector: ArrayLike, matrix: ArrayLike) -> float:
    """Evaluate a finite covariance quadratic form.

    Args:
        vector: Real coefficients of shape ``(k,)``, in energy units.
        matrix: Symmetric covariance of shape ``(k, k)``.

    Returns:
        Variance in squared energy units. Negative roundoff within
        a ``64 * k * eps`` Pauli-covariance roundoff allowance is zero. The
        allowance scales with the squared coefficient one-norm and covariance
        magnitude, so it retains squared energy units.

    Raises:
        ValueError: Inputs have invalid shapes or nonfinite values.
        ArithmeticError: The form is negative beyond the roundoff bound.

    Examples:
        >>> quadratic([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])
        5.0
    """
    x = real_array(vector, "vector", 1)
    block = real_array(matrix, "matrix", 2)
    if block.shape != (len(x), len(x)) or not np.array_equal(block, block.T):
        raise ValueError("Quadratic-form axes must match a symmetric matrix")
    value = float(x @ block @ x)
    scale = float(abs(x) @ abs(block) @ abs(x))
    if not np.isfinite(value) or not np.isfinite(scale):
        raise ValueError("Quadratic-form magnitude exceeds the numeric range")
    entry_scale = max(1.0, float(np.max(abs(block)))) if block.size else 1.0
    arithmetic_scale = max(scale, float(abs(x).sum()) ** 2 * entry_scale)
    allowance = 64 * max(1, len(x)) * np.finfo(float).eps * arithmetic_scale
    if value < -allowance:
        raise ArithmeticError("Negative covariance quadratic form")
    return max(0.0, value)


def variance(
    groups: Sequence[Sequence[int]],
    xs: Sequence[ArrayLike],
    allocation: ArrayLike,
    covariance: CovarianceProvider,
) -> float:
    """Evaluate ``sum(x[g].T @ Sigma[g] @ x[g] / a[g])``.

    Args:
        groups: Pauli term indices for each measurement group.
        xs: One vector of shape ``(len(group),)`` per group, in energy units.
        allocation: Nonnegative shot fractions of shape ``(n_groups,)``.
            Their sum is one. A zero fraction requires a zero coefficient
            block.
        covariance: Reference covariance provider, queried with zero ridge.

    Returns:
        Variance coefficient in squared energy units. The continuous
        estimator variance at ``N`` shots is this value divided by ``N``.

    Raises:
        ValueError: A shape, numeric value or allocation is invalid.

    Examples:
        >>> variance([[0], [1]], [[1.0], [1.0]], [0.5, 0.5],
        ...          Covariance(np.eye(2)))
        4.0
    """
    a = real_array(allocation, "allocation", 1)
    if len(a) != len(groups) or len(xs) != len(groups):
        raise ValueError("Groups, coefficients and allocation must align")
    if np.any(a < 0) or (
        len(a) and not np.isclose(a.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Allocation must lie on the nonnegative simplex")
    value = 0.0
    for group, coefficients, weight in zip(groups, xs, a, strict=True):
        g = index_group(group, covariance.n_terms)
        x = real_array(coefficients, "coefficient block", 1)
        if len(x) != len(g) or (weight == 0 and np.any(x)):
            raise ValueError("Coefficient block and sampled support differ")
        if weight > 0:
            value += quadratic(x, covariance.block(g)) / weight
    if not np.isfinite(value):
        raise ValueError("Variance exceeds the numeric range")
    return float(value)
