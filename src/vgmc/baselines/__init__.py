"""Partition-based reproduction estimators for LF and SI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .._validation import index_group, real_array
from ..covariance import CovarianceProvider, quadratic


def partition_design(
    groups: Sequence[Sequence[int]],
    coefficients: ArrayLike,
    covariance: CovarianceProvider,
) -> dict[str, Any]:
    """Allocate partition shots in proportion to group standard deviations.

    Args:
        groups: Disjoint nonempty term-index groups covering every term.
        coefficients: Real energy coefficients, shape ``(n_terms,)``.
        covariance: Physical Pauli covariance in the same term order.

    Returns:
        Dictionary with ``groups``, local energy coefficients ``x`` and
        normalized fractions ``a``. Zero-variance blocks use a numerical
        allocation floor of 1e-15.

    Raises:
        ValueError: Term axes, partition coverage or covariance are invalid.

    Examples:
        >>> from vgmc import Covariance
        >>> d = partition_design([[0], [1]], [1.0, 2.0], Covariance(np.eye(2)))
        >>> d["a"]
        [0.3333333333333333, 0.6666666666666666]
    """
    c = real_array(coefficients, "coefficients", 1)
    library = [index_group(g, len(c)) for g in groups]
    if covariance.n_terms != len(c) or sorted(
        i for g in library for i in g
    ) != list(range(len(c))):
        raise ValueError("Groups must partition the covariance term axis")
    xs = [c[group] for group in library]
    roots = np.array(
        [
            np.sqrt(quadratic(x, covariance.block(g)))
            for g, x in zip(library, xs, strict=True)
        ]
    )
    if not len(roots):
        allocation = np.empty(0)
    else:
        allocation = (
            roots / roots.sum()
            if roots.sum()
            else np.full(len(roots), 1 / len(roots))
        )
        allocation = np.maximum(allocation, 1e-15)
        allocation /= allocation.sum()
    return dict(
        groups=library, x=[x.tolist() for x in xs], a=allocation.tolist()
    )
