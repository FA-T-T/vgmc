"""Schur-guided QWC growth and complete covariance scores."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from collections.abc import Sequence
from typing import Any
from numpy.typing import ArrayLike
from ._validation import (
    BoolArray,
    FloatArray,
    index_group,
    integer,
    real_array,
)

from .covariance import Covariance, CovarianceProvider
from .grouping import basis_for, visible


def score(
    group: list[int],
    mu: ArrayLike,
    covariance: CovarianceProvider,
    ridge: float,
) -> float:
    """Evaluate the embedded precision score of one group.

    Args:
        group: Distinct term indices of one positive-definite design block.
        mu: Real response vector of shape (n_terms,), in energy units.
        covariance: Physical covariance provider in Hamiltonian order.
        ridge: Nonnegative diagonal design regularization.

    Returns:
        Scalar mu_group.T @ inverse(Sigma_group + ridge * I) @ mu_group,
        in squared energy units.

    Raises:
        ValueError: Response, indices or covariance entries are invalid.
        numpy.linalg.LinAlgError: The design block is not positive definite.

    Examples:
        >>> score([0, 1], np.array([1., 2.]), Covariance(np.eye(2)), 0.0)
        5.0
    """
    group = index_group(group, covariance.n_terms)
    response = real_array(mu, "mu", 1)
    if len(response) != covariance.n_terms:
        raise ValueError("Response and covariance dimensions differ")
    return float(
        response[group]
        @ cho_solve(
            cho_factor(covariance.block(group, ridge)), response[group]
        )
    )


def schur(
    group: list[int],
    term: int,
    mu: ArrayLike,
    covariance: CovarianceProvider,
    ridge: float,
) -> float:
    """Evaluate the score increment from one group extension.

    Args:
        group: Current distinct term indices.
        term: Extension index outside the group.
        mu: Real response vector, shape (n_terms,).
        covariance: Physical covariance provider.
        ridge: Nonnegative diagonal design regularization.

    Returns:
        Squared residual response divided by the best linear prediction
        residual variance, in squared energy units.

    Raises:
        ValueError: Indices or numeric inputs are invalid.
        ArithmeticError: The Schur residual variance is nonpositive.

    Examples:
        >>> schur([0], 1, np.array([1., 2.]), Covariance(np.eye(2)), 0.0)
        4.0
    """
    return float(schur_increments(group, [term], mu, covariance, ridge)[0])


def schur_increments(
    group: Sequence[int],
    options: Sequence[int],
    mu: ArrayLike,
    covariance: CovarianceProvider,
    ridge: float,
) -> FloatArray:
    """Evaluate several Schur score increments with one group factorization.

    Args:
        group: Current QWC term indices.
        options: Distinct extension indices outside the current group.
        mu: Finite target-response vector of shape ``(n_terms,)``.
        covariance: Covariance provider in the same term order.
        ridge: Nonnegative diagonal design regularization.

    Returns:
        Scores of shape ``(len(options),)`` in option order. The denominator
        is the best linear prediction residual variance.

    Raises:
        ValueError: Term indices, response or block entries are invalid.
        ArithmeticError: A computed Schur residual variance is nonpositive.
    """
    group = index_group(group, covariance.n_terms, empty=True)
    options = index_group(options, covariance.n_terms, empty=True)
    if set(group) & set(options):
        raise ValueError("Extension indices must be outside the group")
    response = real_array(mu, "mu", 1)
    if len(response) != covariance.n_terms:
        raise ValueError("Response and covariance dimensions differ")
    if not options:
        return np.empty(0)
    block = covariance.block(group, ridge)
    if isinstance(covariance, Covariance):
        cross = covariance.matrix[np.ix_(group, options)]
        diagonal = covariance.matrix[options, options] + ridge
    else:
        trials = [covariance.block([*group, term], ridge) for term in options]
        cross = np.stack([trial[:-1, -1] for trial in trials], axis=1)
        diagonal = np.array([trial[-1, -1] for trial in trials])
    if not np.isfinite(cross).all() or not np.isfinite(diagonal).all():
        raise ValueError("Growth covariance entries are unavailable")
    if group:
        factor = np.linalg.cholesky(block)
        whitened_cross = solve_triangular(factor, cross, lower=True)
        whitened_response = solve_triangular(
            factor, response[group], lower=True
        )
        residual = response[options] - whitened_response @ whitened_cross
        denominator = diagonal - np.sum(whitened_cross**2, axis=0)
    else:
        residual = response[options]
        denominator = diagonal
    if np.any(denominator <= 0):
        raise ArithmeticError("Nonpositive Schur residual variance")
    return np.asarray(residual**2 / denominator)


def seed_order(mu: ArrayLike, c: ArrayLike) -> list[int]:
    """Interleave response utility with the original Hamiltonian order.

    Args:
        mu: Finite real response vector of shape (n_terms,).
        c: Finite real Hamiltonian coefficients with the same shape.

    Returns:
        Each term index once. Decreasing absolute mu[i] * c[i] determines
        utility order, with input index breaking ties.

    Examples:
        >>> seed_order(np.array([1., 1.]), np.array([1., 2.]))
        [1, 0]
    """
    mu = real_array(mu, "mu", 1)
    c = real_array(c, "c", 1)
    if mu.shape != c.shape:
        raise ValueError("Response and coefficient axes differ")
    order = sorted(range(len(c)), key=lambda i: (-abs(mu[i] * c[i]), i))
    result, seen = [], set()
    for left, right in zip(order, range(len(c)), strict=True):
        for i in (left, right):
            if i not in seen:
                result.append(i)
                seen.add(i)
    return result


def grow(
    seed: int,
    words: list[str],
    growth_mu: ArrayLike,
    scoring_mu: ArrayLike,
    covariance: CovarianceProvider,
    compatibility: BoolArray,
    ridge: float,
    mode: str = "full",
) -> dict[str, Any]:
    """Grow a QWC group from one seed and complete its visible support.

    Args:
        seed: Hamiltonian term index.
        words: Valid equal-width Pauli strings.
        growth_mu: Response used to select extensions, shape (n_terms,).
        scoring_mu: Current response used for final admission scoring.
        covariance: Physical covariance block provider.
        compatibility: Boolean pairwise QWC matrix, shape (n_terms, n_terms).
        ridge: Nonnegative design regularization.
        mode: Full Schur increments or diagonal utility, full or diagonal.

    Returns:
        Dictionary with seed, sorted group support, complete basis and full
        covariance score. Index order breaks equal extension scores.

    Raises:
        ValueError: Mode, indices, response or covariance values are invalid.
        ArithmeticError: A Schur residual variance is nonpositive.
    """
    seed = integer(seed, "seed")
    if seed >= len(words) or covariance.n_terms != len(words):
        raise ValueError("Seed or covariance term axis differs")
    growth_mu = real_array(growth_mu, "growth_mu", 1)
    scoring_mu = real_array(scoring_mu, "scoring_mu", 1)
    if growth_mu.shape != (len(words),) or scoring_mu.shape != growth_mu.shape:
        raise ValueError("Growth response axes differ")
    if (
        compatibility.shape != (len(words), len(words))
        or compatibility.dtype.kind != "b"
    ):
        raise ValueError("Compatibility must be a boolean term matrix")
    group = [seed]
    allowed = compatibility[seed].copy()
    allowed[seed] = False
    while np.any(allowed):
        options = np.flatnonzero(allowed).tolist()
        if mode == "full":
            increments = schur_increments(
                group, options, growth_mu, covariance, ridge
            )
            values = list(zip(increments.tolist(), options, strict=True))
        elif mode == "diagonal":
            values = [
                (
                    float(
                        growth_mu[i] ** 2 / covariance.block([i], ridge)[0, 0]
                    ),
                    i,
                )
                for i in options
            ]
        else:
            raise ValueError("Growth mode must be full or diagonal")
        chosen = min(values, key=lambda item: (-item[0], item[1]))[1]
        group.append(chosen)
        allowed &= compatibility[chosen]
        allowed[chosen] = False
    basis = basis_for(words, group)
    support = sorted(visible(words, basis))
    return dict(
        seed=int(seed),
        group=support,
        basis=basis,
        score=score(support, scoring_mu, covariance, ridge),
    )
