"""Residual checks and lower bounds on a finite candidate pool."""

from collections.abc import Mapping
from typing import Any
from numpy.typing import ArrayLike
from ._validation import integer, real_array

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

LD = np.longdouble


def group_score(
    block: ArrayLike, vector: ArrayLike, settings: Mapping[str, Any]
) -> dict[str, float]:
    """Calculate a covariance score and its residual allowance."""
    a = real_array(block, "block", 2).astype(LD)
    b = real_array(vector, "vector", 1).astype(LD)
    if (
        a.shape != (len(b), len(b))
        or not np.isfinite(a).all()
        or (not np.isfinite(b).all())
    ):
        raise ValueError("Nonfinite or incompatible score inputs")
    if not np.array_equal(a, a.T):
        raise ValueError("Asymmetric frozen covariance")
    n = len(b)
    scale = max(float(np.max(np.sum(abs(a), axis=1))), np.finfo(float).tiny)
    eigen = float(
        eigh(
            np.asarray(a, dtype=float),
            eigvals_only=True,
            subset_by_index=[0, 0],
            check_finite=False,
        )[0]
    )
    eig_error = (
        settings["eigen_error_factor"] * n * np.finfo(float).eps * scale
    )
    lower_eigen = eigen - eig_error
    if lower_eigen <= 0:
        raise ArithmeticError("Positive definiteness unresolved")
    factor = cho_factor(
        np.asarray(a, dtype=float), lower=True, check_finite=False
    )
    z = cho_solve(
        factor, np.asarray(b, dtype=float), check_finite=False
    ).astype(LD)
    for _ in range(settings["linear_refinements"]):
        r = b - a @ z
        trial = z + cho_solve(
            factor, np.asarray(r, dtype=float), check_finite=False
        ).astype(LD)
        if np.max(abs(b - a @ trial)) >= np.max(abs(r)):
            break
        z = trial
    r = b - a @ z
    qlower = 2 * (b @ z) - z @ a @ z
    rerr = (
        settings["roundoff_factor"]
        * n
        * np.finfo(LD).eps
        * (abs(b) + abs(a) @ abs(z))
    )
    rnorm = np.sqrt(np.sum((abs(r) + rerr) ** 2))
    qscale = 2 * np.sum(abs(b * z)) + np.sum(abs(z) * (abs(a) @ abs(z)))
    arithmetic = settings["roundoff_factor"] * n * np.finfo(LD).eps * qscale
    upper = qlower + rnorm * rnorm / LD(lower_eigen) + arithmetic
    if not np.isfinite(upper) or upper < 0:
        raise ArithmeticError("Invalid group score bound")
    return dict(
        score=float(qlower),
        score_upper=float(upper),
        residual_norm=float(rnorm),
        lower_eigenvalue=lower_eigen,
        condition_estimate=scale / lower_eigen,
        residual_allowance=float(upper - qlower),
    )


def mp_score(block: ArrayLike, vector: ArrayLike, digits: int) -> float:
    """Solve the frozen covariance block at the specified decimal precision."""
    import mpmath as mp

    integer(digits, "digits", 20)
    block = real_array(block, "block", 2)
    vector = real_array(vector, "vector", 1)
    if block.shape != (len(vector), len(vector)):
        raise ValueError("Block and vector axes differ")
    context = mp.mp.clone()
    context.dps = digits
    a = context.matrix([[context.mpf(float(x)) for x in row] for row in block])
    b = context.matrix([context.mpf(float(x)) for x in vector])
    z = context.cholesky_solve(a, b)
    return float((b.T * z)[0])


def pool_decision(
    upper: float,
    lower: float,
    initial: float,
    tolerance: float,
    arithmetic_uncertainty: float,
) -> dict[str, Any]:
    """Compare the remaining finite-pool model gain with the configured
    tolerance.
    """
    values = [upper, lower, initial, tolerance, arithmetic_uncertainty]
    if (
        not all((np.isfinite(x) for x in values))
        or min(upper, lower, initial, tolerance) <= 0
    ):
        raise ValueError("Invalid pool decision input")
    if arithmetic_uncertainty < 0 or lower > upper + arithmetic_uncertainty:
        raise ArithmeticError("Inconsistent primal/dual evidence")
    delta = max(0.0, (upper - lower) / initial)
    passed = (
        delta <= tolerance
        and arithmetic_uncertainty / initial <= tolerance / 10
    )
    return dict(
        status=(
            "candidate_pool_tolerance_reached"
            if passed
            else "numerical_resolution_unresolved"
        ),
        verified=bool(passed),
        remaining_model_gain_over_initial=delta,
        tolerance=tolerance,
        arithmetic_uncertainty_over_initial=arithmetic_uncertainty / initial,
    )
