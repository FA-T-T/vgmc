"""Fixed-library conic reference through CVXPY and Clarabel."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_triangular
from scipy.sparse import coo_matrix

from .._validation import index_group, positive, real_array
from ..covariance import CovarianceProvider, as_covariance, quadratic


def solve(
    groups: Sequence[Sequence[int]],
    c: ArrayLike,
    covariance: ArrayLike | CovarianceProvider,
    *,
    ridge: float = 1e-8,
) -> dict[str, Any]:
    """Solve the fixed-library norm program and audit its feasible bounds.

    Args:
        groups: Nonempty local term-index sets covering all target terms.
        c: Real target energy coefficients, shape ``(n_terms,)``.
        covariance: Physical covariance matrix or block provider.
        ridge: Nonnegative diagonal addition to each design block.

    Returns:
        Group coefficients ``x``, simplex fractions ``a``, model upper and
        lower bounds in squared energy units, relative gap and ``verified``.
        Verification requires coefficient recovery and a gap at most 1e-4.

    Raises:
        ValueError: Index, coverage or numerical input checks fail.
        ArithmeticError: The conic solver supplies no primal or dual point.
        numpy.linalg.LinAlgError: A design block is not positive definite.

    Examples:
        >>> result = solve([[0]], [2.0], [[1.0]])
        >>> result["verified"], result["a"]
        (True, [1.0])
    """
    import cvxpy as cp

    c = real_array(c, "c", 1)
    ridge = positive(ridge, "ridge", zero=True)
    library = [index_group(g, len(c)) for g in groups]
    provider = as_covariance(covariance, len(c))
    if set(i for g in library for i in g) != set(range(len(c))):
        raise ValueError("Groups must cover every target coefficient")
    if not np.any(c):
        return dict(
            groups=library,
            x=[np.zeros(len(g)).tolist() for g in library],
            a=(
                np.full(len(library), 1 / len(library)).tolist()
                if library
                else []
            ),
            upper_bound=0.0,
            lower_bound=0.0,
            gap=0.0,
            solver_status="zero_target",
            verified=True,
        )
    blocks = [provider.block(group, ridge) for group in library]
    factors = [np.linalg.cholesky(block) for block in blocks]
    variables = [cp.Variable(len(group)) for group in library]
    selectors = [
        coo_matrix(
            (np.ones(len(g)), (g, np.arange(len(g)))), shape=(len(c), len(g))
        ).tocsr()
        for g in library
    ]
    equality = (
        sum(s @ x for s, x in zip(selectors, variables, strict=True)) == c
    )
    problem = cp.Problem(
        cp.Minimize(
            sum(
                cp.norm(f.T @ x, 2)
                for f, x in zip(factors, variables, strict=True)
            )
        ),
        [equality],
    )
    problem.solve(
        solver="CLARABEL",
        max_iter=500,
        tol_gap_abs=1e-9,
        tol_gap_rel=1e-9,
        tol_feas=1e-9,
        max_threads=1,
    )
    if any(x.value is None for x in variables) or equality.dual_value is None:
        raise ArithmeticError(f"SOCP status: {problem.status}")
    xs = [
        real_array(cast(ArrayLike, x.value), "conic coefficient block", 1)
        for x in variables
    ]
    total = np.zeros_like(c)
    for group, x in zip(library, xs, strict=True):
        np.add.at(total, group, x)
    # A feasible primal point supplies the audited upper bound.
    for i, correction in enumerate(c - total):
        for group, x in zip(library, xs, strict=True):
            if i in group:
                x[group.index(i)] += correction
                break
    roots = np.array(
        [
            float(np.linalg.norm(f.T @ x))
            for f, x in zip(factors, xs, strict=True)
        ]
    )
    allocation = np.maximum(roots / roots.sum(), 1e-15)
    allocation /= allocation.sum()
    dual = -real_array(equality.dual_value, "conic dual", 1)
    norm = max(
        float(np.linalg.norm(solve_triangular(f, dual[g], lower=True)))
        for f, g in zip(factors, library, strict=True)
    )
    if norm == 0:
        raise ArithmeticError("SOCP dual point has zero norm")
    dual /= norm * (1 + 1e-8)
    upper = sum(
        quadratic(x, block) / a
        for x, block, a in zip(xs, blocks, allocation, strict=True)
    )
    lower = max(0.0, float(c @ dual)) ** 2
    gap = upper / lower - 1 if lower else float("inf")
    recovered = np.zeros_like(c)
    for g, x in zip(library, xs, strict=True):
        np.add.at(recovered, g, x)
    error = float(np.max(abs(recovered - c)))
    verified = bool(
        problem.status in ("optimal", "optimal_inaccurate")
        and 0 <= gap <= 1e-4
        and error <= 1e-9 * max(1.0, float(np.max(abs(c))))
    )
    return dict(
        groups=library,
        x=[x.tolist() for x in xs],
        a=allocation.tolist(),
        upper_bound=upper,
        lower_bound=lower,
        gap=gap,
        solver_status=problem.status,
        verified=verified,
        conservation_error=error,
    )
