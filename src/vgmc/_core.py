"""Compact precision assembly, numerical audits and allocation optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

from ._records import AuditResult, StageResult
from ._validation import FloatArray
from .config import SolverConfig
from .covariance import CovarianceProvider, quadratic

ExtendedArray = NDArray[np.longdouble]
Checkpoint = Callable[[AuditResult, int, int], None]


@dataclass(frozen=True)
class Problem:
    """Validated fixed-library data in its original term and group order."""

    groups: list[list[int]]
    c: FloatArray
    covariance: CovarianceProvider
    initial: FloatArray
    config: SolverConfig


class Prepared:
    """Factor covariance blocks and assemble occupied precision entries.

    The assembly has one row per occupied ``K(a)`` entry and one column per
    group. Its CSR row index scales with occupied entries rather than the
    square of the full Pauli count. Local inverse Cholesky features share the
    same factorization and supply the allocation gradient scores.
    """

    def __init__(self, problem: Problem) -> None:
        """Prepare positive-definite group blocks and sparse index maps."""
        start = perf_counter()
        self.problem = problem
        self.n = len(problem.c)
        self.sizes = np.array([len(g) for g in problem.groups], dtype=int)
        self.offsets = np.r_[0, np.cumsum(self.sizes)]
        self.occurrences = np.concatenate(problem.groups).astype(np.int64)
        self.owners = np.repeat(np.arange(len(problem.groups)), self.sizes)
        if np.any(np.bincount(self.occurrences, minlength=self.n) == 0):
            raise ValueError("Groups must cover every coefficient index")
        self.blocks: list[FloatArray] = []
        self.physical_blocks: list[FloatArray] = []
        self.factors: list[FloatArray] = []
        keys, owners, values = [], [], []
        frows, fcols, fvalues = [], [], []
        for k, group in enumerate(problem.groups):
            ix = np.asarray(group)
            physical = problem.covariance.block(group)
            block = physical + problem.config.ridge * np.eye(len(group))
            factor = np.asarray(np.linalg.cholesky(block), dtype=np.float64)
            inverse = cho_solve(
                (factor, True), np.eye(len(group)), check_finite=False
            )
            inverse = (inverse + inverse.T) * 0.5
            self.physical_blocks.append(physical)
            self.blocks.append(block)
            self.factors.append(factor)
            keys.append((ix[:, None] * self.n + ix[None, :]).ravel())
            owners.append(np.full(len(group) ** 2, k))
            values.append(inverse.ravel())
            inverse_factor = solve_triangular(
                factor, np.eye(len(group)), lower=True, check_finite=False
            )
            frows.append(
                np.repeat(
                    np.arange(self.offsets[k], self.offsets[k + 1]), len(group)
                )
            )
            fcols.append(np.tile(ix, len(group)))
            fvalues.append(inverse_factor.ravel())
        self.keys, compact_rows = np.unique(
            np.concatenate(keys), return_inverse=True
        )
        self.assembly = coo_matrix(
            (
                np.concatenate(values),
                (compact_rows, np.concatenate(owners)),
            ),
            shape=(len(self.keys), len(problem.groups)),
        ).tocsr()
        self.assembly_ld = self.assembly.astype(np.longdouble)
        self.indices = (self.keys % self.n).astype(np.int64)
        self.indptr = np.r_[
            0, np.cumsum(np.bincount(self.keys // self.n, minlength=self.n))
        ].astype(np.int64)
        self.sparse_features = coo_matrix(
            (
                np.concatenate(fvalues),
                (np.concatenate(frows), np.concatenate(fcols)),
            ),
            shape=(len(self.occurrences), self.n),
        ).tocsr()
        self.features_ld = self.sparse_features.astype(np.longdouble)
        self.preparation_seconds = perf_counter() - start

    def matrix_sparse(self, allocation: FloatArray) -> csr_matrix:
        """Assemble ``K(a)`` with the fixed occupied-entry CSR pattern."""
        values = self.assembly_ld @ allocation.astype(np.longdouble)
        return csr_matrix(
            (values, self.indices, self.indptr), shape=(self.n, self.n)
        )

    def matrix(self, allocation: FloatArray) -> ExtendedArray:
        """Materialize the precision matrix for a selected dense backend."""
        return np.asarray(self.matrix_sparse(allocation).toarray())

    def audit(
        self,
        allocation: FloatArray,
        mu: ExtendedArray | FloatArray | None = None,
    ) -> AuditResult:
        """Recover feasible coefficients and original-coordinate bounds.

        Args:
            allocation: Strictly positive numerical shot fractions.
            mu: Optional dual vector. Any finite vector defines a lower bound.
                Supplying the evaluator's current vector reuses its solve.

        Returns:
            Coefficients, physical variance, model bounds and relative gap.

        Raises:
            ArithmeticError: Recovered numeric values are inconsistent.
        """
        start = perf_counter()
        a = np.asarray(allocation, dtype=float)
        cfg = self.problem.config
        if a.shape != self.problem.initial.shape or np.any(a <= 0):
            raise ValueError(
                "Audit allocation must match the positive support"
            )
        if not np.isfinite(a).all() or abs(a.sum() - 1) > 1e-10:
            raise ValueError("Audit allocation must sum to one")
        if mu is None:
            matrix = self.matrix_sparse(a)
            factor = splu(matrix.astype(float).tocsc())
            value = factor.solve(self.problem.c).astype(np.longdouble)
            for _ in range(cfg.linear_refinements):
                residual = (
                    self.problem.c.astype(np.longdouble) - matrix @ value
                )
                value += factor.solve(np.asarray(residual, dtype=float))
            mu = value
        dual = np.asarray(mu, dtype=np.longdouble)
        if dual.shape != (self.n,) or not np.isfinite(dual).all():
            raise ValueError("Dual vector must match the finite term space")
        xs: list[ExtendedArray] = []
        scores: list[np.longdouble] = []
        for group, factor, block, weight in zip(
            self.problem.groups, self.factors, self.blocks, a, strict=True
        ):
            rhs = dual[group]
            local = cho_solve(
                (factor, True), rhs.astype(float), check_finite=False
            ).astype(np.longdouble)
            extended_block = block.astype(np.longdouble)
            for _ in range(cfg.coefficient_refinements):
                residual = rhs - extended_block @ local
                local += cho_solve(
                    (factor, True), residual.astype(float), check_finite=False
                ).astype(np.longdouble)
            xs.append(np.longdouble(weight) * local)
            scores.append(rhs @ local)
        x = np.concatenate(xs)
        total = np.zeros(self.n, dtype=np.longdouble)
        np.add.at(total, self.occurrences, x)
        correction = self.problem.c.astype(np.longdouble) - total
        # Repair each term in its most sampled occurrence to control variance
        # amplification from a near-zero allocation.
        occurrence_weights = a[self.owners]
        maximum = np.zeros(self.n)
        np.maximum.at(maximum, self.occurrences, occurrence_weights)
        eligible = occurrence_weights == maximum[self.occurrences]
        positions = np.full(self.n, len(x))
        np.minimum.at(
            positions, self.occurrences[eligible], np.flatnonzero(eligible)
        )
        x[positions] += correction
        recovered = np.split(x, self.offsets[1:-1])
        upper = np.longdouble(0)
        physical = 0.0
        for block, reference, coefficients, weight in zip(
            self.blocks, self.physical_blocks, recovered, a, strict=True
        ):
            upper += np.longdouble(
                coefficients @ block.astype(np.longdouble) @ coefficients
            ) / np.longdouble(weight)
            physical += (
                quadratic(coefficients.astype(float), reference) / weight
            )
        if not np.any(self.problem.c):
            lower = np.longdouble(0)
            gap = 0.0
        else:
            denominator = max(scores) * (1 + cfg.dual_safety_margin)
            if denominator <= 0:
                raise ArithmeticError("Dual score denominator is nonpositive")
            lower = np.longdouble(
                (self.problem.c.astype(np.longdouble) @ dual) ** 2
                / denominator
            )
            gap = float(upper / lower - 1) if lower > 0 else float("inf")
        total[:] = 0
        np.add.at(total, self.occurrences, x)
        if not np.isfinite(upper) or not np.isfinite(physical):
            raise ArithmeticError(
                "Estimator magnitude exceeds the numeric range"
            )
        return AuditResult(
            mu=dual.astype(float).tolist(),
            upper=float(upper),
            lower=float(lower),
            gap=gap,
            physical_variance=float(physical),
            conservation_error=float(np.max(abs(total - self.problem.c))),
            repair_max=float(np.max(abs(correction))),
            allocation=a.tolist(),
            x=[v.astype(float).tolist() for v in recovered],
            audit_seconds=perf_counter() - start,
            bound_kind="original_coordinate_primal_dual_check",
        )


class Evaluator(Protocol):
    """Allocation objective and score interface shared by CPU backends."""

    p: Prepared
    calls: int
    seconds: float
    last_mu: ExtendedArray | None

    def evaluate(self, a: FloatArray) -> tuple[float, FloatArray, float]:
        """Return model objective, group scores and approximate gap."""
        ...


class CPUEvaluator:
    """Evaluate a dense precision matrix with refined Cholesky solves."""

    def __init__(self, prepared: Prepared) -> None:
        """Bind reusable group factors and allocation-independent features."""
        self.p = prepared
        self.calls = 0
        self.seconds = 0.0
        self.last_mu: ExtendedArray | None = None

    def evaluate(self, a: FloatArray) -> tuple[float, FloatArray, float]:
        """Assemble the current precision and evaluate all group scores."""
        start = perf_counter()
        self.calls += 1
        matrix = self.p.matrix(a)
        factor = cho_factor(np.asarray(matrix, dtype=float), lower=True)
        mu = cho_solve(factor, self.p.problem.c).astype(np.longdouble)
        for _ in range(self.p.problem.config.linear_refinements):
            residual = self.p.problem.c.astype(np.longdouble) - matrix @ mu
            mu += cho_solve(factor, np.asarray(residual, dtype=float))
        features = self.p.features_ld @ mu
        scores = np.asarray(
            np.add.reduceat(features * features, self.p.offsets[:-1]),
            dtype=float,
        )
        phi = float(self.p.problem.c.astype(np.longdouble) @ mu)
        self.last_mu = mu
        self.seconds += perf_counter() - start
        if phi <= 0 or not np.isfinite(phi) or not np.isfinite(scores).all():
            raise ArithmeticError("Invalid allocation objective or score")
        return phi, scores, float(scores.max() / phi - 1)


class ReachedGap(Exception):
    """Stop L-BFGS-B after an explicitly stored candidate reaches the gap."""


class EvaluationLimit(Exception):
    """Stop an optimization stage at its objective evaluation budget."""


def optimize(
    evaluator: Evaluator,
    *,
    target_gap: float,
    max_updates: int,
    initial: FloatArray,
    checkpoint: Checkpoint | None = None,
    max_evaluations: int | None = None,
) -> StageResult:
    """Optimize square-coordinate allocations with explicit update budgets."""
    p = evaluator.p
    cfg = p.problem.config
    a = initial.copy()
    floor = cfg.allocation_floor
    a = np.maximum(a, floor)
    a /= a.sum()
    phi, _, approximate_gap = evaluator.evaluate(a)
    scale = phi
    last_a = a
    updates = 0

    def evaluate(theta: FloatArray) -> tuple[float, FloatArray]:
        """Map unconstrained square coordinates to the allocation simplex."""
        nonlocal last_a, approximate_gap
        if max_evaluations is not None and evaluator.calls >= max_evaluations:
            raise EvaluationLimit()
        weights = theta**2 + floor
        allocation = weights / weights.sum()
        value, scores, approximate_gap = evaluator.evaluate(allocation)
        last_a = allocation
        mean = float(allocation @ scores)
        gradient = 2 * theta * (mean - scores) / (weights.sum() * scale)
        if 0 <= approximate_gap <= target_gap:
            raise ReachedGap()
        return value / scale, np.asarray(gradient)

    def accepted(_theta: FloatArray) -> None:
        """Count accepted updates and emit a matching audited allocation."""
        nonlocal updates
        updates += 1
        if checkpoint and updates % cfg.checkpoint_interval == 0:
            checkpoint(
                p.audit(last_a, evaluator.last_mu), updates, evaluator.calls
            )

    try:
        result = minimize(
            evaluate,
            np.sqrt(a),
            method="L-BFGS-B",
            jac=True,
            callback=accepted,
            options=dict(
                maxiter=max_updates,
                maxfun=max_updates * 3,
                ftol=1e-14,
                gtol=1e-12,
                maxls=30,
                maxcor=20,
            ),
        )
        updates = int(result.nit)
        message = str(result.message)
    except ReachedGap:
        message = "approximate_gap_reached"
    except EvaluationLimit:
        message = "evaluation_limit"
    # The final stored dual belongs to last_a, including early trial exits.
    audit = p.audit(last_a, evaluator.last_mu)
    return StageResult(
        method="lbfgs",
        updates=updates,
        evaluations=evaluator.calls,
        approximate_gap=approximate_gap,
        optimizer_message=message,
        audit=audit,
    )
