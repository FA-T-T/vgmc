"""Fixed-library coefficient and shot-allocation optimization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from copy import deepcopy
import logging
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from threadpoolctl import threadpool_limits

from ._boundary import polish
from ._core import CPUEvaluator, Evaluator, Prepared, Problem, optimize
from ._linear import SparseEvaluator
from ._records import AuditResult, SolverResult
from ._validation import index_group, real_array
from .config import SolverConfig
from .covariance import CovarianceProvider, as_covariance

LOGGER = logging.getLogger(__name__)


def solve(
    groups: Sequence[Sequence[int]],
    c: ArrayLike,
    covariance: ArrayLike | CovarianceProvider,
    *,
    config: SolverConfig | None = None,
    ridge: float | None = None,
    relative_gap: float | None = None,
    initial_a: ArrayLike | None = None,
    threads: int | None = None,
    max_updates: int | None = None,
    checkpoint: Callable[[SolverResult], None] | None = None,
) -> SolverResult:
    """Optimize a fixed library and recover a feasible linear estimator.

    Args:
        groups: Distinct term indices within each group. The groups jointly
            cover all ``len(c)`` indices. Their original order is retained.
        c: Real float64-compatible coefficients of shape ``(n_terms,)`` in
            energy units. Identity contributions are handled by the caller.
        covariance: Real array of shape ``(n_terms, n_terms)`` or a block
            provider. Each selected covariance block plus the ridge is
            positive definite.
        config: Immutable numerical options. Individual keyword options
            override its corresponding fields.
        ridge: Optional diagonal design regularization.
        relative_gap: Optional numerical primal/dual gap target.
        initial_a: Nonnegative initial weights of shape ``(n_groups,)``.
            Weights are copied, floored and normalized internally.
        threads: Optional BLAS thread count.
        max_updates: Optional accepted update limit.
        checkpoint: Callback receiving audited progress. Exceptions from
            this callback propagate to the caller.

    Returns:
        A ``SolverResult`` with group coefficient blocks ``x``, shot fractions
        ``a``, dual vector ``mu``, bounds, physical variance and status.
        ``converged`` means the numerical gap and coefficient-recovery checks
        both passed. Budget and stagnation statuses retain the last audit.

    Raises:
        ValueError: Inputs, coverage or options are invalid.
        numpy.linalg.LinAlgError: A selected covariance block is not positive
            definite under the chosen ridge.

    Examples:
        >>> result = solve([[0], [1]], [1.0, 2.0], np.eye(2),
        ...                config=SolverConfig(ridge=0.0),
        ...                initial_a=[1.0, 2.0])
        >>> result["status"]
        'converged'
        >>> round(result["physical_V"], 6)
        9.0
        >>> np.allclose(result["a"], [1 / 3, 2 / 3])
        True
    """
    cfg = config or SolverConfig()
    overrides: dict[str, Any] = {
        k: v
        for k, v in dict(
            ridge=ridge,
            relative_gap=relative_gap,
            threads=threads,
            max_updates=max_updates,
        ).items()
        if v is not None
    }
    if overrides:
        cfg = replace(cfg, **overrides)
    c = real_array(c, "c", 1)
    provider = as_covariance(covariance, len(c))
    library = [index_group(group, len(c)) for group in groups]
    if bool(len(c)) != bool(library):
        raise ValueError("A nonempty coefficient space requires groups")
    if len(library) * cfg.allocation_floor >= 1:
        raise ValueError("Allocation floor exceeds the group simplex")
    a = (
        np.ones(len(library))
        if initial_a is None
        else real_array(initial_a, "initial_a", 1)
    )
    if (
        a.shape != (len(library),)
        or np.any(a < 0)
        or (len(a) and a.sum() <= 0)
    ):
        raise ValueError(
            "Initial allocation must match the nonnegative groups"
        )
    a = np.maximum(a, cfg.allocation_floor)
    if len(a):
        a /= a.sum()
    start = perf_counter()
    history: list[dict[str, Any]] = []
    updates = evaluations = 0

    def pack(audit: AuditResult, status: str = "running") -> SolverResult:
        """Serialize one audited allocation at the public result boundary."""
        return SolverResult(
            groups=library,
            x=audit["x"],
            a=audit["allocation"],
            mu=audit["mu"],
            upper_bound=audit["upper"],
            lower_bound=audit["lower"],
            gap=audit["gap"],
            physical_V=audit["physical_variance"],
            conservation_error=audit["conservation_error"],
            status=status,
            iterations=updates,
            evaluations=evaluations,
            ridge=cfg.ridge,
            relative_gap=cfg.relative_gap,
            solve_seconds=perf_counter() - start,
            settings=cfg.to_dict(),
            numerical_history=history.copy(),
        )

    callback_error: ArithmeticError | None = None

    def publish(record: SolverResult) -> None:
        """Isolate observer state and propagate callback storage failures."""
        nonlocal callback_error
        if checkpoint:
            try:
                checkpoint(deepcopy(record))
            except ArithmeticError as error:
                callback_error = error
                raise

    if not len(c):
        empty = AuditResult(
            mu=[],
            upper=0.0,
            lower=0.0,
            gap=0.0,
            physical_variance=0.0,
            conservation_error=0.0,
            repair_max=0.0,
            allocation=[],
            x=[],
            audit_seconds=0.0,
            bound_kind="exact_zero_target",
        )
        return pack(empty, "converged")
    with threadpool_limits(limits=cfg.threads, user_api="blas"):
        prepared = Prepared(Problem(library, c, provider, a, cfg))
        density = len(prepared.keys) / len(c) ** 2
        evaluator: Evaluator = (
            CPUEvaluator(prepared)
            if len(c) < 256 or density > 0.25
            else SparseEvaluator(prepared)
        )
        audit = prepared.audit(a)
        method = "lbfgs"
        stalled = 0
        last_gap = audit["gap"]

        def emit(
            value: AuditResult, local_updates: int, local_calls: int
        ) -> None:
            """Emit correctly counted progress within an optimizer stage."""
            if checkpoint:
                record = pack(value)
                record["iterations"] = updates + local_updates
                record["evaluations"] = evaluations + local_calls
                publish(record)

        while True:
            scale = max(1.0, float(np.max(abs(c))))
            if (
                0 <= audit["gap"] <= cfg.relative_gap
                and audit["conservation_error"]
                <= cfg.conservation_tolerance * scale
            ):
                status = "converged"
                break
            if cfg.max_updates is not None and updates >= cfg.max_updates:
                status = "update_limit"
                break
            if (
                cfg.max_evaluations is not None
                and evaluations >= cfg.max_evaluations
            ):
                status = "evaluation_limit"
                break
            budget = (
                1000
                if cfg.max_updates is None
                else min(1000, cfg.max_updates - updates)
            )
            evaluation_budget = (
                None
                if cfg.max_evaluations is None
                else cfg.max_evaluations - evaluations
            )
            evaluator.calls = 0
            try:
                if method == "boundary":
                    stage = polish(
                        evaluator,
                        np.asarray(audit["allocation"]),
                        cfg.relative_gap * 0.8,
                        min(budget, 32),
                        emit,
                        evaluation_budget,
                    )
                else:
                    stage = optimize(
                        evaluator,
                        target_gap=cfg.relative_gap * 0.8,
                        max_updates=budget,
                        initial=np.asarray(audit["allocation"]),
                        checkpoint=emit,
                        max_evaluations=evaluation_budget,
                    )
            except ArithmeticError as error:
                if error is callback_error:
                    raise
                result = pack(audit, "numerical_failure")
                result["failure"] = str(error)
                result["evaluations"] += evaluator.calls
                return result
            updates += stage["updates"]
            evaluations += stage["evaluations"]
            audit = stage["audit"]
            history.append(
                dict(
                    method=stage["method"],
                    updates=stage["updates"],
                    approximate_gap=stage["approximate_gap"],
                    optimizer_message=stage["optimizer_message"],
                )
            )
            if checkpoint:
                publish(pack(audit))
            improved = np.isfinite(audit["gap"]) and audit[
                "gap"
            ] < last_gap * (1 - 1e-8)
            stalled = 0 if improved else stalled + 1
            last_gap = audit["gap"]
            if stalled >= cfg.stall_checks and audit["gap"] > cfg.relative_gap:
                status = "numerical_stagnation"
                break
            if stage["optimizer_message"] != "approximate_gap_reached":
                method = "boundary"
    result = pack(audit, status)
    result["preparation"] = dict(
        terms=len(c),
        groups=len(library),
        occupied_entries=len(prepared.keys),
        assembly_nnz=prepared.assembly.nnz,
        assembly_rows=prepared.assembly.shape[0],
        seconds=prepared.preparation_seconds,
        backend=(
            "dense"
            if isinstance(evaluator, CPUEvaluator)
            and not isinstance(evaluator, SparseEvaluator)
            else "sparse"
        ),
    )
    LOGGER.info(
        "Allocation solve: %s, groups=%d, gap=%.3g",
        status,
        len(library),
        result["gap"],
    )
    return result
