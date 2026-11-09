"""Sparse precision solves with reusable preconditioners and residual
checks.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg, splu

from ._core import CPUEvaluator, ExtendedArray, Prepared
from ._validation import FloatArray


class SparseEvaluator(CPUEvaluator):
    """Evaluate occupied precision entries and verified sparse linear
    solves.
    """

    def __init__(self, prepared: Prepared) -> None:
        """Bind one prepared problem and initialize bounded diagnostics."""
        if not isinstance(prepared, Prepared):
            raise TypeError("SparseEvaluator requires a Prepared problem")
        super().__init__(prepared)
        self.cached_solve: Callable[[FloatArray], FloatArray] | None = None
        self.preconditioner_age = 0
        self.diagnostics: deque[dict[str, float | int | bool]] = deque(
            maxlen=128
        )

    def evaluate(self, a: FloatArray) -> tuple[float, FloatArray, float]:
        """Solve the current sparse system and check original-system error."""
        start = perf_counter()
        self.calls += 1
        matrix = self.p.matrix_sparse(a)
        rebuilt = False
        iterations = 0

        def direct() -> Callable[[FloatArray], FloatArray]:
            """Factor an equilibrated current precision matrix."""
            nonlocal rebuilt
            scale = 1 / np.sqrt(np.asarray(matrix.diagonal(), dtype=float))
            system = diags(scale) @ matrix.astype(float) @ diags(scale)
            factor = splu(
                system.tocsc(),
                permc_spec="MMD_AT_PLUS_A",
                options={"Equil": False},
            )

            def solve_factored(rhs: FloatArray) -> FloatArray:
                """Undo diagonal equilibration around the sparse factor
                solve.
                """
                return np.asarray(scale * factor.solve(scale * rhs))

            self.cached_solve = solve_factored
            self.preconditioner_age = 0
            rebuilt = True
            return solve_factored

        if self.cached_solve is None or self.preconditioner_age >= 20:
            direct()
        current = matrix.astype(float)

        def precondition(rhs: FloatArray) -> FloatArray:
            """Apply the currently retained factorization."""
            if self.cached_solve is None:
                raise RuntimeError("Preconditioner missing")
            return self.cached_solve(rhs)

        preconditioner = LinearOperator(current.shape, matvec=precondition)

        def solve(rhs: FloatArray) -> FloatArray:
            """Accept finite Krylov iterates or rebuild the current factors."""
            nonlocal iterations

            def count(_iterate: FloatArray) -> None:
                """Count Krylov iterations in this objective evaluation."""
                nonlocal iterations
                iterations += 1

            x, info = cg(
                current,
                np.asarray(rhs, dtype=float),
                M=preconditioner,
                rtol=1e-10,
                atol=0,
                maxiter=50,
                callback=count,
            )
            if info != 0 or not np.isfinite(x).all():
                return direct()(np.asarray(rhs, dtype=float))
            return np.asarray(x)

        self.preconditioner_age += 1
        mu = solve(self.p.problem.c).astype(np.longdouble)
        for _ in range(self.p.problem.config.linear_refinements):
            residual = self.p.problem.c.astype(np.longdouble) - matrix @ mu
            mu += solve(np.asarray(residual, dtype=float))
        error = backward_error(matrix, mu, self.p.problem.c)
        if not np.isfinite(error) or error > 1e-12:
            # A Krylov success code is followed by a current-system residual
            # check. Rebuilding preserves the same scientific precision matrix.
            current_solve = direct()
            mu = current_solve(self.p.problem.c).astype(np.longdouble)
            for _ in range(self.p.problem.config.linear_refinements):
                residual = self.p.problem.c.astype(np.longdouble) - matrix @ mu
                mu += current_solve(np.asarray(residual, dtype=float))
            error = backward_error(matrix, mu, self.p.problem.c)
        if not np.isfinite(error) or error > 1e-12:
            raise ArithmeticError(f"Linear backward error: {error}")
        feature_values = self.p.features_ld @ mu
        scores = np.asarray(
            np.add.reduceat(feature_values**2, self.p.offsets[:-1]),
            dtype=float,
        )
        phi = float(self.p.problem.c.astype(np.longdouble) @ mu)
        if phi <= 0 or not np.isfinite(phi) or not np.isfinite(scores).all():
            raise ArithmeticError("Invalid allocation objective or score")
        self.last_mu = mu
        self.seconds += perf_counter() - start
        self.diagnostics.append(
            {
                "backward_error": error,
                "nnz": matrix.nnz,
                "cg_iterations": iterations,
                "rebuild": rebuilt,
            }
        )
        return phi, scores, float(scores.max() / phi - 1)


def backward_error(
    matrix: csr_matrix, vector: FloatArray | ExtendedArray, rhs: FloatArray
) -> float:
    """Evaluate componentwise backward error in original precision
    coordinates.
    """
    residual = rhs.astype(np.longdouble) - matrix @ vector
    denominator = abs(matrix) @ abs(vector) + abs(rhs) + np.finfo(float).tiny
    return float(np.max(abs(residual) / denominator))
