"""Simplex descent for allocations near the numerical boundary."""

from __future__ import annotations

import numpy as np

from ._core import Checkpoint, Evaluator
from ._records import StageResult
from ._validation import FloatArray


def polish(
    evaluator: Evaluator,
    initial: FloatArray,
    target_gap: float,
    max_updates: int,
    checkpoint: Checkpoint | None = None,
    max_evaluations: int | None = None,
) -> StageResult:
    """Bracket a simplex directional derivative and audit accepted
    allocations.
    """
    cfg = evaluator.p.problem.config
    a = np.array(initial, dtype=float, copy=True)
    steps: list[dict[str, float | int]] = []
    updates = 0
    gap = float("inf")
    message = "boundary_polish"
    for _ in range(max_updates):
        if max_evaluations is not None and evaluator.calls >= max_evaluations:
            message = "evaluation_limit"
            break
        phi, scores, gap = evaluator.evaluate(a)
        if 0 <= gap <= target_gap:
            break
        j = int(np.argmax(scores))
        vertex = np.full(len(a), cfg.allocation_floor)
        vertex[j] += 1 - vertex.sum()
        direction = vertex - a
        if -float(scores @ direction) >= 0:
            raise ArithmeticError(
                "Simplex directional derivative is nonnegative"
            )

        def slope(
            gamma: float, a: FloatArray = a, direction: FloatArray = direction
        ) -> tuple[float, float]:
            """Evaluate the current simplex search direction."""
            if (
                max_evaluations is not None
                and evaluator.calls >= max_evaluations
            ):
                raise StopIteration
            value, trial_scores, _ = evaluator.evaluate(a + gamma * direction)
            return -float(trial_scores @ direction), value

        try:
            lo = 0.0
            hi = max(1e-10, min(0.1, gap * 0.01))
            derivative, _ = slope(hi)
            while derivative < 0 and hi < 0.5:
                lo, hi = hi, min(0.5, hi * 10)
                derivative, _ = slope(hi)
            for _bracket in range(18):
                mid = (lo + hi) / 2
                derivative, _ = slope(mid)
                if derivative < 0:
                    lo = mid
                else:
                    hi = mid
            gamma = (lo + hi) / 2
            # Numerical cancellation can affect objective differences near the
            # boundary. Backtracking checks the same model at smaller steps.
            for _backtrack in range(cfg.max_backtracks):
                derivative, value = slope(gamma)
                if np.isfinite(value) and value <= phi * (1 + 1e-11):
                    break
                gamma *= 0.5
            else:
                raise ArithmeticError("Boundary line search unresolved")
        except StopIteration:
            message = "evaluation_limit"
            break
        candidate = a + gamma * direction
        steps.append(
            dict(
                index=j,
                before_weight=float(a[j]),
                gamma=gamma,
                before_gap=gap,
                objective_change=value - phi,
            )
        )
        a = candidate
        updates += 1
        if checkpoint and updates % cfg.checkpoint_interval == 0:
            checkpoint(evaluator.p.audit(a), updates, evaluator.calls)
    return StageResult(
        method="simplex_boundary_polish",
        updates=updates,
        evaluations=evaluator.calls,
        audit=evaluator.p.audit(a),
        approximate_gap=gap,
        optimizer_message=message,
        boundary_steps=steps,
    )
