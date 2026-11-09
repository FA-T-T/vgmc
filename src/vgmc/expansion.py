"""Variance-guided group admission, reoptimization and finite-pool stopping."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from collections.abc import Callable, Mapping
from typing import Any
import logging
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from threadpoolctl import threadpool_limits

from .candidates import grow, seed_order
from .certificates import group_score, mp_score, pool_decision
from .covariance import CovarianceProvider, as_covariance, variance
from .config import ExpansionConfig
from ._validation import FloatArray, index_group, integer, real_array
from ._records import SolverResult
from .artifacts import (
    CheckpointStore,
    checkpoint_lock,
    computation_identity,
    identity,
)
from .grouping import basis_for, compatibility_matrix, validate_words
from .solver import solve

LOGGER = logging.getLogger(__name__)


def validate_initial_solution(
    groups: list[list[int]],
    c: ArrayLike,
    covariance: CovarianceProvider,
    solution: SolverResult,
    cfg: ExpansionConfig,
) -> None:
    """Verify a supplied common starting design against the effective model."""
    from .candidates import score

    c = real_array(c, "c", 1)
    if solution["groups"] != groups or solution["status"] != "converged":
        raise ValueError("Initial solution must match the converged library")
    a = real_array(solution["a"], "initial allocation", 1)
    mu = real_array(solution["mu"], "initial dual", 1)
    if len(mu) != len(c):
        raise ValueError("Initial dual dimension differs")
    total = np.zeros_like(c)
    for g, raw_x in zip(groups, solution["x"], strict=True):
        x = real_array(raw_x, "initial coefficient block", 1)
        if len(x) != len(g):
            raise ValueError("Initial coefficient block dimension differs")
        np.add.at(total, g, x)
    tolerance = cfg.solver.conservation_tolerance * max(
        1.0, float(np.max(abs(c)))
    )
    if np.max(abs(total - c)) > tolerance:
        raise ValueError("Initial coefficients fail Hamiltonian recovery")
    upper = variance(groups, solution["x"], a, covariance)
    upper += cfg.solver.ridge * sum(
        float(np.asarray(x) @ np.asarray(x)) / w
        for x, w in zip(solution["x"], a, strict=True)
        if w > 0
    )
    if not np.any(c):
        if upper != 0:
            raise ValueError("Initial zero-target variance differs")
        return
    maximum = max(score(g, mu, covariance, cfg.solver.ridge) for g in groups)
    lower = float(c @ mu) ** 2 / (
        maximum * (1 + cfg.solver.dual_safety_margin)
    )
    gap = upper / lower - 1 if lower > 0 else float("inf")
    if not 0 <= gap <= cfg.terminal_gap * (1 + 1e-6):
        raise ValueError(
            "Initial solution fails its effective-model gap check"
        )
    if not np.isclose(
        upper, solution["upper_bound"], atol=tolerance, rtol=1e-10
    ):
        raise ValueError("Initial model upper bound differs")


def score_interval(
    group: list[int],
    mu: FloatArray,
    covariance: CovarianceProvider,
    settings: Mapping[str, Any] | ExpansionConfig,
) -> tuple[float, float, float]:
    """Check the numerical score interval of a selected group.

    Args:
        group: Term indices of a candidate or retained group.
        mu: Current real response vector of shape (n_terms,).
        covariance: Physical covariance provider.
        settings: Effective expansion and numerical-audit configuration.

    Returns:
        Lower score, upper score and arithmetic allowance, all in squared
        energy units. An unresolved binary-precision eigenvalue check uses
        two independent mpmath decimal contexts on the frozen block.

    Raises:
        ValueError: Shapes, covariance entries or configuration are invalid.
        ArithmeticError: The score remains numerically unresolved.
    """
    cfg = (
        settings
        if isinstance(settings, ExpansionConfig)
        else ExpansionConfig.from_mapping(settings)
    )
    block = covariance.block(group, cfg.solver.ridge)
    try:
        result = group_score(block, mu[group], cfg.audit_settings())
        error = result["residual_allowance"]
        return max(0.0, result["score"] - error), result["score_upper"], error
    except ArithmeticError:
        first = mp_score(block, mu[group], 60)
        second = mp_score(block, mu[group], 90)
        error = abs(first - second) + abs(second) * 1e-12
        return (
            max(0.0, min(first, second) - error),
            max(first, second) + error,
            error,
        )


@checkpoint_lock
def expand(
    words: list[str],
    c: ArrayLike,
    covariance: CovarianceProvider,
    groups: list[list[int]],
    settings: Mapping[str, Any] | ExpansionConfig,
    *,
    initial_a: ArrayLike | None = None,
    initial_solution: SolverResult | None = None,
    arm: str = "A",
    batch_size: int | None = None,
    max_added: int | None = None,
    targets: list[int] | None = None,
    workers: int = 1,
    checkpoint: Path | None = None,
    resume: bool = False,
    certify_admission: bool = True,
    mode: str | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Construct and reoptimize a finite VGMC measurement library.

    Args:
        words: Unique equal-width Pauli strings in Hamiltonian order.
        c: Finite real energy coefficients of shape (n_terms,).
        covariance: Physical covariance block provider.
        groups: Initial QWC library covering every target coefficient.
        settings: Immutable expansion settings or canonical configuration map.
        initial_a: Optional nonnegative initial allocation, shape (n_groups,).
        initial_solution: Optional verified common fixed-library solution.
        arm: A uses current/full growth; B freezes the initial response;
            C uses diagonal growth; D combines frozen and diagonal growth.
        batch_size: Explicit positive admission-batch size.
        max_added: Nonnegative admitted-group limit; None uses pool stopping.
        targets: Increasing added-group counts included as batch boundaries.
        workers: Positive candidate-worker count, recorded in run identity.
        checkpoint: Optional path for atomic verified endpoint checkpoints.
        resume: Legacy shorthand for mode=resume when mode is absent.
        certify_admission: Check candidate score allowances before admission.
        mode: Fresh, reuse, resume or auto checkpoint lifecycle.
        progress: Callback receiving a copy of each committed endpoint.

    Returns:
        JSON-compatible record containing groups, the fixed-library solution,
        trajectory, endpoint status and the final finite-pool audit when run.
        Physical trajectory variance excludes the design ridge. Completed
        endpoints are added_budget_reached, candidate_pool_tolerance_reached
        and zero_target. Unresolved admission retains an explicit status.

    Raises:
        ValueError: Inputs or checkpoint scientific identity differ.
        FileExistsError: Fresh execution targets an existing checkpoint.
        ArithmeticError: Fixed-library optimization remains unresolved.

    Examples:
        >>> state = expand(["X", "Z"], np.array([1., 2.]),
        ...                as_covariance(np.eye(2), 2), [[0], [1]],
        ...                ExpansionConfig(), max_added=0)
        >>> state["status"], len(state["trajectory"])
        ('added_budget_reached', 1)
    """
    cfg = (
        settings
        if isinstance(settings, ExpansionConfig)
        else ExpansionConfig.from_mapping(settings)
    )
    validate_words(words)
    c = real_array(c, "c", 1)
    if len(c) != len(words):
        raise ValueError("Words and coefficients must share term order")
    covariance = as_covariance(covariance, len(c))
    workers = integer(workers, "workers", 1)
    if arm not in ("A", "B", "C", "D"):
        raise ValueError("Invalid growth arm")
    if batch_size is not None:
        batch_size = integer(batch_size, "batch_size", 1)
    if max_added is not None:
        max_added = integer(max_added, "max_added")
    if targets is not None:
        targets = [integer(k, "target") for k in targets]
        if not targets or sorted(set(targets)) != targets:
            raise ValueError("Targets must be distinct and increasing")
    if type(certify_admission) is not bool:
        raise ValueError("certify_admission must be boolean")
    if mode is None:
        mode = "resume" if resume else "fresh"
    if mode not in ("fresh", "reuse", "resume", "auto"):
        raise ValueError("Unknown execution mode")
    groups = [index_group(g, len(c)) for g in groups]
    base_count = len(groups)
    for group in groups:
        basis_for(words, group)
    scientific_initial = (
        None
        if initial_solution is None
        else {
            key: dict(initial_solution)[key]
            for key in (
                "groups",
                "x",
                "a",
                "mu",
                "upper_bound",
                "lower_bound",
                "gap",
                "status",
            )
        }
    )
    binding = identity(
        dict(
            algorithm=computation_identity(),
            words=words,
            coefficients=c,
            covariance=covariance.fingerprint(),
            groups=groups,
            settings=cfg.to_dict(),
            arm=arm,
            batch_size=batch_size,
            max_added=max_added,
            targets=targets,
            workers=workers,
            certify_admission=certify_admission,
            initial_a=initial_a,
            initial_solution=scientific_initial,
        )
    )
    store = CheckpointStore(checkpoint, binding) if checkpoint else None

    def optimize(
        library: list[list[int]], allocation: ArrayLike | None, gap: float
    ) -> SolverResult:
        inner_binding = identity(dict(parent=binding, groups=library, gap=gap))
        inner = (
            None
            if checkpoint is None
            else CheckpointStore(
                checkpoint.with_name(checkpoint.stem + ".inner")
                / (inner_binding + ".json"),
                inner_binding,
            )
        )
        if inner and inner.exists():
            allocation = inner.load().payload["a"]
        result = solve(
            library,
            c,
            covariance,
            config=cfg.solver.with_gap(gap),
            initial_a=allocation,
            checkpoint=(
                (lambda result: inner.save(dict(result))) if inner else None
            ),
        )
        if inner:
            inner.save(dict(result))
        if result["status"] != "converged":
            raise ArithmeticError(
                f"Allocation {result['status']}: gap={result['gap']:.6g}, "
                f"target={gap:.6g}; "
                f"{result.get('failure', 'last audit retained')}"
            )
        return result

    if mode == "auto":
        if store and store.exists():
            loaded = store.load()
            mode = (
                "resume" if loaded.payload["status"] == "running" else "reuse"
            )
        else:
            mode = "fresh"
    if mode in ("reuse", "resume"):
        if store is None:
            raise ValueError("Reuse and resume require a checkpoint path")
        loaded = store.load()
        state = loaded.payload
        if state["status"] != "running":
            if mode == "resume":
                raise ValueError("A completed checkpoint is opened with reuse")
            if state["status"] not in (
                "added_budget_reached",
                "candidate_pool_tolerance_reached",
                "zero_target",
            ):
                raise ValueError(
                    "Only completed scientific endpoints are reusable"
                )
            return state
        if mode == "reuse":
            raise ValueError("Reuse requires a completed endpoint")
        state["recovery"] = dict(
            recovered=loaded.recovered, source=loaded.source
        )
    else:
        if store and store.exists():
            raise FileExistsError(
                "Checkpoint exists; choose a compatible mode"
            )
        if initial_solution is not None:
            validate_initial_solution(
                groups, c, covariance, initial_solution, cfg
            )
        solution = initial_solution or optimize(
            groups, initial_a, cfg.terminal_gap
        )
        state = dict(
            binding=binding,
            groups=groups,
            solution=solution,
            mu0=solution["mu"],
            initial_upper=solution["upper_bound"],
            initial_V=variance(
                groups, solution["x"], solution["a"], covariance
            ),
            trajectory=[],
            status="running",
            initial_groups=base_count,
        )

    def save() -> None:
        if store:
            store.save(state)

    def notify() -> None:
        if progress:
            import copy

            progress(copy.deepcopy(state))

    compatibility = compatibility_matrix(words)
    with (
        threadpool_limits(limits=1, user_api="blas"),
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        while True:
            groups = state["groups"]
            solution = state["solution"]
            added = len(groups) - base_count
            if (
                not state["trajectory"]
                or state["trajectory"][-1]["added"] != added
            ):
                value = variance(
                    groups, solution["x"], solution["a"], covariance
                )
                active_added = int(
                    np.count_nonzero(
                        np.asarray(solution["a"])[base_count:]
                        > cfg.active_threshold
                    )
                )
                state["trajectory"].append(
                    dict(
                        added=added,
                        V=value,
                        V0=state["initial_V"],
                        gain_pct=(
                            100 * (1 - value / state["initial_V"])
                            if state["initial_V"]
                            else 0.0
                        ),
                        active_added=active_added,
                        active_pct=(
                            100 * active_added / added if added else None
                        ),
                        gap=solution["gap"],
                    )
                )
            save()
            notify()
            if not np.any(c):
                state["status"] = "zero_target"
                break
            LOGGER.info(
                "VGMC arm %s: added=%d, V=%.8g",
                arm,
                added,
                state["trajectory"][-1]["V"],
            )
            if max_added is not None and added >= max_added:
                state["status"] = "added_budget_reached"
                break
            target = batch_size or min(
                len(c),
                max(
                    cfg.batch_minimum,
                    min(
                        cfg.batch_maximum,
                        int(np.ceil(cfg.batch_fraction * base_count)),
                    ),
                ),
            )
            if max_added is not None:
                target = min(target, max_added - added)
            if targets:
                future = [k for k in targets if k > added]
                if future:
                    target = min(target, min(future) - added)
            mu = np.asarray(solution["mu"], dtype=float)
            growth_mu = np.asarray(state["mu0"]) if arm in ("B", "D") else mu
            mode = "diagonal" if arm in ("C", "D") else "full"
            seeds = seed_order(mu, c)
            wave = min(len(c), max(2 * workers, 4 * target))
            existing = {tuple(sorted(g)) for g in groups}
            pool: dict[tuple, dict] = {}
            qualified: list[dict] = []
            threshold = max(solution["upper_bound"], float(c @ mu)) * (
                1 + cfg.candidate_score_margin
            )
            generator = partial(
                grow,
                words=words,
                growth_mu=growth_mu,
                scoring_mu=mu,
                covariance=covariance,
                compatibility=compatibility,
                ridge=cfg.solver.ridge,
                mode=mode,
            )
            scanned = 0
            for offset in range(0, len(seeds), wave):
                rows = list(
                    executor.map(generator, seeds[offset : offset + wave])
                )
                scanned += len(rows)
                for row in rows:
                    pool.setdefault(tuple(row["group"]), row)
                proposed = sorted(
                    (
                        r
                        for key, r in pool.items()
                        if key not in existing and r["score"] > threshold
                    ),
                    key=lambda row: (-row["score"], row["group"]),
                )
                qualified = []
                for row in proposed:
                    if certify_admission:
                        lower, upper, _ = score_interval(
                            row["group"], mu, covariance, settings
                        )
                        row.update(score_lower=lower, score_upper=upper)
                        if lower <= threshold:
                            continue
                    qualified.append(row)
                    if len(qualified) >= target:
                        break
                if len(qualified) >= target:
                    break
            if scanned == len(c):
                union = {tuple(sorted(g)): g for g in groups}
                union.update({key: row["group"] for key, row in pool.items()})
                maximum = uncertainty = 0.0
                for group in union.values():
                    _, upper, error = score_interval(
                        group, mu, covariance, settings
                    )
                    maximum = max(maximum, upper)
                    uncertainty = max(uncertainty, error)
                lower = float(c @ mu) ** 2 / (
                    maximum * (1 + cfg.dual_safety_margin)
                )
                allowance = solution["upper_bound"] * (
                    uncertainty / maximum + cfg.dual_safety_margin
                )
                decision = pool_decision(
                    solution["upper_bound"],
                    lower,
                    state["initial_upper"],
                    cfg.pool_gain_over_initial,
                    allowance,
                )
                state["pool_audit"] = dict(
                    decision,
                    lower_bound=lower,
                    candidates=len(union),
                    seeds=scanned,
                )
                if decision["verified"]:
                    state["status"] = decision["status"]
                    break
            if not qualified:
                next_gap = next(
                    (g for g in cfg.boundary_gaps if solution["gap"] > g), None
                )
                if next_gap:
                    state["solution"] = optimize(
                        groups, solution["a"], next_gap
                    )
                    state["trajectory"].pop()
                    save()
                    continue
                state["status"] = "numerical_resolution_unresolved"
                break
            selected = qualified[:target]
            enlarged = groups + [row["group"] for row in selected]
            mass = cfg.new_mass
            allocation = np.r_[
                np.asarray(solution["a"]) * (1 - mass),
                np.full(len(selected), mass / len(selected)),
            ]
            candidate = optimize(enlarged, allocation, cfg.inner_gap)
            if candidate["gap"] > cfg.terminal_gap:
                candidate = optimize(
                    enlarged, candidate["a"], cfg.terminal_gap
                )
            if candidate["upper_bound"] > solution["upper_bound"] * (
                1 + 2 * cfg.terminal_gap
            ):
                raise ArithmeticError(
                    "Reoptimized objective exceeds the comparison tolerance"
                )
            state.update(groups=enlarged, solution=candidate)
            save()
    save()
    notify()
    return state
