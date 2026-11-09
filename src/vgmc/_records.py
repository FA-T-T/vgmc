"""Typed numerical results shared by the solver and experiment boundary."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class AuditResult(TypedDict):
    """Feasible estimator and numerical finite-library bounds."""

    mu: list[float]
    upper: float
    lower: float
    gap: float
    physical_variance: float
    conservation_error: float
    repair_max: float
    allocation: list[float]
    x: list[list[float]]
    audit_seconds: float
    bound_kind: str


class StageResult(TypedDict):
    """One optimization stage and its independently recovered estimator."""

    method: str
    updates: int
    evaluations: int
    approximate_gap: float
    optimizer_message: str
    audit: AuditResult
    boundary_steps: NotRequired[list[dict[str, float | int]]]


class SolverResult(TypedDict):
    """Fixed-library solver result with original term and group order."""

    groups: list[list[int]]
    x: list[list[float]]
    a: list[float]
    mu: list[float]
    upper_bound: float
    lower_bound: float
    gap: float
    physical_V: float
    conservation_error: float
    status: str
    iterations: int
    evaluations: int
    ridge: float
    relative_gap: float
    solve_seconds: float
    numerical_history: NotRequired[list[dict[str, Any]]]
    settings: NotRequired[dict[str, Any]]
    preparation: NotRequired[dict[str, int | float | str]]
    failure: NotRequired[str]
