"""Stable framework-neutral entry point for measurement design."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .config import ExpansionConfig, SolverConfig
from .covariance import CovarianceProvider, as_covariance
from .grouping import basis_for, sorted_insertion, visible
from .measurement import MeasurementDesign, PauliHamiltonian
from .solver import solve


def design_measurements(
    hamiltonian: PauliHamiltonian,
    covariance: ArrayLike | CovarianceProvider,
    *,
    groups: Sequence[Sequence[int]] | None = None,
    config: SolverConfig | None = None,
    expansion: ExpansionConfig | None = None,
    max_added: int = 0,
    initial_a: ArrayLike | None = None,
) -> MeasurementDesign:
    """Build a portable Pauli measurement estimator.

    Args:
        hamiltonian: Real Pauli terms, constant and explicit wire order.
        covariance: Pauli covariance matrix or block provider in term order.
        groups: Optional initial QWC library. The default completes SI
            product bases with all visible Hamiltonian terms.
        config: Fixed-library numerical settings.
        expansion: Candidate-growth settings when ``max_added`` is positive.
            Its solver options take precedence during expansion.
        max_added: Maximum admitted group count. Zero optimizes the initial
            library. Positive values enable bounded VGMC construction.
        initial_a: Optional initial group shot fractions.

    Returns:
        Validated immutable estimator with integer shot-plan construction.

    Raises:
        RuntimeError: Numerical convergence or candidate admission is
        unresolved.
        ValueError: Pauli terms, covariance, groups or configuration are
        invalid.

    Examples:
        >>> h = PauliHamiltonian(("X", "Z"), [1.0, 2.0], constant=0.5)
        >>> d = design_measurements(h, np.eye(2),
        ...                         config=SolverConfig(ridge=0.0),
        ...                         initial_a=[1.0, 2.0])
        >>> d.allocate_shots(30).counts.tolist()
        [20, 10]
    """
    from ._validation import integer

    max_added = integer(max_added, "max_added")
    provider = as_covariance(covariance, hamiltonian.n_terms)
    c = np.asarray(hamiltonian.coefficients)
    if groups is None:
        native = sorted_insertion(list(hamiltonian.words), c) if len(c) else []
        library = []
        seen = set()
        for group in native:
            support = visible(
                list(hamiltonian.words),
                basis_for(list(hamiltonian.words), group),
            )
            if tuple(support) not in seen:
                library.append(support)
                seen.add(tuple(support))
    else:
        library = [list(g) for g in groups]
        for group in library:
            basis_for(list(hamiltonian.words), group)
    if max_added and len(c):
        from .expansion import expand

        options = expansion or ExpansionConfig(solver=config or SolverConfig())
        result = expand(
            list(hamiltonian.words),
            c,
            provider,
            library,
            options,
            max_added=max_added,
            initial_a=initial_a,
        )
        if result["status"] not in (
            "added_budget_reached",
            "candidate_pool_tolerance_reached",
            "zero_target",
        ):
            raise RuntimeError(f"Measurement construction: {result['status']}")
        solution = result["solution"]
    else:
        solution = solve(
            library, c, provider, config=config, initial_a=initial_a
        )
    if solution["status"] != "converged":
        raise RuntimeError(f"Measurement optimization: {solution['status']}")
    return MeasurementDesign.from_result(hamiltonian, solution)
