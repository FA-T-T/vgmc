"""Python reproduction of OGM construction, fitting and pooled estimation.

Wu et al., Quantum 7, 896 (2023), Algorithms 1 and 2 and Appendix D.
The probability fit uses SciPy SLSQP.
"""

from __future__ import annotations
from typing import Any
from numpy.typing import ArrayLike
from .._validation import BoolArray, FloatArray, integer, real_array
from ..covariance import CovarianceProvider
from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.optimize import minimize
from ..grouping import validate_words, compatible
from ..covariance import quadratic

UPSTREAM_COMMIT = "0d8753ab35a3dd3f04901e88b6df35abd8c5cbca"


@dataclass(frozen=True)
class OGMConfig:
    shots: int
    max_rounds: int = 10
    iterations_per_round: int = 10
    final_max_iterations: int = 100
    optimizer_ftol: float = 0.0001
    outer_loss_tolerance: float = 0.001
    prune: bool = True
    sampling_seed: int = 0

    def __post_init__(self) -> None:
        """Validate positive iteration limits and the random stream seed."""
        if type(self.prune) is not bool:
            raise ValueError("prune must be boolean")
        if type(self.sampling_seed) is not int or self.sampling_seed < 0:
            raise ValueError("sampling_seed must be a nonnegative integer")
        for name in (
            "shots",
            "max_rounds",
            "iterations_per_round",
            "final_max_iterations",
        ):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("optimizer_ftol", "outer_loss_tolerance"):
            if (
                not math.isfinite(getattr(self, name))
                or getattr(self, name) <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")


def _coefficients(words: list[str], coefficients: ArrayLike) -> FloatArray:
    validate_words(words)
    c = real_array(coefficients, "coefficients", 1)
    if (
        c.shape != (len(words),)
        or not np.all(np.isfinite(c))
        or np.any(c == 0)
    ):
        raise ValueError(
            "Require one finite nonzero coefficient per unique word"
        )
    if "I" * len(words[0]) in words:
        raise ValueError(
            "Remove the identity term and add its known constant separately"
        )
    return c


def coverage_matrix(words: list[str], bases: list[str]) -> BoolArray:
    """Compute directional coverage of Pauli terms by measurement bases.

    Args:
        words: Nonempty equal-width Pauli words.
        bases: Nonempty equal-width product strings over I, X, Y and Z.

    Returns:
        Boolean array (n_terms, n_bases). A basis identity measures no
        nonidentity term on that wire.

    Raises:
        ValueError: Input widths or Pauli alphabets differ.

    Examples:
        >>> coverage_matrix(["XI", "IZ"], ["XZ", "XI"]).tolist()
        [[True, True], [True, False]]
    """
    if not words or not bases:
        raise ValueError("Require nonempty Pauli terms and measurement bases")
    width = len(words[0])
    if width == 0 or any(
        (len(b) != width or set(b) - set("IXYZ") for b in bases)
    ):
        raise ValueError("Invalid measurement bases")
    if any((len(w) != width or set(w) - set("IXYZ") for w in words)):
        raise ValueError("Invalid Pauli terms")
    term_codes = np.frombuffer(
        "".join(words).encode("ascii"), dtype=np.uint8
    ).reshape(-1, width)
    basis_codes = np.frombuffer(
        "".join(bases).encode("ascii"), dtype=np.uint8
    ).reshape(-1, width)
    covered = np.ones((len(words), len(bases)), dtype=bool)
    for qubit in range(width):
        term = term_codes[:, qubit, None]
        covered &= (term == ord("I")) | (term == basis_codes[None, :, qubit])
    return covered


def initial_distribution(
    words: list[str], coefficients: ArrayLike
) -> dict[str, Any]:
    """Apply OGM forward growth and backward completion.

    Args:
        words: Unique equal-width nonidentity Pauli strings.
        coefficients: Nonzero real energy coefficients, shape (n_terms,).

    Returns:
        Dictionary with bases, covered index groups, normalized initial
        probabilities and coefficient-sorted term order.

    Raises:
        ValueError: Words, coefficient axes or coefficient values are invalid.

    Examples:
        >>> initial_distribution(["XI", "IZ"], [1., 2.])["bases"]
        ['XZ']
    """
    c = _coefficients(words, coefficients)
    order = sorted(range(len(words)), key=lambda i: (-abs(c[i]), i))
    added = set()
    bases, masses, groups = ([], [], [])

    def merge(a: str, b: str) -> str:
        return "".join(
            (y if x == "I" else x for x, y in zip(a, b, strict=True))
        )

    for position, seed in enumerate(order):
        if seed in added:
            continue
        basis, group, mass = (words[seed], [seed], abs(c[seed]))
        added.add(seed)
        for i in order[position + 1 :]:
            if compatible(basis, words[i]):
                basis = merge(basis, words[i])
                group.append(i)
                added.add(i)
                mass += abs(c[i])
        for i in order[:position]:
            if compatible(basis, words[i]):
                basis = merge(basis, words[i])
                group.append(i)
        bases.append(basis)
        groups.append(group)
        masses.append(float(mass))
    p = np.asarray(masses) / sum(masses)
    coverage = coverage_matrix(words, bases)
    assert coverage.any(axis=1).all()
    assert all(
        (
            set(g) == set(np.flatnonzero(coverage[:, k]))
            for k, g in enumerate(groups)
        )
    )
    return {
        "bases": bases,
        "groups": groups,
        "probabilities": p.tolist(),
        "term_order": order,
    }


def diagonal_loss(
    probabilities: ArrayLike,
    coverage: ArrayLike,
    coefficients: ArrayLike,
    shots: int,
) -> tuple[float, FloatArray]:
    """Evaluate the OGM diagonal surrogate and its gradient (Eq. 15).

    Args:
        probabilities: Nonnegative basis fractions, shape (n_bases,).
        coverage: Binary coverage matrix, shape (n_terms, n_bases).
        coefficients: Finite energy coefficients, shape (n_terms,).
        shots: Positive requested measurement count for uncovered-term cost.

    Returns:
        Surrogate value in squared energy units and gradient (n_bases,).
        An uncovered term contributes shots * coefficient ** 2.

    Raises:
        ValueError: Numeric values, shapes or shot count are invalid.

    Examples:
        >>> loss, gradient = diagonal_loss([1.], [[1]], [2.], 10)
        >>> loss, gradient.tolist()
        (4.0, [-4.0])
    """
    p = real_array(probabilities, "probabilities", 1)
    raw_coverage = np.asarray(coverage)
    a = real_array(
        (
            raw_coverage.astype(float)
            if raw_coverage.dtype.kind == "b"
            else raw_coverage
        ),
        "coverage",
        2,
    )
    c2 = np.square(real_array(coefficients, "coefficients", 1))
    integer(shots, "shots", 1)
    if (
        a.shape != (len(c2), len(p))
        or np.any(p < 0)
        or np.any((a != 0) & (a != 1))
    ):
        raise ValueError("Coverage and probability axes differ")
    chi = a @ p
    live = chi > 0
    loss = float(np.sum(c2[live] / chi[live]) + shots * np.sum(c2[~live]))
    gradient = -(a[live].T @ (c2[live] / chi[live] ** 2))
    return (loss, gradient)


def _simplex(p: ArrayLike) -> FloatArray:
    p = real_array(p, "probabilities", 1)
    if (
        p.ndim != 1
        or not len(p)
        or (not np.all(np.isfinite(p)))
        or np.any(p < -1e-12)
    ):
        raise ValueError("Invalid probabilities")
    if abs(p.sum() - 1) > 1e-07:
        raise ValueError("Probability conservation failure")
    p = np.maximum(0, p)
    return p / p.sum()


def _optimize(
    p: FloatArray,
    a: BoolArray,
    c: FloatArray,
    config: OGMConfig,
    maxiter: int,
) -> tuple[FloatArray, dict[str, Any]]:
    old_loss = diagonal_loss(p, a, c, config.shots)[0]
    scale = max(1.0, old_loss)

    def objective(q: FloatArray) -> tuple[float, FloatArray]:
        loss, gradient = diagonal_loss(q, a, c, config.shots)
        return (loss / scale, gradient / scale)

    result = minimize(
        objective,
        p,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(p),
        constraints={
            "type": "eq",
            "fun": lambda q: q.sum() - 1,
            "jac": lambda q: np.ones(len(q)),
        },
        options={"ftol": config.optimizer_ftol / scale, "maxiter": maxiter},
    )
    valid = True
    try:
        candidate = _simplex(result.x)
        new_loss = diagonal_loss(candidate, a, c, config.shots)[0]
        valid = np.isfinite(new_loss) and new_loss <= old_loss + 1e-12 * max(
            1, old_loss
        )
    except ValueError:
        valid = False
    info = {
        "success": bool(result.success and valid),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "incumbent_retained": not bool(valid),
        "old_loss": old_loss,
        "objective_scale": scale,
        "raw_simplex_error": float(result.x.sum() - 1),
    }
    return (candidate if valid else p, info)


def fit(
    words: list[str], coefficients: ArrayLike, config: OGMConfig
) -> dict[str, Any]:
    """Fit the reproduced OGM distribution and construct its shot schedule.

    Args:
        words: Unique equal-width nonidentity Pauli strings.
        coefficients: Nonzero finite energy coefficients, shape (n_terms,).
        config: Immutable SLSQP, pruning, shot-count and random-seed options.

    Returns:
        JSON-compatible bases, probabilities, integer schedule, optimizer
        history, termination reason and configuration. The diagonal fitting
        objective is evaluated independently from pooled physical variance.

    Raises:
        ValueError: Words, coefficients or configuration are invalid.

    Examples:
        >>> fit(["Z"], [2.], OGMConfig(shots=10))["schedule"]["actual_shots"]
        10
    """
    c = _coefficients(words, coefficients)
    initial = initial_distribution(words, c)
    bases, p = (initial["bases"].copy(), np.array(initial["probabilities"]))
    history, previous_loss, success = ([], 0.0, False)
    termination = "outer_iteration_limit"
    for step in range(1, config.max_rounds + 1):
        order = np.argsort(-p, kind="stable")
        p, bases = (p[order], [bases[i] for i in order])
        a = coverage_matrix(words, bases)
        length, count = (0, 0)
        for probability in p:
            length += 1
            count += math.ceil(config.shots * float(probability))
            if count >= config.shots:
                break
        cut = False
        if config.prune and length < len(p):
            proposal = p[:length] / p[:length].sum()
            if (
                diagonal_loss(proposal, a[:, :length], c, config.shots)[0]
                < diagonal_loss(p, a, c, config.shots)[0]
            ):
                p, bases, a = (proposal, bases[:length], a[:, :length])
                cut = True
        if not cut and success:
            termination = "author_no_improving_prune_after_optimizer_success"
            break
        p, info = _optimize(
            p, a, c, config, config.iterations_per_round * step
        )
        loss = diagonal_loss(p, a, c, config.shots)[0]
        success = info["success"]
        history.append(
            {
                "round": step,
                "groups": len(p),
                "pruned": cut,
                "loss": loss,
                **info,
            }
        )
        if abs(previous_loss - loss) < config.outer_loss_tolerance:
            termination = "author_diagonal_loss_stable"
            break
        previous_loss = loss
    if not success:
        a = coverage_matrix(words, bases)
        p, info = _optimize(p, a, c, config, config.final_max_iterations)
        success = info["success"]
        history.append(
            {
                "round": "final",
                "groups": len(p),
                "loss": diagonal_loss(p, a, c, config.shots)[0],
                **info,
            }
        )
    coverage = coverage_matrix(words, bases)
    return {
        "method": "OGM",
        "variant": "paper-v1-partially-derandomized-pooled",
        "estimator": "appendix-D-pooled",
        "schedule": partial_derandomized_counts(
            p, config.shots, config.sampling_seed
        ),
        "initialization": "algorithm-1-direct-as-paper-table-1",
        "config": asdict(config),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_role": "reference-implementation",
        "iteration_policy": "configured-slsqp-iterations",
        "bases": bases,
        "probabilities": p.tolist(),
        "initial": initial,
        "history": history,
        "uncovered_indices": np.flatnonzero(coverage @ p == 0).tolist(),
        "diagonal_loss": diagonal_loss(p, coverage, c, config.shots)[0],
        "optimizer_success": success,
        "termination": termination,
        "global_optimality_certified": False,
        "design_uses_covariance": False,
    }


def partial_derandomized_counts(
    probabilities: ArrayLike, shots: int, seed: int
) -> dict[str, Any]:
    """Construct the reproduced partially derandomized OGM schedule.

    Args:
        probabilities: Nonnegative normalized fractions, shape (n_bases,).
        shots: Positive requested joint-measurement count.
        seed: Nonnegative integer random-stream seed.

    Returns:
        JSON record with integer counts, requested_shots and actual_shots.
        Bernoulli residual rounding and the recorded low-probability fill
        determine the actual total used by the pooled estimator.

    Raises:
        ValueError: Simplex, shot count or seed is invalid.

    Examples:
        >>> partial_derandomized_counts([0.4, 0.6], 10, 0)["counts"]
        [4, 6]
    """
    if type(shots) is not int or shots <= 0:
        raise ValueError("shots must be a positive integer")
    integer(seed, "seed")
    p = _simplex(probabilities)
    expected = p * shots
    counts = np.floor(expected).astype(np.int64)
    rng = np.random.default_rng(seed)
    order = np.argsort(-p, kind="stable")
    for i in order:
        if counts.sum() >= shots:
            break
        counts[i] += rng.random() < expected[i] - math.floor(expected[i])
    for i in order:
        if counts.sum() >= shots:
            break
        if 0 < expected[i] < 1:
            counts[i] += 1
    return {
        "counts": counts.tolist(),
        "requested_shots": shots,
        "actual_shots": int(counts.sum()),
        "seed": seed,
        "sampling_rule": "paper-algorithm-2-with-stated-low-probability-fill",
    }


def evaluate_counts(
    words: list[str],
    coefficients: ArrayLike,
    bases: list[str],
    counts: ArrayLike,
    cov: CovarianceProvider,
    means: ArrayLike,
) -> dict[str, Any]:
    """Evaluate pooled OGM variance and omitted-term bias.

    Args:
        words: Unique nonidentity Pauli words.
        coefficients: Nonzero real energy coefficients, shape (n_terms,).
        bases: Product measurement strings in the same wire order.
        counts: Nonnegative integer-valued counts, shape (n_bases,).
        cov: Reference Pauli covariance provider.
        means: Reference Pauli expectations, shape (n_terms,).

    Returns:
        Record with variance, bias, mse, rmse, actual shots and V_effective.
        V_effective equals actual shots times conditional pooled variance.
        Bias and RMSE use energy units; variance and MSE use squared units.

    Raises:
        ValueError: Pauli inputs, counts, means or axes are invalid.

    Examples:
        >>> from vgmc import Covariance
        >>> r = evaluate_counts(["Z"], [2.], ["Z"], [4],
        ...                     Covariance([[1.]]), [0.])
        >>> r["variance"], r["V_effective"], r["bias"]
        (1.0, 4.0, -0.0)
    """
    c = _coefficients(words, coefficients)
    a = coverage_matrix(words, bases)
    n = np.asarray(counts)
    if (
        n.shape != (len(bases),)
        or not np.all(np.isfinite(n))
        or np.any(n < 0)
        or np.any(n != np.floor(n))
        or (n.sum() <= 0)
    ):
        raise ValueError("Invalid integer counts")
    mu = real_array(means, "means", 1)
    if mu.shape != c.shape:
        raise ValueError("Invalid Pauli means")
    hits = a @ n
    weights = np.divide(c, hits, out=np.zeros_like(c), where=hits > 0)
    variance = 0.0
    for k, count in enumerate(n):
        if count:
            g = np.flatnonzero(a[:, k]).tolist()
            variance += float(count) * quadratic(weights[g], cov.block(g, 0.0))
    bias = -float(c[hits == 0] @ mu[hits == 0])
    return {
        "sampling_rule": "native-pooled-conditional-on-counts",
        "shots": int(n.sum()),
        "variance": variance,
        "V_effective": float(n.sum()) * variance,
        "bias": bias,
        "mse": variance + bias**2,
        "rmse": math.sqrt(variance + bias**2),
        "uncovered_indices": np.flatnonzero(hits == 0).tolist(),
    }
