"""Joint bitstring sampling for frozen coefficient and allocation designs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

from numpy.typing import ArrayLike

from ._validation import FloatArray, IntArray, index_group, integer, real_array
from .artifacts import (
    CheckpointStore,
    checkpoint_lock,
    computation_identity,
    identity,
)
from .config import SamplingConfig
from .grouping import validate_words

import numpy as np

from .covariance import quadratic
from .grouping import basis_for


def stream_id(molecule: str, covariance: str, method: str) -> int:
    """Return the experiment's deterministic measurement stream identifier."""
    method = "VOCA" if method == "VGMC" else method
    identifier = f"{molecule}_{covariance}_{method}_finite"
    return int.from_bytes(
        hashlib.sha256(identifier.encode()).digest()[:8], "little"
    )


def basis_distribution(state: ArrayLike, basis: str) -> FloatArray:
    """Compute the joint distribution of a product-basis measurement.

    Args:
        state: Finite normalized pure state, shape (2 ** n_qubits,).
            Character zero corresponds to the most significant state bit.
        basis: Complete XYZ measurement string in the same wire order.

    Returns:
        Float64 probabilities, shape (2 ** n_qubits,), in binary outcome
        order. Y measurement applies S-dagger followed by Hadamard.

    Raises:
        ValueError: Basis, state shape, finiteness or unit norm is invalid.

    Examples:
        >>> basis_distribution(np.array([1., 0.]), "Z").tolist()
        [1.0, 0.0]
    """
    raw = np.asarray(state)
    if (
        raw.ndim != 1
        or raw.size != 2 ** len(basis)
        or raw.dtype.kind not in "iufc"
    ):
        raise ValueError("State shape must be (2 ** len(basis),)")
    if not basis or set(basis) - set("XYZ") or not np.isfinite(raw).all():
        raise ValueError("A finite state and complete XYZ basis are required")
    norm = float(np.vdot(raw, raw).real)
    if abs(norm - 1.0) > 1e-10:
        raise ValueError("State vector must be normalized")
    h = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    tensor = np.asarray(state, dtype=complex).reshape((2,) * len(basis))
    for q, axis in enumerate(basis):
        rotation = {"Z": np.eye(2), "X": h, "Y": h @ np.diag([1.0, -1j])}[axis]
        tensor = np.moveaxis(np.tensordot(rotation, tensor, axes=(1, q)), 0, q)
    probabilities = abs(tensor.reshape(-1)) ** 2
    return probabilities / probabilities.sum()


def score_outcomes(
    words: list[str], group: list[int], coefficients: ArrayLike
) -> FloatArray:
    """Compute a frozen group's energy score for every joint outcome.

    Args:
        words: Valid Pauli strings in Hamiltonian order.
        group: QWC term indices measured by one complete product basis.
        coefficients: Real local energy coefficients, shape (len(group),).

    Returns:
        Float64 energy scores in most-significant-bit-first binary order,
        shape (2 ** n_qubits,). Each term contributes its bit parity.

    Raises:
        ValueError: Pauli words, group indices or coefficient axes differ.

    Examples:
        >>> score_outcomes(["ZI", "IZ"], [0, 1], [1., 2.]).tolist()
        [3.0, -1.0, 1.0, -3.0]
    """
    validate_words(words)
    group = index_group(group, len(words))
    basis_for(words, group)
    xs = real_array(coefficients, "coefficients", 1)
    if len(xs) != len(group):
        raise ValueError("Outcome coefficients must match the group")
    n = len(words[0])
    indices = np.arange(2**n, dtype=int)
    parity = np.array([(-1) ** i.bit_count() for i in range(2**n)])
    scores = np.zeros(2**n)
    for i, coefficient in zip(group, xs, strict=True):
        mask = sum(
            1 << (n - 1 - q) for q, axis in enumerate(words[i]) if axis != "I"
        )
        scores += coefficient * parity[indices & mask]
    return scores


def integer_counts(
    allocation: ArrayLike, coefficients: Sequence[ArrayLike], budget: int
) -> IntArray:
    """Round shot fractions while retaining every nonzero estimator block.

    Args:
        allocation: Nonnegative weights of shape ``(n_groups,)``. Weights
            are copied and normalized internally.
        coefficients: Real local coefficient vectors in group order.
        budget: Nonnegative integer number of shots.

    Returns:
        Integer counts of shape ``(n_groups,)`` summing to ``budget``.
        Every nonzero coefficient block receives at least one shot.

    Raises:
        ValueError: Weights, shapes or budget differ from the input contract.

    Examples:
        >>> integer_counts([2.0, 1.0], [[1.0], [1.0]], 11).tolist()
        [7, 4]
    """
    budget = integer(budget, "budget")
    a = real_array(allocation, "allocation", 1)
    if len(a) != len(coefficients) or np.any(a < 0):
        raise ValueError(
            "Allocation must match nonnegative coefficient blocks"
        )
    xs = [real_array(x, "coefficient block", 1) for x in coefficients]
    if any(a[i] == 0 and np.any(x) for i, x in enumerate(xs)):
        raise ValueError("A nonzero coefficient block requires allocation")
    if not len(a):
        if budget:
            raise ValueError("An empty design has zero measurement shots")
        return np.zeros(0, dtype=np.int64)
    if not np.isfinite(a.sum()) or a.sum() <= 0:
        raise ValueError("Allocation must have a finite positive total")
    a /= a.sum()
    mandatory = np.array([int(np.any(x != 0)) for x in xs], dtype=np.int64)
    if budget < mandatory.sum():
        raise ValueError("Budget is smaller than the estimator support")
    raw = (budget - mandatory.sum()) * a
    extra = np.floor(raw).astype(np.int64)
    left = int(budget - mandatory.sum() - extra.sum())
    order = np.argsort(-(raw - extra), kind="stable")
    extra[order[:left]] += 1
    return mandatory + extra


def summarize(
    errors: ArrayLike, bootstrap: np.random.Generator, repetitions: int
) -> dict[str, Any]:
    """Calculate RMSE, empirical bias and a percentile bootstrap interval.

    Args:
        errors: Finite energy-estimation errors, shape (replicates,).
        bootstrap: Explicit NumPy random generator, advanced by this call.
        repetitions: Positive number of bootstrap resamples.

    Returns:
        Dictionary with rmse, empirical_bias and two-element rmse_ci95.
        All values use the input energy units. Resamples are generated in
        bounded chunks while retaining the ordered RNG stream.

    Raises:
        ValueError: Errors, generator or resample count are invalid.

    Examples:
        >>> summarize(np.array([1., -1.]), np.random.default_rng(1), 10)
        {'rmse': 1.0, 'rmse_ci95': [1.0, 1.0], 'empirical_bias': 0.0}
    """
    errors = real_array(errors, "errors", 1)
    repetitions = integer(repetitions, "repetitions", 1)
    if not len(errors) or not isinstance(bootstrap, np.random.Generator):
        raise ValueError(
            "Bootstrap requires observations and a NumPy Generator"
        )
    estimates = np.empty(repetitions)
    chunk = min(256, max(1, 2**20 // len(errors)))
    for begin in range(0, repetitions, chunk):
        end = min(repetitions, begin + chunk)
        indices = bootstrap.integers(
            len(errors), size=(end - begin, len(errors))
        )
        estimates[begin:end] = np.sqrt(np.mean(errors[indices] ** 2, axis=1))
    interval = np.quantile(estimates, [0.025, 0.975])
    return dict(
        rmse=float(np.sqrt(np.mean(errors**2))),
        rmse_ci95=interval.tolist(),
        empirical_bias=float(errors.mean()),
    )


@checkpoint_lock
def sample(
    data: Mapping[str, Any],
    design: Mapping[str, Any],
    settings: Mapping[str, Any] | SamplingConfig,
    *,
    molecule: str,
    covariance: str,
    method: str,
    checkpoint: Path | None = None,
    resume: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Sample joint energy outcomes for a frozen molecular design.

    Args:
        data: Verified molecular arrays, including reference state, means,
            coefficients and covariance provider in the same term order.
        design: Groups, local energy coefficients x and shot fractions a.
        settings: Validated sampling configuration or canonical JSON mapping.
        molecule: Dataset name used in the deterministic stream mapping.
        covariance: Design source identifier in the stream mapping.
        method: Method label in the stream mapping.
        checkpoint: Optional group-cursor, partial sums and RNG checkpoint.
        resume: Restore a matching checkpoint and skip committed groups.
        progress: Callback with completed and total groups after checkpointing.

    Returns:
        One record per shot budget containing replicate errors, actual shots,
        analytic RMSE/bias, empirical RMSE/bias and the bootstrap interval.
        Energies are in Hartree for the supplied molecular datasets.

    Raises:
        ValueError: Design, settings or checkpoint identity is invalid.
        ArithmeticError: Bitstring moments disagree with reference moments.
        FileExistsError: Fresh sampling targets an existing checkpoint.
    """
    cfg = (
        settings
        if isinstance(settings, SamplingConfig)
        else SamplingConfig.from_mapping(settings)
    )
    settings = cfg.to_dict()
    from .measurement import MeasurementDesign, PauliHamiltonian

    validated = MeasurementDesign(
        PauliHamiltonian(data["words"], data["coefficients"]),
        design["groups"],
        design["x"],
        design["a"],
    )
    groups = [list(g) for g in validated.groups]
    xs, allocation = validated.coefficients, validated.allocation
    total = np.zeros_like(data["coefficients"])
    for group, x in zip(groups, xs, strict=True):
        np.add.at(total, group, x)
    if (
        np.max(abs(total - data["coefficients"]))
        > settings["conservation_tolerance"]
    ):
        raise ArithmeticError("Design coefficients fail conservation")
    budgets, replicates = settings["budgets"], settings["replicates"]
    counts = [integer_counts(allocation, xs, budget) for budget in budgets]
    stream = stream_id(molecule, covariance, method)
    rngs = [
        np.random.default_rng(
            np.random.SeedSequence([settings["seed"], stream, b])
        )
        for b in budgets
    ]
    sums = np.zeros((len(budgets), replicates))
    variances = np.zeros(len(budgets))
    expected = 0.0
    words = data["words"]
    binding = identity(
        dict(
            algorithm=computation_identity(),
            words=words,
            coefficients=data["coefficients"],
            means=data["exact_means"],
            state_sha256=hashlib.sha256(
                np.asarray(data["state"]).tobytes()
            ).hexdigest(),
            covariance=data["reference_covariance"].fingerprint(),
            design=dict(groups=groups, x=xs, a=allocation),
            settings=settings,
            molecule=molecule,
            covariance_source=covariance,
            method=method,
        )
    )
    store = CheckpointStore(checkpoint, binding) if checkpoint else None
    cursor = 0
    if resume:
        if store is None:
            raise ValueError("Resume requires a checkpoint path")
        saved = store.load().payload
        cursor = integer(saved["cursor"], "cursor")
        sums = real_array(saved["sums"], "sums", 2)
        variances = real_array(saved["variances"], "variances", 1)
        expected = float(saved["expected"])
        if (
            cursor > len(groups)
            or sums.shape != (len(budgets), replicates)
            or variances.shape != (len(budgets),)
        ):
            raise ValueError("Sampling checkpoint shapes differ")
        if len(saved["rng_states"]) != len(rngs) or not np.isfinite(expected):
            raise ValueError("Sampling checkpoint state differs")
        for rng, state in zip(rngs, saved["rng_states"], strict=True):
            rng.bit_generator.state = state
    elif store and store.exists():
        raise FileExistsError("Sampling checkpoint exists; choose resume")
    for k, (group, coefficients) in enumerate(zip(groups, xs, strict=True)):
        if k < cursor:
            continue
        x = np.asarray(coefficients)
        probabilities = basis_distribution(
            data["state"], basis_for(words, group)
        )
        values = score_outcomes(words, group, x)
        mean = float(probabilities @ values)
        value = float(probabilities @ (values - mean) ** 2)
        expected += mean
        reference = quadratic(x, data["reference_covariance"].block(group))
        if not np.isclose(value, reference, atol=1e-8, rtol=1e-6):
            raise ArithmeticError("Bitstring variance and covariance disagree")
        if abs(mean - float(x @ data["exact_means"][group])) > 1e-8:
            raise ArithmeticError("Bitstring mean and Pauli means disagree")
        unique, inverse = np.unique(values, return_inverse=True)
        masses = np.asarray(
            np.bincount(inverse, weights=probabilities, minlength=len(unique)),
            dtype=np.float64,
        )
        masses /= masses.sum()
        for i, (n, rng) in enumerate(zip(counts, rngs, strict=True)):
            shots = int(n[k])
            if shots == 0:
                continue
            variances[i] += value / shots
            for begin in range(0, replicates, settings["replicate_chunk"]):
                end = min(replicates, begin + settings["replicate_chunk"])
                draws = rng.multinomial(shots, masses, size=end - begin)
                sums[i, begin:end] += draws @ unique / shots
        if (k + 1) % cfg.checkpoint_groups == 0 or k + 1 == len(groups):
            if store:
                store.save(
                    dict(
                        cursor=k + 1,
                        sums=sums.tolist(),
                        variances=variances.tolist(),
                        expected=expected,
                        rng_states=[rng.bit_generator.state for rng in rngs],
                    )
                )
            if progress:
                progress(k + 1, len(groups))
    target = float(data["coefficients"] @ data["exact_means"])
    bootstrap = np.random.default_rng(
        np.random.SeedSequence([settings["bootstrap_seed"], stream])
    )
    rows = []
    for budget, estimates, value, n in zip(
        budgets, sums, variances, counts, strict=True
    ):
        errors = estimates - target
        metrics = summarize(
            errors, bootstrap, settings["bootstrap_replicates"]
        )
        rows.append(
            dict(
                molecule=molecule,
                covariance=covariance,
                method=method,
                budget=budget,
                actual_shots=int(n.sum()),
                errors=errors.tolist(),
                analytic_rmse=float(np.sqrt(value + (expected - target) ** 2)),
                analytic_bias=expected - target,
                settings=int(np.count_nonzero(n)),
                accuracy_threshold=cfg.epsilon,
                rmse_ci95_below_threshold=metrics["rmse_ci95"][1]
                <= cfg.epsilon,
                **metrics,
            )
        )
    return rows
