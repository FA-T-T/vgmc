"""Framework-neutral Pauli Hamiltonians, designs and finite-shot plans."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._records import SolverResult
from ._validation import FloatArray, IntArray, index_group, integer, real_array
from .covariance import CovarianceProvider, quadratic
from .grouping import basis_for, validate_words
from .sampling import integer_counts


@dataclass(frozen=True, init=False)
class PauliHamiltonian:
    """Represent a real Pauli Hamiltonian with an explicit wire order.

    Args:
        words: Unique nonidentity Pauli strings. Character zero corresponds
            to ``wire_order[0]`` and the most significant state-vector bit.
        coefficients: Real energy coefficients of shape ``(n_terms,)``.
        constant: Identity coefficient in the same energy units.
        wire_order: Unique hashable framework wire labels. Defaults to
            ``range(n_qubits)`` for a nonempty term list.

    Examples:
        >>> h = PauliHamiltonian(("XI", "ZZ"), [0.5, -1.0], constant=2.0)
        >>> h.wire_order
        (0, 1)
        >>> h.n_terms
        2
    """

    words: tuple[str, ...]
    coefficients: FloatArray
    constant: float
    wire_order: tuple[Hashable, ...]

    def __init__(
        self,
        words: Sequence[str],
        coefficients: ArrayLike,
        constant: float = 0.0,
        wire_order: Sequence[Hashable] | None = None,
    ) -> None:
        """Copy numerical input and validate Pauli and wire identities."""
        words = tuple(words)
        c = real_array(coefficients, "coefficients", 1)
        constant = real_array([constant], "constant", 1)[0]
        if len(words) != len(c):
            raise ValueError("Words and coefficients must share term order")
        if words:
            validate_words(list(words))
            if any(set(word) == {"I"} for word in words):
                raise ValueError("Store the identity coefficient in constant")
        wires = wire_order
        if wires is None:
            wires = tuple(range(len(words[0]))) if words else ()
        wires = tuple(wires)
        if len(set(wires)) != len(wires):
            raise ValueError("Wire labels must be unique")
        if words and len(wires) != len(words[0]):
            raise ValueError("Wire order and Pauli width differ")
        c.flags.writeable = False
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "coefficients", c)
        object.__setattr__(self, "constant", float(constant))
        object.__setattr__(self, "wire_order", wires)

    @property
    def n_terms(self) -> int:
        """Number of nonidentity Pauli terms."""
        return len(self.words)

    @property
    def n_qubits(self) -> int:
        """Number of wires in the declared tensor-product order."""
        return len(self.wire_order or ())

    def to_record(self) -> dict[str, Any]:
        """Serialize coefficients and serializable wire labels for storage."""
        return dict(
            words=list(self.words),
            coefficients=np.asarray(self.coefficients).tolist(),
            constant=self.constant,
            wire_order=list(self.wire_order or ()),
        )


@dataclass(frozen=True, init=False)
class MeasurementDesign:
    """Freeze group coefficients and continuous shot allocations.

    Args:
        hamiltonian: Global coefficient and wire identities.
        groups: QWC term-index tuples. Overlap between groups is allowed.
        coefficients: One real energy coefficient vector per group.
        allocation: Nonnegative group fractions that sum to one.
        status: Solver or supplied-design status retained for inspection.
        gap: Optional numerical relative gap reported by the solver.

    The constructor checks coefficient recovery independently of the solver.
    Arrays are copied and made read-only.
    """

    hamiltonian: PauliHamiltonian
    groups: tuple[tuple[int, ...], ...]
    coefficients: tuple[FloatArray, ...]
    allocation: FloatArray
    status: str
    gap: float | None

    def __init__(
        self,
        hamiltonian: PauliHamiltonian,
        groups: Sequence[Sequence[int]],
        coefficients: Sequence[ArrayLike],
        allocation: ArrayLike,
        status: str = "provided",
        gap: float | None = None,
    ) -> None:
        """Validate QWC groups, the simplex and Hamiltonian recovery."""
        h = hamiltonian
        groups = tuple(tuple(index_group(g, h.n_terms)) for g in groups)
        xs = tuple(real_array(x, "coefficient block", 1) for x in coefficients)
        a = real_array(allocation, "allocation", 1)
        if len(groups) != len(xs) or a.shape != (len(groups),):
            raise ValueError("Group, coefficient and allocation axes differ")
        if groups and (
            np.any(a < 0) or not np.isclose(a.sum(), 1, atol=1e-12, rtol=0)
        ):
            raise ValueError("Allocation must lie on the nonnegative simplex")
        if h.n_terms and not groups:
            raise ValueError("Nonidentity Hamiltonian terms require groups")
        recovered = np.zeros(h.n_terms)
        for group, x, weight in zip(groups, xs, a, strict=True):
            basis_for(list(h.words), list(group))
            if len(x) != len(group) or (weight == 0 and np.any(x)):
                raise ValueError(
                    "Coefficient block and sampled support differ"
                )
            np.add.at(recovered, list(group), x)
            x.flags.writeable = False
        c = np.asarray(h.coefficients)
        scale = max(1.0, float(np.max(abs(c)))) if len(c) else 1.0
        if not np.allclose(recovered, c, atol=1e-9 * scale, rtol=0):
            raise ValueError("Local coefficients must recover the Hamiltonian")
        if gap is not None and (not np.isfinite(gap) or gap < 0):
            raise ValueError("A reported numerical gap must be nonnegative")
        a.flags.writeable = False
        object.__setattr__(self, "hamiltonian", h)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "gap", gap)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "coefficients", xs)
        object.__setattr__(self, "allocation", a)

    @classmethod
    def from_result(
        cls, hamiltonian: PauliHamiltonian, result: SolverResult
    ) -> MeasurementDesign:
        """Validate a numerical solver result as a portable measurement
        design.
        """
        return cls(
            hamiltonian,
            tuple(map(tuple, result["groups"])),
            tuple(result["x"]),
            result["a"],
            result["status"],
            result["gap"],
        )

    @property
    def bases(self) -> tuple[str, ...]:
        """Complete product bases with unused wire axes filled by Z."""
        return tuple(
            basis_for(list(self.hamiltonian.words), list(g))
            for g in self.groups
        )

    def allocate_shots(self, budget: int) -> ShotPlan:
        """Create an integer measurement plan from a total shot budget.

        Args:
            budget: Nonnegative integer shot budget. Each nonzero estimator
                block requires one shot. An exactly constant estimator uses
                zero quantum shots.

        Returns:
            A plan with frozen integer counts and an energy postprocessor.

        Examples:
            >>> h = PauliHamiltonian(("X",), [2.0], constant=1.0)
            >>> design = MeasurementDesign(h, ((0,),), ([2.0],), [1.0])
            >>> design.allocate_shots(5).counts.tolist()
            [5]
        """
        integer(budget, "budget")
        if not any(np.any(x) for x in self.coefficients):
            return ShotPlan(self, np.zeros(len(self.groups), dtype=np.int64))
        return ShotPlan(
            self, integer_counts(self.allocation, self.coefficients, budget)
        )

    def to_record(self) -> dict[str, Any]:
        """Serialize the complete frozen estimator for framework exchange."""
        return dict(
            hamiltonian=self.hamiltonian.to_record(),
            groups=[list(g) for g in self.groups],
            x=[np.asarray(x).tolist() for x in self.coefficients],
            a=np.asarray(self.allocation).tolist(),
            status=self.status,
            gap=self.gap,
        )


@dataclass(frozen=True, init=False)
class ShotPlan:
    """Associate a frozen estimator with integer group shot counts.

    Args:
        design: Continuous estimator and wire identities.
        counts: Nonnegative integer array of shape ``(n_groups,)``.
    """

    design: MeasurementDesign
    counts: IntArray

    def __init__(self, design: MeasurementDesign, counts: ArrayLike) -> None:
        """Copy counts and check support of every nonzero coefficient block."""
        object.__setattr__(self, "design", design)
        raw = np.asarray(counts)
        if raw.ndim != 1 or len(raw) != len(self.design.groups):
            raise ValueError("Shot counts must match the group axis")
        counts = np.array(
            [integer(n, "shot count") for n in raw], dtype=np.int64
        )
        if any(
            n == 0 and np.any(x)
            for n, x in zip(counts, self.design.coefficients, strict=True)
        ):
            raise ValueError("Nonzero estimator blocks require positive shots")
        counts.flags.writeable = False
        object.__setattr__(self, "counts", counts)

    @property
    def total_shots(self) -> int:
        """Actual number of joint bitstring measurements in the plan."""
        return int(np.asarray(self.counts).sum())

    def estimate(self, samples: Sequence[ArrayLike | None]) -> float:
        """Estimate energy from joint bitstrings in the declared wire order.

        Args:
            samples: One binary array of shape ``(counts[g], n_qubits)`` per
                group. A zero-shot group accepts ``None``.

        Returns:
            Scalar energy including the Hamiltonian's identity coefficient.

        Raises:
            ValueError: Group count, shot count, wire axis or bit values
            differ.

        Examples:
            >>> h = PauliHamiltonian(("Z",), [2.0], constant=1.0)
            >>> d = MeasurementDesign(h, ((0,),), ([2.0],), [1.0])
            >>> d.allocate_shots(2).estimate([[[0], [1]]])
            1.0
        """
        if len(samples) != len(self.design.groups):
            raise ValueError("Sample groups must match the measurement plan")
        h = self.design.hamiltonian
        energy = h.constant
        for group, xs, count, sample in zip(
            self.design.groups,
            self.design.coefficients,
            np.asarray(self.counts),
            samples,
            strict=True,
        ):
            if count == 0:
                if sample is not None and np.asarray(sample).shape != (
                    0,
                    h.n_qubits,
                ):
                    raise ValueError(
                        "A zero-shot group requires empty samples"
                    )
                continue
            bits = np.asarray(sample)
            if (
                bits.shape != (count, h.n_qubits)
                or bits.dtype.kind not in "biuf"
            ):
                raise ValueError(
                    "Samples must have explicit shot and wire axes"
                )
            if not np.all((bits == 0) | (bits == 1)):
                raise ValueError("Samples must contain binary bit values")
            for index, coefficient in zip(group, xs, strict=True):
                axes = [
                    i for i, axis in enumerate(h.words[index]) if axis != "I"
                ]
                parity = 1 - 2 * (bits[:, axes].sum(axis=1) % 2)
                energy += float(coefficient) * float(parity.mean())
        return float(energy)

    def variance(self, covariance: CovarianceProvider) -> float:
        """Evaluate integer-plan energy variance from reference covariance."""
        if covariance.n_terms != self.design.hamiltonian.n_terms:
            raise ValueError("Covariance term identity differs")
        total = 0.0
        for group, x, count in zip(
            self.design.groups,
            self.design.coefficients,
            np.asarray(self.counts),
            strict=True,
        ):
            if count:
                total += quadratic(x, covariance.block(group)) / count
        return float(total)
