"""PennyLane Pauli conversion and frozen product-basis measurement tapes."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any, TYPE_CHECKING

import numpy as np

from ..measurement import PauliHamiltonian, ShotPlan

if TYPE_CHECKING:
    from pennylane.operation import Operator
    from pennylane.tape import QuantumScript


def _qml() -> Any:
    """Import the optional PennyLane dependency at the adapter boundary."""
    try:
        import pennylane as qml
    except ImportError as error:
        raise ImportError(
            "Install vgmc[pennylane] to use this adapter"
        ) from error
    return qml


def from_pennylane(
    operator: Operator, *, wire_order: Sequence[Hashable] | None = None
) -> PauliHamiltonian:
    """Convert a real PennyLane Pauli operator and preserve its wire mapping.

    Args:
        operator: PennyLane operator expressible as a PauliSentence.
        wire_order: Explicit tensor-product order. Extra idle wires are
            allowed. The default is ``operator.wires`` order.

    Returns:
        Hamiltonian with duplicate Pauli terms combined and the identity
        coefficient stored separately as ``constant``. Exactly zero summed
        terms are removed during PauliSentence canonicalization.

    Raises:
        ValueError: Coefficients are non-real or wires are missing/duplicated.

    Examples:
        >>> import pennylane as qml
        >>> h = from_pennylane(2 * qml.X("a") + 3 * qml.I("a"))
        >>> h.words, h.constant, h.wire_order
        (('X',), 3.0, ('a',))
    """
    qml = _qml()
    wires = tuple(operator.wires) if wire_order is None else tuple(wire_order)
    if len(set(wires)) != len(wires) or not set(operator.wires).issubset(
        wires
    ):
        raise ValueError("Wire order must contain each operator wire once")
    sentence = qml.pauli.pauli_sentence(operator)
    combined: dict[str, float] = {}
    constant = 0.0
    for pauli_word, coefficient in sentence.items():
        scalar = np.asarray(coefficient)
        if scalar.ndim != 0 or scalar.dtype.kind not in "iufc":
            raise ValueError(
                "Operator coefficients must be concrete real scalars"
            )
        value = complex(scalar)
        if not np.isfinite(value) or value.imag != 0:
            raise ValueError("Operator coefficients must be finite and real")
        word = "".join(pauli_word.get(wire, "I") for wire in wires)
        if not pauli_word:
            constant += value.real
        else:
            combined[word] = combined.get(word, 0.0) + value.real
    words = tuple(word for word, value in combined.items() if value != 0)
    return PauliHamiltonian(
        words,
        [combined[word] for word in words],
        constant=constant,
        wire_order=wires,
    )


def to_pennylane(hamiltonian: PauliHamiltonian) -> Operator:
    """Convert a declared wire-ordered Hamiltonian back to PennyLane.

    Args:
        hamiltonian: Backend-neutral real Pauli representation.

    Returns:
        A PennyLane operator with the same matrix in ``wire_order``.

    Raises:
        ValueError: A constant Hamiltonian has no declared wire.
    """
    qml = _qml()
    wires = hamiltonian.wire_order or ()
    if not wires:
        raise ValueError(
            "A PennyLane constant operator requires a declared wire"
        )
    coefficients = []
    observables = []
    with qml.QueuingManager.stop_recording():
        for word, coefficient in zip(
            hamiltonian.words, hamiltonian.coefficients, strict=True
        ):
            factors = [
                getattr(qml, axis)(wire)
                for axis, wire in zip(word, wires, strict=True)
                if axis != "I"
            ]
            observables.append(
                qml.prod(*factors) if len(factors) > 1 else factors[0]
            )
            coefficients.append(float(coefficient))
        if hamiltonian.constant != 0 or not observables:
            observables.append(qml.I(wires[0]))
            coefficients.append(hamiltonian.constant)
        return qml.dot(coefficients, observables)


def make_tapes(
    operations: Sequence[Operator], plan: ShotPlan
) -> tuple[list[QuantumScript], Callable[[Sequence[Any]], float]]:
    """Create one finite-shot PennyLane tape per sampled product basis.

    Args:
        operations: State-preparation operations in execution order. Their
            wires must belong to the plan's declared wire order.
        plan: Frozen coefficients, integer counts and framework wire labels.

    Returns:
        ``(tapes, postprocess)``. Execute the tapes on a caller-owned device
        with ``qml.execute(tapes, device, diff_method=None)``. Pass those
        results to ``postprocess`` to obtain scalar energy including the
        constant contribution.

    Raises:
        ValueError: State-preparation wires or returned sample shapes differ.

    The classical allocation decision is frozen before device execution.
    Device selection, circuit differentiation and parameter updates remain
    with the caller.
    """
    qml = _qml()
    wires = tuple(plan.design.hamiltonian.wire_order or ())
    if any(
        not set(operation.wires).issubset(wires) for operation in operations
    ):
        raise ValueError("Preparation operations must use the declared wires")
    tapes = []
    measured_groups = []
    with qml.QueuingManager.stop_recording():
        for index, (basis, count) in enumerate(
            zip(plan.design.bases, np.asarray(plan.counts), strict=True)
        ):
            if count == 0:
                continue
            rotated = list(operations)
            for axis, wire in zip(basis, wires, strict=True):
                if axis == "Y":
                    rotated.append(qml.adjoint(qml.S)(wire))
                if axis in "XY":
                    rotated.append(qml.Hadamard(wire))
            tapes.append(
                qml.tape.QuantumScript(
                    rotated, [qml.sample(wires=wires)], shots=int(count)
                )
            )
            measured_groups.append(index)

    def postprocess(results: Sequence[Any]) -> float:
        """Restore explicit shot/wire axes and apply the frozen estimator."""
        if len(results) != len(measured_groups):
            raise ValueError("Results must match the emitted tape count")
        samples: list[Any] = [None] * len(plan.design.groups)
        for index, raw in zip(measured_groups, results, strict=True):
            count = int(np.asarray(plan.counts)[index])
            array = np.asarray(raw)
            expected = (count, len(wires))
            if array.shape != expected:
                squeezed = tuple(axis for axis in expected if axis != 1)
                if array.shape != squeezed:
                    raise ValueError("PennyLane sample shot/wire axes differ")
                array = array.reshape(expected)
            samples[index] = array
        return plan.estimate(samples)

    return tapes, postprocess
