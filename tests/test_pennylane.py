"""Actual PennyLane conversion, tape execution and sample postprocessing."""

import numpy as np
import pytest

from vgmc import (
    MeasurementDesign,
    SolverConfig,
    StateCovariance,
    design_measurements,
)
from vgmc.adapters.pennylane import from_pennylane, make_tapes, to_pennylane


qml = pytest.importorskip("pennylane")


def test_roundtrip_preserves_noncontiguous_wire_order_and_identity() -> None:
    operator = (
        0.7 * qml.X("left") @ qml.Y(9) + 0.3 * qml.Z(9) + 2 * qml.I("left")
    )
    wires = (9, "idle", "left")
    h = from_pennylane(operator, wire_order=wires)
    assert h.constant == 2
    assert h.wire_order == wires
    np.testing.assert_allclose(
        qml.matrix(to_pennylane(h), wire_order=wires),
        qml.matrix(operator, wire_order=wires),
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize(
    "axis,state",
    [
        ("X", np.array([1, 1]) / np.sqrt(2)),
        ("Y", np.array([1, 1j]) / np.sqrt(2)),
        ("Z", np.array([1, 0])),
    ],
)
def test_basis_rotation_has_correct_sign(axis: str, state: np.ndarray) -> None:
    operator = 2 * getattr(qml, axis)("q") + 3 * qml.I("q")
    h = from_pennylane(operator)
    d = MeasurementDesign(h, ((0,),), ([2],), [1])
    for shots in (1, 8):
        tapes, process = make_tapes(
            [qml.StatePrep(state, wires=["q"])], d.allocate_shots(shots)
        )
        results = qml.execute(
            tapes,
            qml.device("default.qubit", wires=["q"], seed=12),
            diff_method=None,
        )
        assert process(results) == 5


def test_frozen_estimator_executes_with_multiple_wires() -> None:
    h = from_pennylane(
        qml.Z("a") + 2 * qml.Z("b") + 4 * qml.I("a"), wire_order=["b", "a"]
    )
    provider = StateCovariance(h, [1, 0, 0, 0])
    design = design_measurements(h, provider, config=SolverConfig(ridge=1e-8))
    tapes, process = make_tapes([], design.allocate_shots(16))
    results = qml.execute(
        tapes,
        qml.device("default.qubit", wires=["a", "b"], seed=8),
        diff_method=None,
    )
    assert process(results) == 7


def test_adapter_rejects_nonreal_and_missing_wires() -> None:
    with pytest.raises(ValueError):
        from_pennylane(1j * qml.X(0))
    with pytest.raises(ValueError):
        from_pennylane(qml.X(0), wire_order=[1])


def test_constant_tapes_and_bad_sample_shapes() -> None:
    h = from_pennylane(4 * qml.I("q"))
    design = design_measurements(h, np.empty((0, 0)))
    tapes, process = make_tapes([], design.allocate_shots(10))
    assert tapes == []
    assert process([]) == 4
