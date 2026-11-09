"""Design and execute a frozen two-wire PennyLane energy estimator."""

from __future__ import annotations

import json

import numpy as np
import pennylane as qml
from numpy.typing import NDArray

from vgmc import StateCovariance, design_measurements
from vgmc.adapters.pennylane import from_pennylane, make_tapes


def main() -> None:
    """Print sampled energy, analytic energy and predicted standard error."""
    wires = ("a", "b")
    device = qml.device("default.qubit", wires=wires, seed=41)

    @qml.qnode(device)
    def prepare() -> NDArray[np.complex128]:
        """Return the reference state in the declared tensor-product order."""
        qml.RY(0.31, wires="a")
        qml.RX(-0.2, wires="b")
        return qml.state()

    state = prepare()
    operator = (
        0.7 * qml.X("a")
        + 0.2 * qml.Z("a") @ qml.Z("b")
        - 0.3 * qml.Y("b")
        + 0.5 * qml.I("a")
    )
    hamiltonian = from_pennylane(operator, wire_order=wires)
    covariance = StateCovariance(hamiltonian, state)
    design = design_measurements(hamiltonian, covariance, max_added=2)
    plan = design.allocate_shots(5000)
    operations = [qml.RY(0.31, wires="a"), qml.RX(-0.2, wires="b")]
    tapes, process = make_tapes(operations, plan)
    energy = process(qml.execute(tapes, device, diff_method=None))
    exact = float(
        np.vdot(state, qml.matrix(operator, wire_order=wires) @ state).real
    )
    stderr = float(np.sqrt(plan.variance(covariance)))
    print(
        json.dumps(
            dict(
                energy=energy,
                exact_energy=exact,
                predicted_standard_error=stderr,
                actual_shots=plan.total_shots,
                bases=design.bases,
                status=design.status,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
