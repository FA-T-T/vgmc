# Sources

## Molecular Hamiltonians

The five molecular Hamiltonians are supplied by the original authors in the repository associated with “Deterministic improvements of quantum measurements with grouping of compatible operators, non-local transformations, and covariance estimates”. [Author repository](https://github.com/ThomsonYen/VQE_Hamiltonian), [source publication](https://doi.org/10.1038/s41534-023-00683-y)

The molecular inputs use the Bravyi–Kitaev transform of the supplied fermionic operators.

| Molecule | Fermionic source | Qubits | Nonidentity Pauli terms |
| --- | --- | --- | --- |
| H2 | [ham_lib/h2_fer.bin](https://github.com/ThomsonYen/VQE_Hamiltonian/blob/7e7a7cd672890ea24fa8c9a9387b0514426994cb/ham_lib/h2_fer.bin) | 4 | 14 |
| LiH | [ham_lib/lih_fer.bin](https://github.com/ThomsonYen/VQE_Hamiltonian/blob/7e7a7cd672890ea24fa8c9a9387b0514426994cb/ham_lib/lih_fer.bin) | 12 | 630 |
| BeH2 | [ham_lib/beh2_fer.bin](https://github.com/ThomsonYen/VQE_Hamiltonian/blob/7e7a7cd672890ea24fa8c9a9387b0514426994cb/ham_lib/beh2_fer.bin) | 14 | 665 |
| H2O | [ham_lib/h2o_fer.bin](https://github.com/ThomsonYen/VQE_Hamiltonian/blob/7e7a7cd672890ea24fa8c9a9387b0514426994cb/ham_lib/h2o_fer.bin) | 14 | 1085 |
| NH3 | [ham_lib/nh3_fer.bin](https://github.com/ThomsonYen/VQE_Hamiltonian/blob/7e7a7cd672890ea24fa8c9a9387b0514426994cb/ham_lib/nh3_fer.bin) | 16 | 3608 |

The Pauli strings, real coefficients and states are stored in `data/inputs/<molecule>/state.npz`. Pauli character zero corresponds to the most significant state-vector bit. Coefficients use Hartree. The identity coefficient is stored separately in `data/hamiltonians.json` and adds to the nonidentity energy estimate. The same file records the source hashes, packaged input hashes and electron counts.

## Measurement methods

| Component | Primary source | Use |
| --- | --- | --- |
| VGMC | This repository. | Fixed-library coefficient recovery, allocation optimization and finite-pool group construction. |
| OGM | Wu et al., [Quantum 7, 896 (2023)](https://doi.org/10.22331/q-2023-01-13-896), [reference implementation](https://github.com/GillianOoO/Overlapped-grouping-measurement/tree/0d8753ab35a3dd3f04901e88b6df35abd8c5cbca). | Python reproduction of construction, fitting and pooled estimation. |
| ICS | [Tequila commit 93c361b](https://github.com/tequilahub/tequila/tree/93c361b6821ae6790fd0fc3d0b7a0ebfaa312be4). | External implementation called by the adapter. |
| LF | Verteletskyi, Yen and Izmaylov, [J. Chem. Phys. 152, 124114 (2020)](https://doi.org/10.1063/1.5141458). | Conflict-graph coloring with index tie order. |
| SI | Crawford et al., [Quantum 5, 385 (2021)](https://doi.org/10.22331/q-2021-01-20-385). | Coefficient-sorted insertion. |

## PennyLane

The adapter uses PennyLane 0.45.1 and its Pauli conversion, tape and execution interfaces.

- [PauliSentence conversion](https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli.pauli_sentence.html).
- [QuantumScript](https://docs.pennylane.ai/en/stable/code/api/pennylane.tape.QuantumScript.html).
- [Tape execution](https://docs.pennylane.ai/en/stable/code/api/pennylane.execute.html).
