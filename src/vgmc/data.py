"""Verified reproduction inputs and frozen coefficient/allocation designs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, TypedDict

import numpy as np

from ._validation import FloatArray, index_group, real_array
from .artifacts import file_hash, read_json, safe_path, write_json
from .covariance import Covariance
from .grouping import validate_words
from .measurement import MeasurementDesign, PauliHamiltonian
from .state import ComplexArray

__all__ = [
    "ExperimentInput",
    "load_design",
    "load_input",
    "read_json",
    "verify_file",
    "write_json",
]


class ExperimentInput(TypedDict):
    """Typed molecular arrays in their frozen Hamiltonian term order."""

    words: list[str]
    coefficients: FloatArray
    state: ComplexArray
    cisd_state: ComplexArray
    exact_means: FloatArray
    cisd_means: FloatArray
    design_covariance: Covariance
    reference_covariance: Covariance
    library: dict[str, list[list[int]]]
    initial_a: FloatArray


def verify_file(root: Path, relative: str | Path) -> Path:
    """Check a selected repository input against ``DATA_MANIFEST.json``.

    Args:
        root: Explicit reproduction repository root.
        relative: Root-relative input path.

    Returns:
        Ordinary file path after SHA-256 verification.

    Raises:
        FileNotFoundError: The input or manifest is absent.
        ValueError: Path scope, manifest membership or checksum differs.
    """
    path = safe_path(root, relative)
    manifest = read_json(safe_path(root, "DATA_MANIFEST.json"))
    hashes = manifest.get("sha256", {})
    expected = hashes.get(str(Path(relative)))
    if expected is None:
        raise ValueError("Input is outside the registered file manifest")
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_hash(path) != expected:
        raise ValueError(f"Input checksum differs: {relative}")
    return path


def _molecule_name(molecule: str) -> str:
    """Validate a flat reproduction dataset identifier."""
    if not isinstance(molecule, str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_-]*", molecule
    ):
        raise ValueError("Molecule must be a flat dataset identifier")
    return molecule


def load_input(
    root: Path, molecule: str, covariance: str = "exact"
) -> ExperimentInput:
    """Read verified molecular inputs and reconstruct symmetric covariance.

    Args:
        root: Reproduction repository containing data and its manifest.
        molecule: Registered molecular dataset, such as ``H2`` or ``LiH``.
        covariance: ``exact`` or ``cisd`` design covariance source. Reference
            variance always uses the supplied exact covariance.

    Returns:
        Typed input with coefficients ``(n_terms,)``, pure states
        ``(2 ** n_qubits,)``, Pauli means, covariance providers and group
        libraries. Input arrays and provider matrices are read-only.

    Raises:
        ValueError: Identity, shape, numerical values or covariance differ.
        FileNotFoundError: A registered file is missing.
    """
    molecule = _molecule_name(molecule)
    if covariance not in ("exact", "cisd"):
        raise ValueError("Covariance must be exact or cisd")
    directory = Path("data/inputs") / molecule
    paths = {
        name: verify_file(root, directory / name)
        for name in (
            "state.npz",
            "covariance.npz",
            "library.json",
            "warm.json",
        )
    }
    with np.load(paths["state.npz"], allow_pickle=False) as archive:
        if set(archive.files) != {
            "paulis",
            "coefficients",
            "state",
            "cisd_state",
            "exact_means",
            "cisd_means",
        }:
            raise ValueError("Molecular input fields differ")
        raw = {key: archive[key] for key in archive.files}
    words = raw.pop("paulis").tolist()
    validate_words(words)
    n = len(words)
    c = real_array(raw["coefficients"], "coefficients", 1)
    exact_means = real_array(raw["exact_means"], "exact_means", 1)
    cisd_means = real_array(raw["cisd_means"], "cisd_means", 1)
    if any(value.shape != (n,) for value in (c, exact_means, cisd_means)):
        raise ValueError("Molecular term axes differ")
    states = []
    for key in ("state", "cisd_state"):
        state = np.asarray(raw[key])
        if (
            state.shape != (2 ** len(words[0]),)
            or not np.isfinite(state).all()
        ):
            raise ValueError("Molecular state shape or values differ")
        if abs(float(np.vdot(state, state).real) - 1) > 1e-10:
            raise ValueError("Molecular state must be normalized")
        states.append(np.array(state, dtype=np.complex128, copy=True))
    with np.load(paths["covariance.npz"], allow_pickle=False) as archive:
        if set(archive.files) != {"exact_upper", "cisd_upper"}:
            raise ValueError("Covariance archive fields differ")
        indices = np.triu_indices(n)

        def matrix(label: str) -> Covariance:
            """Recover a symmetric matrix from its stored upper triangle."""
            values = archive[label + "_upper"]
            if values.shape != (n * (n + 1) // 2,):
                raise ValueError("Covariance triangle axis differs")
            value = np.empty((n, n))
            value[indices] = values
            value[(indices[1], indices[0])] = values
            return Covariance(value)

        exact = matrix("exact")
        design = exact if covariance == "exact" else matrix("cisd")
    library = read_json(paths["library.json"])
    if set(library) != {"groups", "si", "lf"}:
        raise ValueError("Initial library fields differ")
    library = {
        key: [index_group(group, n) for group in groups]
        for key, groups in library.items()
    }
    initial_a = real_array(
        read_json(paths["warm.json"])[covariance], "initial_a", 1
    )
    if (
        initial_a.shape != (len(library["groups"]),)
        or np.any(initial_a < 0)
        or not np.isclose(initial_a.sum(), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("Initial allocation axes or normalization differ")
    for array in (c, exact_means, cisd_means, initial_a, *states):
        array.flags.writeable = False
    return ExperimentInput(
        words=words,
        coefficients=c,
        state=states[0],
        cisd_state=states[1],
        exact_means=exact_means,
        cisd_means=cisd_means,
        design_covariance=design,
        reference_covariance=exact,
        library=library,
        initial_a=initial_a,
    )


def load_design(
    root: Path,
    molecule: str,
    covariance: str,
    method: str,
    *,
    data: ExperimentInput | None = None,
) -> dict[str, Any]:
    """Read and validate a registered frozen E2 estimator.

    Args:
        root: Reproduction repository root.
        molecule: Registered molecular identifier.
        covariance: ``exact`` or ``cisd`` design source.
        method: ``ICS``, ``SOCP`` or ``VGMC``.
        data: Previously loaded matching inputs, reused across methods.

    Returns:
        Frozen result dictionary with validated ``groups``, ``x`` and ``a``.

    Raises:
        ValueError: Identity, support, allocation or coefficient recovery
        differs.
    """
    molecule = _molecule_name(molecule)
    if covariance not in ("exact", "cisd") or method not in (
        "ICS",
        "SOCP",
        "VGMC",
    ):
        raise ValueError("Unknown frozen design selection")
    path = verify_file(
        root,
        Path("data/designs") / f"{molecule}_{covariance}_{method}.json.gz",
    )
    result = read_json(path)
    if any(
        result.get(key) != value
        for key, value in (
            ("molecule", molecule),
            ("covariance", covariance),
            ("method", method),
        )
    ):
        raise ValueError("Frozen design metadata differs")
    inputs = data or load_input(root, molecule, covariance)
    h = PauliHamiltonian(tuple(inputs["words"]), inputs["coefficients"])
    MeasurementDesign(
        h, tuple(map(tuple, result["groups"])), tuple(result["x"]), result["a"]
    )
    return result
