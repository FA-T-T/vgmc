"""QWC compatibility, product bases and reproduced SI and LF partitions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from ._validation import BoolArray, index_group, real_array


def validate_words(words: list[str]) -> None:
    """Validate unique Pauli strings in a common tensor order.

    Args:
        words: Nonempty list of equal-width strings over I, X, Y and Z.

    Raises:
        ValueError: The list, alphabet, width or uniqueness is invalid.

    Examples:
        >>> validate_words(["XI", "ZZ"])
    """
    if not words or not words[0] or len(set(words)) != len(words):
        raise ValueError("Pauli words must be nonempty and unique")
    if any(len(w) != len(words[0]) or set(w) - set("IXYZ") for w in words):
        raise ValueError("Invalid Pauli words")


def compatible(left: str, right: str) -> bool:
    """Test qubit-wise commutation of two Pauli words.

    Args:
        left: First IXYZ string.
        right: Second IXYZ string in the same wire order.

    Returns:
        True when widths agree and each axis agrees or contains an identity.

    Examples:
        >>> compatible("XI", "XZ"), compatible("X", "Y")
        (True, False)
    """
    return len(left) == len(right) and all(
        a == "I" or b == "I" or a == b
        for a, b in zip(left, right, strict=True)
    )


def basis_for(
    words: list[str], group: list[int], unspecified_axis: str = "Z"
) -> str:
    """Complete the product basis of a QWC group.

    Args:
        words: Valid Pauli words in Hamiltonian order.
        group: Distinct integer term indices; an empty group is allowed.
        unspecified_axis: X, Y or Z used on axes containing only identities.

    Returns:
        Complete XYZ string in the original wire order.

    Raises:
        ValueError: Indices, completion axis or QWC compatibility are invalid.

    Examples:
        >>> basis_for(["XI", "IZ"], [0, 1])
        'XZ'
    """
    if (
        not words
        or len(unspecified_axis) != 1
        or unspecified_axis not in "XYZ"
    ):
        raise ValueError("A Pauli list and one completion axis are required")
    group = index_group(group, len(words), empty=True)
    basis = ["I"] * len(words[0])
    for i in group:
        if not compatible("".join(basis), words[i]):
            raise ValueError("Group must be QWC")
        basis = [
            a if b == "I" else b for a, b in zip(basis, words[i], strict=True)
        ]
    return "".join(unspecified_axis if axis == "I" else axis for axis in basis)


def visible(words: list[str], basis: str) -> list[int]:
    """List Hamiltonian terms covered by a complete product basis.

    Args:
        words: Valid Pauli words in Hamiltonian order.
        basis: Complete XYZ string with the same width.

    Returns:
        Covered indices in ascending Hamiltonian index order.

    Examples:
        >>> visible(["XI", "IZ", "YY"], "XZ")
        [0, 1]
    """
    if not words or len(basis) != len(words[0]) or set(basis) - set("XYZ"):
        raise ValueError("Visibility requires a complete product basis")
    return [i for i, word in enumerate(words) if compatible(word, basis)]


def compatibility_matrix(words: list[str]) -> BoolArray:
    """Construct the pairwise QWC compatibility matrix.

    Args:
        words: Unique equal-width Pauli words.

    Returns:
        Boolean array of shape (n_terms, n_terms), including a true diagonal.

    Raises:
        ValueError: Pauli word validation fails.

    Examples:
        >>> compatibility_matrix(["X", "Z"]).tolist()
        [[True, False], [False, True]]
    """
    validate_words(words)
    encoded = np.array(
        [["IXYZ".index(a) for a in w] for w in words], dtype=np.uint8
    )
    matrix = np.ones((len(words), len(words)), dtype=bool)
    for axis in encoded.T:
        matrix &= (
            (axis[:, None] == 0)
            | (axis[None, :] == 0)
            | (axis[:, None] == axis[None, :])
        )
    return matrix


def sorted_insertion(
    words: list[str], coefficients: ArrayLike
) -> list[list[int]]:
    """Insert terms in decreasing absolute coefficient order.

    Args:
        words: Unique equal-width Pauli words.
        coefficients: Finite real energy coefficients, shape (n_terms,).

    Returns:
        Disjoint QWC index groups. Index order breaks coefficient ties;
        each term enters the first compatible group.

    Raises:
        ValueError: Words or coefficient axes are invalid.

    Examples:
        >>> sorted_insertion(["XI", "IZ", "ZZ"], np.array([1., 2., 3.]))
        [[2, 1], [0]]
    """
    validate_words(words)
    coefficients = real_array(coefficients, "coefficients", 1)
    if coefficients.shape != (len(words),):
        raise ValueError("Coefficients must match Pauli word order")
    groups: list[list[int]] = []
    for i in sorted(
        range(len(words)), key=lambda j: (-abs(coefficients[j]), j)
    ):
        for group in groups:
            if all(compatible(words[i], words[j]) for j in group):
                group.append(i)
                break
        else:
            groups.append([i])
    return groups


def largest_first(words: list[str]) -> list[list[int]]:
    """Color the QWC conflict graph in decreasing degree order.

    Args:
        words: Unique equal-width Pauli words.

    Returns:
        Disjoint QWC index groups. Input index breaks degree ties, and
        the first available color receives each term.

    Raises:
        ValueError: Pauli word validation fails.

    Examples:
        >>> largest_first(["X", "Y", "Z"])
        [[0], [1], [2]]
    """
    adjacency = ~compatibility_matrix(words)
    degrees = adjacency.sum(axis=1)
    order = sorted(range(len(words)), key=lambda i: (-degrees[i], i))
    groups: list[list[int]] = []
    for i in order:
        for group in groups:
            if not np.any(adjacency[i, group]):
                group.append(int(i))
                break
        else:
            groups.append([int(i)])
    return groups
