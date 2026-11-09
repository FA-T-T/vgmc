"""ICS evaluation through the referenced Tequila repository."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from .._validation import integer, positive, real_array
from ..grouping import validate_words

from ..covariance import Covariance

REPOSITORY = "https://github.com/tequilahub/tequila"
COMMIT = "93c361b6821ae6790fd0fc3d0b7a0ebfaa312be4"


def binary_terms(words: list[str], coefficients: ArrayLike) -> list[Any]:
    """Convert Pauli words to Tequila's binary representation."""
    from tequila.grouping.binary_rep import BinaryPauliString

    coefficients = real_array(coefficients, "coefficients", 1)
    return [
        BinaryPauliString(
            np.array(
                [int(a in "XY") for a in word] + [int(a in "YZ") for a in word]
            ),
            float(c),
        )
        for word, c in zip(words, coefficients, strict=True)
    ]


def author_library(
    words: list[str], coefficients: ArrayLike
) -> tuple[Any, list[list[int]], list[list[int]]]:
    """Construct the Tequila SI partition and overlapping coefficient
    library.
    """
    from tequila.grouping.overlapping_methods import OverlappingGroups

    terms = binary_terms(words, coefficients)
    lookup = {term.binary_tuple(): i for i, term in enumerate(terms)}
    obj = OverlappingGroups.init_from_binary_terms(terms, condition="qwc")
    si = [
        [lookup[term.binary_tuple()] for term in group]
        for group in obj.no_groups
    ]
    groups = [group.copy() for group in si]
    for term, positions in zip(obj.o_terms, obj.term_exists_in, strict=True):
        for position in positions:
            i = lookup[term.binary_tuple()]
            if i not in groups[position]:
                groups[position].append(i)
    return obj, si, groups


class CovarianceLookup:
    """Expose frozen covariance entries through Tequila's lookup interface."""

    def __init__(
        self, terms: list[Any], covariance: Covariance, ridge: float
    ) -> None:
        self.indices = {term.binary_tuple(): i for i, term in enumerate(terms)}
        self.matrix = covariance.matrix
        self.ridge = ridge

    def __contains__(self, key: tuple) -> bool:
        return key[0] in self.indices and key[1] in self.indices

    def __getitem__(self, key: tuple) -> float:
        i, j = (self.indices[k] for k in key)
        return float(self.matrix[i, j]) + (self.ridge if i == j else 0.0)


def ics(
    words: list[str],
    coefficients: ArrayLike,
    covariance: Covariance,
    *,
    iterations: int = 5,
    ridge: float = 1e-8,
) -> dict[str, Any]:
    """Run coefficient and allocation updates through external Tequila.

    Args:
        words: Unique equal-width nonidentity Pauli strings.
        coefficients: Real energy coefficients, shape (n_terms,).
        covariance: Frozen dense Pauli covariance in Hamiltonian order.
        iterations: Positive number of alternating ICS updates.
        ridge: Nonnegative diagonal addition to Tequila covariance lookup.

    Returns:
        Dictionary with overlapping groups, local energy coefficients x,
        normalized shot fractions a and the completed update count.

    Raises:
        ImportError: The pinned external Tequila package is absent.
        ValueError: Input dimensions, iteration count or ridge are invalid.

    The repository and commit used for this interface are module constants.
    """
    from tequila.grouping.overlapping_methods import get_opt_sample_size

    integer(iterations, "iterations", 1)
    positive(ridge, "ridge", zero=True)
    validate_words(words)
    coefficients = real_array(coefficients, "coefficients", 1)
    if covariance.n_terms != len(words):
        raise ValueError("Covariance term axis differs")
    obj, _, groups = author_library(words, coefficients)
    terms = binary_terms(words, coefficients)
    lookup = {term.binary_tuple(): i for i, term in enumerate(terms)}
    covdict = CovarianceLookup(terms, covariance, ridge)
    current = obj.no_groups
    for _ in range(iterations):
        allocation = get_opt_sample_size(current, covdict)
        coeff = obj.optimize_pauli_coefficients(covdict, allocation)
        current = obj.overlapping_groups_from_coeff(coeff)
    allocation = get_opt_sample_size(current, covdict)
    xs = []
    for group, terms in zip(groups, current, strict=True):
        mapping = {
            lookup[term.binary_tuple()]: float(term.coeff) for term in terms
        }
        xs.append([mapping.get(i, 0.0) for i in group])
    return dict(
        groups=groups,
        x=xs,
        a=np.asarray(allocation).tolist(),
        iterations=iterations,
    )
