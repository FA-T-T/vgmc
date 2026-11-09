"""Validation at numerical and framework boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def real_array(value: ArrayLike, name: str, ndim: int) -> FloatArray:
    """Copy a finite real array with the requested number of axes.

    Args:
        value: Numeric input. Exactly real complex arrays are accepted after
            checking their imaginary components.
        name: Argument name used in error messages.
        ndim: Required number of axes.

    Returns:
        New C-contiguous float64 array with ``ndim`` axes.

    Raises:
        ValueError: Shape, dtype, finiteness or imaginary parts differ from
            the real numerical input contract.
    """
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "iufc":
        raise ValueError(f"{name} must be a {ndim}-dimensional real array")
    if np.iscomplexobj(array):
        if np.any(array.imag != 0):
            raise ValueError(f"{name} must have real values")
        array = array.real
    out = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must have finite values")
    return out


def integer(value: object, name: str, minimum: int = 0) -> int:
    """Validate an integer scalar and its inclusive lower bound."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def index_group(
    group: Sequence[int], size: int, *, empty: bool = False
) -> list[int]:
    """Validate distinct integer indices in ``[0, size)``."""
    indices = [integer(i, "group index") for i in group]
    if (not indices and not empty) or len(set(indices)) != len(indices):
        raise ValueError("Groups require distinct indices")
    if any(i >= size for i in indices):
        raise ValueError("Group index exceeds the term count")
    return indices


def positive(value: object, name: str, *, zero: bool = False) -> float:
    """Validate a finite real scalar and its sign."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0 or (result == 0 and not zero):
        raise ValueError(f"{name} has an invalid sign or magnitude")
    return result
