"""On-demand covariance blocks from an explicitly ordered pure state."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import hashlib
from threading import RLock

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, index_group, integer, positive
from .measurement import PauliHamiltonian

ComplexArray = NDArray[np.complex128]


def apply_pauli(word: str, state: ComplexArray) -> ComplexArray:
    """Apply a Pauli word with its first wire as the most significant bit.

    Args:
        word: Pauli characters in state-vector tensor order.
        state: Complex array of shape ``(2 ** len(word),)``.

    Returns:
        New complex128 state vector after the Pauli action.
    """
    n = len(word)
    indices = np.arange(len(state), dtype=np.uint64)
    flip = sum(1 << (n - q - 1) for q, axis in enumerate(word) if axis in "XY")
    phase_mask = sum(
        1 << (n - q - 1) for q, axis in enumerate(word) if axis in "YZ"
    )
    parity = np.array([int(i).bit_count() % 2 for i in indices & phase_mask])
    phase = (1j ** word.count("Y")) * (1 - 2 * parity)
    result = np.empty_like(state)
    result[indices ^ flip] = phase * state
    return result


class StateCovariance:
    """Provide Pauli covariance blocks from a frozen normalized pure state.

    Args:
        hamiltonian: Pauli words and the state-vector wire order.
        state: Finite complex array of shape ``(2 ** n_qubits,)`` and unit
            norm within ``1e-10``. The copied state is normalized explicitly.
        cache_bytes: Byte budget for cached centered Pauli state vectors.

    Blocks are real Gram matrices of centered Pauli actions. The cache is
    bounded by vector bytes and does not materialize a global covariance.

    Examples:
        >>> h = PauliHamiltonian(("X", "Z"), [1.0, 2.0])
        >>> provider = StateCovariance(h, [1.0, 0.0])
        >>> provider.block([0, 1]).tolist()
        [[1.0, 0.0], [0.0, 0.0]]
    """

    def __init__(
        self,
        hamiltonian: PauliHamiltonian,
        state: ArrayLike,
        *,
        cache_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        """Validate a state and initialize an LRU cache of centered actions."""
        raw = np.asarray(state)
        if raw.ndim != 1 or raw.shape != (2**hamiltonian.n_qubits,):
            raise ValueError("State shape must match the declared wire order")
        if raw.dtype.kind not in "iufc" or not np.isfinite(raw).all():
            raise ValueError("State must be finite and numeric")
        value = np.array(raw, dtype=np.complex128, copy=True)
        norm = float(np.vdot(value, value).real)
        if abs(norm - 1) > 1e-10:
            raise ValueError("State vector must be normalized")
        value /= np.sqrt(norm)
        value.flags.writeable = False
        self._state = np.frombuffer(value.tobytes(), dtype=np.complex128)
        self._words = hamiltonian.words
        self._capacity = integer(cache_bytes, "cache_bytes") // value.nbytes
        self._cache: OrderedDict[int, ComplexArray] = OrderedDict()
        self._lock = RLock()
        digest = hashlib.sha256(value.tobytes())
        digest.update("|".join(self.words).encode())
        self._identity = digest.hexdigest()

    @property
    def state(self) -> ComplexArray:
        """Frozen normalized state in the declared wire tensor order."""
        return self._state

    @property
    def words(self) -> tuple[str, ...]:
        """Frozen Pauli term identities in covariance index order."""
        return self._words

    @property
    def n_terms(self) -> int:
        """Number of frozen Pauli words in the provider."""
        return len(self._words)

    def _centered(self, index: int) -> ComplexArray:
        """Compute or retrieve one centered Pauli action."""
        with self._lock:
            if index in self._cache:
                self._cache.move_to_end(index)
                return self._cache[index]
        action = apply_pauli(self.words[index], self.state)
        mean = float(np.vdot(self.state, action).real)
        vector = action - mean * self.state
        vector.flags.writeable = False
        with self._lock:
            if self._capacity:
                self._cache[index] = vector
                if len(self._cache) > self._capacity:
                    self._cache.popitem(last=False)
        return vector

    def block(self, group: Sequence[int], ridge: float = 0.0) -> FloatArray:
        """Return a real symmetric Gram block plus diagonal regularization."""
        group = index_group(group, self.n_terms, empty=True)
        ridge = positive(ridge, "ridge", zero=True)
        if not group:
            return np.empty((0, 0))
        vectors = np.stack([self._centered(i) for i in group])
        block = (vectors.conj() @ vectors.T).real
        block = (block + block.T) * 0.5
        block.flat[:: len(group) + 1] += ridge
        return np.asarray(block, dtype=np.float64)

    def fingerprint(self) -> str:
        """Return the effective state and Pauli term identity."""
        return self._identity
