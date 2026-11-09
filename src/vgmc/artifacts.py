"""Content identities, atomic JSON commits and recoverable checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from contextlib import nullcontext
from functools import wraps
import hashlib
import gzip
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, ParamSpec, TypeVar
from typing import TextIO

import numpy as np

JsonObject = dict[str, Any]
P = ParamSpec("P")
R = TypeVar("R")


def _json_default(value: object) -> object:
    """Convert arrays and NumPy scalars at the JSON artifact boundary."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def canonical_json(value: object) -> bytes:
    """Encode scientific metadata deterministically as UTF-8 JSON.

    Args:
        value: JSON-compatible data, NumPy arrays or numeric NumPy scalars.

    Returns:
        Sorted, compact UTF-8 bytes. Numeric NaN and infinity are rejected.

    Examples:
        >>> canonical_json({"b": 2, "a": 1}).decode()
        '{"a":1,"b":2}'
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    ).encode()


def identity(value: object) -> str:
    """Return the SHA-256 identity of canonical scientific metadata."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: Path) -> str:
    """Hash file contents in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def computation_identity() -> JsonObject:
    """Identify computation sources and the numerical execution environment.

    Source hashing excludes plotting and experiment presentation. It includes
    scientific entry points, persistence and covariance/estimator interfaces.
    """
    directory = Path(__file__).parent
    excluded = {"plots.py"}
    sources = {
        str(path.relative_to(directory)): file_hash(path)
        for path in sorted(directory.rglob("*.py"))
        if path.name not in excluded
    }
    versions: dict[str, Any] = {}
    for name in (
        "numpy",
        "scipy",
        "mpmath",
        "threadpoolctl",
        "cvxpy",
        "clarabel",
        "pandas",
        "tequila-basic",
        "PennyLane",
    ):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
            continue
        versions[name] = {"version": distribution.version}
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            vcs = json.loads(direct_url).get("vcs_info", {})
            if vcs:
                versions[name]["vcs_commit"] = vcs.get("commit_id")
    return dict(
        sources=sources,
        versions=versions,
        python=platform.python_version(),
        platform=platform.system(),
        machine=platform.machine(),
        longdouble_mantissa=np.finfo(np.longdouble).nmant,
    )


def safe_path(root: Path, relative: str | Path) -> Path:
    """Resolve a project-relative path while rejecting traversal and symlinks.

    Args:
        root: Explicit project or run directory.
        relative: Relative path inside the root.

    Returns:
        A path whose existing components are ordinary files/directories.

    Raises:
        ValueError: A component escapes the root or follows a symlink.
    """
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Paths must remain relative to the explicit root")
    if root.is_symlink():
        raise ValueError("Project root must be an ordinary directory")
    root = root.resolve()
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError("Artifact paths must use ordinary components")
    if not current.resolve().is_relative_to(root):
        raise ValueError("Artifact path leaves the explicit root")
    return current


def atomic_write(path: Path, contents: bytes) -> None:
    """Publish bytes through a flushed temporary file and atomic replacement.

    Args:
        path: Destination file on a local POSIX filesystem.
        contents: Complete serialized content.

    Raises:
        ValueError: The destination or an existing parent is a symlink.
        OSError: File or directory synchronization fails.
    """
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Artifact paths must use ordinary components")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    """Atomically publish strict JSON with deterministic serialization."""
    atomic_write(path, canonical_json(value) + b"\n")


def read_json(path: Path) -> Any:
    """Read strict JSON and reject numeric NaN or infinity."""

    def reject(value: str) -> None:
        """Reject non-JSON numeric constants at the parsing boundary."""
        raise ValueError(f"Invalid JSON numeric constant: {value}")

    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            text = stream.read()
    else:
        text = path.read_text()
    return json.loads(text, parse_constant=reject)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A verified state and the checkpoint generation used to restore it."""

    payload: JsonObject
    recovered: bool
    source: str


class IdentityMismatchError(ValueError):
    """A valid artifact belongs to different scientific inputs."""


class FileLock:
    """Hold an exclusive local POSIX lock for one artifact writer."""

    def __init__(self, path: Path) -> None:
        """Select the lock file beside the protected artifact."""
        self.path = path
        self._stream: TextIO | None = None

    def __enter__(self) -> FileLock:
        """Acquire the lock or reject an already active writer."""
        import fcntl

        if self.path.is_symlink() or any(
            p.is_symlink() for p in self.path.parents
        ):
            raise ValueError("Lock paths must use ordinary components")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("r" if self.path.exists() else "w")
        try:
            fcntl.flock(self._stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._stream.close()
            self._stream = None
            raise RuntimeError(
                "A writer already holds the artifact lock"
            ) from error
        return self

    def __exit__(self, *exception: object) -> None:
        """Release the operating-system lock on normal or exceptional exit."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class CheckpointStore:
    """Retain two checksummed checkpoint generations for one scientific task.

    Args:
        path: Primary checkpoint path. Its parent is the task directory.
        binding: Scientific identity, including code and numerical inputs.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()).resolve() / "checkpoint.json"
        >>> store = CheckpointStore(path, "example-input")
        >>> store.save({"cursor": 1})
        >>> store.load().payload["cursor"]
        1
    """

    def __init__(self, path: Path, binding: str) -> None:
        """Bind primary and previous checkpoint paths to a scientific task."""
        self.path = path
        self.previous = path.with_name(path.stem + ".previous.json")
        self.binding = binding

    def _read(self, path: Path) -> JsonObject:
        """Validate one checkpoint envelope and its payload digest."""
        if path.is_symlink():
            raise ValueError("Checkpoint must be an ordinary file")
        envelope = read_json(path)
        if not isinstance(envelope, dict) or set(envelope) != {
            "schema",
            "binding",
            "payload",
            "sha256",
        }:
            raise ValueError("Checkpoint envelope schema differs")
        if envelope["schema"] != 1 or not isinstance(
            envelope["payload"], dict
        ):
            raise ValueError("Checkpoint payload schema differs")
        signed = {k: envelope[k] for k in ("schema", "binding", "payload")}
        if identity(signed) != envelope["sha256"]:
            raise ValueError("Checkpoint content checksum differs")
        if envelope["binding"] != self.binding:
            raise IdentityMismatchError(
                "Checkpoint scientific identity differs"
            )
        return envelope["payload"]

    def exists(self) -> bool:
        """Report whether either committed generation exists."""
        return self.path.exists() or self.previous.exists()

    def load(self) -> LoadedCheckpoint:
        """Load the newest valid generation and report any recovery.

        Returns:
            Verified state, source filename and recovery flag.

        Raises:
            FileNotFoundError: Both generations are missing.
            ValueError: Both generations are invalid or belong to other inputs.
        """
        errors = []
        for candidate in (self.path, self.previous):
            if not candidate.exists():
                continue
            try:
                payload = self._read(candidate)
                return LoadedCheckpoint(
                    payload, candidate != self.path, candidate.name
                )
            except IdentityMismatchError:
                raise
            except (ValueError, OSError, KeyError, TypeError) as error:
                errors.append(f"{candidate.name}: {error}")
        if not errors:
            raise FileNotFoundError("Checkpoint generations are missing")
        raise ValueError("Checkpoint recovery failed: " + "; ".join(errors))

    def save(self, payload: JsonObject) -> None:
        """Publish a validated state and retain the last valid generation."""
        signed = dict(schema=1, binding=self.binding, payload=payload)
        contents = (
            canonical_json(dict(signed, sha256=identity(signed))) + b"\n"
        )
        if self.path.exists():
            try:
                self._read(self.path)
            except IdentityMismatchError:
                raise
            except (ValueError, OSError, KeyError, TypeError):
                pass
            else:
                atomic_write(self.previous, self.path.read_bytes())
        atomic_write(self.path, contents)


def checkpoint_lock(function: Callable[P, R]) -> Callable[P, R]:
    """Serialize writers using the function's explicit checkpoint keyword.

    Completed-checkpoint reuse performs a read without creating a lock file.
    Run-level task execution additionally holds its coordinator lock.
    """

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        """Hold the local checkpoint lock for fresh or resumed computation."""
        path = kwargs.get("checkpoint")
        lock = (
            FileLock(path.with_suffix(".lock"))
            if isinstance(path, Path) and kwargs.get("mode") != "reuse"
            else nullcontext()
        )
        with lock:
            return function(*args, **kwargs)

    return wrapped
