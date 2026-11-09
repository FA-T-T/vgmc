"""Single-coordinator run lifecycle and committed task results."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

from .artifacts import (
    FileLock,
    canonical_json,
    computation_identity,
    file_hash,
    identity,
    read_json,
    safe_path,
    write_json,
)

RunMode = Literal["fresh", "reuse", "resume", "auto"]


@dataclass(frozen=True)
class RunDecision:
    """Selected lifecycle action and its input-based explanation."""

    action: str
    reason: str
    compute: bool


class RunStore:
    """Manage one immutable scientific run and its task commits.

    Args:
        path: Explicit new or existing run directory on local POSIX storage.
        scientific: Effective inputs, configuration and random-stream mapping.
        tasks: Unique expected task identifiers in canonical execution order.
        mode: ``fresh``, ``reuse``, ``resume`` or ``auto``.
        allow_compute: Explicit permission for fresh or resumed computation.

    A completed task file is independently checksummed. The final manifest
    lists all expected tasks and is published after their files. Reuse reads
    a verified completed manifest. Resume skips committed tasks and continues
    compatible algorithm checkpoints.
    """

    def __init__(
        self,
        path: Path,
        scientific: dict[str, Any],
        tasks: list[str],
        *,
        mode: RunMode = "fresh",
        allow_compute: bool = False,
    ) -> None:
        """Freeze run identity and the expected task set before execution."""
        if mode not in ("fresh", "reuse", "resume", "auto"):
            raise ValueError("Unknown run mode")
        if len(tasks) != len(set(tasks)) or not tasks:
            raise ValueError("A run requires distinct task identities")
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError("Run paths must use ordinary components")
        self.path = path.resolve()
        self.scientific = dict(scientific)
        self.tasks = sorted(tasks)
        self.mode = mode
        self.allow_compute = allow_compute
        self.frozen = json.loads(
            canonical_json(
                dict(
                    schema=1,
                    scientific=scientific,
                    implementation=computation_identity(),
                    tasks=self.tasks,
                )
            )
        )
        self.scientific = json.loads(canonical_json(self.frozen["scientific"]))
        self.binding = identity(self.frozen)
        self._lock: FileLock | None = None
        self.decision: RunDecision | None = None

    def plan(self) -> RunDecision:
        """Select a mode without creating files or starting computation."""
        exists = self.path.exists()
        if self.mode == "fresh":
            if exists:
                raise FileExistsError(
                    "Fresh mode requires a new run directory"
                )
            return RunDecision("fresh", "new explicit run directory", True)
        if not exists:
            if self.mode == "auto":
                return RunDecision("fresh", "compatible run is absent", True)
            raise FileNotFoundError("Requested run directory is absent")
        frozen = read_json(safe_path(self.path, "identity.json"))
        if frozen != self.frozen:
            raise ValueError(
                "Run inputs, task set, code or environment changed"
            )
        complete = safe_path(self.path, "manifest.json").exists()
        if self.mode == "reuse" or (self.mode == "auto" and complete):
            self.verify()
            return RunDecision(
                "reuse", "verified completed task manifest", False
            )
        if complete:
            raise ValueError("A completed run is opened with reuse")
        return RunDecision("resume", "matching incomplete task set", True)

    def __enter__(self) -> RunStore:
        """Acquire the coordinator lock and enter the selected lifecycle
        mode.
        """
        decision = self.plan()
        if decision.compute and not self.allow_compute:
            raise PermissionError(
                "Fresh and resumed runs require allow_compute"
            )
        if decision.action == "fresh":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.mkdir(exist_ok=False)
        self._lock = FileLock(safe_path(self.path, ".coordinator.lock"))
        self._lock.__enter__()
        try:
            if decision.action == "fresh":
                write_json(safe_path(self.path, "identity.json"), self.frozen)
            if decision.compute:
                self._state("running")
        except BaseException:
            self._lock.__exit__()
            self._lock = None
            raise
        self.decision = decision
        return self

    def __exit__(self, kind: object, error: object, traceback: object) -> None:
        """Record interruption/failure and release the coordinator lock."""
        try:
            if self.decision and self.decision.compute:
                if kind is not None:
                    state = (
                        "interrupted"
                        if isinstance(error, (KeyboardInterrupt, TimeoutError))
                        else "failed"
                    )
                    self._state(state, str(error))
                elif not safe_path(self.path, "manifest.json").exists():
                    self._state("interrupted", "task set is incomplete")
        finally:
            if self._lock:
                self._lock.__exit__()
                self._lock = None

    def _state(self, status: str, detail: str = "") -> None:
        """Publish the advisory run status after scientific commits."""
        write_json(
            safe_path(self.path, "status.json"),
            dict(schema=1, binding=self.binding, status=status, detail=detail),
        )

    def _task_path(self, task: str) -> Path:
        """Map a declared task identifier to a safe content-independent
        path.
        """
        if task not in self.tasks:
            raise ValueError("Task is outside the frozen task set")
        return safe_path(self.path, Path("tasks") / (identity(task) + ".json"))

    def read_task(self, task: str) -> dict[str, Any] | None:
        """Read a task commit after schema, identity and checksum
        validation.
        """
        path = self._task_path(task)
        if not path.exists():
            return None
        envelope = read_json(path)
        if not isinstance(envelope, dict) or set(envelope) != {
            "schema",
            "binding",
            "task",
            "payload",
            "sha256",
        }:
            raise ValueError("Task commit schema differs")
        signed = {
            k: envelope[k] for k in ("schema", "binding", "task", "payload")
        }
        if envelope["schema"] != 1 or identity(signed) != envelope["sha256"]:
            raise ValueError("Task commit checksum differs")
        if envelope["binding"] != self.binding or envelope["task"] != task:
            raise ValueError("Task scientific identity differs")
        if not isinstance(envelope["payload"], dict):
            raise ValueError("Task payload must be a JSON object")
        return envelope["payload"]

    def commit_task(self, task: str, payload: dict[str, Any]) -> None:
        """Publish a task once and reject conflicting repeated submissions."""
        if not self.decision or not self.decision.compute:
            raise PermissionError("Task commits require a computing lifecycle")
        previous = self.read_task(task)
        if previous is not None:
            if identity(previous) != identity(payload):
                raise ValueError(
                    "Repeated task submission contains different data"
                )
            return
        signed = dict(
            schema=1, binding=self.binding, task=task, payload=payload
        )
        write_json(
            self._task_path(task), dict(signed, sha256=identity(signed))
        )

    def checkpoint_path(self, task: str) -> Path:
        """Return the algorithm checkpoint path for a declared task."""
        self._task_path(task)
        return safe_path(
            self.path, Path("checkpoints") / (identity(task) + ".json")
        )

    def complete(self) -> None:
        """Verify all tasks and publish the completed task manifest."""
        if not self.decision or not self.decision.compute:
            raise PermissionError("Completion requires a computing lifecycle")
        references = {}
        for task in self.tasks:
            if self.read_task(task) is None:
                raise ValueError("Completion requires every frozen task")
            references[task] = file_hash(self._task_path(task))
        signed = dict(schema=1, binding=self.binding, tasks=references)
        write_json(
            safe_path(self.path, "manifest.json"),
            dict(signed, sha256=identity(signed)),
        )
        self._state("completed")

    def verify(self) -> dict[str, Any]:
        """Verify a completed manifest against every committed task file."""
        manifest = read_json(safe_path(self.path, "manifest.json"))
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema",
            "binding",
            "tasks",
            "sha256",
        }:
            raise ValueError("Run manifest schema differs")
        signed = {k: manifest[k] for k in ("schema", "binding", "tasks")}
        if manifest["schema"] != 1 or identity(signed) != manifest["sha256"]:
            raise ValueError("Run manifest checksum differs")
        if manifest["binding"] != self.binding or set(
            manifest["tasks"]
        ) != set(self.tasks):
            raise ValueError("Run manifest identity or task set differs")
        for task, checksum in manifest["tasks"].items():
            if (
                self.read_task(task) is None
                or file_hash(self._task_path(task)) != checksum
            ):
                raise ValueError("Committed task content differs")
        return manifest
