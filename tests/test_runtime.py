"""Lifecycle, interruption, corruption and idempotent task commits."""

import json
from pathlib import Path

import pytest

from vgmc.artifacts import CheckpointStore, FileLock, safe_path
from vgmc.runtime import RunStore


def test_checkpoint_falls_back_and_rejects_all_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    store = CheckpointStore(path, "input-1")
    store.save({"cursor": 1})
    store.save({"cursor": 2})
    path.write_text("broken")
    loaded = store.load()
    assert loaded.recovered
    assert loaded.payload == {"cursor": 1}
    store.previous.write_text("broken")
    with pytest.raises(ValueError):
        store.load()


def test_changed_binding_cannot_restore_a_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    CheckpointStore(path, "one").save({"cursor": 1})
    with pytest.raises(ValueError):
        CheckpointStore(path, "two").load()
    before = path.read_bytes()
    with pytest.raises(ValueError):
        CheckpointStore(path, "two").save({"cursor": 2})
    assert path.read_bytes() == before


def test_interrupted_write_preserves_last_valid_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vgmc.artifacts as artifacts

    path = tmp_path / "checkpoint.json"
    store = CheckpointStore(path, "same")
    store.save({"cursor": 1})
    original = artifacts.os.replace

    def fail(source: object, target: object) -> None:
        if Path(target) == path:
            raise OSError("injected publication failure")
        original(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises(OSError):
        store.save({"cursor": 2})
    assert store.load().payload == {"cursor": 1}


def test_run_resume_reuse_and_idempotent_commit(tmp_path: Path) -> None:
    path = tmp_path / "run"
    scientific = {"seed": 7, "grid": (1, 2)}
    with pytest.raises(KeyboardInterrupt):
        with RunStore(path, scientific, ["A", "B"], allow_compute=True) as run:
            run.commit_task("A", {"value": 1})
            raise KeyboardInterrupt()
    with RunStore(
        path, scientific, ["B", "A"], mode="resume", allow_compute=True
    ) as run:
        assert run.read_task("A") == {"value": 1}
        run.commit_task("A", {"value": 1})
        with pytest.raises(ValueError):
            run.commit_task("A", {"value": 2})
        run.commit_task("B", {"value": 3})
        run.complete()
    before = {str(p): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    with RunStore(path, scientific, ["A", "B"], mode="reuse") as run:
        run.verify()
    after = {str(p): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    assert before == after
    with pytest.raises(FileExistsError):
        RunStore(path, scientific, ["A", "B"], mode="fresh").plan()
    with pytest.raises(ValueError):
        RunStore(path, {"seed": 8}, ["A", "B"], mode="reuse").plan()


def test_task_corruption_and_missing_data_fail_reuse(tmp_path: Path) -> None:
    path = tmp_path / "run"
    with RunStore(path, {}, ["A"], allow_compute=True) as run:
        run.commit_task("A", {"value": 1})
        run.complete()
    task = next((path / "tasks").glob("*.json"))
    value = json.loads(task.read_text())
    value["payload"]["value"] = 2
    task.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        RunStore(path, {}, ["A"], mode="reuse").plan()


def test_lock_and_output_path_rules(tmp_path: Path) -> None:
    with FileLock(tmp_path / ".lock"):
        with pytest.raises(RuntimeError):
            with FileLock(tmp_path / ".lock"):
                pass
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../escape")
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_path(tmp_path, "link/file.json")
