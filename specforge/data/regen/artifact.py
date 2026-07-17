"""Atomic local artifact lifecycle and integrity verification."""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .contracts import assert_text_trajectory_value, canonical_digest, canonical_json
from .errors import ArtifactError

ARTIFACT_SCHEMA_VERSION = 1
STATES = frozenset(
    {
        "PLANNED",
        "RUNNING",
        "COMPLETE_UNVALIDATED",
        "VALIDATED",
        "FINALIZED",
        "QUARANTINED",
    }
)


@dataclass(frozen=True)
class FileIdentity:
    path: str
    bytes: int
    sha256: str
    rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "rows": self.rows,
        }


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path

    @classmethod
    def from_uri(cls, value: str | Path) -> "ArtifactLayout":
        raw = str(value)
        if "://" in raw and not raw.startswith("file://"):
            raise ArtifactError(
                f"artifact URI {raw!r} is not supported by the local artifact store"
            )
        path = Path(raw[7:] if raw.startswith("file://") else raw)
        return cls(path.expanduser().resolve())

    @property
    def recipe(self) -> Path:
        return self.root / "recipe.json"

    @property
    def plan(self) -> Path:
        return self.root / "run-plan.json"

    @property
    def tasks(self) -> Path:
        return self.root / "tasks.jsonl"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def rejects(self) -> Path:
        return self.root / "rejects"

    @property
    def attempts(self) -> Path:
        return self.root / "attempts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def shard_dir(self, shard_index: int) -> Path:
        return self.work / "shards" / f"shard-{shard_index:05d}"

    def attempt_dir(self, shard_index: int) -> Path:
        return self.shard_dir(shard_index) / "attempts"


_TEMPORARY_SEQUENCE = itertools.count()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temporary name must be unique per writer, not per process: local
    # shard workers may share a process across threads.
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}."
        f"{next(_TEMPORARY_SEQUENCE)}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    assert_text_trajectory_value(value)
    atomic_write_text(path, canonical_json(value) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path} is not a JSON object")
    try:
        assert_text_trajectory_value(value, path=str(path))
    except Exception as exc:
        raise ArtifactError(str(exc)) from exc
    return value


def file_identity(path: Path, *, relative_to: Path | None = None) -> FileIdentity:
    digest = hashlib.sha256()
    size = 0
    rows = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            rows += chunk.count(b"\n")
    name = str(path.relative_to(relative_to)) if relative_to else str(path)
    return FileIdentity(
        path=name,
        bytes=size,
        sha256=digest.hexdigest(),
        rows=rows if path.suffix == ".jsonl" else None,
    )


def initialize_layout(layout: ArtifactLayout) -> None:
    if layout.root.exists() and any(layout.root.iterdir()):
        raise ArtifactError(
            f"artifact directory is not empty: {layout.root}; choose a fresh output"
        )
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.work.mkdir(parents=True, exist_ok=True)
    layout.reports.mkdir(parents=True, exist_ok=True)


def write_state(layout: ArtifactLayout, state: str, **metadata: Any) -> None:
    if state not in STATES:
        raise ArtifactError(f"unknown artifact state {state!r}")
    atomic_write_json(
        layout.state,
        {"schema_version": ARTIFACT_SCHEMA_VERSION, "state": state, **metadata},
    )


def read_state(layout: ArtifactLayout) -> str:
    value = read_json(layout.state)
    state = value.get("state")
    if state not in STATES:
        raise ArtifactError(f"{layout.state}: invalid state {state!r}")
    return str(state)


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArtifactError(f"cannot acquire exclusive lock {path}") from exc
        yield


def verify_manifest(path_or_root: str | Path) -> dict[str, Any]:
    path = Path(path_or_root)
    if path.is_dir():
        path = path / "manifest.json"
    manifest = read_json(path)
    if manifest.get("state") != "FINALIZED":
        raise ArtifactError(f"{path}: artifact is not FINALIZED")
    recorded_digest = manifest.get("artifact_digest")
    body = dict(manifest)
    body.pop("artifact_digest", None)
    actual_digest = canonical_digest(body)
    if recorded_digest != actual_digest:
        raise ArtifactError(
            f"{path}: artifact digest mismatch ({recorded_digest} != {actual_digest})"
        )
    root = path.parent
    seen: set[str] = set()
    for entry in manifest.get("files", []):
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise ArtifactError(f"{path}: invalid or duplicate payload path")
        seen.add(relative)
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ArtifactError(f"{path}: unsafe payload path {relative!r}")
        file_path = root / candidate
        if not file_path.is_file():
            raise ArtifactError(f"artifact payload is missing: {file_path}")
        actual = file_identity(file_path, relative_to=root)
        if (
            actual.sha256 != entry.get("sha256")
            or actual.bytes != entry.get("bytes")
            or actual.rows != entry.get("rows")
        ):
            raise ArtifactError(f"artifact payload digest mismatch: {file_path}")
    return manifest


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactLayout",
    "FileIdentity",
    "atomic_write_json",
    "atomic_write_text",
    "exclusive_lock",
    "file_identity",
    "initialize_layout",
    "read_json",
    "read_state",
    "verify_manifest",
    "write_state",
]
