"""Artifact publication to object storage with atomic manifest visibility.

A published artifact becomes visible only through its manifest object, and
the manifest is written last with a conditional (create-only) put. A failed
or racing publisher therefore can never expose a manifest that references
missing or corrupt payload: readers that find no manifest refuse the prefix,
and readers that find one verify every referenced file digest on download.

``BlobStore`` is the minimal contract a concrete backend must honor. The
local-directory implementation exists for tests and single-host use; cloud
adapters (multipart uploads, retry policy) implement the same interface
without changing publication semantics.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .artifact import ArtifactLayout, verify_manifest
from .contracts import canonical_json
from .errors import ArtifactError

MANIFEST_KEY = "manifest.json"


class BlobStore(ABC):
    """Minimal blob interface: byte objects addressed by string keys."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def put_if_absent(self, key: str, data: bytes) -> bool:
        """Atomically create the object; False if the key already exists."""

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...


class LocalDirectoryBlobStore(BlobStore):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise ArtifactError(f"blob key escapes the store root: {key!r}")
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)

    def put_if_absent(self, key: str, data: bytes) -> bool:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            os.write(handle, data)
            os.fsync(handle)
        finally:
            os.close(handle)
        return True

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise ArtifactError(f"blob not found: {key!r}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()


def _manifest_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ArtifactError("manifest lists no payload files")
    return files


def publish_to_store(
    artifact: str | Path | ArtifactLayout,
    store: BlobStore,
    prefix: str,
    *,
    fault_hook: Any = None,
) -> dict[str, Any]:
    """Upload a FINALIZED local artifact; the manifest object is written last.

    Publication is idempotent: republishing the identical artifact succeeds,
    while a prefix already holding a different artifact digest is refused.
    """

    layout = (
        artifact
        if isinstance(artifact, ArtifactLayout)
        else ArtifactLayout.from_uri(artifact)
    )
    manifest = verify_manifest(layout.manifest)
    prefix = prefix.strip("/")

    for entry in _manifest_files(manifest):
        relative = entry["path"]
        data = (layout.root / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"] or len(data) != entry["bytes"]:
            raise ArtifactError(f"local payload changed after finalize: {relative}")
        key = f"{prefix}/{relative}"
        if store.exists(key):
            remote = store.get(key)
            if hashlib.sha256(remote).hexdigest() != entry["sha256"]:
                raise ArtifactError(
                    f"prefix already holds different bytes for {relative}"
                )
        else:
            store.put(key, data)
        if fault_hook is not None:
            fault_hook("after_file", relative)

    encoded = (canonical_json(manifest) + "\n").encode("utf-8")
    manifest_key = f"{prefix}/{MANIFEST_KEY}"
    if not store.put_if_absent(manifest_key, encoded):
        existing = store.get(manifest_key)
        try:
            recorded = read_manifest_bytes(existing)
        except ArtifactError as exc:
            raise ArtifactError(
                f"prefix {prefix!r} holds a corrupt manifest"
            ) from exc
        if recorded["artifact_digest"] != manifest["artifact_digest"]:
            raise ArtifactError(
                f"prefix {prefix!r} already holds a different artifact "
                f"({recorded['artifact_digest'][:16]}…)"
            )
    return manifest


def read_manifest_bytes(data: bytes) -> dict[str, Any]:
    import json

    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactError("manifest must be a JSON object")
    body = dict(manifest)
    recorded = body.pop("artifact_digest", None)
    from .contracts import canonical_digest

    if recorded != canonical_digest(body):
        raise ArtifactError("manifest digest does not match its content")
    return manifest


def fetch_from_store(
    store: BlobStore, prefix: str, destination: str | Path
) -> dict[str, Any]:
    """Download and byte-verify a published artifact; refuse partial prefixes."""

    prefix = prefix.strip("/")
    manifest_key = f"{prefix}/{MANIFEST_KEY}"
    if not store.exists(manifest_key):
        raise ArtifactError(
            f"prefix {prefix!r} has no published manifest; refusing partial data"
        )
    manifest = read_manifest_bytes(store.get(manifest_key))

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in _manifest_files(manifest):
        relative = entry["path"]
        data = store.get(f"{prefix}/{relative}")
        if (
            hashlib.sha256(data).hexdigest() != entry["sha256"]
            or len(data) != entry["bytes"]
        ):
            raise ArtifactError(f"published file failed verification: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (destination / MANIFEST_KEY).write_bytes(
        (canonical_json(manifest) + "\n").encode("utf-8")
    )
    from .artifact import atomic_write_json

    atomic_write_json(
        destination / "state.json",
        {
            "state": "FINALIZED",
            "artifact_digest": manifest["artifact_digest"],
            "recipe_digest": manifest["recipe_digest"],
            "plan_digest": manifest["plan_digest"],
        },
    )
    return verify_manifest(destination / MANIFEST_KEY)


__all__ = [
    "BlobStore",
    "LocalDirectoryBlobStore",
    "MANIFEST_KEY",
    "fetch_from_store",
    "publish_to_store",
    "read_manifest_bytes",
]
