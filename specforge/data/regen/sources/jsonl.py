"""Local JSONL source adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..contracts import iter_jsonl, jsonl_identity
from ..errors import ContractError
from ..recipe import SourceSpec
from .base import SourceIdentity, SourceRow


class JsonlSource:
    def __init__(self, source_name: str, spec: SourceSpec) -> None:
        raw_path = spec.config.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ContractError(
                f"source {source_name!r}: jsonl config.path must be a non-empty string"
            )
        self.source_name = source_name
        self.path = Path(raw_path).expanduser().resolve()
        if not self.path.is_file():
            raise ContractError(f"source {source_name!r}: file not found: {self.path}")
        file_identity = jsonl_identity(self.path)
        self._identity = SourceIdentity(
            adapter="jsonl",
            # Local placement is runtime-only. Content identity makes identical
            # bytes portable across mounts without publishing a private path.
            locator="content-addressed-jsonl",
            fingerprint=f"sha256:{file_identity.sha256}",
            rows=file_identity.rows,
        )

    @property
    def identity(self) -> SourceIdentity:
        return self._identity

    def iter_rows(self):
        position = 0
        for _, value in iter_jsonl(self.path):
            yield SourceRow(position=position, value=value)
            position += 1


def create_jsonl_source(source_name: str, spec: SourceSpec) -> JsonlSource:
    return JsonlSource(source_name, spec)


__all__ = ["JsonlSource", "create_jsonl_source"]
