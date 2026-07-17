"""Source adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class SourceIdentity:
    adapter: str
    locator: str
    fingerprint: str
    rows: int | None = None
    revision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "locator": self.locator,
            "fingerprint": self.fingerprint,
            "rows": self.rows,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class SourceRow:
    position: int
    value: Mapping[str, Any]


class SourceAdapter(Protocol):
    @property
    def identity(self) -> SourceIdentity: ...

    def iter_rows(self) -> Iterable[SourceRow]: ...


__all__ = ["SourceAdapter", "SourceIdentity", "SourceRow"]
