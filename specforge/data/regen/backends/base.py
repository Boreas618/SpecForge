"""Generation backend boundary for decoded text trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ..contracts import GenerationRequest


@dataclass(frozen=True)
class BackendResponse:
    message: Mapping[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw_token_ids: tuple[int, ...] | None = None


class GenerationBackend(Protocol):
    name: str
    model: str

    def generate(self, request: GenerationRequest) -> BackendResponse: ...


__all__ = ["BackendResponse", "GenerationBackend"]
