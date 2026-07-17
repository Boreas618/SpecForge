"""Generation backend boundary for decoded text trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ..contracts import GenerationRequest, assert_text_trajectory_value


@dataclass(frozen=True)
class BackendResponse:
    message: Mapping[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw_token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        assert_text_trajectory_value(self.message, path="$.backend_response.message")
        assert_text_trajectory_value(self.usage, path="$.backend_response.usage")


class GenerationBackend(Protocol):
    name: str
    model: str

    def generate(self, request: GenerationRequest) -> BackendResponse: ...


__all__ = ["BackendResponse", "GenerationBackend"]
