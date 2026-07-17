"""Serving-format codec boundary."""

from __future__ import annotations

from typing import Protocol

from ..backends.base import BackendResponse
from ..contracts import GenerationRequest, GenerationResult


class ConversationCodec(Protocol):
    name: str

    def decode(
        self,
        response: BackendResponse,
        request: GenerationRequest,
        *,
        backend: str,
        model: str,
    ) -> GenerationResult: ...


__all__ = ["ConversationCodec"]
