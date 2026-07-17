"""Operation contracts for ordered regeneration stages."""

from __future__ import annotations

from typing import Protocol

from ..contracts import GenerationResult, RecordEnvelope
from ..recipe import StageSpec


class OperationContext(Protocol):
    def generate(
        self,
        envelope: RecordEnvelope,
        stage: StageSpec,
        *,
        messages: list[dict],
        tools: list[dict],
        variant: str,
        generation_ordinal: int,
    ) -> GenerationResult: ...


class Operation(Protocol):
    def run(
        self,
        envelope: RecordEnvelope,
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]: ...


__all__ = ["Operation", "OperationContext"]
