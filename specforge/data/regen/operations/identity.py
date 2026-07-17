"""Deterministic no-generation operation."""

from __future__ import annotations

from ..contracts import RecordEnvelope, StageEvent
from ..recipe import StageSpec
from .base import OperationContext


class IdentityOperation:
    def run(
        self,
        envelope: RecordEnvelope,
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]:
        return [
            envelope.with_payload(
                dict(envelope.payload),
                event=StageEvent(
                    stage_id=stage.id,
                    operation=stage.operation,
                    generator=None,
                    variant=envelope.key.variant,
                ),
            )
        ]


def create_identity_operation() -> IdentityOperation:
    return IdentityOperation()


__all__ = ["IdentityOperation", "create_identity_operation"]
