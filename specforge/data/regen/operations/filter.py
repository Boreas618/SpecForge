"""Deterministic semantic trajectory filters."""

from __future__ import annotations

from ..contracts import RecordEnvelope, StageEvent
from ..errors import ContractError, PolicyReject
from ..recipe import StageSpec
from .base import OperationContext


class FilterOperation:
    def run(
        self,
        envelope: RecordEnvelope,
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]:
        assistants = [
            message
            for message in envelope.payload.get("conversations", [])
            if message.get("role") == "assistant"
        ]
        minimum = stage.config.get("min_assistant_chars", 0)
        required = stage.config.get("required_substring")
        require_reasoning = bool(stage.config.get("require_reasoning", False))
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise ContractError("filter min_assistant_chars must be non-negative")
        if required is not None and not isinstance(required, str):
            raise ContractError("filter required_substring must be text")
        combined = "".join(str(message.get("content", "")) for message in assistants)
        if len(combined) < minimum:
            raise PolicyReject("assistant text is below the configured minimum length")
        if required is not None and required not in combined:
            raise PolicyReject("assistant text lacks the configured substring")
        if require_reasoning and any(
            not isinstance(message.get("reasoning_content"), str)
            or not message["reasoning_content"].strip()
            for message in assistants
        ):
            raise PolicyReject("an assistant turn lacks required reasoning")
        return [
            envelope.with_payload(
                envelope.payload,
                event=StageEvent(
                    stage_id=stage.id,
                    operation=stage.operation,
                    generator=None,
                    variant=envelope.key.variant,
                    metadata={"passed": True},
                ),
            )
        ]


def create_filter_operation() -> FilterOperation:
    return FilterOperation()


__all__ = ["FilterOperation", "create_filter_operation"]
