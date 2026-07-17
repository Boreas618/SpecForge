"""On-policy text/reasoning and recorded-tool replay operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..contracts import RecordEnvelope, RecordKey, StageEvent
from ..errors import ContractError, PolicyReject
from ..recipe import StageSpec
from ..records.messages import preserved_digest
from .base import OperationContext


def _variant_name(envelope: RecordEnvelope, stage: StageSpec, index: int) -> str:
    if stage.candidates == 1:
        return envelope.key.variant
    return f"{envelope.key.variant}.{stage.id}.candidate-{index}"


def _call_names(message: Mapping[str, Any]) -> list[str]:
    return [str(call["function"]["name"]) for call in message.get("tool_calls", [])]


def _call_ids(message: Mapping[str, Any]) -> list[str | None]:
    return [call.get("id") for call in message.get("tool_calls", [])]


class ReplayAssistantsOperation:
    """Replace each stored assistant turn at its original trajectory position."""

    def run(
        self,
        envelope: RecordEnvelope,
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]:
        source_messages = envelope.payload.get("conversations")
        if not isinstance(source_messages, list):
            raise ContractError("replay operation requires payload.conversations")
        tools = list(envelope.payload.get("tools") or [])
        has_agentic = bool(tools) or any(
            message.get("role") == "tool" or message.get("tool_calls")
            for message in source_messages
        )
        if has_agentic and stage.tool_policy == "reject":
            raise PolicyReject(
                "source contains a tool trajectory but stage tool_policy is reject"
            )

        outputs: list[RecordEnvelope] = []
        for candidate_index in range(stage.candidates):
            variant = _variant_name(envelope, stage, candidate_index)
            history: list[dict[str, Any]] = []
            generation_provenance: list[dict[str, Any]] = []
            generation_ordinal = 0
            active_id_map: dict[str, str] = {}
            for source_message in source_messages:
                role = source_message.get("role")
                if role != "assistant":
                    preserved = deepcopy(source_message)
                    if role == "tool":
                        old_id = preserved.get("tool_call_id")
                        if not isinstance(old_id, str) or old_id not in active_id_map:
                            raise PolicyReject(
                                "recorded tool result cannot be bound to the regenerated call"
                            )
                        preserved["tool_call_id"] = active_id_map[old_id]
                    history.append(preserved)
                    continue

                generation_ordinal += 1
                result = context.generate(
                    envelope,
                    stage,
                    messages=history,
                    tools=tools,
                    variant=variant,
                    generation_ordinal=generation_ordinal,
                )
                assistant = deepcopy(dict(result.message))
                original_names = _call_names(source_message)
                generated_names = _call_names(assistant)
                if stage.tool_policy == "reject" and generated_names:
                    raise PolicyReject(
                        "generator emitted tool calls but stage tool_policy is reject"
                    )
                active_id_map = {}
                if stage.tool_policy in {"preserve_shape", "replay"}:
                    if generated_names != original_names:
                        raise PolicyReject(
                            "regenerated tool-call name/order differs from the source shape"
                        )
                    original_ids = _call_ids(source_message)
                    generated_ids = _call_ids(assistant)
                    for index, (old_id, new_id) in enumerate(
                        zip(original_ids, generated_ids)
                    ):
                        if not isinstance(old_id, str) or not isinstance(new_id, str):
                            raise PolicyReject(
                                f"tool call {index} lacks a stable source/generated id"
                            )
                        active_id_map[old_id] = new_id
                history.append(assistant)
                result_meta = result.to_dict()
                result_meta.pop("message", None)
                generation_provenance.append(result_meta)

            if generation_ordinal == 0:
                raise PolicyReject(
                    "replay_assistants found no source assistant turn; use complete_prompt"
                )
            payload = dict(envelope.payload)
            payload["conversations"] = history
            if preserved_digest(payload) != envelope.source_preserved_digest:
                raise ContractError("replay operation changed source-authored messages")
            key = RecordKey(
                source=envelope.key.source,
                source_id=envelope.key.source_id,
                variant=variant,
            )
            outputs.append(
                envelope.with_payload(
                    payload,
                    key=key,
                    event=StageEvent(
                        stage_id=stage.id,
                        operation=stage.operation,
                        generator=stage.generator,
                        variant=variant,
                        generation_count=generation_ordinal,
                        metadata={"generations": generation_provenance},
                    ),
                )
            )
        return outputs


class CompletePromptOperation:
    def run(
        self,
        envelope: RecordEnvelope,
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]:
        source_messages = envelope.payload.get("conversations")
        if not isinstance(source_messages, list):
            raise ContractError("complete_prompt requires payload.conversations")
        if any(message.get("role") == "assistant" for message in source_messages):
            raise PolicyReject("complete_prompt accepts prompt-only source rows")
        if any(message.get("role") == "tool" for message in source_messages):
            raise PolicyReject(
                "complete_prompt cannot start from an unresolved tool result"
            )
        tools = list(envelope.payload.get("tools") or [])
        outputs: list[RecordEnvelope] = []
        for candidate_index in range(stage.candidates):
            variant = _variant_name(envelope, stage, candidate_index)
            result = context.generate(
                envelope,
                stage,
                messages=deepcopy(source_messages),
                tools=tools,
                variant=variant,
                generation_ordinal=1,
            )
            if result.message.get("tool_calls"):
                raise PolicyReject(
                    "complete_prompt emitted an unexecuted tool call; use a tool-loop operation"
                )
            payload = dict(envelope.payload)
            payload["conversations"] = [
                *deepcopy(source_messages),
                deepcopy(dict(result.message)),
            ]
            if preserved_digest(payload) != envelope.source_preserved_digest:
                raise ContractError("complete_prompt changed source-authored messages")
            result_meta = result.to_dict()
            result_meta.pop("message", None)
            key = RecordKey(
                source=envelope.key.source,
                source_id=envelope.key.source_id,
                variant=variant,
            )
            outputs.append(
                envelope.with_payload(
                    payload,
                    key=key,
                    event=StageEvent(
                        stage_id=stage.id,
                        operation=stage.operation,
                        generator=stage.generator,
                        variant=variant,
                        generation_count=1,
                        metadata={"generations": [result_meta]},
                    ),
                )
            )
        return outputs


def create_replay_operation() -> ReplayAssistantsOperation:
    return ReplayAssistantsOperation()


def create_complete_operation() -> CompletePromptOperation:
    return CompletePromptOperation()


__all__ = [
    "CompletePromptOperation",
    "ReplayAssistantsOperation",
    "create_complete_operation",
    "create_replay_operation",
]
