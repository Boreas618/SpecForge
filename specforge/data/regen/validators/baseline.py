"""Baseline finalized-row validators."""

from __future__ import annotations

from ..contracts import Finding, RecordEnvelope, assert_text_trajectory_value
from ..errors import ContractError
from ..records.messages import preserved_digest, preserved_projection


def _source_turns_preserved(envelope: RecordEnvelope) -> bool:
    """Source-authored turns survive unchanged.

    Exact projection equality is the common case. Tool-executing operations
    may extend the trajectory, so a source projection that survives as a
    prefix also passes — but only when every appended non-assistant turn is
    an environment-produced tool result.
    """

    if preserved_digest(envelope.payload) == envelope.source_preserved_digest:
        return True
    projection = preserved_projection(envelope.payload)
    for length in range(len(projection), -1, -1):
        from ..contracts import canonical_digest

        if canonical_digest(projection[:length]) == envelope.source_preserved_digest:
            return all(
                message.get("role") == "tool" for message in projection[length:]
            )
    return False


class BaselineValidator:
    name = "baseline"

    def validate(self, envelope: RecordEnvelope) -> list[Finding]:
        findings: list[Finding] = []
        try:
            assert_text_trajectory_value(envelope.payload)
        except ContractError as exc:
            findings.append(
                Finding(
                    self.name, "non_text_payload", str(exc), record_key=envelope.key
                )
            )
            return findings
        messages = envelope.payload.get("conversations")
        if not isinstance(messages, list) or not messages:
            findings.append(
                Finding(
                    self.name,
                    "missing_conversation",
                    "payload has no conversation",
                    record_key=envelope.key,
                )
            )
            return findings
        assistants = [
            message for message in messages if message.get("role") == "assistant"
        ]
        if not assistants:
            findings.append(
                Finding(
                    self.name,
                    "missing_assistant",
                    "conversation has no generated assistant message",
                    record_key=envelope.key,
                )
            )
        for index, message in enumerate(assistants):
            content = message.get("content")
            reasoning = message.get("reasoning_content")
            if not isinstance(content, str):
                findings.append(
                    Finding(
                        self.name,
                        "assistant_content_not_text",
                        f"assistant {index} content is not text",
                        record_key=envelope.key,
                    )
                )
        call_ids = {
            call.get("id")
            for message in assistants
            for call in message.get("tool_calls", [])
            if isinstance(call.get("id"), str)
        }
        for index, message in enumerate(messages):
            if (
                message.get("role") == "tool"
                and message.get("tool_call_id") not in call_ids
            ):
                findings.append(
                    Finding(
                        self.name,
                        "unbound_tool_result",
                        f"tool result {index} is not bound to an assistant call",
                        record_key=envelope.key,
                    )
                )
            if reasoning is not None and not isinstance(reasoning, str):
                findings.append(
                    Finding(
                        self.name,
                        "assistant_reasoning_not_text",
                        f"assistant {index} reasoning_content is not text",
                        record_key=envelope.key,
                    )
                )
        if not _source_turns_preserved(envelope):
            findings.append(
                Finding(
                    self.name,
                    "source_preservation",
                    "source-authored messages changed",
                    record_key=envelope.key,
                )
            )
        return findings


def create_baseline_validator(recipe=None) -> BaselineValidator:
    return BaselineValidator()


__all__ = ["BaselineValidator", "create_baseline_validator"]
