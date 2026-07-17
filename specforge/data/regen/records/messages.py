"""Text-only OpenAI and ShareGPT record normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts import assert_text_trajectory_value, canonical_digest
from ..errors import ContractError

ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}
ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(frozen=True)
class NormalizedRecord:
    source_id: Any
    payload: dict[str, Any]


def _normalize_arguments(value: Any, *, where: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"{where}: tool arguments are invalid JSON: {exc}"
            ) from exc
    if not isinstance(value, dict):
        raise ContractError(f"{where}: tool arguments must be a JSON object")
    assert_text_trajectory_value(value, path=where)
    return dict(value)


def _normalize_tool_calls(value: Any, *, where: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ContractError(f"{where}: tool_calls must be a list")
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(value):
        call_where = f"{where}[{index}]"
        if not isinstance(call, Mapping):
            raise ContractError(f"{call_where}: tool call must be an object")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise ContractError(f"{call_where}.function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ContractError(f"{call_where}.function.name must be non-empty")
        call_id = call.get("id")
        if call_id is not None and not isinstance(call_id, str):
            raise ContractError(f"{call_where}.id must be a string")
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _normalize_arguments(
                        function.get("arguments", {}),
                        where=f"{call_where}.function.arguments",
                    ),
                },
            }
        )
    return calls


def normalize_message(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    raw_role = value.get("role", value.get("from"))
    role = ROLE_ALIASES.get(str(raw_role).lower()) if raw_role is not None else None
    if role not in ROLES:
        raise ContractError(f"{where}: unsupported role {raw_role!r}")
    content = value.get("content", value.get("value", ""))
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ContractError(
            f"{where}.content must be text, got {type(content).__name__}"
        )
    message: dict[str, Any] = {"role": role, "content": content}

    reasoning = value.get("reasoning_content")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise ContractError(f"{where}.reasoning_content must be text")
        message["reasoning_content"] = reasoning

    calls = _normalize_tool_calls(value.get("tool_calls"), where=f"{where}.tool_calls")
    if calls:
        if role != "assistant":
            raise ContractError(
                f"{where}: only assistant messages may contain tool_calls"
            )
        message["tool_calls"] = calls

    for field in ("tool_call_id", "name"):
        item = value.get(field)
        if item is not None:
            if role != "tool" or not isinstance(item, str) or not item:
                raise ContractError(
                    f"{where}.{field} is valid only as non-empty text on tool messages"
                )
            message[field] = item

    assert_text_trajectory_value(message, path=where)
    return message


def _normalize_tools(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ContractError("$.tools must be a list")
    tools: list[dict[str, Any]] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, Mapping):
            raise ContractError(f"$.tools[{index}] must be an object")
        function = tool.get("function")
        if tool.get("type", "function") != "function" or not isinstance(
            function, Mapping
        ):
            raise ContractError(f"$.tools[{index}] must declare a function")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ContractError(f"$.tools[{index}].function.name must be non-empty")
        normalized = {
            "type": "function",
            "function": {
                key: item
                for key, item in function.items()
                if key in {"name", "description", "parameters", "strict"}
            },
        }
        assert_text_trajectory_value(normalized, path=f"$.tools[{index}]")
        tools.append(normalized)
    return tools


def _validate_conversation(messages: list[dict[str, Any]]) -> None:
    if not messages:
        raise ContractError("conversation is empty")
    if messages[0]["role"] in {"assistant", "tool"}:
        raise ContractError("conversation starts with an assistant/tool message")
    if not any(message["role"] == "user" for message in messages):
        raise ContractError("conversation has no user message")

    known_calls: dict[str, str] = {}
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            for call in message.get("tool_calls", []):
                call_id = call.get("id")
                if call_id:
                    if call_id in known_calls:
                        raise ContractError(f"duplicate tool call id {call_id!r}")
                    known_calls[call_id] = call["function"]["name"]
        elif message["role"] == "tool":
            call_id = message.get("tool_call_id")
            if call_id and call_id not in known_calls:
                raise ContractError(
                    f"tool message {index} references unknown call id {call_id!r}"
                )


def normalize_openai_record(
    row: Mapping[str, Any], *, source_name: str, position: int
) -> NormalizedRecord:
    raw_messages = row.get("conversations", row.get("messages"))
    if not isinstance(raw_messages, list):
        raise ContractError("row must contain a conversations/messages list")
    messages = [
        normalize_message(message, where=f"$.conversations[{index}]")
        for index, message in enumerate(raw_messages)
    ]
    _validate_conversation(messages)
    source_id = row.get("id", position)
    payload: dict[str, Any] = {
        "id": source_id,
        "conversations": messages,
    }
    tools = _normalize_tools(row.get("tools"))
    if tools:
        payload["tools"] = tools
    metadata = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "id",
            "conversations",
            "messages",
            "tools",
            "status",
            "error",
            "regeneration",
            "regen_contract",
        }
    }
    if metadata:
        assert_text_trajectory_value(metadata, path="$.metadata")
        payload["metadata"] = metadata
    assert_text_trajectory_value(payload)
    return NormalizedRecord(source_id=source_id, payload=payload)


def normalize_sharegpt_record(
    row: Mapping[str, Any], *, source_name: str, position: int
) -> NormalizedRecord:
    return normalize_openai_record(row, source_name=source_name, position=position)


def preserved_projection(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return source-authored turns that replay operations must not alter."""

    projection = []
    for message in payload.get("conversations", []):
        if message.get("role") != "assistant":
            projection.append(
                {
                    key: value
                    for key, value in message.items()
                    if key
                    in {
                        "role",
                        "content",
                        "reasoning_content",
                        "name",
                    }
                }
            )
    return projection


def preserved_digest(payload: Mapping[str, Any]) -> str:
    return canonical_digest(preserved_projection(payload))


__all__ = [
    "NormalizedRecord",
    "normalize_message",
    "normalize_openai_record",
    "normalize_sharegpt_record",
    "preserved_digest",
    "preserved_projection",
]
