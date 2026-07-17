"""Lossless structured-chat response codec."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Mapping

from ..backends.base import BackendResponse
from ..contracts import (
    GenerationRequest,
    GenerationResult,
    canonical_digest,
    canonical_json,
)
from ..errors import ContractError, FailureCategory, GenerationError
from ..recipe import GeneratorSpec
from ..records.messages import normalize_message


class StructuredChatCodec:
    name = "structured_chat"

    def __init__(
        self,
        spec: GeneratorSpec,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.history_reasoning = spec.config.get("history_reasoning", "preserve")
        if self.history_reasoning not in {"preserve", "strip"}:
            raise ContractError(
                "codec history_reasoning must be 'preserve' or 'strip'"
            )

    def prepare_request(self, request: GenerationRequest) -> GenerationRequest:
        # Serving stacks that drop earlier-turn thinking (for example Qwen chat
        # templates) must not receive stored reasoning in the request history.
        # The captured artifact keeps the complete reasoning either way.
        if self.history_reasoning == "preserve":
            return request
        messages = tuple(
            {
                key: value
                for key, value in message.items()
                if key != "reasoning_content"
            }
            if message.get("role") == "assistant"
            else message
            for message in request.messages
        )
        return replace(request, messages=messages)

    def decode(
        self,
        response: BackendResponse,
        request: GenerationRequest,
        *,
        backend: str,
        model: str,
    ) -> GenerationResult:
        if response.finish_reason == "length":
            raise GenerationError(
                "completion was truncated (finish_reason=length)",
                category=FailureCategory.GENERATION_INVALID,
            )
        try:
            message = normalize_message(
                dict(response.message), where="$.generation_result.message"
            )
        except Exception as exc:
            raise GenerationError(str(exc)) from None
        content = message.get("content")
        if not isinstance(content, str):
            raise GenerationError("assistant content must be text")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise GenerationError("assistant reasoning_content must be text")
        control_tokens = self.spec.config.get("control_tokens", [])
        if not isinstance(control_tokens, list) or any(
            not isinstance(token, str) or not token for token in control_tokens
        ):
            raise GenerationError("codec control_tokens must be non-empty strings")
        for field_name in ("content", "reasoning_content"):
            field_value = message.get(field_name)
            if isinstance(field_value, str):
                leaked = next(
                    (token for token in control_tokens if token in field_value), None
                )
                if leaked is not None:
                    raise GenerationError(
                        f"assistant {field_name} contains a serving control token"
                    )

        reasoning_policy = self.spec.sampling.reasoning
        if reasoning_policy == "required" and not (
            isinstance(reasoning, str) and reasoning.strip()
        ):
            raise GenerationError("assistant reasoning_content is required and empty")
        if reasoning_policy == "disabled" and isinstance(reasoning, str) and reasoning:
            raise GenerationError(
                "assistant emitted reasoning although thinking was disabled"
            )
        if not content.strip() and not message.get("tool_calls"):
            raise GenerationError(
                "assistant emitted neither visible text nor tool calls"
            )

        raw_digest = None
        raw_count = None
        if response.raw_token_ids is not None:
            raw_digest = hashlib.sha256(
                canonical_json(list(response.raw_token_ids)).encode("utf-8")
            ).hexdigest()
            raw_count = len(response.raw_token_ids)
        return GenerationResult(
            message=message,
            finish_reason=response.finish_reason,
            backend=backend,
            model=model,
            codec=self.name,
            exactness="structured_chat",
            usage=dict(response.usage),
            request_digest=canonical_digest(request.to_dict()),
            request_seed=request.seed,
            raw_token_digest=raw_digest,
            raw_token_count=raw_count,
            raw_token_ids=response.raw_token_ids,
        )


def create_structured_chat_codec(
    spec: GeneratorSpec, runtime: Mapping[str, Any] | None = None
) -> StructuredChatCodec:
    return StructuredChatCodec(spec, runtime)


__all__ = ["StructuredChatCodec", "create_structured_chat_codec"]
