"""Strict client-side Kimi K3 reasoning grammar codec."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Mapping

from ..backends.base import BackendResponse
from ..contracts import GenerationRequest, GenerationResult
from ..errors import GenerationError
from ..recipe import GeneratorSpec
from .structured_chat import StructuredChatCodec

K3_CONTROL_TOKENS = (
    "<|open|>",
    "<|close|>",
    "<|sep|>",
    "<|end_of_msg|>",
    "[BOS]",
    "[EOS]",
    "<|kimi_image_placeholder|>",
)
_THINK_TO_RESPONSE = "<|close|>think<|sep|><|open|>response<|sep|>"
_RESPONSE_TAIL = "<|close|>response<|sep|>"
_MESSAGE_TAIL = "<|close|>message<|sep|>"


def reject_k3_control_tokens(value: str, where: str) -> None:
    if any(token in value for token in K3_CONTROL_TOKENS):
        raise GenerationError(f"{where} contains a Kimi K3 control token")


def parse_k3_completion(content: str) -> tuple[str, str]:
    if content.count(_THINK_TO_RESPONSE) != 1:
        raise GenerationError(
            "Kimi K3 response lacks a unique think/response transition"
        )
    reasoning, rest = content.split(_THINK_TO_RESPONSE, 1)
    if rest.endswith(_RESPONSE_TAIL + _MESSAGE_TAIL):
        answer = rest[: -len(_RESPONSE_TAIL + _MESSAGE_TAIL)]
    elif rest.endswith(_RESPONSE_TAIL):
        answer = rest[: -len(_RESPONSE_TAIL)]
    else:
        raise GenerationError("Kimi K3 response does not close its response block")
    reject_k3_control_tokens(reasoning, "parsed reasoning")
    reject_k3_control_tokens(answer, "parsed answer")
    return reasoning, answer


class KimiK3Codec(StructuredChatCodec):
    name = "kimi_k3"

    def prepare_request(self, request: GenerationRequest) -> GenerationRequest:
        for message in request.messages:
            for field in ("content", "reasoning_content"):
                value = message.get(field)
                if isinstance(value, str):
                    reject_k3_control_tokens(value, f"outgoing {field}")
        return request

    def decode(
        self,
        response: BackendResponse,
        request: GenerationRequest,
        *,
        backend: str,
        model: str,
    ) -> GenerationResult:
        message = dict(response.message)
        if message.get("reasoning_content") is not None:
            raise GenerationError(
                "Kimi K3 server already returned split reasoning; refusing a double parse"
            )
        raw = message.get("content")
        if not isinstance(raw, str):
            raise GenerationError("Kimi K3 raw response must be text")
        if response.finish_reason != "length":
            reasoning, answer = parse_k3_completion(raw)
            message["reasoning_content"] = reasoning
            message["content"] = answer
        result = super().decode(
            BackendResponse(
                message=message,
                finish_reason=response.finish_reason,
                usage=response.usage,
                raw_token_ids=response.raw_token_ids,
            ),
            request,
            backend=backend,
            model=model,
        )
        encoded = raw.encode("utf-8")
        return replace(
            result,
            codec=self.name,
            exactness="kimi_k3_grammar",
            raw_evidence_digest=hashlib.sha256(encoded).hexdigest(),
            raw_evidence_bytes=len(encoded),
        )


def create_kimi_k3_codec(
    spec: GeneratorSpec, runtime: Mapping[str, Any] | None = None
) -> KimiK3Codec:
    return KimiK3Codec(spec, runtime)


__all__ = [
    "K3_CONTROL_TOKENS",
    "KimiK3Codec",
    "create_kimi_k3_codec",
    "parse_k3_completion",
    "reject_k3_control_tokens",
]
