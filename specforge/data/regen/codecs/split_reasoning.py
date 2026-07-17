"""Strict delimiter-based split-reasoning response codec."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Mapping

from ..backends.base import BackendResponse
from ..contracts import GenerationRequest, GenerationResult
from ..errors import GenerationError
from ..recipe import GeneratorSpec
from .structured_chat import StructuredChatCodec


class SplitReasoningCodec(StructuredChatCodec):
    name = "split_reasoning"

    def decode(
        self,
        response: BackendResponse,
        request: GenerationRequest,
        *,
        backend: str,
        model: str,
    ) -> GenerationResult:
        message = dict(response.message)
        raw = message.get("content")
        if message.get("reasoning_content") is None:
            if not isinstance(raw, str):
                raise GenerationError("split-reasoning response content must be text")
            delimiters = self.spec.config.get("reasoning_delimiters")
            if not isinstance(delimiters, Mapping):
                raise GenerationError(
                    "split_reasoning requires config.reasoning_delimiters"
                )
            start = delimiters.get("start")
            end = delimiters.get("end")
            if (
                not isinstance(start, str)
                or not start
                or not isinstance(end, str)
                or not end
            ):
                raise GenerationError("reasoning delimiters must be non-empty text")
            if (
                not raw.startswith(start)
                or raw.count(start) != 1
                or raw.count(end) != 1
            ):
                raise GenerationError(
                    "response does not match the reasoning delimiter grammar"
                )
            reasoning, content = raw[len(start) :].split(end, 1)
            message["reasoning_content"] = reasoning
            message["content"] = content
        decoded = super().decode(
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
        if isinstance(raw, str):
            encoded = raw.encode("utf-8")
            return replace(
                decoded,
                codec=self.name,
                exactness="delimiter_roundtrip",
                raw_evidence_digest=hashlib.sha256(encoded).hexdigest(),
                raw_evidence_bytes=len(encoded),
            )
        return replace(
            decoded,
            codec=self.name,
            exactness="delimiter_roundtrip",
        )


def create_split_reasoning_codec(
    spec: GeneratorSpec, runtime: Mapping[str, Any] | None = None
) -> SplitReasoningCodec:
    return SplitReasoningCodec(spec, runtime)


__all__ = ["SplitReasoningCodec", "create_split_reasoning_codec"]
