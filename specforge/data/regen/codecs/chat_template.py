"""Tokenizer chat-template renderability codec."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..backends.base import BackendResponse
from ..contracts import GenerationRequest, GenerationResult, canonical_digest
from ..errors import FailureCategory, GenerationError
from ..recipe import GeneratorSpec
from .structured_chat import StructuredChatCodec


class ChatTemplateCodec(StructuredChatCodec):
    name = "chat_template"

    def __init__(
        self,
        spec: GeneratorSpec,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(spec, runtime)
        runtime = runtime or {}
        self.tokenizer = runtime.get("tokenizer")
        if self.tokenizer is None:
            tokenizer_name = spec.tokenizer or spec.model
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                revision=spec.revision or None,
                trust_remote_code=False,
            )

    def decode(
        self,
        response: BackendResponse,
        request: GenerationRequest,
        *,
        backend: str,
        model: str,
    ) -> GenerationResult:
        result = super().decode(response, request, backend=backend, model=model)
        messages = [dict(message) for message in request.messages]
        messages.append(dict(result.message))
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tools=[dict(tool) for tool in request.tools] or None,
                tokenize=True,
                add_generation_prompt=False,
            )
        except Exception as exc:
            raise GenerationError(
                f"chat template could not render the stored trajectory: {type(exc).__name__}",
                category=FailureCategory.GENERATION_INVALID,
            ) from None
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if not isinstance(rendered, list) or any(
            isinstance(token, bool) or not isinstance(token, int) for token in rendered
        ):
            raise GenerationError("chat template did not return integer token IDs")
        return replace(
            result,
            codec=self.name,
            exactness="chat_template_renderable",
            raw_evidence_digest=canonical_digest(rendered),
        )


def create_chat_template_codec(
    spec: GeneratorSpec, runtime: Mapping[str, Any] | None = None
) -> ChatTemplateCodec:
    return ChatTemplateCodec(spec, runtime)


__all__ = ["ChatTemplateCodec", "create_chat_template_codec"]
