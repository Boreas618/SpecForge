"""OpenAI-compatible chat-completions backend.

Endpoints and credentials are runtime-only mappings supplied to workers. They
are never accepted from a recipe or copied into results and diagnostics.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import GenerationRequest
from ..endpoints import pool_from_runtime
from ..errors import FailureCategory, GenerationError
from ..recipe import GeneratorSpec
from ..records.messages import normalize_message
from .base import BackendResponse
from .http import (
    post_json,
    runtime_api_key,
    runtime_endpoints,
)


def _completion_url(endpoint: str) -> str:
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


class OpenAIChatBackend:
    name = "openai_chat"

    def __init__(
        self,
        spec: GeneratorSpec,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.model = spec.model
        # Keep the caller's mapping so a shared endpoint pool (health,
        # circuit breakers, in-flight budgets) spans every worker it reaches.
        self.runtime = runtime if isinstance(runtime, dict) else dict(runtime or {})
        self.endpoints = runtime_endpoints(self.runtime)
        self.pool = pool_from_runtime(self.runtime, self.endpoints)
        self.api_key = runtime_api_key(self.runtime)
        self.timeout = float(self.runtime.get("timeout", 300.0))
        if self.timeout <= 0:
            raise GenerationError(
                "runtime timeout must be positive",
                category=FailureCategory.TRANSPORT_TERMINAL,
            )

    def generate(self, request: GenerationRequest) -> BackendResponse:
        sampling = request.sampling
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in request.messages],
            "stream": False,
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "max_tokens": sampling["max_tokens"],
        }
        if sampling.get("top_k") is not None:
            payload["top_k"] = sampling["top_k"]
        if sampling.get("stop"):
            payload["stop"] = sampling["stop"]
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.tools:
            payload["tools"] = [dict(tool) for tool in request.tools]
        extra = sampling.get("extra") or {}
        request_extra = self.spec.config.get("request_extra") or {}
        if not isinstance(extra, dict) or not isinstance(request_extra, dict):
            raise GenerationError("sampling/request extra values must be objects")
        payload.update(extra)
        payload.update(request_extra)

        with self.pool.lease(request) as endpoint:
            body = post_json(
                _completion_url(endpoint),
                payload,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GenerationError("chat response has no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise GenerationError("chat response choice is not an object")
        raw_message = choice.get("message")
        if not isinstance(raw_message, Mapping):
            raise GenerationError("chat response has no message object")
        selected = {
            "role": "assistant",
            "content": raw_message.get("content"),
            "reasoning_content": raw_message.get("reasoning_content"),
            "tool_calls": raw_message.get("tool_calls"),
        }
        if selected["content"] is None:
            selected["content"] = ""
        message = normalize_message(selected, where="$.response.message")
        usage_raw = body.get("usage")
        usage = {}
        if isinstance(usage_raw, Mapping):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage_raw.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    usage[key] = value
        return BackendResponse(
            message=message,
            finish_reason=str(choice.get("finish_reason") or "unknown"),
            usage=usage,
        )


def create_openai_chat_backend(
    spec: GeneratorSpec, runtime: Mapping[str, Any] | None = None
) -> OpenAIChatBackend:
    return OpenAIChatBackend(spec, runtime)


__all__ = ["OpenAIChatBackend", "create_openai_chat_backend"]
