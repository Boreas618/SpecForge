"""Deterministic fake text backend used by CPU tests and dry runs."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import GenerationRequest
from ..errors import FailureCategory, GenerationError
from ..recipe import GeneratorSpec
from .base import BackendResponse


class FakeBackend:
    name = "fake"

    def __init__(
        self,
        spec: GeneratorSpec,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.model = spec.model
        runtime = runtime or {}
        self.remaining_failures = int(runtime.get("fail_first_attempts", 0))
        if self.remaining_failures < 0:
            raise ValueError("fail_first_attempts must be non-negative")

    def generate(self, request: GenerationRequest) -> BackendResponse:
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise GenerationError(
                "synthetic retryable transport failure",
                category=FailureCategory.TRANSPORT_RETRYABLE,
            )
        last_user = next(
            (
                message.get("content", "")
                for message in reversed(request.messages)
                if message.get("role") == "user"
            ),
            "",
        )
        fields = {
            "prompt": last_user,
            "seed": request.seed,
            "ordinal": request.generation_ordinal,
            "variant": request.variant,
            "stage": request.stage_id,
        }
        content_template = self.spec.config.get(
            "content_template", "answer:{prompt}:{variant}:{ordinal}"
        )
        reasoning_template = self.spec.config.get("reasoning_template")
        message: dict[str, Any] = {
            "role": "assistant",
            "content": str(content_template).format(**fields),
        }
        if reasoning_template is not None:
            message["reasoning_content"] = str(reasoning_template).format(**fields)
        configured_calls = self.spec.config.get("tool_calls")
        calls_by_ordinal = self.spec.config.get("tool_calls_by_ordinal")
        if isinstance(calls_by_ordinal, dict):
            configured_calls = calls_by_ordinal.get(str(request.generation_ordinal))
        if configured_calls is not None:
            message["tool_calls"] = configured_calls
        return BackendResponse(
            message=message,
            finish_reason=str(self.spec.config.get("finish_reason", "stop")),
            usage={"fake": True},
        )


def create_fake_backend(
    spec: GeneratorSpec, runtime: Mapping[str, Any] | None = None
) -> FakeBackend:
    return FakeBackend(spec, runtime)


__all__ = ["FakeBackend", "create_fake_backend"]
