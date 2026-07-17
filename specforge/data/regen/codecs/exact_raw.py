"""Generic transient raw-token codec with a mandatory exact rebuild proof.

Model-family renderers and parsers are runtime/plugin concerns. The core codec
only defines their contract and enforces equality; this keeps private templates
and campaign paths out of SpecForge while retaining the prototype's strongest
agentic guarantee.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Mapping, Sequence

from ..backends.base import BackendResponse
from ..contracts import (
    GenerationRequest,
    GenerationResult,
    canonical_digest,
    canonical_json,
)
from ..errors import FailureCategory, GenerationError
from ..recipe import GeneratorSpec
from ..records.messages import normalize_message


class ExactRawTokenCodec:
    name = "exact_raw_tokens"

    def __init__(
        self,
        spec: GeneratorSpec,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        runtime = runtime or {}
        self.render_request = runtime.get("render_request")
        self.parse_response = runtime.get("parse_response")
        self.rebuild_response = runtime.get("rebuild_response")
        for name, value in (
            ("render_request", self.render_request),
            ("parse_response", self.parse_response),
            ("rebuild_response", self.rebuild_response),
        ):
            if not callable(value):
                raise GenerationError(
                    f"exact_raw_tokens codec runtime requires callable {name}",
                    category=FailureCategory.CAPABILITY_ERROR,
                )
        declared = spec.config.get("format_identity")
        actual = runtime.get("format_identity")
        if not isinstance(declared, str) or not declared:
            raise GenerationError(
                "exact_raw_tokens requires generator.config.format_identity",
                category=FailureCategory.CAPABILITY_ERROR,
            )
        if actual is not None and actual != declared:
            raise GenerationError(
                "raw-token codec runtime format identity does not match recipe",
                category=FailureCategory.CAPABILITY_ERROR,
            )

    @staticmethod
    def _ids(value: Sequence[int], where: str) -> tuple[int, ...]:
        try:
            result = tuple(value)
        except TypeError:
            raise GenerationError(f"{where} did not return a token sequence") from None
        if any(
            isinstance(token, bool) or not isinstance(token, int) for token in result
        ):
            raise GenerationError(f"{where} returned non-integer token IDs")
        return result

    def prepare_request(self, request: GenerationRequest) -> GenerationRequest:
        rendered = self.render_request(
            request.messages,
            request.tools,
            request.sampling,
        )
        return replace(
            request,
            input_token_ids=self._ids(rendered, "render_request"),
        )

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
        accepted = self.spec.config.get("accepted_finish_reasons", ["stop"])
        if response.finish_reason not in accepted:
            raise GenerationError(
                f"unexpected raw generation finish reason {response.finish_reason!r}"
            )
        if response.raw_token_ids is None:
            raise GenerationError("raw-token backend returned no raw token IDs")
        sampled = self._ids(response.raw_token_ids, "backend")
        try:
            raw_message = self.parse_response(sampled, request)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                f"raw response grammar rejected: {type(exc).__name__}"
            ) from None
        if not isinstance(raw_message, Mapping):
            raise GenerationError("raw response parser returned a non-object message")
        message = normalize_message(raw_message, where="$.raw_response.message")
        try:
            rebuilt_value = self.rebuild_response(message, request)
        except Exception as exc:
            raise GenerationError(
                f"raw response rebuild failed: {type(exc).__name__}",
                category=FailureCategory.EXACT_REBUILD,
            ) from None
        rebuilt = self._ids(rebuilt_value, "rebuild_response")
        if rebuilt != sampled:
            limit = min(len(rebuilt), len(sampled))
            divergence = next(
                (index for index in range(limit) if rebuilt[index] != sampled[index]),
                limit,
            )
            raise GenerationError(
                "exact rebuild diverged at token "
                f"{divergence} (rebuilt={len(rebuilt)}, sampled={len(sampled)})",
                category=FailureCategory.EXACT_REBUILD,
            )
        raw_digest = hashlib.sha256(
            canonical_json(list(sampled)).encode("utf-8")
        ).hexdigest()
        return GenerationResult(
            message=message,
            finish_reason=response.finish_reason,
            backend=backend,
            model=model,
            codec=self.name,
            exactness="token_identical",
            usage=dict(response.usage),
            request_digest=canonical_digest(request.to_dict()),
            request_seed=request.seed,
            raw_token_digest=raw_digest,
            raw_token_count=len(sampled),
            raw_token_ids=sampled,
        )


def create_exact_raw_codec(
    spec: GeneratorSpec, runtime: Mapping[str, Any] | None = None
) -> ExactRawTokenCodec:
    return ExactRawTokenCodec(spec, runtime)


__all__ = ["ExactRawTokenCodec", "create_exact_raw_codec"]
