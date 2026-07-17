"""SGLang-compatible ``/generate`` backend for transient raw token evidence."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import GenerationRequest
from ..errors import FailureCategory, GenerationError
from ..recipe import GeneratorSpec
from .base import BackendResponse
from ..endpoints import pool_from_runtime
from .http import post_json, runtime_api_key, runtime_endpoints


class SGLangRawBackend:
    name = "sglang_raw"

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
        if request.input_token_ids is None:
            raise GenerationError(
                "raw-token backend requires codec-rendered input token IDs"
            )
        sampling = request.sampling
        params: dict[str, Any] = {
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "max_new_tokens": sampling["max_tokens"],
            "skip_special_tokens": False,
            "no_stop_trim": True,
        }
        if sampling.get("top_k") is not None:
            params["top_k"] = sampling["top_k"]
        if sampling.get("stop"):
            params["stop"] = sampling["stop"]
        if request.seed is not None:
            params["seed"] = request.seed
        extra = sampling.get("extra") or {}
        if not isinstance(extra, dict):
            raise GenerationError("sampling.extra must be an object")
        params.update(extra)
        with self.pool.lease(request) as endpoint:
            body = post_json(
                endpoint + "/generate",
                {
                    "input_ids": list(request.input_token_ids),
                    "sampling_params": params,
                },
                api_key=self.api_key,
                timeout=self.timeout,
            )
        raw_ids = body.get("output_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise GenerationError("raw generation response has no output_ids")
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in raw_ids
        ):
            raise GenerationError("raw generation output_ids must be integers")
        output_ids = tuple(raw_ids)
        finish = (body.get("meta_info") or {}).get("finish_reason", "unknown")
        if isinstance(finish, Mapping):
            finish = finish.get("type", "unknown")
        terminal_id = self.spec.config.get("terminal_token_id")
        if terminal_id is not None:
            if isinstance(terminal_id, bool) or not isinstance(terminal_id, int):
                raise GenerationError("terminal_token_id must be an integer")
            if output_ids and output_ids[-1] == terminal_id:
                output_ids = output_ids[:-1]
        return BackendResponse(
            finish_reason=str(finish),
            usage={"output_tokens": len(output_ids)},
            raw_token_ids=output_ids,
        )


def create_sglang_raw_backend(
    spec: GeneratorSpec, runtime: Mapping[str, Any] | None = None
) -> SGLangRawBackend:
    return SGLangRawBackend(spec, runtime)


__all__ = ["SGLangRawBackend", "create_sglang_raw_backend"]
