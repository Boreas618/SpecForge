"""Small secret-safe JSON HTTP transport shared by remote backends."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..contracts import GenerationRequest
from ..errors import FailureCategory, GenerationError

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429})


def runtime_endpoints(runtime: Mapping[str, Any]) -> tuple[str, ...]:
    raw = runtime.get("endpoints", runtime.get("endpoint"))
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = []
    endpoints: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise GenerationError(
                "runtime endpoints must be non-empty strings",
                category=FailureCategory.TRANSPORT_TERMINAL,
            )
        normalized = value.rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise GenerationError(
                "runtime endpoint is not a valid HTTP(S) URL",
                category=FailureCategory.TRANSPORT_TERMINAL,
            )
        endpoints.append(normalized)
    if not endpoints:
        raise GenerationError(
            "no runtime inference endpoint was supplied",
            category=FailureCategory.TRANSPORT_TERMINAL,
        )
    return tuple(endpoints)


def select_endpoint(endpoints: tuple[str, ...], request: GenerationRequest) -> str:
    # Endpoint placement is deliberately absent from request identity. This
    # stable choice merely avoids scheduler timing affecting pool selection.
    selector = (
        request.seed
        if request.seed is not None
        else int(request.record_key.digest[:16], 16)
    )
    return endpoints[selector % len(endpoints)]


def runtime_api_key(runtime: Mapping[str, Any]) -> str | None:
    value = runtime.get("api_key")
    if value is not None:
        if not isinstance(value, str):
            raise GenerationError(
                "runtime api_key must be text",
                category=FailureCategory.TRANSPORT_TERMINAL,
            )
        return value
    environment_name = runtime.get("api_key_env")
    if environment_name is None:
        return None
    if not isinstance(environment_name, str) or not environment_name:
        raise GenerationError(
            "runtime api_key_env must be a non-empty name",
            category=FailureCategory.TRANSPORT_TERMINAL,
        )
    return os.environ.get(environment_name)


def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            encoded = response.read()
    except HTTPError as exc:
        category = (
            FailureCategory.TRANSPORT_RETRYABLE
            if exc.code in _RETRYABLE_STATUS or exc.code >= 500
            else FailureCategory.TRANSPORT_TERMINAL
        )
        # Do not copy provider response bodies: they may echo prompts or secrets.
        raise GenerationError(
            f"inference service returned HTTP {exc.code}", category=category
        ) from None
    except (URLError, TimeoutError, OSError):
        raise GenerationError(
            "inference service was unreachable",
            category=FailureCategory.TRANSPORT_RETRYABLE,
        ) from None
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise GenerationError(
            "inference service returned invalid JSON",
            category=FailureCategory.TRANSPORT_TERMINAL,
        ) from None
    if not isinstance(value, dict):
        raise GenerationError(
            "inference service returned a non-object JSON response",
            category=FailureCategory.TRANSPORT_TERMINAL,
        )
    return value


__all__ = [
    "post_json",
    "runtime_api_key",
    "runtime_endpoints",
    "select_endpoint",
]
