"""Dependency-light contracts for text and reasoning trajectory regeneration.

This module deliberately imports no torch, transformers, distributed runtime, or
model code. Finalized payloads are JSON values and are guarded against hidden-state
or tensor-feature fields at every serialization boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .errors import ContractError

SCHEMA_VERSION = 1
SEED_STRATEGY = "sha256_record_stage_variant_ordinal_v1"
SHARD_STRATEGY_CONTIGUOUS = "contiguous_v1"

_FORBIDDEN_TENSOR_FIELDS = frozenset(
    {
        "hiddenstate",
        "hiddenstates",
        "logit",
        "logits",
        "embedding",
        "embeddings",
        "kvcache",
        "pastkeyvalue",
        "pastkeyvalues",
        "tensor",
        "tensors",
        "tensorfeature",
        "tensorfeatures",
        "featuretensor",
        "featuretensors",
    }
)
_FIELD_NORMALIZER = re.compile(r"[^a-z0-9]+")


def canonical_json(value: Any) -> str:
    """Serialize JSON injectively for identity/provenance hashing."""

    assert_text_trajectory_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_field(value: str) -> str:
    return _FIELD_NORMALIZER.sub("", value.lower())


def assert_text_trajectory_value(value: Any, *, path: str = "$") -> None:
    """Reject non-JSON and tensor-feature-shaped values recursively.

    Field-name rejection is intentionally enforced even for plain Python lists. It
    prevents tensor data from being laundered through JSON before finalization while
    keeping ordinary token IDs and tool arguments legal where their contracts allow
    them.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(
                f"{path}: non-finite floats are not valid artifact data"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path}: object key {key!r} is not a string")
            normalized = _normalized_field(key)
            if normalized in _FORBIDDEN_TENSOR_FIELDS:
                raise ContractError(
                    f"{path}.{key}: tensor/hidden-state fields are forbidden in "
                    "regeneration payloads"
                )
            assert_text_trajectory_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_text_trajectory_value(item, path=f"{path}[{index}]")
        return
    raise ContractError(
        f"{path}: {type(value).__name__} is not JSON text/trajectory data"
    )


def type_preserving_id(value: Any) -> str:
    """Return a stable identity where integer 1 and string ``"1"`` differ."""

    assert_text_trajectory_value(value, path="$.id")
    return canonical_json({"type": type(value).__name__, "value": value})


@dataclass(frozen=True)
class RecordKey:
    source: str
    source_id: Any
    variant: str = "base"

    def __post_init__(self) -> None:
        if not self.source or not isinstance(self.source, str):
            raise ContractError("record source must be a non-empty string")
        if not self.variant or not isinstance(self.variant, str):
            raise ContractError("record variant must be a non-empty string")
        type_preserving_id(self.source_id)

    @property
    def canonical(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "variant": self.variant,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecordKey":
        return cls(
            source=value["source"],
            source_id=value["source_id"],
            variant=value.get("variant", "base"),
        )


@dataclass(frozen=True)
class StageEvent:
    stage_id: str
    operation: str
    generator: str | None
    variant: str
    generation_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "stage_id": self.stage_id,
            "operation": self.operation,
            "generator": self.generator,
            "variant": self.variant,
            "generation_count": self.generation_count,
            "metadata": dict(self.metadata),
        }
        assert_text_trajectory_value(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageEvent":
        return cls(
            stage_id=value["stage_id"],
            operation=value["operation"],
            generator=value.get("generator"),
            variant=value["variant"],
            generation_count=int(value.get("generation_count", 0)),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RecordEnvelope:
    key: RecordKey
    input_position: int
    payload: Mapping[str, Any]
    source_fingerprint: str
    source_payload_digest: str
    source_preserved_digest: str
    stage_history: tuple[StageEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.input_position < 0:
            raise ContractError("input_position must be non-negative")
        if not self.source_fingerprint:
            raise ContractError("source_fingerprint must be non-empty")
        assert_text_trajectory_value(self.payload, path="$.payload")
        actual = canonical_digest(self.payload)
        if not self.source_payload_digest:
            raise ContractError("source_payload_digest must be non-empty")
        if not self.source_preserved_digest:
            raise ContractError("source_preserved_digest must be non-empty")
        # Stage-derived envelopes intentionally keep the original source digest, so
        # equality is required only before the first operation.
        if not self.stage_history and actual != self.source_payload_digest:
            raise ContractError(
                "source_payload_digest does not match the normalized source payload"
            )

    def with_payload(
        self,
        payload: Mapping[str, Any],
        *,
        key: RecordKey | None = None,
        event: StageEvent,
    ) -> "RecordEnvelope":
        return RecordEnvelope(
            key=key or self.key,
            input_position=self.input_position,
            payload=dict(payload),
            source_fingerprint=self.source_fingerprint,
            source_payload_digest=self.source_payload_digest,
            source_preserved_digest=self.source_preserved_digest,
            stage_history=(*self.stage_history, event),
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "key": self.key.to_dict(),
            "input_position": self.input_position,
            "payload": dict(self.payload),
            "source_fingerprint": self.source_fingerprint,
            "source_payload_digest": self.source_payload_digest,
            "source_preserved_digest": self.source_preserved_digest,
            "stage_history": [event.to_dict() for event in self.stage_history],
        }
        assert_text_trajectory_value(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecordEnvelope":
        return cls(
            key=RecordKey.from_dict(value["key"]),
            input_position=int(value["input_position"]),
            payload=dict(value["payload"]),
            source_fingerprint=value["source_fingerprint"],
            source_payload_digest=value["source_payload_digest"],
            source_preserved_digest=value["source_preserved_digest"],
            stage_history=tuple(
                StageEvent.from_dict(event) for event in value.get("stage_history", [])
            ),
        )


@dataclass(frozen=True)
class GenerationRequest:
    record_key: RecordKey
    stage_id: str
    variant: str
    generation_ordinal: int
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...] = ()
    sampling: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    # Transient serving evidence. The values never appear in to_dict or an
    # artifact; only their digest/count participate in request identity.
    input_token_ids: tuple[int, ...] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.generation_ordinal <= 0:
            raise ContractError("generation_ordinal must be positive")
        assert_text_trajectory_value(self.to_dict(), path="$.generation_request")

    def to_dict(self) -> dict[str, Any]:
        input_token_digest = None
        input_token_count = None
        if self.input_token_ids is not None:
            if any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in self.input_token_ids
            ):
                raise ContractError("input_token_ids must contain integers")
            input_token_digest = canonical_digest(list(self.input_token_ids))
            input_token_count = len(self.input_token_ids)
        return {
            "record_key": self.record_key.to_dict(),
            "stage_id": self.stage_id,
            "variant": self.variant,
            "generation_ordinal": self.generation_ordinal,
            "messages": [dict(message) for message in self.messages],
            "tools": [dict(tool) for tool in self.tools],
            "sampling": dict(self.sampling),
            "seed": self.seed,
            "input_token_digest": input_token_digest,
            "input_token_count": input_token_count,
        }


@dataclass(frozen=True)
class GenerationResult:
    message: Mapping[str, Any]
    finish_reason: str
    backend: str
    model: str
    codec: str
    exactness: str = "structured_chat"
    usage: Mapping[str, Any] = field(default_factory=dict)
    request_digest: str | None = None
    request_seed: int | None = None
    raw_evidence_digest: str | None = None
    raw_evidence_bytes: int | None = None
    raw_token_digest: str | None = None
    raw_token_count: int | None = None
    # Transient exact-rebuild evidence. It is intentionally omitted by to_dict.
    raw_token_ids: tuple[int, ...] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        assert_text_trajectory_value(self.message, path="$.generation_result.message")
        assert_text_trajectory_value(self.usage, path="$.generation_result.usage")
        if self.raw_token_ids is not None:
            if any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in self.raw_token_ids
            ):
                raise ContractError("raw_token_ids must contain integers")
            digest = hashlib.sha256(
                canonical_json(list(self.raw_token_ids)).encode("utf-8")
            ).hexdigest()
            if self.raw_token_digest is not None and self.raw_token_digest != digest:
                raise ContractError("raw_token_digest does not match raw_token_ids")
            if self.raw_token_count is not None and self.raw_token_count != len(
                self.raw_token_ids
            ):
                raise ContractError("raw_token_count does not match raw_token_ids")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "message": dict(self.message),
            "finish_reason": self.finish_reason,
            "backend": self.backend,
            "model": self.model,
            "codec": self.codec,
            "exactness": self.exactness,
            "usage": dict(self.usage),
            "request_digest": self.request_digest,
            "request_seed": self.request_seed,
            "raw_evidence_digest": self.raw_evidence_digest,
            "raw_evidence_bytes": self.raw_evidence_bytes,
            "raw_token_digest": self.raw_token_digest,
            "raw_token_count": self.raw_token_count,
        }
        assert_text_trajectory_value(value)
        return value


@dataclass(frozen=True)
class Finding:
    validator: str
    code: str
    message: str
    severity: str = "error"
    record_key: RecordKey | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator": self.validator,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "record_key": self.record_key.to_dict() if self.record_key else None,
        }


@dataclass(frozen=True)
class JsonlIdentity:
    rows: int
    sha256: str
    bytes: int


def jsonl_identity(path: str | Path) -> JsonlIdentity:
    digest = hashlib.sha256()
    rows = 0
    size = 0
    with Path(path).open("rb") as handle:
        for line in handle:
            digest.update(line)
            size += len(line)
            rows += bool(line.strip())
    return JsonlIdentity(rows=rows, sha256=digest.hexdigest(), bytes=size)


def deterministic_generation_seed(
    base_seed: int,
    record_key: RecordKey,
    stage_id: str,
    variant: str,
    generation_ordinal: int,
) -> int:
    if generation_ordinal <= 0:
        raise ContractError("generation_ordinal must be positive")
    payload = {
        "strategy": SEED_STRATEGY,
        "base_seed": base_seed,
        "record_key": record_key.to_dict(),
        "stage_id": stage_id,
        "variant": variant,
        "generation_ordinal": generation_ordinal,
    }
    return (
        int.from_bytes(
            hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()[:4],
            "big",
        )
        & 0x7FFFFFFF
    )


def contiguous_shard_bounds(
    total_rows: int, num_shards: int, shard_index: int
) -> tuple[int, int]:
    if total_rows < 0:
        raise ContractError("total_rows must be non-negative")
    if num_shards <= 0:
        raise ContractError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ContractError("shard_index must be in [0, num_shards)")
    base, remainder = divmod(total_rows, num_shards)
    start = shard_index * base + min(shard_index, remainder)
    return start, start + base + int(shard_index < remainder)


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ContractError(
                    f"{path}:{line_number}: expected a JSON object, got "
                    f"{type(value).__name__}"
                )
            assert_text_trajectory_value(value, path=f"{path}:{line_number}")
            yield line_number, value


def jsonl_line(value: Mapping[str, Any]) -> str:
    assert_text_trajectory_value(value)
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    )


__all__ = [
    "Finding",
    "GenerationRequest",
    "GenerationResult",
    "JsonlIdentity",
    "RecordEnvelope",
    "RecordKey",
    "SCHEMA_VERSION",
    "SEED_STRATEGY",
    "SHARD_STRATEGY_CONTIGUOUS",
    "StageEvent",
    "assert_text_trajectory_value",
    "canonical_digest",
    "canonical_json",
    "contiguous_shard_bounds",
    "deterministic_generation_seed",
    "iter_jsonl",
    "jsonl_identity",
    "jsonl_line",
    "type_preserving_id",
]
