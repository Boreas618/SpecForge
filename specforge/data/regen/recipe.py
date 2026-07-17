"""Typed, secret-free recipes for text regeneration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import SCHEMA_VERSION, canonical_digest
from .errors import ContractError

_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SECRET_FIELDS = frozenset(
    {
        "apikey",
        "authorization",
        "authtoken",
        "bearertoken",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)
_RUNTIME_ENDPOINT_FIELDS = frozenset(
    {"baseurl", "endpoint", "endpoints", "server", "serveraddress", "serverurl"}
)
_NORMALIZE_FIELD = re.compile(r"[^a-z0-9]+")


def _normalized_field(value: str) -> str:
    return _NORMALIZE_FIELD.sub("", value.lower())


def _reject_sensitive_values(
    value: Any,
    *,
    path: str,
    reject_endpoints: bool = False,
) -> Any:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_field(str(key))
            if normalized in _SECRET_FIELDS:
                raise ValueError(f"{path}.{key}: secrets are runtime-only")
            if reject_endpoints and normalized in _RUNTIME_ENDPOINT_FIELDS:
                raise ValueError(
                    f"{path}.{key}: inference endpoints are runtime-only; "
                    "use an endpoint-pool label in the recipe"
                )
            _reject_sensitive_values(
                item,
                path=f"{path}.{key}",
                reject_endpoints=reject_endpoints,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_values(
                item,
                path=f"{path}[{index}]",
                reject_endpoints=reject_endpoints,
            )
    return value


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectionSpec(_StrictModel):
    mode: Literal["all", "indices", "sample"] = "all"
    indices: list[int] = Field(default_factory=list)
    sample: int | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> "SelectionSpec":
        if self.mode == "all":
            if self.indices or self.sample is not None:
                raise ValueError("selection mode 'all' accepts no indices/sample")
        elif self.mode == "indices":
            if not self.indices:
                raise ValueError("selection mode 'indices' requires indices")
            if len(set(self.indices)) != len(self.indices) or min(self.indices) < 0:
                raise ValueError("selection indices must be unique and non-negative")
            if self.sample is not None:
                raise ValueError("selection indices and sample are mutually exclusive")
        else:
            if self.sample is None or self.sample <= 0:
                raise ValueError("selection mode 'sample' requires a positive sample")
            if self.indices:
                raise ValueError("selection sample and indices are mutually exclusive")
        return self


class SourceSpec(_StrictModel):
    adapter: str = "jsonl"
    record_adapter: str = "openai_messages"
    config: dict[str, Any] = Field(default_factory=dict)
    selection: SelectionSpec = Field(default_factory=SelectionSpec)

    @field_validator("adapter", "record_adapter")
    @classmethod
    def _valid_component_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError(f"invalid component name {value!r}")
        return value

    @field_validator("config")
    @classmethod
    def _source_config_is_secret_free(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_sensitive_values(value, path="source.config")


class SamplingSpec(_StrictModel):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 8192
    stop: list[str] = Field(default_factory=list)
    reasoning: Literal["preserve", "required", "disabled", "optional"] = "preserve"
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_sampling(self) -> "SamplingSpec":
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        _reject_sensitive_values(self.extra, path="sampling.extra")
        return self


class GeneratorSpec(_StrictModel):
    backend: str
    model: str
    revision: str = ""
    tokenizer: str = ""
    codec: str = "structured_chat"
    endpoint_pool: str = "default"
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("backend", "codec", "endpoint_pool")
    @classmethod
    def _valid_component_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError(f"invalid component name {value!r}")
        return value

    @field_validator("model")
    @classmethod
    def _model_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must be non-empty")
        return value

    @field_validator("config")
    @classmethod
    def _generator_config_is_runtime_free(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_sensitive_values(
            value, path="generator.config", reject_endpoints=True
        )


class StageSpec(_StrictModel):
    id: str
    operation: str
    generator: str | None = None
    candidates: int = 1
    tool_policy: Literal["reject", "preserve_shape", "replay", "execute"] = "reject"
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "operation")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError(f"invalid stage/component name {value!r}")
        return value

    @model_validator(mode="after")
    def _valid_stage(self) -> "StageSpec":
        if self.candidates <= 0:
            raise ValueError("stage candidates must be positive")
        _reject_sensitive_values(self.config, path=f"workflow.{self.id}.config")
        return self


class ValidationSpec(_StrictModel):
    profiles: list[str] = Field(default_factory=lambda: ["baseline"])
    #: per-profile parameters, keyed by profile name (e.g. the tokenizer and
    #: chat-template identity used by ``specforge_loss_mask``).
    config: dict[str, Any] = Field(default_factory=dict)
    max_unresolved_error_rate: float = 0.0
    max_policy_reject_rate: float = 1.0

    @model_validator(mode="after")
    def _valid_thresholds(self) -> "ValidationSpec":
        if not self.profiles:
            raise ValueError("at least one validation profile is required")
        if not 0 <= self.max_unresolved_error_rate <= 1:
            raise ValueError("max_unresolved_error_rate must be in [0, 1]")
        if not 0 <= self.max_policy_reject_rate <= 1:
            raise ValueError("max_policy_reject_rate must be in [0, 1]")
        _reject_sensitive_values(self.config, path="validation.config")
        return self


class RetrySpec(_StrictModel):
    max_attempts: int = 3
    categories: list[Literal["transport_retryable"]] = Field(
        default_factory=lambda: ["transport_retryable"]
    )
    #: base delay between retry attempts; doubles per attempt. Scheduling
    #: only — request seeds and identities never depend on it.
    backoff_seconds: float = 0.0

    @model_validator(mode="after")
    def _valid_retry(self) -> "RetrySpec":
        if self.max_attempts <= 0:
            raise ValueError("retry max_attempts must be positive")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("retry categories must be unique")
        if self.backoff_seconds < 0:
            raise ValueError("retry backoff_seconds must be non-negative")
        return self


class ExecutionSpec(_StrictModel):
    retry: RetrySpec = Field(default_factory=RetrySpec)


class OutputSpec(_StrictModel):
    uri: str
    format: Literal["jsonl"] = "jsonl"
    shards: int = 1
    #: shard-ownership strategy: static contiguous ranges or stable key-hash
    #: partitions. Both are pinned by the plan; topology never moves tasks.
    partitioning: Literal["contiguous", "hash"] = "contiguous"

    @model_validator(mode="after")
    def _valid_output(self) -> "OutputSpec":
        if not self.uri.strip():
            raise ValueError("output uri must be non-empty")
        if self.shards <= 0:
            raise ValueError("output shards must be positive")
        return self


class RegenerationRecipe(_StrictModel):
    version: Literal[1] = SCHEMA_VERSION
    seed: int = 0
    #: explicitly named third-party component plugins; loaded identity is
    #: recorded in the plan snapshot. Nothing is discovered implicitly.
    plugins: list[str] = Field(default_factory=list)
    sources: dict[str, SourceSpec]
    generators: dict[str, GeneratorSpec] = Field(default_factory=dict)
    workflow: list[StageSpec]
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    output: OutputSpec

    @model_validator(mode="after")
    def _validate_graph(self) -> "RegenerationRecipe":
        if not self.sources:
            raise ValueError("recipe requires at least one source")
        if not self.workflow:
            raise ValueError("recipe requires at least one workflow stage")
        if len(set(self.plugins)) != len(self.plugins):
            raise ValueError("plugins must be unique")
        for plugin in self.plugins:
            if not _NAME.fullmatch(plugin):
                raise ValueError(f"invalid plugin name {plugin!r}")
        for name in (*self.sources.keys(), *self.generators.keys()):
            if not _NAME.fullmatch(name):
                raise ValueError(f"invalid resource name {name!r}")
        stage_ids = [stage.id for stage in self.workflow]
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("workflow stage ids must be unique")
        for stage in self.workflow:
            if stage.generator is not None and stage.generator not in self.generators:
                raise ValueError(
                    f"stage {stage.id!r} references unknown generator "
                    f"{stage.generator!r}"
                )
        return self

    def canonical_payload(self, *, for_identity: bool = False) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude_none=False)
        if for_identity:
            # Artifact location is runtime placement, not dataset semantics.
            payload["output"] = dict(payload["output"])
            payload["output"].pop("uri", None)
        return payload

    @property
    def digest(self) -> str:
        return canonical_digest(self.canonical_payload(for_identity=True))


def _parse_override_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(
    recipe: RegenerationRecipe, overrides: list[str]
) -> RegenerationRecipe:
    raw = recipe.model_dump(mode="json")
    for override in overrides:
        if "=" not in override:
            raise ContractError(f"override {override!r} must be path=value")
        path, encoded = override.split("=", 1)
        keys = path.split(".")
        node: Any = raw
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise ContractError(f"override path {path!r} does not exist")
            node = node[key]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise ContractError(f"override path {path!r} does not exist")
        node[keys[-1]] = _parse_override_value(encoded)
    return RegenerationRecipe.model_validate(raw)


def load_recipe(
    path: str | Path, overrides: list[str] | None = None
) -> RegenerationRecipe:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            raw = yaml.safe_load(handle)
        else:
            raw = json.load(handle)
    recipe = RegenerationRecipe.model_validate(raw)
    return apply_overrides(recipe, overrides or [])


__all__ = [
    "GeneratorSpec",
    "ExecutionSpec",
    "OutputSpec",
    "RegenerationRecipe",
    "RetrySpec",
    "SamplingSpec",
    "SelectionSpec",
    "SourceSpec",
    "StageSpec",
    "ValidationSpec",
    "apply_overrides",
    "load_recipe",
]
