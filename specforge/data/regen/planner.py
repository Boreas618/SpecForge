"""Resolve a recipe into an immutable, deterministic row-task plan."""

from __future__ import annotations

import hashlib
import heapq
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .artifact import (
    ArtifactLayout,
    atomic_write_json,
    file_identity,
    initialize_layout,
    write_state,
)
from .contracts import (
    RecordEnvelope,
    RecordKey,
    canonical_digest,
    canonical_json,
    contiguous_shard_bounds,
    jsonl_line,
)
from .errors import ArtifactError, CapabilityError, ContractError
from .partitioning import build_shard_specs
from .recipe import RegenerationRecipe, SourceSpec
from .records.messages import NormalizedRecord, preserved_digest
from .registry import (
    BACKENDS,
    CODECS,
    OPERATIONS,
    RECORD_ADAPTERS,
    SOURCE_ADAPTERS,
    TOOL_ENVIRONMENTS,
    VALIDATORS,
    load_builtin_components,
    load_plugin_components,
    registry_snapshot,
)
from .sources.base import SourceIdentity, SourceRow

PLAN_SCHEMA_VERSION = 1
TASK_SCHEMA_VERSION = 1
SELECTION_STRATEGY = "sha256_priority_v1"


def resolve_recipe(recipe: RegenerationRecipe) -> RegenerationRecipe:
    """Resolve every planning-time automatic choice to a registered name."""

    raw = recipe.model_dump(mode="json")
    for generator in raw["generators"].values():
        if generator["codec"] == "auto":
            generator["codec"] = (
                "exact_raw_tokens"
                if generator["backend"] == "sglang_raw"
                else "structured_chat"
            )
    return RegenerationRecipe.model_validate(raw)


def _pin_source_identities(
    recipe: RegenerationRecipe,
) -> tuple[RegenerationRecipe, list[tuple[str, SourceSpec, Any]]]:
    """Instantiate sources and replace runtime locators with pinned identity."""

    sources = []
    raw = recipe.model_dump(mode="json")
    for source_name, source_spec in recipe.sources.items():
        registration = SOURCE_ADAPTERS.resolve(source_spec.adapter)
        source = registration.factory(source_name, source_spec)
        identity = source.identity
        raw["sources"][source_name]["config"] = {
            "resolved_locator": identity.locator,
            "fingerprint": identity.fingerprint,
            "rows": identity.rows,
            "revision": identity.revision,
        }
        sources.append((source_name, source_spec, source))
    return RegenerationRecipe.model_validate(raw), sources


@dataclass(frozen=True)
class PlannedTask:
    ordinal: int
    source_order: int
    source_position: int
    envelope: RecordEnvelope

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_SCHEMA_VERSION,
            "ordinal": self.ordinal,
            "source_order": self.source_order,
            "source_position": self.source_position,
            "task_key": self.envelope.key.canonical,
            "envelope": self.envelope.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlannedTask":
        envelope = RecordEnvelope.from_dict(value["envelope"])
        if value.get("task_key") != envelope.key.canonical:
            raise ArtifactError("task key does not match its envelope")
        return cls(
            ordinal=int(value["ordinal"]),
            source_order=int(value["source_order"]),
            source_position=int(value["source_position"]),
            envelope=envelope,
        )


def _validate_components(recipe: RegenerationRecipe) -> dict[str, Any]:
    load_builtin_components()
    load_plugin_components(recipe.plugins)
    for source in recipe.sources.values():
        SOURCE_ADAPTERS.resolve(
            source.adapter, required_capabilities={"text_trajectory"}
        )
        RECORD_ADAPTERS.resolve(
            source.record_adapter, required_capabilities={"text_trajectory"}
        )
    for generator in recipe.generators.values():
        if generator.backend != "fake" and not generator.revision:
            raise ContractError(
                f"generator model {generator.model!r} requires a pinned revision"
            )
        backend_required = {"chat", "text_trajectory"}
        if generator.sampling.reasoning == "required":
            backend_required.add("reasoning_split")
        backend = BACKENDS.resolve(
            generator.backend, required_capabilities=backend_required
        )
        codec = CODECS.resolve(
            generator.codec, required_capabilities={"text_trajectory"}
        )
        if (
            "exact_rebuild" in codec.capabilities
            and "raw_token_ids" not in backend.capabilities
        ):
            raise CapabilityError(
                f"codec {generator.codec!r} requires a raw-token generation backend"
            )
        if (
            "raw_token_ids" in backend.capabilities
            and "exact_rebuild" not in codec.capabilities
        ):
            raise CapabilityError(
                f"raw-token backend {generator.backend!r} requires an exact-rebuild codec"
            )
    for stage in recipe.workflow:
        operation = OPERATIONS.resolve(stage.operation)
        if "chat" in operation.capabilities and stage.generator is None:
            raise CapabilityError(
                f"stage {stage.id!r} operation {stage.operation!r} requires a generator"
            )
        if stage.candidates > 1 and "multiple_candidates" not in operation.capabilities:
            raise CapabilityError(
                f"stage {stage.id!r} does not support multiple candidates"
            )
        if stage.tool_policy != "reject" and "tools" not in operation.capabilities:
            raise CapabilityError(
                f"stage {stage.id!r} tool policy is not supported by {stage.operation!r}"
            )
        if stage.tool_policy == "execute":
            OPERATIONS.resolve(
                stage.operation, required_capabilities={"tool_execution"}
            )
            environment_name = stage.config.get("environment")
            if not isinstance(environment_name, str) or not environment_name:
                raise ContractError(
                    f"stage {stage.id!r} tool execution requires "
                    "config.environment"
                )
            required = {"tool_execution"}
            if stage.config.get("side_effect_policy", "forbid") == "forbid":
                required.add("no_side_effects")
            TOOL_ENVIRONMENTS.resolve(
                environment_name, required_capabilities=required
            )
        elif "tool_execution" in operation.capabilities:
            raise CapabilityError(
                f"stage {stage.id!r} operation {stage.operation!r} executes "
                "tools and requires tool_policy 'execute'"
            )
        if stage.generator is not None:
            generator = recipe.generators[stage.generator]
            backend = BACKENDS.resolve(generator.backend)
            codec = CODECS.resolve(generator.codec)
            if (
                stage.candidates > 1
                and "multiple_candidates" not in backend.capabilities
            ):
                raise CapabilityError(
                    f"backend {generator.backend!r} does not support multiple candidates"
                )
            if stage.tool_policy != "reject":
                for kind, registration in (("backend", backend), ("codec", codec)):
                    if "tools" not in registration.capabilities:
                        raise CapabilityError(
                            f"{kind} {registration.name!r} does not support tools"
                        )
    for profile in recipe.validation.profiles:
        registration = VALIDATORS.resolve(
            profile, required_capabilities={"text_trajectory"}
        )
        # Constructing the validator surfaces profile configuration errors at
        # plan time instead of after generation; heavyweight renderer and
        # tokenizer imports stay deferred to the first validated row.
        registration.factory(recipe)
    return registry_snapshot()


def _normalize_row(
    source_name: str,
    source_identity: SourceIdentity,
    source_order: int,
    row: SourceRow,
    record_factory,
) -> PlannedTask:
    normalized: NormalizedRecord = record_factory(
        row.value, source_name=source_name, position=row.position
    )
    payload_digest = canonical_digest(normalized.payload)
    envelope = RecordEnvelope(
        key=RecordKey(source=source_name, source_id=normalized.source_id),
        input_position=row.position,
        payload=normalized.payload,
        source_fingerprint=source_identity.fingerprint,
        source_payload_digest=payload_digest,
        source_preserved_digest=preserved_digest(normalized.payload),
    )
    return PlannedTask(
        ordinal=-1,
        source_order=source_order,
        source_position=row.position,
        envelope=envelope,
    )


def _sample_rank(seed: int, task: PlannedTask) -> int:
    payload = {
        "strategy": SELECTION_STRATEGY,
        "seed": seed,
        "record_key": task.envelope.key.to_dict(),
    }
    return int.from_bytes(
        hashlib.sha256(canonical_json(payload).encode("utf-8")).digest(), "big"
    )


def _selected_tasks(
    source_name: str,
    source_spec: SourceSpec,
    source_identity: SourceIdentity,
    source_order: int,
    rows: Iterable[SourceRow],
    record_factory,
) -> Iterator[PlannedTask]:
    selection = source_spec.selection
    seen: set[str] = set()

    def normalize(row: SourceRow) -> PlannedTask:
        task = _normalize_row(
            source_name,
            source_identity,
            source_order,
            row,
            record_factory,
        )
        key = task.envelope.key.canonical
        if key in seen:
            raise ContractError(f"source {source_name!r} contains duplicate id {key}")
        seen.add(key)
        return task

    if selection.mode == "all":
        for row in rows:
            yield normalize(row)
        return

    if selection.mode == "indices":
        wanted = set(selection.indices)
        for row in rows:
            if row.position in wanted:
                yield normalize(row)
                wanted.remove(row.position)
                if not wanted:
                    break
        if wanted:
            raise ContractError(
                f"source {source_name!r}: selection indices not found: {sorted(wanted)}"
            )
        return

    sample_size = int(selection.sample or 0)
    seed = selection.seed if selection.seed is not None else 0
    heap: list[tuple[int, str, PlannedTask]] = []
    eligible = 0
    for row in rows:
        task = normalize(row)
        eligible += 1
        rank = _sample_rank(seed, task)
        item = (-rank, task.envelope.key.canonical, task)
        if len(heap) < sample_size:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    if eligible < sample_size:
        raise ContractError(
            f"source {source_name!r}: sample {sample_size} exceeds eligible rows {eligible}"
        )
    selected = [item[2] for item in heap]
    selected.sort(key=lambda task: task.source_position)
    yield from selected


def plan_recipe(recipe: RegenerationRecipe) -> ArtifactLayout:
    recipe = resolve_recipe(recipe)
    registry = _validate_components(recipe)
    recipe, pinned_sources = _pin_source_identities(recipe)
    layout = ArtifactLayout.from_uri(recipe.output.uri)
    initialize_layout(layout)
    # Output placement is runtime state, not dataset semantics. Workers restore
    # the artifact root while loading this identity recipe.
    atomic_write_json(layout.recipe, recipe.canonical_payload(for_identity=True))

    temporary_tasks = layout.tasks.with_name(f".{layout.tasks.name}.{os.getpid()}.tmp")
    source_reports: list[dict[str, Any]] = []
    task_count = 0
    try:
        with temporary_tasks.open("w", encoding="utf-8") as handle:
            for source_order, (source_name, source_spec, source) in enumerate(
                pinned_sources
            ):
                record_registration = RECORD_ADAPTERS.resolve(
                    source_spec.record_adapter
                )
                selected_count = 0
                for task in _selected_tasks(
                    source_name,
                    source_spec,
                    source.identity,
                    source_order,
                    source.iter_rows(),
                    record_registration.factory,
                ):
                    task = PlannedTask(
                        ordinal=task_count,
                        source_order=task.source_order,
                        source_position=task.source_position,
                        envelope=task.envelope,
                    )
                    handle.write(jsonl_line(task.to_dict()))
                    task_count += 1
                    selected_count += 1
                source_reports.append(
                    {
                        "name": source_name,
                        **source.identity.to_dict(),
                        "record_adapter": source_spec.record_adapter,
                        "selection": source_spec.selection.model_dump(mode="json"),
                        "selected_rows": selected_count,
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_tasks, layout.tasks)
    finally:
        if temporary_tasks.exists():
            temporary_tasks.unlink()

    tasks_identity = file_identity(layout.tasks, relative_to=layout.root)
    shards = build_shard_specs(
        recipe.output.partitioning, task_count, recipe.output.shards
    )

    plan_body = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "recipe_digest": recipe.digest,
        "task_count": task_count,
        "tasks": tasks_identity.to_dict(),
        "sources": source_reports,
        "shards": shards,
        "registries": registry,
        "selection_strategy": SELECTION_STRATEGY,
    }
    plan = {**plan_body, "plan_digest": canonical_digest(plan_body)}
    atomic_write_json(layout.plan, plan)
    write_state(
        layout,
        "PLANNED",
        recipe_digest=recipe.digest,
        plan_digest=plan["plan_digest"],
        task_count=task_count,
    )
    return layout


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "PlannedTask",
    "SELECTION_STRATEGY",
    "plan_recipe",
    "resolve_recipe",
]
