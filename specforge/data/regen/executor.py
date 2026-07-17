"""Crash-safe static-shard execution and exact resume."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .artifact import (
    ArtifactLayout,
    atomic_write_json,
    exclusive_lock,
    file_identity,
    read_json,
    read_state,
    write_state,
)
from .contracts import canonical_digest, iter_jsonl
from .errors import (
    ArtifactError,
    CapabilityError,
    ContractError,
    FailureCategory,
    GenerationError,
    PolicyReject,
    safe_diagnostic,
)
from .observability import METRICS, log_event
from .partitioning import shard_member, validate_shard_spec
from .pipeline import Pipeline
from .planner import PlannedTask
from .recipe import RegenerationRecipe

ATTEMPT_SCHEMA_VERSION = 1
SIDECAR_SCHEMA_VERSION = 1
RESOLUTION_POLICY = "latest_attempt_v1"


def load_artifact_recipe(layout: ArtifactLayout) -> RegenerationRecipe:
    raw = read_json(layout.recipe)
    # Artifact placement is intentionally omitted from identity recipes. Older
    # planned artifacts may contain it; both representations load identically.
    output = dict(raw.get("output") or {})
    output.setdefault("uri", str(layout.root))
    raw["output"] = output
    return RegenerationRecipe.model_validate(raw)


def _verified_plan(layout: ArtifactLayout) -> dict[str, Any]:
    plan = read_json(layout.plan)
    recorded = plan.get("plan_digest")
    body = dict(plan)
    body.pop("plan_digest", None)
    actual = canonical_digest(body)
    if recorded != actual:
        raise ArtifactError(f"run plan digest mismatch ({recorded!r} != {actual!r})")
    tasks = file_identity(layout.tasks, relative_to=layout.root).to_dict()
    if tasks != plan.get("tasks"):
        raise ArtifactError("tasks.jsonl no longer matches the immutable run plan")
    return plan


def _shard_spec(plan: Mapping[str, Any], shard_index: int) -> dict[str, Any]:
    for value in plan.get("shards", []):
        if value.get("index") == shard_index:
            return validate_shard_spec(value)
    raise ArtifactError(f"run plan has no shard {shard_index}")


def _sidecar_body(plan: Mapping[str, Any], shard: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "plan_digest": plan["plan_digest"],
        "recipe_digest": plan["recipe_digest"],
        "tasks_sha256": plan["tasks"]["sha256"],
        "resolution_policy": RESOLUTION_POLICY,
    }
    body.update(validate_shard_spec(shard))
    body["shard_index"] = body.pop("index")
    return body


def _ensure_sidecar(path: Path, expected: Mapping[str, Any]) -> None:
    if path.exists():
        actual = read_json(path)
        if actual != dict(expected):
            raise ArtifactError(f"incompatible shard sidecar: {path}")
        return
    atomic_write_json(path, dict(expected))


def _attempt_path(directory: Path, ordinal: int, attempt: int) -> Path:
    return directory / f"task-{ordinal:012d}-attempt-{attempt:04d}.json"


def iter_attempts(directory: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    if not directory.exists():
        return
    for path in sorted(directory.glob("task-*-attempt-*.json")):
        yield path, read_json(path)


def _existing_attempts(
    directory: Path,
    *,
    plan_digest: str,
    shard: Mapping[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    shard_index = int(shard["index"])
    by_task: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[int, int]] = set()
    for path, attempt in iter_attempts(directory):
        if attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
            raise ArtifactError(f"{path}: unsupported attempt schema")
        if attempt.get("plan_digest") != plan_digest:
            raise ArtifactError(f"{path}: foreign plan digest")
        if attempt.get("shard_index") != shard_index:
            raise ArtifactError(f"{path}: foreign shard index")
        ordinal = attempt.get("task_ordinal")
        task_key = attempt.get("task_key")
        number = attempt.get("attempt")
        if (
            not isinstance(ordinal, int)
            or not isinstance(task_key, str)
            or not shard_member(shard, ordinal, task_key)
        ):
            raise ArtifactError(f"{path}: foreign task position")
        if not isinstance(number, int) or number <= 0:
            raise ArtifactError(f"{path}: invalid attempt number")
        identity = (ordinal, number)
        if identity in seen:
            raise ArtifactError(f"duplicate attempt identity {identity}")
        seen.add(identity)
        by_task.setdefault(ordinal, []).append(attempt)
    for ordinal, attempts in by_task.items():
        attempts.sort(key=lambda value: value["attempt"])
        expected = list(range(1, len(attempts) + 1))
        actual = [value["attempt"] for value in attempts]
        if actual != expected:
            raise ArtifactError(
                f"task {ordinal} has non-contiguous attempt numbers {actual}"
            )
    return by_task


def _iter_tasks(layout: ArtifactLayout, shard: Mapping[str, Any]):
    if shard.get("strategy") == "hash":
        yield from _iter_hash_tasks(layout, shard)
        return
    start, end = int(shard["start"]), int(shard["end"])
    expected = start
    for _, raw in iter_jsonl(layout.tasks):
        ordinal = raw.get("ordinal")
        if not isinstance(ordinal, int):
            raise ArtifactError("tasks.jsonl contains a non-integer ordinal")
        if ordinal < start:
            continue
        if ordinal >= end:
            break
        if ordinal != expected:
            raise ArtifactError(
                f"tasks.jsonl expected ordinal {expected}, found {ordinal}"
            )
        yield PlannedTask.from_dict(raw)
        expected += 1
    if expected != end:
        raise ArtifactError(
            f"tasks.jsonl ended at ordinal {expected}; shard requires {end}"
        )


def _iter_hash_tasks(layout: ArtifactLayout, shard: Mapping[str, Any]):
    expected = 0
    for _, raw in iter_jsonl(layout.tasks):
        ordinal = raw.get("ordinal")
        task_key = raw.get("task_key")
        if not isinstance(ordinal, int) or not isinstance(task_key, str):
            raise ArtifactError("tasks.jsonl contains an invalid task record")
        if ordinal != expected:
            raise ArtifactError(
                f"tasks.jsonl expected ordinal {expected}, found {ordinal}"
            )
        expected += 1
        # Membership is checked on the cheap raw fields so non-member rows
        # skip full task validation.
        if shard_member(shard, ordinal, task_key):
            yield PlannedTask.from_dict(raw)


def _category(exc: BaseException) -> FailureCategory:
    if isinstance(exc, PolicyReject):
        return FailureCategory.POLICY_REJECT
    if isinstance(exc, GenerationError):
        return exc.category
    if isinstance(exc, CapabilityError):
        return FailureCategory.CAPABILITY_ERROR
    if isinstance(exc, ContractError):
        return FailureCategory.SOURCE_INVALID
    return FailureCategory.INTERNAL_ERROR


def _status(category: FailureCategory | None) -> str:
    if category is None:
        return "success"
    if category == FailureCategory.POLICY_REJECT:
        return "policy_reject"
    if category in {
        FailureCategory.SOURCE_INVALID,
        FailureCategory.GENERATION_INVALID,
        FailureCategory.EXACT_REBUILD,
    }:
        return "terminal_reject"
    return "unresolved_error"


def _attempt_record(
    *,
    plan_digest: str,
    shard_index: int,
    task: PlannedTask,
    attempt: int,
    outputs: list | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    category = _category(error) if error is not None else None
    diagnostic = None
    if error is not None:
        diagnostic = (
            type(error).__name__
            if category == FailureCategory.INTERNAL_ERROR
            else safe_diagnostic(error)
        )
    value: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "resolution_policy": RESOLUTION_POLICY,
        "plan_digest": plan_digest,
        "shard_index": shard_index,
        "task_ordinal": task.ordinal,
        "task_key": task.envelope.key.canonical,
        "attempt": attempt,
        "status": _status(category),
        "category": category.value if category else None,
        "diagnostic": diagnostic,
        "request_digest": getattr(error, "request_digest", None),
        "request_seed": getattr(error, "request_seed", None),
        "outputs": [item.to_dict() for item in (outputs or [])],
    }
    return value


def _journal_identity(directory: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for path, _ in iter_attempts(directory):
        identity = file_identity(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(identity.sha256.encode("ascii"))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def run_worker(
    artifact: str | Path | ArtifactLayout,
    shard_index: int,
    *,
    runtime: Mapping[str, Mapping[str, Any]] | None = None,
    retry_categories: set[str] | None = None,
    fault_hook: Callable[[str, PlannedTask, int], None] | None = None,
) -> dict[str, Any]:
    """Run or exactly resume one immutable static shard.

    ``retry_categories`` is used by the explicit retry command. It schedules one
    additional attempt for an already resolved error in a listed retry-eligible
    category, while ordinary resume respects the recipe's attempt budget.
    """

    layout = (
        artifact
        if isinstance(artifact, ArtifactLayout)
        else ArtifactLayout.from_uri(artifact)
    )
    plan = _verified_plan(layout)
    recipe = load_artifact_recipe(layout)
    if recipe.digest != plan["recipe_digest"]:
        raise ArtifactError("recipe no longer matches the immutable run plan")
    shard = _shard_spec(plan, shard_index)
    shard_dir = layout.shard_dir(shard_index)
    attempt_dir = layout.attempt_dir(shard_index)
    shard_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    with exclusive_lock(shard_dir / ".worker.lock"):
        _ensure_sidecar(shard_dir / "sidecar.json", _sidecar_body(plan, shard))
        existing = _existing_attempts(
            attempt_dir,
            plan_digest=plan["plan_digest"],
            shard=shard,
        )
        pipeline = Pipeline(recipe, runtime)
        wrote = 0
        for task in _iter_tasks(layout, shard):
            history = list(existing.get(task.ordinal, []))
            latest = history[-1] if history else None
            explicit_retry = bool(
                retry_categories
                and latest
                and latest.get("status") == "unresolved_error"
                and latest.get("category") in retry_categories
            )
            if retry_categories is not None:
                if not explicit_retry:
                    continue
                remaining_attempts = 1
            elif latest:
                if latest["status"] != "unresolved_error":
                    continue
                if latest.get("category") not in recipe.execution.retry.categories:
                    continue
                if len(history) >= recipe.execution.retry.max_attempts:
                    continue
                remaining_attempts = recipe.execution.retry.max_attempts - len(history)
            else:
                remaining_attempts = recipe.execution.retry.max_attempts

            for _ in range(remaining_attempts):
                attempt_number = len(history) + 1
                if fault_hook:
                    fault_hook("before_generate", task, attempt_number)
                error: BaseException | None = None
                outputs = []
                try:
                    outputs = pipeline.run(task.envelope)
                    if not outputs:
                        raise PolicyReject("workflow emitted no output variants")
                except (
                    Exception
                ) as exc:  # journal first; fatal categories re-raise below
                    error = exc
                record = _attempt_record(
                    plan_digest=plan["plan_digest"],
                    shard_index=shard_index,
                    task=task,
                    attempt=attempt_number,
                    outputs=outputs,
                    error=error,
                )
                if fault_hook:
                    fault_hook("before_commit", task, attempt_number)
                atomic_write_json(
                    _attempt_path(attempt_dir, task.ordinal, attempt_number), record
                )
                METRICS.increment(f"attempts.{record['status']}")
                log_event(
                    "attempt_committed",
                    shard_index=shard_index,
                    task_ordinal=task.ordinal,
                    attempt=attempt_number,
                    status=record["status"],
                    category=record["category"],
                )
                history.append(record)
                wrote += 1
                if fault_hook:
                    fault_hook("after_commit", task, attempt_number)
                category = _category(error) if error is not None else None
                if category in {
                    FailureCategory.CAPABILITY_ERROR,
                    FailureCategory.INTERNAL_ERROR,
                }:
                    raise error
                if (
                    error is None
                    or retry_categories is not None
                    or category.value not in recipe.execution.retry.categories
                ):
                    break
                backoff = recipe.execution.retry.backoff_seconds
                if backoff > 0:
                    time.sleep(backoff * (2 ** (attempt_number - 1)))
            existing[task.ordinal] = history

        count, digest = _journal_identity(attempt_dir)
        completion = {
            **_sidecar_body(plan, shard),
            "attempt_segments": count,
            "attempt_segments_digest": digest,
        }
        atomic_write_json(shard_dir / "complete.json", completion)

    state = read_state(layout)
    if state == "PLANNED":
        write_state(
            layout,
            "RUNNING",
            recipe_digest=plan["recipe_digest"],
            plan_digest=plan["plan_digest"],
        )
    return {
        **{key: value for key, value in shard.items() if key != "index"},
        "shard_index": shard_index,
        "attempts_written": wrote,
        "attempt_segments": count,
        "attempt_segments_digest": digest,
    }


def run_local(
    artifact: str | Path | ArtifactLayout,
    *,
    runtime: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    layout = (
        artifact
        if isinstance(artifact, ArtifactLayout)
        else ArtifactLayout.from_uri(artifact)
    )
    plan = _verified_plan(layout)
    return [
        run_worker(layout, shard["index"], runtime=runtime) for shard in plan["shards"]
    ]


def prepare_retry(artifact: str | Path | ArtifactLayout) -> ArtifactLayout:
    """Invalidate derived publication candidates before an explicit retry."""

    layout = (
        artifact
        if isinstance(artifact, ArtifactLayout)
        else ArtifactLayout.from_uri(artifact)
    )
    if layout.manifest.exists() or read_state(layout) == "FINALIZED":
        raise ArtifactError(
            "a finalized artifact is immutable; create a new recipe/run"
        )
    with exclusive_lock(layout.work / ".artifact.lock"):
        for directory in (layout.data, layout.rejects, layout.attempts):
            if directory.exists():
                shutil.rmtree(directory)
        for path in (
            layout.reports / "summary.json",
            layout.reports / "validation.json",
        ):
            if path.exists():
                path.unlink()
        plan = _verified_plan(layout)
        write_state(
            layout,
            "RUNNING",
            recipe_digest=plan["recipe_digest"],
            plan_digest=plan["plan_digest"],
            retry=True,
        )
    return layout


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "RESOLUTION_POLICY",
    "SIDECAR_SCHEMA_VERSION",
    "iter_attempts",
    "load_artifact_recipe",
    "prepare_retry",
    "run_local",
    "run_worker",
]
