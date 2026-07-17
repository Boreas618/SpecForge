"""Bounded-memory attempt resolution, validation, and atomic publication."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .artifact import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactLayout,
    atomic_write_json,
    file_identity,
    read_json,
    read_state,
    verify_manifest,
    write_state,
)
from .contracts import (
    Finding,
    RecordEnvelope,
    canonical_digest,
    canonical_json,
    contiguous_shard_bounds,
    iter_jsonl,
    jsonl_line,
)
from .errors import ArtifactError, FailureCategory
from .executor import (
    RESOLUTION_POLICY,
    _journal_identity,
    _shard_spec,
    _sidecar_body,
    _verified_plan,
    iter_attempts,
    load_artifact_recipe,
)
from .partitioning import assign_shard, hash_partition
from .planner import PlannedTask
from .registry import VALIDATORS, load_builtin_components

FINALIZER_SCHEMA_VERSION = 1
MAX_FINDING_EXAMPLES = 1000


def _layout(value: str | Path | ArtifactLayout) -> ArtifactLayout:
    return (
        value if isinstance(value, ArtifactLayout) else ArtifactLayout.from_uri(value)
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(jsonl_line(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fresh_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _open_index(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript("""
        CREATE TABLE tasks (
            ordinal INTEGER PRIMARY KEY,
            task_key TEXT NOT NULL UNIQUE,
            source_order INTEGER NOT NULL,
            source_position INTEGER NOT NULL,
            source_name TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            shard_index INTEGER NOT NULL
        );
        CREATE INDEX tasks_by_shard ON tasks(shard_index, ordinal);
        CREATE TABLE attempts (
            task_ordinal INTEGER NOT NULL,
            attempt INTEGER NOT NULL,
            shard_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            category TEXT,
            diagnostic TEXT,
            record_json TEXT NOT NULL,
            PRIMARY KEY (task_ordinal, attempt),
            FOREIGN KEY (task_ordinal) REFERENCES tasks(ordinal)
        );
        CREATE TABLE results (
            result_key TEXT PRIMARY KEY,
            source_order INTEGER NOT NULL,
            source_position INTEGER NOT NULL,
            variant TEXT NOT NULL,
            envelope_json TEXT NOT NULL
        );
        """)
    return connection


def _index_tasks(
    connection: sqlite3.Connection,
    layout: ArtifactLayout,
    plan: Mapping[str, Any],
) -> None:
    expected = 0
    shards = plan["shards"]
    with connection:
        for _, raw in iter_jsonl(layout.tasks):
            task = PlannedTask.from_dict(raw)
            if task.ordinal != expected:
                raise ArtifactError(
                    f"tasks.jsonl expected ordinal {expected}, found {task.ordinal}"
                )
            connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task.ordinal,
                    task.envelope.key.canonical,
                    task.source_order,
                    task.source_position,
                    task.envelope.key.source,
                    task.envelope.source_fingerprint,
                    assign_shard(shards, task.ordinal, task.envelope.key.canonical),
                ),
            )
            expected += 1
    if expected != plan["task_count"]:
        raise ArtifactError(f"task count mismatch ({expected} != {plan['task_count']})")


def _verify_completion(
    layout: ArtifactLayout,
    plan: Mapping[str, Any],
    shard: Mapping[str, int],
) -> None:
    shard_dir = layout.shard_dir(shard["index"])
    expected = _sidecar_body(plan, shard)
    sidecar_path = shard_dir / "sidecar.json"
    complete_path = shard_dir / "complete.json"
    if not sidecar_path.is_file() or read_json(sidecar_path) != expected:
        raise ArtifactError(f"shard {shard['index']} has no compatible sidecar")
    if not complete_path.is_file():
        raise ArtifactError(f"shard {shard['index']} is incomplete")
    actual = read_json(complete_path)
    count, digest = _journal_identity(layout.attempt_dir(shard["index"]))
    completion = {
        **expected,
        "attempt_segments": count,
        "attempt_segments_digest": digest,
    }
    if actual != completion:
        raise ArtifactError(
            f"shard {shard['index']} completion sidecar or attempts were modified"
        )


def _index_attempts(
    connection: sqlite3.Connection,
    layout: ArtifactLayout,
    plan: Mapping[str, Any],
) -> None:
    with connection:
        for shard_value in plan["shards"]:
            shard = _shard_spec(plan, int(shard_value["index"]))
            _verify_completion(layout, plan, shard)
            previous_ordinal: int | None = None
            previous_attempt = 0
            for path, attempt in iter_attempts(layout.attempt_dir(shard["index"])):
                if attempt.get("plan_digest") != plan["plan_digest"]:
                    raise ArtifactError(f"{path}: foreign plan digest")
                if attempt.get("resolution_policy") != RESOLUTION_POLICY:
                    raise ArtifactError(f"{path}: foreign resolution policy")
                if attempt.get("shard_index") != shard["index"]:
                    raise ArtifactError(f"{path}: foreign shard index")
                ordinal = attempt.get("task_ordinal")
                number = attempt.get("attempt")
                if not isinstance(ordinal, int):
                    raise ArtifactError(f"{path}: foreign task position")
                if not isinstance(number, int) or number <= 0:
                    raise ArtifactError(f"{path}: invalid attempt number")
                if ordinal != previous_ordinal:
                    previous_ordinal = ordinal
                    previous_attempt = 0
                expected_number = previous_attempt + 1
                if number != expected_number:
                    raise ArtifactError(
                        f"{path}: expected attempt {expected_number}, found {number}"
                    )
                previous_attempt = number
                task = connection.execute(
                    "SELECT task_key, shard_index FROM tasks WHERE ordinal = ?",
                    (ordinal,),
                ).fetchone()
                if task is None or attempt.get("task_key") != task[0]:
                    raise ArtifactError(f"{path}: task identity mismatch")
                if int(task[1]) != shard["index"]:
                    raise ArtifactError(f"{path}: foreign task position")
                status = attempt.get("status")
                category = attempt.get("category")
                outputs = attempt.get("outputs")
                if status not in {
                    "success",
                    "policy_reject",
                    "terminal_reject",
                    "unresolved_error",
                }:
                    raise ArtifactError(f"{path}: invalid status {status!r}")
                if not isinstance(outputs, list):
                    raise ArtifactError(f"{path}: outputs must be a list")
                if status == "success" and (category is not None or not outputs):
                    raise ArtifactError(f"{path}: malformed success attempt")
                if status != "success" and (not isinstance(category, str) or outputs):
                    raise ArtifactError(f"{path}: malformed non-success attempt")
                try:
                    connection.execute(
                        "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            ordinal,
                            number,
                            shard["index"],
                            status,
                            category,
                            attempt.get("diagnostic"),
                            canonical_json(attempt),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ArtifactError(f"{path}: duplicate attempt identity") from exc


def _latest_rows(connection: sqlite3.Connection):
    return connection.execute("""
        SELECT t.ordinal, t.task_key, t.source_order, t.source_position,
               t.source_name, t.source_fingerprint,
               a.attempt, a.status, a.category, a.diagnostic, a.record_json
          FROM tasks AS t
          LEFT JOIN attempts AS a
            ON a.task_ordinal = t.ordinal
           AND a.attempt = (
               SELECT MAX(a2.attempt)
                 FROM attempts AS a2
                WHERE a2.task_ordinal = t.ordinal
           )
         ORDER BY t.ordinal
        """)


def _resolve_latest(connection: sqlite3.Connection) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    exactness_counts: Counter[str] = Counter()
    resolved_retries = 0

    with connection:
        for row in _latest_rows(connection):
            (
                ordinal,
                task_key,
                source_order,
                source_position,
                source_name,
                source_fingerprint,
                attempt_number,
                status,
                category,
                diagnostic,
                record_json,
            ) = row
            if status is None:
                status = "unresolved_error"
                category = FailureCategory.INTERNAL_ERROR.value
                diagnostic = "planned task has no committed attempt"
            status_counts[status] += 1
            source_counts.setdefault(source_name, Counter())[status] += 1
            if category:
                category_counts[category] += 1
            if attempt_number and attempt_number > 1 and status == "success":
                resolved_retries += 1
            if status != "success":
                continue
            attempt = json.loads(record_json)
            for raw_envelope in attempt["outputs"]:
                envelope = RecordEnvelope.from_dict(raw_envelope)
                if envelope.key.source != source_name:
                    raise ArtifactError(
                        f"task {ordinal} output changed source namespace"
                    )
                if envelope.input_position != source_position:
                    raise ArtifactError(
                        f"task {ordinal} output changed source position"
                    )
                if envelope.source_fingerprint != source_fingerprint:
                    raise ArtifactError(
                        f"task {ordinal} output changed source fingerprint"
                    )
                for event in envelope.stage_history:
                    for generation in event.metadata.get("generations", []):
                        exactness = generation.get("exactness")
                        if exactness:
                            exactness_counts[str(exactness)] += 1
                try:
                    connection.execute(
                        "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
                        (
                            envelope.key.canonical,
                            source_order,
                            source_position,
                            envelope.key.variant,
                            canonical_json(envelope.to_dict()),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ArtifactError(
                        f"duplicate final record key {envelope.key.canonical}"
                    ) from exc

    total = sum(status_counts.values())
    return {
        "planned_tasks": total,
        "output_rows": connection.execute("SELECT COUNT(*) FROM results").fetchone()[0],
        "status_counts": dict(sorted(status_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(source_counts.items())
        },
        "resolved_retries": resolved_retries,
        "exactness_counts": dict(sorted(exactness_counts.items())),
    }


def _result_values(
    connection: sqlite3.Connection, start: int, end: int
) -> Iterator[dict[str, Any]]:
    cursor = connection.execute(
        """
        SELECT envelope_json
          FROM results
         ORDER BY source_order, source_position, variant, result_key
         LIMIT ? OFFSET ?
        """,
        (end - start, start),
    )
    for (encoded,) in cursor:
        yield json.loads(encoded)


def _reject_values(
    connection: sqlite3.Connection, shard_index: int
) -> Iterator[dict[str, Any]]:
    query = """
        SELECT t.ordinal, t.task_key, a.attempt, a.status, a.category, a.diagnostic
          FROM tasks AS t
          LEFT JOIN attempts AS a
            ON a.task_ordinal = t.ordinal
           AND a.attempt = (
               SELECT MAX(a2.attempt)
                 FROM attempts AS a2
                WHERE a2.task_ordinal = t.ordinal
           )
         WHERE t.shard_index = ?
         ORDER BY t.ordinal
    """
    for ordinal, key, attempt, status, category, diagnostic in connection.execute(
        query, (shard_index,)
    ):
        if status == "success":
            continue
        yield {
            "task_ordinal": ordinal,
            "task_key": key,
            "attempt": attempt,
            "status": status or "unresolved_error",
            "category": category or FailureCategory.INTERNAL_ERROR.value,
            "diagnostic": diagnostic or "planned task has no committed attempt",
        }


def _attempt_values(
    connection: sqlite3.Connection, shard_index: int
) -> Iterator[dict[str, Any]]:
    for (record_json,) in connection.execute(
        """
        SELECT record_json FROM attempts
         WHERE shard_index = ?
         ORDER BY task_ordinal, attempt
        """,
        (shard_index,),
    ):
        yield json.loads(record_json)


def resolve_attempts(
    artifact: str | Path | ArtifactLayout,
) -> dict[str, Any]:
    """Resolve attempts and write deterministic candidate artifact payloads."""

    layout = _layout(artifact)
    state = read_state(layout)
    if state == "FINALIZED":
        raise ArtifactError("a finalized artifact is immutable")
    plan = _verified_plan(layout)
    recipe = load_artifact_recipe(layout)
    if recipe.digest != plan["recipe_digest"]:
        raise ArtifactError("recipe no longer matches the immutable run plan")

    index_path = layout.work / "finalize.sqlite3"
    connection = _open_index(index_path)
    try:
        _index_tasks(connection, layout, plan)
        _index_attempts(connection, layout, plan)
        summary = _resolve_latest(connection)

        _fresh_directory(layout.data)
        _fresh_directory(layout.rejects)
        _fresh_directory(layout.attempts)
        for part in range(recipe.output.shards):
            start, end = contiguous_shard_bounds(
                summary["output_rows"], recipe.output.shards, part
            )
            _write_jsonl(
                layout.data / f"part-{part:05d}.jsonl",
                _result_values(connection, start, end),
            )
        for shard in plan["shards"]:
            index = int(shard["index"])
            _write_jsonl(
                layout.rejects / f"part-{index:05d}.jsonl",
                _reject_values(connection, index),
            )
            _write_jsonl(
                layout.attempts / f"part-{index:05d}.jsonl",
                _attempt_values(connection, index),
            )

        summary = {
            "schema_version": FINALIZER_SCHEMA_VERSION,
            "resolution_policy": RESOLUTION_POLICY,
            "recipe_digest": plan["recipe_digest"],
            "plan_digest": plan["plan_digest"],
            **summary,
        }
        atomic_write_json(layout.reports / "summary.json", summary)
        validation_path = layout.reports / "validation.json"
        if validation_path.exists():
            validation_path.unlink()
        if layout.manifest.exists():
            layout.manifest.unlink()
        write_state(
            layout,
            "COMPLETE_UNVALIDATED",
            recipe_digest=plan["recipe_digest"],
            plan_digest=plan["plan_digest"],
            planned_tasks=summary["planned_tasks"],
            output_rows=summary["output_rows"],
        )
        return summary
    finally:
        connection.close()


def _data_envelopes(layout: ArtifactLayout) -> Iterator[RecordEnvelope]:
    for path in sorted(layout.data.glob("part-*.jsonl")):
        for _, value in iter_jsonl(path):
            yield RecordEnvelope.from_dict(value)


def validate_artifact(
    artifact: str | Path | ArtifactLayout,
) -> dict[str, Any]:
    layout = _layout(artifact)
    state = read_state(layout)
    if state not in {"COMPLETE_UNVALIDATED", "VALIDATED", "QUARANTINED"}:
        raise ArtifactError(
            f"validation requires COMPLETE_UNVALIDATED/VALIDATED/QUARANTINED, got {state}"
        )
    recipe = load_artifact_recipe(layout)
    summary = read_json(layout.reports / "summary.json")
    load_builtin_components()
    validators = [
        VALIDATORS.resolve(profile).factory(recipe)
        for profile in recipe.validation.profiles
    ]
    code_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    profile_checked: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    checked = 0
    for envelope in _data_envelopes(layout):
        checked += 1
        for validator in validators:
            # Expensive validators may declare a deterministic sample; the
            # baseline structural profile never samples.
            modulus = int(getattr(validator, "sample_modulus", 1) or 1)
            if modulus > 1 and (
                hash_partition(envelope.key.canonical, modulus) != 0
            ):
                continue
            profile_checked[validator.name] += 1
            for finding in validator.validate(envelope):
                if not isinstance(finding, Finding):
                    raise ArtifactError("validator returned a non-Finding value")
                code_counts[f"{finding.validator}:{finding.code}"] += 1
                severity_counts[finding.severity] += 1
                if len(examples) < MAX_FINDING_EXAMPLES:
                    examples.append(finding.to_dict())

    planned = summary["planned_tasks"]
    unresolved = summary["status_counts"].get("unresolved_error", 0)
    policy_rejects = summary["status_counts"].get("policy_reject", 0)
    unresolved_rate = unresolved / planned if planned else 0.0
    policy_reject_rate = policy_rejects / planned if planned else 0.0
    gates = {
        "validators": severity_counts.get("error", 0) == 0,
        "coverage": sum(summary["status_counts"].values()) == planned,
        "unresolved_error_rate": (
            unresolved_rate <= recipe.validation.max_unresolved_error_rate
        ),
        "policy_reject_rate": (
            policy_reject_rate <= recipe.validation.max_policy_reject_rate
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": FINALIZER_SCHEMA_VERSION,
        "recipe_digest": summary["recipe_digest"],
        "plan_digest": summary["plan_digest"],
        "profiles": recipe.validation.profiles,
        "checked_rows": checked,
        "profile_checked_rows": dict(sorted(profile_checked.items())),
        "finding_counts": dict(sorted(code_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "finding_examples": examples,
        "rates": {
            "unresolved_error": unresolved_rate,
            "policy_reject": policy_reject_rate,
        },
        "thresholds": {
            "max_unresolved_error_rate": recipe.validation.max_unresolved_error_rate,
            "max_policy_reject_rate": recipe.validation.max_policy_reject_rate,
        },
        "gates": gates,
        "passed": passed,
    }
    atomic_write_json(layout.reports / "validation.json", report)
    write_state(
        layout,
        "VALIDATED" if passed else "QUARANTINED",
        recipe_digest=summary["recipe_digest"],
        plan_digest=summary["plan_digest"],
        validation_passed=passed,
    )
    return report


def _immutable_files(layout: ArtifactLayout) -> list[dict[str, Any]]:
    paths = [layout.recipe, layout.plan, layout.tasks]
    for directory in (layout.data, layout.rejects, layout.attempts, layout.reports):
        paths.extend(path for path in directory.rglob("*") if path.is_file())
    identities = [
        file_identity(path, relative_to=layout.root).to_dict()
        for path in sorted(paths, key=lambda value: str(value.relative_to(layout.root)))
    ]
    return identities


def publish_artifact(
    artifact: str | Path | ArtifactLayout,
) -> dict[str, Any]:
    layout = _layout(artifact)
    if layout.manifest.exists():
        return verify_manifest(layout.manifest)
    state = read_state(layout)
    if state != "VALIDATED":
        raise ArtifactError(f"publication requires VALIDATED state, got {state}")
    plan = _verified_plan(layout)
    recipe = load_artifact_recipe(layout)
    summary = read_json(layout.reports / "summary.json")
    validation = read_json(layout.reports / "validation.json")
    if not validation.get("passed"):
        raise ArtifactError("validation report did not pass")
    body = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "state": "FINALIZED",
        "recipe_digest": plan["recipe_digest"],
        "plan_digest": plan["plan_digest"],
        "sources": plan["sources"],
        "generators": {
            name: {
                "backend": spec.backend,
                "model": spec.model,
                "revision": spec.revision,
                "tokenizer": spec.tokenizer,
                "codec": spec.codec,
            }
            for name, spec in sorted(recipe.generators.items())
        },
        "counts": {
            key: summary[key]
            for key in (
                "planned_tasks",
                "output_rows",
                "status_counts",
                "category_counts",
                "source_counts",
                "resolved_retries",
            )
        },
        "exactness_counts": summary["exactness_counts"],
        "validation": {
            "profiles": validation["profiles"],
            "gates": validation["gates"],
            "finding_counts": validation["finding_counts"],
            "passed": validation["passed"],
        },
        "shards": plan["shards"],
        "files": _immutable_files(layout),
        "creation_tool": "specforge.data.regen",
        "parents": [],
    }
    manifest = {**body, "artifact_digest": canonical_digest(body)}
    atomic_write_json(layout.manifest, manifest)
    write_state(
        layout,
        "FINALIZED",
        artifact_digest=manifest["artifact_digest"],
        recipe_digest=plan["recipe_digest"],
        plan_digest=plan["plan_digest"],
    )
    return verify_manifest(layout.manifest)


def finalize_artifact(
    artifact: str | Path | ArtifactLayout,
) -> dict[str, Any]:
    """Resolve, validate, and atomically publish a local v1 artifact."""

    layout = _layout(artifact)
    if layout.manifest.exists():
        return verify_manifest(layout.manifest)
    resolve_attempts(layout)
    report = validate_artifact(layout)
    if not report["passed"]:
        raise ArtifactError("artifact validation failed; state is QUARANTINED")
    return publish_artifact(layout)


def inspect_artifact(
    artifact: str | Path | ArtifactLayout,
) -> dict[str, Any]:
    layout = _layout(artifact)
    if layout.manifest.exists():
        return verify_manifest(layout.manifest)
    value: dict[str, Any] = {"state": read_state(layout)}
    if layout.plan.exists():
        plan = _verified_plan(layout)
        value.update(
            {
                "recipe_digest": plan["recipe_digest"],
                "plan_digest": plan["plan_digest"],
                "task_count": plan["task_count"],
                "shards": plan["shards"],
            }
        )
    if (layout.reports / "summary.json").exists():
        value["summary"] = read_json(layout.reports / "summary.json")
    if (layout.reports / "validation.json").exists():
        value["validation"] = read_json(layout.reports / "validation.json")
    return value


__all__ = [
    "FINALIZER_SCHEMA_VERSION",
    "finalize_artifact",
    "inspect_artifact",
    "publish_artifact",
    "resolve_attempts",
    "validate_artifact",
]
