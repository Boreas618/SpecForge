"""Restart reconciliation and workspace hygiene for running artifacts.

Committed attempt segments are immutable evidence and are never rewritten or
merged here — the finalizer's SQLite index is the compacted view of the
journals. Maintenance is limited to removing uncommitted temporary files
left by killed writers, returning expired leases to the pool, and reporting
shard progress so operators can restart with confidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact import ArtifactLayout, read_json, read_state
from .executor import _verified_plan, iter_attempts
from .leases import ShardLeaseStore
from .observability import log_event


def cleanup_orphans(artifact: str | Path | ArtifactLayout) -> list[str]:
    """Remove uncommitted temporary files; committed evidence is untouched."""

    layout = (
        artifact
        if isinstance(artifact, ArtifactLayout)
        else ArtifactLayout.from_uri(artifact)
    )
    removed: list[str] = []
    if not layout.root.exists():
        return removed
    for path in layout.root.rglob(".*.tmp"):
        if path.is_file():
            path.unlink()
            removed.append(str(path.relative_to(layout.root)))
    if removed:
        log_event("orphans_removed", artifact_state=read_state(layout), count=len(removed))
    return sorted(removed)


def reconcile_artifact(
    artifact: str | Path | ArtifactLayout,
    *,
    lease_store: ShardLeaseStore | None = None,
) -> dict[str, Any]:
    """Report shard progress and reclaim expired leases after a restart."""

    layout = (
        artifact
        if isinstance(artifact, ArtifactLayout)
        else ArtifactLayout.from_uri(artifact)
    )
    plan = _verified_plan(layout)
    orphans = cleanup_orphans(layout)

    shards = []
    for shard in plan["shards"]:
        index = int(shard["index"])
        shard_dir = layout.shard_dir(index)
        attempt_dir = layout.attempt_dir(index)
        attempts = sum(1 for _ in iter_attempts(attempt_dir))
        complete_path = shard_dir / "complete.json"
        shards.append(
            {
                "index": index,
                "has_sidecar": (shard_dir / "sidecar.json").is_file(),
                "attempt_segments": attempts,
                "complete": complete_path.is_file()
                and read_json(complete_path).get("plan_digest")
                == plan["plan_digest"],
            }
        )

    leases: dict[str, Any] | None = None
    store = lease_store
    if store is None and (layout.work / "leases.sqlite3").is_file():
        store = ShardLeaseStore(
            layout.work / "leases.sqlite3",
            plan_digest=plan["plan_digest"],
            shard_indexes=[int(shard["index"]) for shard in plan["shards"]],
        )
    if store is not None:
        leases = store.reconcile()

    report = {
        "state": read_state(layout),
        "plan_digest": plan["plan_digest"],
        "orphans_removed": orphans,
        "shards": shards,
        "incomplete_shards": [s["index"] for s in shards if not s["complete"]],
        "leases": leases,
    }
    log_event(
        "artifact_reconciled",
        state=report["state"],
        incomplete_shards=len(report["incomplete_shards"]),
        orphans_removed=len(orphans),
    )
    return report


__all__ = ["cleanup_orphans", "reconcile_artifact"]
