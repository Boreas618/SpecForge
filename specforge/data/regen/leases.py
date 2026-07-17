"""Regeneration-owned shard lease, heartbeat, and reconcile interface.

Ownership of every task is pinned by the immutable run plan; leases only
schedule which worker executes a shard next. A lease carries a monotonically
increasing fence so a worker that lost its lease (expiry, restart, network
partition) can never complete or extend a shard that was reclaimed by
another worker. This interface is deliberately independent of any
feature-capture runtime contract.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifact import ArtifactLayout
from .errors import ArtifactError, RegenerationError


class LeaseLostError(RegenerationError):
    """The worker no longer owns the shard it is executing."""


class ShardLeaseStore:
    """SQLite-backed lease table shared by cooperating local workers.

    Remote deployments can substitute any store honoring the same claim /
    heartbeat / complete / reconcile fencing contract.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        plan_digest: str,
        shard_indexes: list[int],
        lease_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if lease_seconds <= 0:
            raise ArtifactError("lease_seconds must be positive")
        self.path = Path(path)
        self.plan_digest = plan_digest
        self.lease_seconds = float(lease_seconds)
        self.clock = clock
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS leases (
                        shard_index INTEGER PRIMARY KEY,
                        plan_digest TEXT NOT NULL,
                        worker_id TEXT,
                        fence INTEGER NOT NULL DEFAULT 0,
                        expires_at REAL,
                        completed INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                recorded = connection.execute(
                    "SELECT DISTINCT plan_digest FROM leases"
                ).fetchall()
                if recorded and recorded != [(plan_digest,)]:
                    raise ArtifactError(
                        "lease store belongs to a different run plan"
                    )
                for shard_index in shard_indexes:
                    connection.execute(
                        "INSERT OR IGNORE INTO leases "
                        "(shard_index, plan_digest) VALUES (?, ?)",
                        (int(shard_index), plan_digest),
                    )
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def claim(self, worker_id: str) -> Mapping[str, Any] | None:
        """Atomically claim one incomplete shard; return its lease or None."""

        now = self.clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT shard_index, fence FROM leases
                 WHERE completed = 0
                   AND (worker_id IS NULL OR expires_at < ?)
                 ORDER BY shard_index
                 LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            shard_index, fence = int(row[0]), int(row[1]) + 1
            connection.execute(
                """
                UPDATE leases
                   SET worker_id = ?, fence = ?, expires_at = ?
                 WHERE shard_index = ?
                """,
                (worker_id, fence, now + self.lease_seconds, shard_index),
            )
            connection.commit()
            return {"shard_index": shard_index, "fence": fence}
        finally:
            connection.close()

    def _owned(
        self, connection: sqlite3.Connection, worker_id: str, lease: Mapping[str, Any]
    ) -> bool:
        row = connection.execute(
            "SELECT worker_id, fence, completed FROM leases WHERE shard_index = ?",
            (int(lease["shard_index"]),),
        ).fetchone()
        return (
            row is not None
            and row[0] == worker_id
            and int(row[1]) == int(lease["fence"])
            and not row[2]
        )

    def heartbeat(self, worker_id: str, lease: Mapping[str, Any]) -> bool:
        """Extend the lease; False means it was reclaimed and must be dropped."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not self._owned(connection, worker_id, lease):
                connection.rollback()
                return False
            connection.execute(
                "UPDATE leases SET expires_at = ? WHERE shard_index = ?",
                (self.clock() + self.lease_seconds, int(lease["shard_index"])),
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def complete(self, worker_id: str, lease: Mapping[str, Any]) -> bool:
        """Mark the shard complete iff this worker still holds the fence."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not self._owned(connection, worker_id, lease):
                connection.rollback()
                return False
            connection.execute(
                "UPDATE leases SET completed = 1, worker_id = NULL, "
                "expires_at = NULL WHERE shard_index = ?",
                (int(lease["shard_index"]),),
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def release(self, worker_id: str, lease: Mapping[str, Any]) -> None:
        """Voluntarily return an incomplete shard to the pool."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if self._owned(connection, worker_id, lease):
                connection.execute(
                    "UPDATE leases SET worker_id = NULL, expires_at = NULL "
                    "WHERE shard_index = ?",
                    (int(lease["shard_index"]),),
                )
            connection.commit()
        finally:
            connection.close()

    def reconcile(self) -> dict[str, Any]:
        """Report progress and return expired leases to the pool."""

        now = self.clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            expired = [
                int(row[0])
                for row in connection.execute(
                    "SELECT shard_index FROM leases "
                    "WHERE completed = 0 AND worker_id IS NOT NULL "
                    "AND expires_at < ?",
                    (now,),
                )
            ]
            connection.execute(
                "UPDATE leases SET worker_id = NULL, expires_at = NULL "
                "WHERE completed = 0 AND worker_id IS NOT NULL "
                "AND expires_at < ?",
                (now,),
            )
            counts = dict(
                connection.execute(
                    "SELECT CASE WHEN completed = 1 THEN 'completed' "
                    "WHEN worker_id IS NOT NULL THEN 'leased' "
                    "ELSE 'pending' END AS state, COUNT(*) "
                    "FROM leases GROUP BY state"
                ).fetchall()
            )
            connection.commit()
        finally:
            connection.close()
        return {
            "expired_reclaimed": expired,
            "completed": int(counts.get("completed", 0)),
            "leased": int(counts.get("leased", 0)),
            "pending": int(counts.get("pending", 0)),
        }


def run_leased(
    artifact: str | Path | ArtifactLayout,
    worker_id: str,
    *,
    runtime: Mapping[str, Mapping[str, Any]] | None = None,
    lease_store: ShardLeaseStore | None = None,
    lease_seconds: float = 60.0,
    clock: Callable[[], float] = time.time,
    fault_hook: Callable[..., None] | None = None,
) -> list[dict[str, Any]]:
    """Claim, execute, and complete plan shards until the pool is drained.

    The lease heartbeats on every attempt commit. If the lease was reclaimed
    (for example after a stall longer than ``lease_seconds``), the worker
    stops writing that shard immediately and moves on; the reclaiming worker
    resumes from the committed attempt journal exactly.
    """

    from .executor import _verified_plan, run_worker

    layout = (
        artifact
        if isinstance(artifact, ArtifactLayout)
        else ArtifactLayout.from_uri(artifact)
    )
    plan = _verified_plan(layout)
    store = lease_store or ShardLeaseStore(
        layout.work / "leases.sqlite3",
        plan_digest=plan["plan_digest"],
        shard_indexes=[int(shard["index"]) for shard in plan["shards"]],
        lease_seconds=lease_seconds,
        clock=clock,
    )

    results: list[dict[str, Any]] = []
    while True:
        lease = store.claim(worker_id)
        if lease is None:
            break

        def guarded_hook(stage: str, task: Any, attempt: int) -> None:
            if stage == "before_commit" and not store.heartbeat(worker_id, lease):
                raise LeaseLostError(
                    f"worker {worker_id} lost shard {lease['shard_index']}"
                )
            if fault_hook is not None:
                fault_hook(stage, task, attempt)

        try:
            result = run_worker(
                layout,
                int(lease["shard_index"]),
                runtime=runtime,
                fault_hook=guarded_hook,
            )
        except LeaseLostError:
            continue
        if store.complete(worker_id, lease):
            results.append({**result, "worker_id": worker_id})
    return results


__all__ = ["LeaseLostError", "ShardLeaseStore", "run_leased"]
