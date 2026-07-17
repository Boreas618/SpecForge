"""Shard assignment strategies for immutable run plans.

Static contiguous ranges are the default. Hash partitioning assigns tasks to
shards by a stable digest of the type-preserving task key, so membership is
independent of task ordering and worker count. Both strategies pin ownership
in the plan; execution topology can only change scheduling.
"""

from __future__ import annotations

import bisect
import hashlib
from typing import Any, Mapping, Sequence

from .contracts import contiguous_shard_bounds
from .errors import ArtifactError

HASH_STRATEGY_VERSION = "sha256_key_mod_v1"


def hash_partition(task_key: str, modulus: int) -> int:
    """Stable hash partition of a canonical task key."""

    if modulus <= 0:
        raise ArtifactError("hash partition modulus must be positive")
    digest = hashlib.sha256(task_key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulus


def build_shard_specs(
    partitioning: str, task_count: int, shards: int
) -> list[dict[str, Any]]:
    """Materialize the plan's immutable shard ownership records."""

    if partitioning == "hash":
        return [
            {
                "index": index,
                "strategy": "hash",
                "hash_strategy": HASH_STRATEGY_VERSION,
                "modulus": shards,
            }
            for index in range(shards)
        ]
    if partitioning == "contiguous":
        specs = []
        for index in range(shards):
            start, end = contiguous_shard_bounds(task_count, shards, index)
            specs.append(
                {
                    "index": index,
                    "strategy": "contiguous",
                    "start": start,
                    "end": end,
                }
            )
        return specs
    raise ArtifactError(f"unknown partitioning strategy {partitioning!r}")


def validate_shard_spec(shard: Mapping[str, Any]) -> dict[str, Any]:
    strategy = shard.get("strategy", "contiguous")
    if strategy == "contiguous":
        return {
            "index": int(shard["index"]),
            "strategy": "contiguous",
            "start": int(shard["start"]),
            "end": int(shard["end"]),
        }
    if strategy == "hash":
        if shard.get("hash_strategy") != HASH_STRATEGY_VERSION:
            raise ArtifactError(
                f"unsupported hash strategy {shard.get('hash_strategy')!r}"
            )
        return {
            "index": int(shard["index"]),
            "strategy": "hash",
            "hash_strategy": HASH_STRATEGY_VERSION,
            "modulus": int(shard["modulus"]),
        }
    raise ArtifactError(f"unknown shard strategy {strategy!r}")


def shard_member(shard: Mapping[str, Any], ordinal: int, task_key: str) -> bool:
    """Return whether a planned task belongs to a shard."""

    if shard.get("strategy") == "hash":
        return hash_partition(task_key, int(shard["modulus"])) == int(shard["index"])
    return int(shard["start"]) <= ordinal < int(shard["end"])


def assign_shard(
    shards: Sequence[Mapping[str, Any]], ordinal: int, task_key: str
) -> int:
    """Return the single owning shard index for a planned task."""

    if not shards:
        raise ArtifactError("run plan has no shards")
    strategy = shards[0].get("strategy", "contiguous")
    if strategy == "hash":
        return hash_partition(task_key, int(shards[0]["modulus"]))
    starts = [int(shard["start"]) for shard in shards]
    position = bisect.bisect_right(starts, ordinal) - 1
    if position < 0 or not shard_member(shards[position], ordinal, task_key):
        raise ArtifactError(f"task ordinal {ordinal} is not owned by any shard")
    return int(shards[position]["index"])


__all__ = [
    "HASH_STRATEGY_VERSION",
    "assign_shard",
    "build_shard_specs",
    "hash_partition",
    "shard_member",
    "validate_shard_spec",
]
