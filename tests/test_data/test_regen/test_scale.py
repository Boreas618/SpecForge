"""R4 distributed-scale gates: partitioning, leases, stores, endpoint pools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from specforge.data.regen.artifact import ArtifactLayout, verify_manifest
from specforge.data.regen.contracts import GenerationRequest, RecordKey
from specforge.data.regen.endpoints import EndpointPool
from specforge.data.regen.errors import ArtifactError, GenerationError
from specforge.data.regen.executor import run_local, run_worker
from specforge.data.regen.finalize import finalize_artifact, validate_artifact
from specforge.data.regen.leases import ShardLeaseStore, run_leased
from specforge.data.regen.maintenance import cleanup_orphans, reconcile_artifact
from specforge.data.regen.partitioning import (
    assign_shard,
    build_shard_specs,
    hash_partition,
    shard_member,
)
from specforge.data.regen.planner import plan_recipe
from specforge.data.regen.recipe import RegenerationRecipe
from specforge.data.regen.stores import (
    LocalDirectoryBlobStore,
    fetch_from_store,
    publish_to_store,
)

ROWS = 12


def _write_rows(path: Path, rows: int = ROWS) -> None:
    payload = [
        {
            "id": f"row-{index}",
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"stale {index}"},
            ],
        }
        for index in range(rows)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in payload), encoding="utf-8"
    )


def _recipe(
    source: Path, output: Path, *, shards: int = 4, partitioning: str = "contiguous"
) -> RegenerationRecipe:
    return RegenerationRecipe.model_validate(
        {
            "version": 1,
            "seed": 7,
            "sources": {
                "chat": {
                    "adapter": "jsonl",
                    "record_adapter": "openai_messages",
                    "config": {"path": str(source)},
                }
            },
            "generators": {
                "teacher": {
                    "backend": "fake",
                    "model": "teacher",
                    "revision": "immutable-1",
                    "codec": "structured_chat",
                    "config": {"content_template": "answer:{prompt}:{ordinal}"},
                }
            },
            "workflow": [
                {
                    "id": "regenerate",
                    "operation": "replay_assistants",
                    "generator": "teacher",
                }
            ],
            "validation": {"profiles": ["baseline"]},
            "output": {
                "uri": str(output),
                "shards": shards,
                "partitioning": partitioning,
            },
        }
    )


def _final_rows(layout: ArtifactLayout) -> list[dict]:
    rows = []
    for path in sorted(layout.data.glob("part-*.jsonl")):
        rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    return rows


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --- hash partitioning ------------------------------------------------------


def test_hash_partition_is_stable_and_shards_partition_the_plan():
    assert hash_partition("key-a", 8) == hash_partition("key-a", 8)
    shards = build_shard_specs("hash", 100, 8)
    for ordinal, key in enumerate(f"key-{i}" for i in range(100)):
        owners = [
            shard["index"] for shard in shards if shard_member(shard, ordinal, key)
        ]
        assert len(owners) == 1
        assert assign_shard(shards, ordinal, key) == owners[0]


def test_hash_and_contiguous_partitioning_produce_identical_artifacts(
    tmp_path: Path,
):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    contiguous = plan_recipe(
        _recipe(source, tmp_path / "contiguous", partitioning="contiguous")
    )
    hashed = plan_recipe(_recipe(source, tmp_path / "hash", partitioning="hash"))
    run_local(contiguous)
    run_local(hashed)
    finalize_artifact(contiguous)
    finalize_artifact(hashed)
    assert _final_rows(contiguous) == _final_rows(hashed)


def test_hash_shards_reject_foreign_attempts(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    layout = plan_recipe(_recipe(source, tmp_path / "hash", partitioning="hash"))
    run_local(layout)
    donors = [
        shard for shard in range(4) if list(layout.attempt_dir(shard).glob("*.json"))
    ]
    moved = next(layout.attempt_dir(donors[0]).glob("*.json"))
    target_dir = layout.attempt_dir(donors[1])
    (target_dir / moved.name).write_bytes(moved.read_bytes())
    moved.unlink()
    with pytest.raises(ArtifactError):
        finalize_artifact(layout)


# --- leases -----------------------------------------------------------------


def test_leased_workers_drain_the_pool_and_match_static_output(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    static = plan_recipe(_recipe(source, tmp_path / "static"))
    leased = plan_recipe(_recipe(source, tmp_path / "leased"))
    run_local(static)

    first = run_leased(leased, "worker-a")
    second = run_leased(leased, "worker-b")
    assert len(first) == 4 and second == []

    finalize_artifact(static)
    finalize_artifact(leased)
    assert _final_rows(static) == _final_rows(leased)


def test_expired_lease_is_reclaimed_and_completion_is_fenced(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    layout = plan_recipe(_recipe(source, tmp_path / "artifact"))
    plan_digest = json.loads(layout.plan.read_text())["plan_digest"]
    clock = Clock()
    store = ShardLeaseStore(
        tmp_path / "leases.sqlite3",
        plan_digest=plan_digest,
        shard_indexes=[0, 1, 2, 3],
        lease_seconds=10,
        clock=clock,
    )

    stale = store.claim("worker-a")
    assert stale is not None and stale["shard_index"] == 0
    clock.now += 11  # worker-a stalls past its lease

    report = store.reconcile()
    assert report["expired_reclaimed"] == [0]

    fresh = store.claim("worker-b")
    assert fresh is not None and fresh["shard_index"] == 0
    # The stalled worker can no longer heartbeat, complete, or extend.
    assert not store.heartbeat("worker-a", stale)
    assert not store.complete("worker-a", stale)
    assert store.complete("worker-b", fresh)
    # Completion is terminal: nobody can claim shard 0 again.
    assert {store.claim("worker-c")["shard_index"] for _ in range(3)} == {1, 2, 3}


def test_killed_leased_worker_is_resumed_exactly_by_another(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    reference = plan_recipe(_recipe(source, tmp_path / "reference"))
    run_local(reference)
    finalize_artifact(reference)

    layout = plan_recipe(_recipe(source, tmp_path / "artifact"))
    clock = Clock()
    plan_digest = json.loads(layout.plan.read_text())["plan_digest"]
    store = ShardLeaseStore(
        layout.work / "leases.sqlite3",
        plan_digest=plan_digest,
        shard_indexes=[0, 1, 2, 3],
        lease_seconds=10,
        clock=clock,
    )

    committed = 0

    def die_mid_shard(stage, task, attempt):
        nonlocal committed
        if stage == "after_commit":
            committed += 1
            if committed == 2:
                raise KeyboardInterrupt("simulated worker death")

    with pytest.raises(KeyboardInterrupt):
        run_leased(
            layout,
            "worker-a",
            lease_store=store,
            clock=clock,
            fault_hook=die_mid_shard,
        )
    clock.now += 11
    results = run_leased(layout, "worker-b", lease_store=store, clock=clock)
    assert len(results) == 4

    finalize_artifact(layout)
    assert _final_rows(layout) == _final_rows(reference)


# --- object store publication -----------------------------------------------


def _finalized_artifact(tmp_path: Path, rows: int = ROWS) -> ArtifactLayout:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.jsonl"
    _write_rows(source, rows)
    layout = plan_recipe(_recipe(source, tmp_path / "artifact"))
    run_local(layout)
    finalize_artifact(layout)
    return layout


def test_store_publish_fetch_roundtrip_verifies_bytes(tmp_path: Path):
    layout = _finalized_artifact(tmp_path)
    store = LocalDirectoryBlobStore(tmp_path / "bucket")
    manifest = publish_to_store(layout, store, "datasets/run-1")
    fetched = fetch_from_store(store, "datasets/run-1", tmp_path / "download")
    assert fetched["artifact_digest"] == manifest["artifact_digest"]
    assert verify_manifest(tmp_path / "download")


def test_failed_publisher_exposes_no_manifest(tmp_path: Path):
    layout = _finalized_artifact(tmp_path)
    store = LocalDirectoryBlobStore(tmp_path / "bucket")

    def crash(stage, relative):
        raise KeyboardInterrupt("publisher died mid-upload")

    with pytest.raises(KeyboardInterrupt):
        publish_to_store(layout, store, "datasets/run-1", fault_hook=crash)
    with pytest.raises(ArtifactError, match="no published manifest"):
        fetch_from_store(store, "datasets/run-1", tmp_path / "download")


def test_publish_race_is_idempotent_but_conflicts_are_refused(tmp_path: Path):
    layout = _finalized_artifact(tmp_path)
    store = LocalDirectoryBlobStore(tmp_path / "bucket")
    first = publish_to_store(layout, store, "datasets/run-1")
    second = publish_to_store(layout, store, "datasets/run-1")
    assert first["artifact_digest"] == second["artifact_digest"]

    other = _finalized_artifact(tmp_path / "other", rows=5)
    with pytest.raises(ArtifactError, match="different"):
        publish_to_store(other, store, "datasets/run-1")


def test_tampered_published_file_fails_fetch(tmp_path: Path):
    layout = _finalized_artifact(tmp_path)
    store = LocalDirectoryBlobStore(tmp_path / "bucket")
    publish_to_store(layout, store, "datasets/run-1")
    victim = next((tmp_path / "bucket" / "datasets/run-1/data").glob("*.jsonl"))
    victim.write_text(victim.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="failed verification"):
        fetch_from_store(store, "datasets/run-1", tmp_path / "download")


# --- endpoint pool ----------------------------------------------------------


def _request(seed: int = 0) -> GenerationRequest:
    return GenerationRequest(
        record_key=RecordKey(source="s", source_id="1"),
        stage_id="stage",
        variant="base",
        generation_ordinal=1,
        messages=({"role": "user", "content": "hi"},),
        sampling={},
        seed=seed,
    )


def test_circuit_breaker_opens_fails_over_and_recovers():
    clock = Clock()
    pool = EndpointPool(
        ("http://a", "http://b"),
        failure_threshold=2,
        cooldown_seconds=30,
        clock=clock,
    )
    request = _request(seed=0)  # prefers endpoint index 0

    from specforge.data.regen.errors import FailureCategory

    for _ in range(2):
        with pytest.raises(GenerationError):
            with pool.lease(request) as endpoint:
                assert endpoint == "http://a"
                raise GenerationError(
                    "boom", category=FailureCategory.TRANSPORT_RETRYABLE
                )
    # Circuit for a is open: the same preference now lands on b.
    with pool.lease(request) as endpoint:
        assert endpoint == "http://b"

    clock.now += 31  # cool-down elapsed: one probe is admitted and recovers
    with pool.lease(request) as endpoint:
        assert endpoint == "http://a"
    assert not pool.snapshot()["endpoint-0"]["circuit_open"]


def test_bounded_inflight_saturation_is_a_retryable_error():
    # Real clock: the acquire deadline must elapse while the slot stays held.
    pool = EndpointPool(
        ("http://a",),
        max_inflight=1,
        acquire_timeout=0.05,
    )
    request = _request()
    with pool.lease(request):
        with pytest.raises(GenerationError) as info:
            with pool.lease(request):
                pass
    assert info.value.category.value == "transport_retryable"
    # The slot was released; acquisition works again.
    with pool.lease(request) as endpoint:
        assert endpoint == "http://a"


# --- sampled validators and reconciliation -----------------------------------


def test_expensive_validator_sampling_is_deterministic(tmp_path: Path):
    from specforge.data.regen import registry
    from specforge.data.regen.finalize import resolve_attempts

    class SampledValidator:
        name = "sampled-test"
        sample_modulus = 3

        def validate(self, envelope):
            return []

    registry.load_builtin_components()
    registry.VALIDATORS.register(
        "sampled-test",
        lambda recipe=None: SampledValidator(),
        version="1",
        capabilities={"text_trajectory"},
    )
    try:
        source = tmp_path / "source.jsonl"
        _write_rows(source)
        raw = _recipe(source, tmp_path / "artifact").model_dump(mode="json")
        raw["validation"]["profiles"] = ["baseline", "sampled-test"]
        layout = plan_recipe(RegenerationRecipe.model_validate(raw))
        run_local(layout)
        resolve_attempts(layout)

        first = validate_artifact(layout)["profile_checked_rows"]
        second = validate_artifact(layout)["profile_checked_rows"]
        assert first == second
        assert first["baseline"] == ROWS  # the baseline never samples
        assert 0 < first["sampled-test"] < ROWS
    finally:
        registry.VALIDATORS._entries.pop("sampled-test", None)


def test_validation_report_counts_sampled_rows(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    recipe = _recipe(source, tmp_path / "artifact")
    layout = plan_recipe(recipe)
    run_local(layout)
    from specforge.data.regen.finalize import resolve_attempts

    resolve_attempts(layout)
    report = validate_artifact(layout)
    assert report["profile_checked_rows"] == {"baseline": ROWS}
    assert report["passed"]


def test_cli_leased_worker_and_reconcile(tmp_path: Path, capsys):
    import argparse

    from specforge.data.regen.cli import configure_regen_parser, run_regen_command

    source = tmp_path / "source.jsonl"
    _write_rows(source)
    layout = plan_recipe(_recipe(source, tmp_path / "artifact"))

    parser = argparse.ArgumentParser()
    configure_regen_parser(parser)
    args = parser.parse_args(
        [
            "worker",
            "--plan",
            str(layout.plan),
            "--leased",
            "--worker-id",
            "cli-worker",
        ]
    )
    assert run_regen_command(args) == 0
    worker_report = json.loads(capsys.readouterr().out)
    assert worker_report["worker_id"] == "cli-worker"
    assert len(worker_report["shards"]) == 4

    args = parser.parse_args(["reconcile", "--artifact", str(layout.root)])
    assert run_regen_command(args) == 0
    reconcile_report = json.loads(capsys.readouterr().out)
    assert reconcile_report["incomplete_shards"] == []
    assert reconcile_report["leases"]["completed"] == 4


def test_reconcile_cleans_orphans_and_reports_incomplete_shards(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    layout = plan_recipe(_recipe(source, tmp_path / "artifact"))
    run_worker(layout, 0)
    stale = layout.work / ".stale-write.12345.tmp"
    stale.write_text("partial", encoding="utf-8")

    report = reconcile_artifact(layout)
    assert not stale.exists()
    assert report["orphans_removed"] == ["work/.stale-write.12345.tmp"]
    assert report["incomplete_shards"] == [1, 2, 3]
    assert cleanup_orphans(layout) == []
