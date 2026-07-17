from __future__ import annotations

import json
from pathlib import Path

import pytest

from specforge.data.regen.artifact import ArtifactLayout, verify_manifest
from specforge.data.regen.errors import ArtifactError
from specforge.data.regen.executor import run_local, run_worker
from specforge.data.regen.finalize import finalize_artifact
from specforge.data.regen.planner import plan_recipe
from specforge.data.regen.recipe import RegenerationRecipe


def _write_rows(path: Path) -> None:
    rows = [
        {
            "id": "multi-turn",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "source first"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "source second"},
            ],
        },
        {
            "id": "reasoning",
            "messages": [
                {"role": "user", "content": "reason"},
                {
                    "role": "assistant",
                    "content": "source answer",
                    "reasoning_content": "source reasoning",
                },
            ],
        },
        {
            "id": "prompt",
            "messages": [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "source prompt answer"},
            ],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _recipe(source: Path, output: Path, shards: int = 5) -> RegenerationRecipe:
    return RegenerationRecipe.model_validate(
        {
            "version": 1,
            "seed": 123,
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
                    "model": "teacher-revision",
                    "revision": "immutable-1",
                    "codec": "structured_chat",
                    "sampling": {"reasoning": "required"},
                    "config": {
                        "content_template": "answer:{prompt}:{ordinal}",
                        "reasoning_template": "trace:{prompt}:{ordinal}",
                    },
                }
            },
            "workflow": [
                {
                    "id": "regenerate",
                    "operation": "replay_assistants",
                    "generator": "teacher",
                }
            ],
            "validation": {
                "profiles": ["baseline"],
                "max_unresolved_error_rate": 0,
                "max_policy_reject_rate": 1,
            },
            "output": {"uri": str(output), "shards": shards},
        }
    )


def _final_rows(layout: ArtifactLayout) -> list[dict]:
    rows = []
    for path in sorted(layout.data.glob("part-*.jsonl")):
        rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    return rows


def test_fake_backend_interrupt_resume_validate_finalize(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    layout = plan_recipe(_recipe(source, tmp_path / "artifact"))

    crashed = False

    def fault(phase, task, attempt):
        nonlocal crashed
        if not crashed and phase == "before_commit":
            crashed = True
            raise KeyboardInterrupt("simulated process death")

    with pytest.raises(KeyboardInterrupt):
        run_worker(layout, 0, fault_hook=fault)
    # No partial attempt segment was published; an ordinary resume is exact.
    assert not list(layout.attempt_dir(0).glob("*.json"))

    run_local(layout)
    manifest = finalize_artifact(layout)
    assert manifest == verify_manifest(layout.root)
    assert manifest["state"] == "FINALIZED"
    assert manifest["counts"]["planned_tasks"] == 3
    assert manifest["counts"]["output_rows"] == 3
    assert manifest["counts"]["status_counts"] == {"success": 3}

    rows = _final_rows(layout)
    first = next(row for row in rows if row["key"]["source_id"] == "multi-turn")
    messages = first["payload"]["conversations"]
    assert [message["content"] for message in messages] == [
        "Be concise.",
        "first",
        "answer:first:1",
        "second",
        "answer:second:2",
    ]
    assert messages[2]["reasoning_content"] == "trace:first:1"
    assert messages[4]["reasoning_content"] == "trace:second:2"
    assert all("source first" not in json.dumps(row) for row in rows)
    # Raw tensor/model-feature fields can never enter final payloads.
    assert "hidden_state" not in json.dumps(manifest)


def test_shard_topology_does_not_change_final_rows(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    one = plan_recipe(_recipe(source, tmp_path / "one", shards=1))
    five = plan_recipe(_recipe(source, tmp_path / "five", shards=5))
    run_local(one)
    run_local(five)
    finalize_artifact(one)
    finalize_artifact(five)
    assert _final_rows(one) == _final_rows(five)


def test_attempt_tampering_is_rejected_before_merge(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    layout = plan_recipe(_recipe(source, tmp_path / "artifact", shards=1))
    run_local(layout)
    attempt = next(layout.attempt_dir(0).glob("*.json"))
    raw = json.loads(attempt.read_text(encoding="utf-8"))
    raw["task_ordinal"] = 999
    attempt.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ArtifactError, match="modified"):
        finalize_artifact(layout)


def test_final_payload_tampering_is_detected(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    layout = plan_recipe(_recipe(source, tmp_path / "artifact", shards=1))
    run_local(layout)
    finalize_artifact(layout)
    data = next(layout.data.glob("*.jsonl"))
    data.write_text(data.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="payload digest mismatch"):
        verify_manifest(layout.root)


def test_retryable_attempts_resolve_without_changing_request_identity(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(source)
    layout = plan_recipe(_recipe(source, tmp_path / "artifact", shards=1))
    result = run_worker(
        layout,
        0,
        runtime={"teacher": {"fail_first_attempts": 2}},
    )
    assert result["attempts_written"] == 5  # 3 for the first task, then 1 + 1
    attempts = sorted(layout.attempt_dir(0).glob("task-000000000000-*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in attempts]
    assert [record["status"] for record in records] == [
        "unresolved_error",
        "unresolved_error",
        "success",
    ]
    success_digest = records[-1]["outputs"][0]["stage_history"][0]["metadata"][
        "generations"
    ][0]["request_digest"]
    assert [record["request_digest"] for record in records[:-1]] == [
        success_digest,
        success_digest,
    ]
    manifest = finalize_artifact(layout)
    assert manifest["counts"]["resolved_retries"] == 1
