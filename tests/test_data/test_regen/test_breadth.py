"""R5 breadth gates: tool loops, critique/revision, pairs, plugins."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from specforge.data.artifact import open_dataset_artifact
from specforge.data.regen.artifact import ArtifactLayout
from specforge.data.regen.errors import CapabilityError, ContractError
from specforge.data.regen.executor import run_local
from specforge.data.regen.finalize import finalize_artifact
from specforge.data.regen.planner import plan_recipe
from specforge.data.regen.recipe import RegenerationRecipe
from specforge.data.regen import registry

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "look something up",
        "parameters": {"type": "object"},
    },
}


def _write_tool_rows(path: Path) -> None:
    rows = [
        {
            "id": f"task-{index}",
            "tools": [SEARCH_TOOL],
            "messages": [{"role": "user", "content": f"find item {index}"}],
        }
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _tool_recipe(source: Path, output: Path) -> RegenerationRecipe:
    return RegenerationRecipe.model_validate(
        {
            "version": 1,
            "seed": 5,
            "sources": {
                "agentic": {
                    "adapter": "jsonl",
                    "record_adapter": "openai_messages",
                    "config": {"path": str(source)},
                }
            },
            "generators": {
                "agent": {
                    "backend": "fake",
                    "model": "agent-model",
                    "revision": "immutable-1",
                    "codec": "structured_chat",
                    "config": {
                        "content_template": "step:{ordinal}:{prompt}",
                        "tool_calls_by_ordinal": {
                            "1": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": {"query": "widget"},
                                    },
                                }
                            ]
                        },
                    },
                }
            },
            "workflow": [
                {
                    "id": "act",
                    "operation": "execute_tool_loop",
                    "generator": "agent",
                    "tool_policy": "execute",
                    "config": {
                        "environment": "deterministic",
                        "environment_config": {
                            "results": {"search": "found:{query}"}
                        },
                        "max_tool_rounds": 4,
                    },
                }
            ],
            "validation": {"profiles": ["baseline"]},
            "output": {"uri": str(output), "shards": 1},
        }
    )


def _final_payloads(layout: ArtifactLayout) -> list[dict]:
    rows = []
    for path in sorted(layout.data.glob("part-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line))
    return rows


# --- deterministic tool loop --------------------------------------------------


def test_deterministic_tool_loop_records_evidence_and_binds_results(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_tool_rows(source)
    layout = plan_recipe(_tool_recipe(source, tmp_path / "artifact"))
    run_local(layout)
    manifest = finalize_artifact(layout)
    assert manifest["counts"]["status_counts"] == {"success": 3}

    rows = _final_payloads(layout)
    for row in rows:
        messages = row["payload"]["conversations"]
        roles = [message["role"] for message in messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert messages[1]["tool_calls"][0]["function"]["name"] == "search"
        assert messages[2] == {
            "role": "tool",
            "content": "found:widget",
            "tool_call_id": "call-1",
            "name": "search",
        }
        event = row["stage_history"][-1]
        executions = event["metadata"]["tool_executions"]
        assert len(executions) == 1
        assert executions[0]["environment"] == "deterministic"
        assert executions[0]["environment_version"] == "1"
        assert executions[0]["result_digest"]

    # Determinism: a fresh run of the same recipe produces identical rows.
    second = plan_recipe(_tool_recipe(source, tmp_path / "again"))
    run_local(second)
    finalize_artifact(second)
    assert _final_payloads(second) == rows


def test_forbidden_side_effects_are_rejected_at_plan_time(tmp_path):
    class SideEffectEnvironment:
        name = "network"
        version = "1"
        side_effects = "sandboxed"

        def declared_tools(self):
            return ("search",)

        def execute(self, name, arguments, *, timeout_seconds):
            return "live"

    registry.load_builtin_components()
    registry.TOOL_ENVIRONMENTS.register(
        "network-test",
        lambda config=None: SideEffectEnvironment(),
        version="1",
        capabilities={"tool_execution"},  # deliberately no no_side_effects
    )
    try:
        source = tmp_path / "source.jsonl"
        _write_tool_rows(source)
        raw = _tool_recipe(source, tmp_path / "artifact").model_dump(mode="json")
        raw["workflow"][0]["config"]["environment"] = "network-test"
        with pytest.raises(CapabilityError, match="no_side_effects"):
            plan_recipe(RegenerationRecipe.model_validate(raw))

        # Explicitly sandboxed policy admits the declared environment.
        raw["workflow"][0]["config"]["side_effect_policy"] = "sandboxed"
        raw["output"]["uri"] = str(tmp_path / "sandboxed")
        assert plan_recipe(RegenerationRecipe.model_validate(raw))
    finally:
        registry.TOOL_ENVIRONMENTS._entries.pop("network-test", None)


def test_tool_execution_requires_execute_policy(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_tool_rows(source)
    raw = _tool_recipe(source, tmp_path / "artifact").model_dump(mode="json")
    raw["workflow"][0]["tool_policy"] = "preserve_shape"
    with pytest.raises(CapabilityError, match="execute"):
        plan_recipe(RegenerationRecipe.model_validate(raw))


# --- critique / revise / candidate pair ---------------------------------------


def _write_chat_rows(path: Path) -> None:
    rows = [
        {
            "id": f"chat-{index}",
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"stale {index}"},
            ],
        }
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _critique_recipe(source: Path, output: Path) -> RegenerationRecipe:
    return RegenerationRecipe.model_validate(
        {
            "version": 1,
            "seed": 9,
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
                    "config": {"content_template": "draft:{prompt}:{variant}"},
                },
                "critic": {
                    "backend": "fake",
                    "model": "critic",
                    "revision": "immutable-1",
                    "codec": "structured_chat",
                    "config": {"content_template": "too vague"},
                },
                "editor": {
                    "backend": "fake",
                    "model": "editor",
                    "revision": "immutable-1",
                    "codec": "structured_chat",
                    "config": {"content_template": "revised:{stage}"},
                },
                "judge": {
                    "backend": "fake",
                    "model": "judge",
                    "revision": "immutable-1",
                    "codec": "structured_chat",
                    "config": {
                        "content_template": '{{"chosen": 0, "rejected": 1}}'
                    },
                },
            },
            "workflow": [
                {
                    "id": "draft",
                    "operation": "replay_assistants",
                    "generator": "teacher",
                    "candidates": 2,
                },
                {"id": "review", "operation": "critique", "generator": "critic"},
                {"id": "polish", "operation": "revise", "generator": "editor"},
                {"id": "pair", "operation": "candidate_pair", "generator": "judge"},
            ],
            "validation": {"profiles": ["baseline"]},
            "output": {"uri": str(output), "shards": 1},
        }
    )


def test_critique_revision_pair_workflow_preserves_stage_provenance(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_chat_rows(source)
    layout = plan_recipe(_critique_recipe(source, tmp_path / "artifact"))
    run_local(layout)
    manifest = finalize_artifact(layout)
    assert manifest["counts"]["status_counts"] == {"success": 3}

    rows = _final_payloads(layout)
    for row in rows:
        payload = row["payload"]
        stages = [event["stage_id"] for event in row["stage_history"]]
        assert stages == ["draft", "review", "polish", "pair"]

        # The revision replaced the draft but kept it losslessly.
        final = payload["conversations"][-1]
        assert final["content"] == "revised:polish"
        assert payload["revisions"][0]["previous"]["content"].startswith("draft:")
        assert payload["critiques"][0] == {
            "stage_id": "review",
            "generator": "critic",
            "content": "too vague",
        }
        # The rejected sibling trajectory survives as typed structure.
        assert payload["preference"]["stage_id"] == "pair"
        assert payload["rejected_conversations"][-1]["content"] == "revised:polish"
        pair_event = row["stage_history"][-1]
        assert len(pair_event["metadata"]["candidate_keys"]) == 2


def test_preference_artifact_is_consumed_by_the_typed_reader(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_chat_rows(source)
    layout = plan_recipe(_critique_recipe(source, tmp_path / "artifact"))
    run_local(layout)
    finalize_artifact(layout)

    artifact = open_dataset_artifact(layout.root)
    records = list(artifact.iter_preference_records())
    assert len(records) == 3
    for record in records:
        assert record["chosen_conversations"][-1]["role"] == "assistant"
        assert record["rejected_conversations"][-1]["role"] == "assistant"
        assert record["preference"]["chosen_key"] != record["preference"][
            "rejected_key"
        ]

    # A plain conversation artifact is not silently coerced into pairs.
    plain_source = tmp_path / "plain.jsonl"
    _write_chat_rows(plain_source)
    plain_raw = _critique_recipe(plain_source, tmp_path / "plain-artifact")
    raw = plain_raw.model_dump(mode="json")
    raw["workflow"] = [raw["workflow"][0]]
    raw["workflow"][0]["candidates"] = 1
    plain = plan_recipe(RegenerationRecipe.model_validate(raw))
    run_local(plain)
    finalize_artifact(plain)
    plain_artifact = open_dataset_artifact(plain.root)
    with pytest.raises(Exception, match="not a preference record"):
        list(plain_artifact.iter_preference_records())


# --- third-party plugins -------------------------------------------------------


def test_plugin_discovery_is_explicit_versioned_and_negotiated(
    tmp_path, monkeypatch
):
    calls = []

    def register_plugin():
        calls.append("registered")
        registry.TOOL_ENVIRONMENTS.register(
            "vendor-env",
            lambda config=None: SimpleNamespace(
                name="vendor-env",
                version="2.1",
                side_effects="none",
                declared_tools=lambda: ("search",),
                execute=lambda name, arguments, timeout_seconds: "vendor",
            ),
            version="2.1",
            capabilities={"tool_execution", "no_side_effects"},
        )

    entry_point = SimpleNamespace(
        name="vendor_plugin",
        value="vendor_pkg:register",
        load=lambda: register_plugin,
        dist=SimpleNamespace(name="vendor-pkg", version="2.1.0"),
    )
    monkeypatch.setattr(
        registry, "_iter_entry_points", lambda group: (entry_point,)
    )

    try:
        source = tmp_path / "source.jsonl"
        _write_tool_rows(source)
        raw = _tool_recipe(source, tmp_path / "artifact").model_dump(mode="json")
        raw["plugins"] = ["vendor_plugin"]
        raw["workflow"][0]["config"]["environment"] = "vendor-env"
        raw["workflow"][0]["config"].pop("environment_config")
        layout = plan_recipe(RegenerationRecipe.model_validate(raw))

        assert calls == ["registered"]
        plan = json.loads(layout.plan.read_text(encoding="utf-8"))
        assert plan["registries"]["plugins"]["vendor_plugin"] == {
            "entry_point": "vendor_pkg:register",
            "distribution": "vendor-pkg",
            "version": "2.1.0",
        }
        assert plan["registries"]["environments"]["vendor-env"]["version"] == "2.1"

        # An unknown plugin is refused, never silently skipped.
        raw["plugins"] = ["missing_plugin"]
        raw["output"]["uri"] = str(tmp_path / "other")
        with pytest.raises(ContractError, match="missing_plugin"):
            plan_recipe(RegenerationRecipe.model_validate(raw))
    finally:
        registry.TOOL_ENVIRONMENTS._entries.pop("vendor-env", None)
        registry._LOADED_PLUGINS.pop("vendor_plugin", None)
