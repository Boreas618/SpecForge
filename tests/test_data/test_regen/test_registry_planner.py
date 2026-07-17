from __future__ import annotations

import json
from pathlib import Path

import pytest

from specforge.data.regen.errors import CapabilityError, ContractError
from specforge.data.regen.planner import plan_recipe
from specforge.data.regen.recipe import RegenerationRecipe
from specforge.data.regen.registry import ComponentRegistry, load_builtin_components


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _recipe(source: Path, output: Path, **source_changes) -> RegenerationRecipe:
    source_spec = {
        "adapter": "jsonl",
        "record_adapter": "openai_messages",
        "config": {"path": str(source)},
        "selection": {"mode": "all"},
    }
    source_spec.update(source_changes)
    return RegenerationRecipe.model_validate(
        {
            "version": 1,
            "seed": 41,
            "sources": {"alpha": source_spec},
            "generators": {
                "teacher": {
                    "backend": "fake",
                    "model": "fake-v1",
                    "codec": "structured_chat",
                }
            },
            "workflow": [
                {
                    "id": "replay",
                    "operation": "replay_assistants",
                    "generator": "teacher",
                }
            ],
            "output": {"uri": str(output), "shards": 5},
        }
    )


def test_registry_rejects_collisions_and_missing_capabilities():
    registry = ComponentRegistry("example")
    registry.register("one", lambda: None, capabilities={"a"})
    with pytest.raises(ContractError, match="duplicate"):
        registry.register("one", lambda: None)
    with pytest.raises(CapabilityError, match="lacks capabilities: b"):
        registry.resolve("one", required_capabilities={"a", "b"})
    with pytest.raises(CapabilityError, match="available: one"):
        registry.resolve("missing")


def test_builtin_loading_is_idempotent():
    load_builtin_components()
    load_builtin_components()


def test_planner_streams_stable_selection_and_contiguous_shards(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(
        source,
        [
            {
                "id": index,
                "messages": [
                    {"role": "user", "content": f"question-{index}"},
                    {"role": "assistant", "content": "old"},
                ],
            }
            for index in range(17)
        ],
    )
    recipe = _recipe(
        source,
        tmp_path / "artifact",
        selection={"mode": "sample", "sample": 7, "seed": 19},
    )
    layout = plan_recipe(recipe)
    plan = json.loads(layout.plan.read_text(encoding="utf-8"))
    tasks = [
        json.loads(line)
        for line in layout.tasks.read_text(encoding="utf-8").splitlines()
    ]

    assert plan["task_count"] == 7
    assert [task["ordinal"] for task in tasks] == list(range(7))
    assert [task["source_position"] for task in tasks] == sorted(
        task["source_position"] for task in tasks
    )
    owned = [
        ordinal
        for shard in plan["shards"]
        for ordinal in range(shard["start"], shard["end"])
    ]
    assert owned == list(range(7))
    # Runtime placement is deliberately absent from the immutable identity recipe.
    persisted_recipe = json.loads(layout.recipe.read_text(encoding="utf-8"))
    assert "uri" not in persisted_recipe["output"]


def test_planner_rejects_duplicate_type_preserving_source_ids(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(
        source,
        [
            {"id": 1, "messages": [{"role": "user", "content": "one"}]},
            {"id": 1, "messages": [{"role": "user", "content": "duplicate"}]},
        ],
    )
    with pytest.raises(ContractError, match="duplicate id"):
        plan_recipe(_recipe(source, tmp_path / "artifact"))


def test_integer_and_string_ids_do_not_collide(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(
        source,
        [
            {"id": 1, "messages": [{"role": "user", "content": "integer"}]},
            {"id": "1", "messages": [{"role": "user", "content": "string"}]},
        ],
    )
    layout = plan_recipe(_recipe(source, tmp_path / "artifact"))
    assert len(layout.tasks.read_text(encoding="utf-8").splitlines()) == 2
