from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from specforge.data.regen.recipe import (
    RegenerationRecipe,
    apply_overrides,
    load_recipe,
)


def recipe_dict(tmp_path):
    return {
        "version": 1,
        "seed": 17,
        "sources": {
            "chat": {
                "adapter": "jsonl",
                "record_adapter": "openai_messages",
                "config": {"path": str(tmp_path / "input.jsonl")},
                "selection": {"mode": "all"},
            }
        },
        "generators": {
            "teacher": {
                "backend": "fake",
                "model": "test/teacher",
                "revision": "abc123",
                "codec": "structured_chat",
                "sampling": {"temperature": 0, "max_tokens": 32},
            }
        },
        "workflow": [
            {
                "id": "answer",
                "operation": "replay_assistants",
                "generator": "teacher",
            }
        ],
        "validation": {"profiles": ["baseline"]},
        "output": {"uri": str(tmp_path / "artifact"), "shards": 2},
    }


def test_recipe_digest_is_order_independent_and_semantic(tmp_path):
    raw = recipe_dict(tmp_path)
    first = RegenerationRecipe.model_validate(raw)
    reordered = {key: raw[key] for key in reversed(list(raw))}
    second = RegenerationRecipe.model_validate(reordered)
    assert first.digest == second.digest

    changed = first.model_copy(
        update={
            "seed": 18,
        }
    )
    assert first.digest != changed.digest


@pytest.mark.parametrize(
    "field",
    ["api_key", "token", "password", "authorization", "base_url", "endpoint"],
)
def test_generator_recipe_rejects_secrets_and_runtime_endpoints(tmp_path, field):
    raw = recipe_dict(tmp_path)
    raw["generators"]["teacher"]["config"] = {field: "do-not-serialize"}
    with pytest.raises(ValidationError, match="runtime-only"):
        RegenerationRecipe.model_validate(raw)


def test_recipe_rejects_unknown_generator_and_duplicate_stage(tmp_path):
    raw = recipe_dict(tmp_path)
    raw["workflow"][0]["generator"] = "missing"
    with pytest.raises(ValidationError, match="unknown generator"):
        RegenerationRecipe.model_validate(raw)

    raw = recipe_dict(tmp_path)
    raw["workflow"].append(dict(raw["workflow"][0]))
    with pytest.raises(ValidationError, match="stage ids"):
        RegenerationRecipe.model_validate(raw)


def test_recipe_load_and_typed_override(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe_dict(tmp_path)), encoding="utf-8")
    recipe = load_recipe(path, ["output.shards=4", "seed=99"])
    assert recipe.output.shards == 4
    assert recipe.seed == 99

    with pytest.raises(ValueError, match="does not exist"):
        apply_overrides(recipe, ["output.unknown=1"])
