from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from specforge.config.schema import DataConfig
from specforge.data.artifact import open_dataset_artifact, preprocessing_cache_key
from specforge.data.regen.errors import ArtifactError
from specforge.data.regen.executor import run_local
from specforge.data.regen.finalize import finalize_artifact
from specforge.data.regen.planner import plan_recipe
from specforge.data.regen.recipe import RegenerationRecipe


def _build(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "reasoning",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "old"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recipe = RegenerationRecipe.model_validate(
        {
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
                    "model": "fake",
                    "sampling": {"reasoning": "required"},
                    "config": {
                        "content_template": "answer",
                        "reasoning_template": "complete trace",
                    },
                }
            },
            "workflow": [
                {
                    "id": "replay",
                    "operation": "replay_assistants",
                    "generator": "teacher",
                }
            ],
            "output": {"uri": str(tmp_path / "artifact")},
        }
    )
    layout = plan_recipe(recipe)
    return layout


def test_normal_consumer_requires_finalized_verified_manifest(tmp_path: Path):
    layout = _build(tmp_path)
    with pytest.raises(ArtifactError):
        open_dataset_artifact(layout.root)
    run_local(layout)
    finalize_artifact(layout)

    artifact = open_dataset_artifact(layout.manifest)
    rows = list(artifact.iter_text_records())
    assert rows[0]["conversations"][-1] == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "complete trace",
    }
    assert artifact.provenance()["dataset_artifact_digest"] == artifact.digest

    data = next(layout.data.glob("*.jsonl"))
    data.write_text(data.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="digest mismatch"):
        open_dataset_artifact(layout.root)


def test_artifact_digest_participates_in_text_cache_identity():
    fields = {"max_length": 4096, "chat_template": "qwen", "tokenizer": "rev-a"}
    first = preprocessing_cache_key("artifact-a", fields)
    assert first == preprocessing_cache_key("artifact-a", fields)
    assert first != preprocessing_cache_key("artifact-b", fields)
    assert first != preprocessing_cache_key(
        "artifact-a", {**fields, "max_length": 8192}
    )


def test_data_config_has_one_explicit_boundary():
    value = DataConfig(dataset_artifact="/artifact/manifest.json", chat_template="qwen")
    assert value.dataset_artifact
    with pytest.raises(ValidationError, match="exactly one"):
        DataConfig(
            dataset_artifact="/artifact/manifest.json",
            chat_template="qwen",
            prompts_path="prompts.jsonl",
        )
    with pytest.raises(ValidationError, match="chat_template"):
        DataConfig(dataset_artifact="/artifact/manifest.json")


def _training_cfg(tmp_path: Path, artifact_uri: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        data=SimpleNamespace(
            dataset_artifact=artifact_uri,
            chat_template="qwen",
            max_length=2048,
            is_preformatted=False,
            train_only_last_turn=False,
            max_prompts=None,
            cache_dir=str(tmp_path / "cache"),
            allow_unverified_regenerated_jsonl=False,
        ),
        model=SimpleNamespace(
            target_model_path="org/target",
            draft_model_config="draft.json",
            draft_checkpoint_path="",
            draft_num_hidden_layers=None,
            draft_block_size=None,
            input_modality="text",
        ),
        training=SimpleNamespace(strategy="eagle3"),
    )


def test_assembly_materializes_verified_artifact_view(tmp_path: Path):
    from specforge.training.assembly import (
        _materialize_artifact_view,
        _prompt_semantic_identity,
        dataset_artifact_checkpoint_extra,
    )

    layout = _build(tmp_path)
    run_local(layout)
    manifest = finalize_artifact(layout)
    cfg = _training_cfg(tmp_path, str(layout.root))

    view_path, cache_key = _materialize_artifact_view(cfg)
    artifact = open_dataset_artifact(layout.root)
    assert cache_key == preprocessing_cache_key(
        artifact.digest, _prompt_semantic_identity(cfg)
    )
    rows = [
        json.loads(line)
        for line in Path(view_path).read_text(encoding="utf-8").splitlines()
    ]
    assert rows == list(artifact.iter_text_records())
    # The view is content-addressed by the artifact digest.
    assert artifact.digest in view_path

    provenance = dataset_artifact_checkpoint_extra(cfg)
    assert provenance == {
        "dataset_artifact_digest": manifest["artifact_digest"],
        "dataset_recipe_digest": manifest["recipe_digest"],
    }

    # A tampered payload is refused before any row is materialized.
    data = next(layout.data.glob("*.jsonl"))
    data.write_text(data.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    Path(view_path).unlink()
    with pytest.raises(ArtifactError, match="digest mismatch"):
        _materialize_artifact_view(cfg)

    # The light provenance reader still refuses a tampered manifest.
    manifest_path = layout.manifest
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["counts"]["output_rows"] = 999
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ArtifactError, match="digest mismatch"):
        dataset_artifact_checkpoint_extra(cfg)


def test_same_jsonl_bytes_have_path_independent_planned_identity(tmp_path: Path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _build(first_root)
    second_source = first_root / "source.jsonl"
    copied = second_root / "renamed.jsonl"
    copied.write_bytes(second_source.read_bytes())
    recipe = RegenerationRecipe.model_validate(
        {
            "sources": {
                "chat": {
                    "adapter": "jsonl",
                    "record_adapter": "openai_messages",
                    "config": {"path": str(copied)},
                }
            },
            "generators": {
                "teacher": {
                    "backend": "fake",
                    "model": "fake",
                    "sampling": {"reasoning": "required"},
                    "config": {
                        "content_template": "answer",
                        "reasoning_template": "complete trace",
                    },
                }
            },
            "workflow": [
                {
                    "id": "replay",
                    "operation": "replay_assistants",
                    "generator": "teacher",
                }
            ],
            "output": {"uri": str(second_root / "artifact")},
        }
    )
    second = plan_recipe(recipe)
    first_plan = json.loads(first.plan.read_text(encoding="utf-8"))
    second_plan = json.loads(second.plan.read_text(encoding="utf-8"))
    assert first_plan["recipe_digest"] == second_plan["recipe_digest"]
    assert first_plan["sources"] == second_plan["sources"]
    assert str(first_root) not in first.recipe.read_text(encoding="utf-8")
