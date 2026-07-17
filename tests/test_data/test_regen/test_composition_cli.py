from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from specforge.data.regen.executor import run_local
from specforge.data.regen.finalize import finalize_artifact
from specforge.data.regen.planner import plan_recipe
from specforge.data.regen.recipe import RegenerationRecipe


def _source(path: Path, identifier: int, prompt: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": identifier,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "old"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_two_sources_two_generators_candidate_selection(tmp_path: Path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _source(first, 1, "alpha")
    _source(second, 1, "beta")
    recipe = RegenerationRecipe.model_validate(
        {
            "seed": 8,
            "sources": {
                "first": {
                    "adapter": "jsonl",
                    "record_adapter": "openai_messages",
                    "config": {"path": str(first)},
                },
                "second": {
                    "adapter": "jsonl",
                    "record_adapter": "openai_messages",
                    "config": {"path": str(second)},
                },
            },
            "generators": {
                "teacher": {
                    "backend": "fake",
                    "model": "teacher",
                    "config": {"content_template": "{prompt}:{variant}"},
                },
                "judge": {
                    "backend": "fake",
                    "model": "judge",
                    "config": {"content_template": "1"},
                },
            },
            "workflow": [
                {
                    "id": "candidates",
                    "operation": "replay_assistants",
                    "generator": "teacher",
                    "candidates": 2,
                },
                {
                    "id": "choose",
                    "operation": "select_candidate",
                    "generator": "judge",
                },
            ],
            "output": {"uri": str(tmp_path / "artifact"), "shards": 2},
        }
    )
    layout = plan_recipe(recipe)
    run_local(layout)
    manifest = finalize_artifact(layout)
    assert manifest["counts"]["output_rows"] == 2

    rows = [
        json.loads(line)
        for path in sorted(layout.data.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["key"]["source"] for row in rows} == {"first", "second"}
    assert all("candidate-1" in row["key"]["variant"] for row in rows)
    choose = rows[0]["stage_history"][-1]
    assert choose["metadata"]["selected_index"] == 1
    assert len(choose["metadata"]["candidate_keys"]) == 2
    assert choose["metadata"]["judge_generation"]["message"]["content"] == "1"


def test_cli_plan_run_validate_finalize_inspect(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _source(source, 1, "hello")
    artifact = tmp_path / "artifact"
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
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
                        "config": {"content_template": "new:{prompt}"},
                    }
                },
                "workflow": [
                    {
                        "id": "replay",
                        "operation": "replay_assistants",
                        "generator": "teacher",
                    }
                ],
                "output": {"uri": str(artifact)},
            }
        ),
        encoding="utf-8",
    )

    def invoke(*arguments: str) -> dict:
        completed = subprocess.run(
            [sys.executable, "-m", "specforge.cli", "data", "regen", *arguments],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    assert invoke("plan", "--config", str(recipe))["state"] == "PLANNED"
    assert invoke("run", "--config", str(recipe))["output_rows"] == 1
    assert invoke("validate", "--artifact", str(artifact))["passed"] is True
    finalized = invoke("finalize", "--artifact", str(artifact))
    assert finalized["state"] == "FINALIZED"
    inspected = invoke("inspect", "--artifact", str(artifact), "--json")
    assert inspected["artifact_digest"] == finalized["artifact_digest"]
