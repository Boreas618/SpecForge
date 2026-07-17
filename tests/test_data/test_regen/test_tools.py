from __future__ import annotations

import json
from pathlib import Path

from specforge.data.regen.executor import run_local
from specforge.data.regen.finalize import finalize_artifact
from specforge.data.regen.planner import plan_recipe
from specforge.data.regen.recipe import RegenerationRecipe


def test_recorded_tool_results_rebind_to_regenerated_calls(tmp_path: Path):
    source = tmp_path / "tools.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "tool-row",
                "messages": [
                    {"role": "user", "content": "weather"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "old",
                        "tool_calls": [
                            {
                                "id": "old-call",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": {"city": "Paris"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "old-call",
                        "name": "weather",
                        "content": "sunny",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recipe = RegenerationRecipe.model_validate(
        {
            "sources": {
                "tools": {
                    "adapter": "jsonl",
                    "record_adapter": "openai_messages",
                    "config": {"path": str(source)},
                }
            },
            "generators": {
                "teacher": {
                    "backend": "fake",
                    "model": "fake",
                    "codec": "structured_chat",
                    "sampling": {"reasoning": "required"},
                    "config": {
                        "content_template": "",
                        "reasoning_template": "new reasoning",
                        "tool_calls": [
                            {
                                "id": "new-call",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": {"city": "Paris"},
                                },
                            }
                        ],
                    },
                }
            },
            "workflow": [
                {
                    "id": "replay",
                    "operation": "replay_assistants",
                    "generator": "teacher",
                    "tool_policy": "preserve_shape",
                }
            ],
            "output": {"uri": str(tmp_path / "artifact")},
        }
    )
    layout = plan_recipe(recipe)
    run_local(layout)
    finalize_artifact(layout)
    row = json.loads(next(layout.data.glob("*.jsonl")).read_text(encoding="utf-8"))
    messages = row["payload"]["conversations"]
    assert messages[1]["reasoning_content"] == "new reasoning"
    assert messages[1]["tool_calls"][0]["id"] == "new-call"
    assert messages[2] == {
        "role": "tool",
        "content": "sunny",
        "tool_call_id": "new-call",
        "name": "weather",
    }
