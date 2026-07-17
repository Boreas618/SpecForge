import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.regenerate_train_data import main as regenerate_main
from scripts.regenerate_train_data import set_skipped, validate_regen_input
from scripts.validate_regenerated_data import validate_dataset, validate_row


def make_row(row_id="row-1", content="answer"):
    return {
        "id": row_id,
        "status": "success",
        "conversations": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": content},
        ],
    }


class TestValidateRegeneratedData(TestCase):
    def test_valid_non_reasoning_dataset_allows_duplicate_ids_with_warning(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text(
                "".join(json.dumps(make_row("same")) + "\n" for _ in range(2)),
                encoding="utf-8",
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                summary = validate_dataset(
                    path,
                    expect_non_reasoning=True,
                    strict_think_markers=True,
                )

            self.assertEqual(summary.rows, 2)
            self.assertEqual(summary.assistant_messages, 2)
            self.assertEqual(summary.duplicate_rows, 1)
            self.assertEqual(len(caught), 1)
            self.assertIn("duplicates are allowed", str(caught[0].message))

    def test_rejects_incomplete_training_conversation(self):
        row = make_row()
        row["conversations"].append({"role": "user", "content": "follow-up"})

        with self.assertRaisesRegex(ValueError, "must end with an assistant"):
            validate_row(
                row,
                expect_non_reasoning=True,
                strict_think_markers=True,
            )

    def test_rejects_nonempty_reasoning_content(self):
        row = make_row()
        row["conversations"][-1]["reasoning_content"] = "hidden reasoning"

        with self.assertRaisesRegex(ValueError, "reasoning_content"):
            validate_row(
                row,
                expect_non_reasoning=True,
                strict_think_markers=False,
            )

    def test_reasoning_mode_requires_reasoning_content(self):
        row = make_row()

        with self.assertRaisesRegex(ValueError, "empty reasoning_content"):
            validate_row(
                row,
                expect_non_reasoning=False,
                expect_reasoning=True,
                strict_think_markers=True,
            )

        row["conversations"][-1]["reasoning_content"] = "structured reasoning"
        self.assertEqual(
            validate_row(
                row,
                expect_non_reasoning=False,
                expect_reasoning=True,
                strict_think_markers=True,
            ),
            1,
        )

    def test_strict_mode_rejects_thinking_markers_in_reasoning(self):
        row = make_row()
        row["conversations"][-1]["reasoning_content"] = "<think>hidden</think>"

        with self.assertRaisesRegex(ValueError, "reasoning_content"):
            validate_row(
                row,
                expect_non_reasoning=False,
                expect_reasoning=True,
                strict_think_markers=True,
            )

    def test_strict_mode_rejects_thinking_markers(self):
        row = make_row(content="<THINK>hidden</THINK> visible")

        with self.assertRaisesRegex(ValueError, "thinking marker"):
            validate_row(
                row,
                expect_non_reasoning=True,
                strict_think_markers=True,
            )

    def test_rejects_non_success_rows(self):
        row = make_row()
        row["status"] = "error"

        with self.assertRaisesRegex(ValueError, "status must be 'success'"):
            validate_row(
                row,
                expect_non_reasoning=False,
                strict_think_markers=False,
            )

    def test_rejects_rows_the_pipeline_would_reject(self):
        row = make_row()
        row["conversations"][-1]["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": "{not json"},
            }
        ]

        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            validate_row(
                row,
                expect_non_reasoning=True,
                strict_think_markers=False,
            )


class FakeServer:
    """Deterministic OpenAI-compatible chat responses for the wrapper."""

    def __init__(self, reasoning=None, content="regenerated answer"):
        self.reasoning = reasoning
        self.content = content
        self.requests = []

    def post_json(self, url, payload, *, api_key=None, timeout=None):
        if payload.get("max_tokens", 0) > 1:
            # Probe requests use max_tokens=1 and are not generation traffic.
            self.requests.append(json.loads(json.dumps(payload)))
        message = {"role": "assistant", "content": self.content}
        if self.reasoning is not None:
            message["reasoning_content"] = self.reasoning
        return {
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }


class TestRegenerationWrapper(TestCase):
    """The deprecated script is a wrapper over specforge.data.regen."""

    def _write_input(self, path: Path, rows) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def _run(self, tmpdir: str, rows, extra_args=(), server=None):
        directory = Path(tmpdir)
        input_path = directory / "input.jsonl"
        output_path = directory / "output.jsonl"
        self._write_input(input_path, rows)
        server = server or FakeServer()
        argv = [
            "--model",
            "org/test-model",
            "--model-revision",
            "abc123",
            "--server-address",
            "localhost:30000",
            "--concurrency",
            "2",
            "--input-file-path",
            str(input_path),
            "--output-file-path",
            str(output_path),
            *extra_args,
        ]
        with patch(
            "specforge.data.regen.backends.openai_chat.post_json",
            server.post_json,
        ):
            regenerate_main(argv)
        return directory, output_path, server

    def _read_jsonl(self, path: Path):
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_input_precheck_rejects_bad_role_order_and_content(self):
        bad_order = {
            "conversations": [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ]
        }
        empty_content = {"conversations": [{"role": "user", "content": "  "}]}

        self.assertIn("role order", validate_regen_input(bad_order))
        self.assertIn("non-empty string", validate_regen_input(empty_content))

    def test_non_object_input_can_be_recorded_as_skipped(self):
        skipped = set_skipped([], "Expected a JSON object")

        self.assertEqual(skipped["status"], "skipped")
        self.assertEqual(skipped["data"], [])

    def test_wrapper_produces_finalized_artifact_and_legacy_files(self):
        rows = [
            {
                "id": str(index),
                "extra_field": f"metadata-{index}",
                "conversations": [
                    {"role": "user", "content": f"question {index}"},
                    {"role": "assistant", "content": "stale answer"},
                ],
            }
            for index in range(3)
        ]
        with TemporaryDirectory() as tmpdir:
            directory, output_path, server = self._run(tmpdir, rows)
            outputs = self._read_jsonl(output_path)
            artifact_dir = directory / "output.regen-artifact"

            self.assertEqual(len(outputs), 3)
            for index, row in enumerate(outputs):
                self.assertEqual(row["id"], str(index))
                self.assertEqual(row["status"], "success")
                self.assertEqual(row["extra_field"], f"metadata-{index}")
                self.assertEqual(
                    row["conversations"][-1]["content"], "regenerated answer"
                )
            manifest = json.loads(
                (artifact_dir / "replay" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["state"], "FINALIZED")
            self.assertEqual(manifest["counts"]["status_counts"]["success"], 3)
            self.assertEqual(
                manifest["generators"]["teacher"]["revision"], "abc123"
            )
            self.assertEqual(
                self._read_jsonl(directory / "output_error.jsonl"), []
            )
            self.assertEqual(
                self._read_jsonl(directory / "output_skipped.jsonl"), []
            )

    def test_num_samples_limits_planned_rows_and_skips_invalid_input(self):
        rows = [[]] + [
            {
                "id": str(index),
                "conversations": [{"role": "user", "content": f"question {index}"}],
            }
            for index in range(4)
        ]
        with TemporaryDirectory() as tmpdir:
            directory, output_path, server = self._run(
                tmpdir, rows, extra_args=["--num-samples", "2"]
            )

            self.assertEqual(len(server.requests), 2)
            self.assertEqual(len(self._read_jsonl(output_path)), 2)
            skipped = self._read_jsonl(directory / "output_skipped.jsonl")
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["status"], "skipped")

    def test_disable_reasoning_skips_residual_think_output(self):
        rows = [
            {
                "id": "0",
                "conversations": [{"role": "user", "content": "question"}],
            }
        ]
        server = FakeServer(content="<think>hidden</think>answer")
        with TemporaryDirectory() as tmpdir:
            directory, output_path, _ = self._run(
                tmpdir,
                rows,
                extra_args=["--reasoning", "disable"],
                server=server,
            )

            self.assertEqual(self._read_jsonl(output_path), [])
            skipped = self._read_jsonl(directory / "output_skipped.jsonl")
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["status"], "skipped")
            self.assertIn("control token", skipped[0]["error"])

    def test_save_reasoning_sets_contract_and_strips_history_reasoning(self):
        rows = [
            {
                "id": "0",
                "conversations": [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "old first answer"},
                    {"role": "user", "content": "second question"},
                    {"role": "assistant", "content": "old second answer"},
                ],
            }
        ]
        server = FakeServer(reasoning="fresh reasoning")
        with TemporaryDirectory() as tmpdir:
            directory, output_path, _ = self._run(
                tmpdir,
                rows,
                extra_args=["--reasoning", "save"],
                server=server,
            )

            self.assertEqual(len(server.requests), 2)
            for request in server.requests:
                self.assertTrue(
                    request["chat_template_kwargs"]["enable_thinking"]
                )
            second_request = server.requests[1]
            history_assistant = second_request["messages"][1]
            self.assertEqual(history_assistant["role"], "assistant")
            self.assertEqual(history_assistant["content"], "regenerated answer")
            self.assertNotIn("reasoning_content", history_assistant)

            outputs = self._read_jsonl(output_path)
            self.assertEqual(len(outputs), 1)
            for message in outputs[0]["conversations"]:
                if message["role"] == "assistant":
                    self.assertEqual(
                        message["reasoning_content"], "fresh reasoning"
                    )

    def test_save_reasoning_rejects_missing_reasoning(self):
        rows = [
            {
                "id": "0",
                "conversations": [{"role": "user", "content": "question"}],
            }
        ]
        with TemporaryDirectory() as tmpdir:
            directory, output_path, _ = self._run(
                tmpdir,
                rows,
                extra_args=["--reasoning", "save"],
                server=FakeServer(reasoning=None),
            )

            self.assertEqual(self._read_jsonl(output_path), [])
            skipped = self._read_jsonl(directory / "output_skipped.jsonl")
            self.assertEqual(len(skipped), 1)
            self.assertIn("reasoning_content", skipped[0]["error"])

    def test_resume_reuses_committed_attempts_without_new_generation(self):
        rows = [
            {
                "id": str(index),
                "conversations": [{"role": "user", "content": f"question {index}"}],
            }
            for index in range(2)
        ]
        with TemporaryDirectory() as tmpdir:
            directory, output_path, first_server = self._run(tmpdir, rows)
            self.assertEqual(len(first_server.requests), 2)

            resume_server = FakeServer(content="different answer")
            directory, output_path, resume_server = self._run(
                tmpdir, rows, extra_args=["--resume"], server=resume_server
            )

            self.assertEqual(len(resume_server.requests), 0)
            outputs = self._read_jsonl(output_path)
            self.assertEqual(len(outputs), 2)
            self.assertEqual(
                outputs[0]["conversations"][-1]["content"], "regenerated answer"
            )

    def test_mixed_replay_and_completion_rows_run_in_one_invocation(self):
        rows = [
            {
                "id": "replay-row",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "stale"},
                ],
            },
            {
                "id": "prompt-row",
                "conversations": [{"role": "user", "content": "prompt only"}],
            },
            {
                "id": "mixed-row",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "stale"},
                    {"role": "user", "content": "trailing"},
                ],
            },
        ]
        with TemporaryDirectory() as tmpdir:
            directory, output_path, server = self._run(tmpdir, rows)
            outputs = self._read_jsonl(output_path)

            self.assertEqual(
                [row["id"] for row in outputs], ["replay-row", "prompt-row"]
            )
            for row in outputs:
                self.assertEqual(
                    row["conversations"][-1]["content"], "regenerated answer"
                )
            skipped = self._read_jsonl(directory / "output_skipped.jsonl")
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["id"], "mixed-row")
            self.assertIn("trailing user turn", skipped[0]["error"])

    def test_duplicate_ids_survive_regeneration_with_original_ids(self):
        rows = [
            {
                "id": "same",
                "conversations": [{"role": "user", "content": f"question {index}"}],
            }
            for index in range(2)
        ]
        with TemporaryDirectory() as tmpdir:
            _, output_path, _ = self._run(tmpdir, rows)
            outputs = self._read_jsonl(output_path)

            self.assertEqual([row["id"] for row in outputs], ["same", "same"])
            self.assertEqual(
                [row["status"] for row in outputs], ["success", "success"]
            )


class TestQwenRegenerationRecipe(TestCase):
    """The example scripts select recipes and drive `specforge data regen`."""

    RECIPES = Path(__file__).resolve().parents[2] / (
        "examples/data_regeneration/recipes"
    )

    def _make_fake_python(self, directory: Path) -> Path:
        fake_python = directory / "fake_python"
        fake_python.write_text(
            f"""#!{sys.executable}
import json
import os
import sys

args = sys.argv[1:]
if args[:2] == ["-m", "specforge"]:
    with open(os.environ["CAPTURE_ARGS"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps(args) + "\\n")
    command = args[4] if len(args) > 4 else ""
    if command == "inspect":
        error_rows = int(os.environ.get("FAKE_ERROR_ROWS", "0"))
        drop_rows = int(os.environ.get("FAKE_DROP_ROWS", "0"))
        planned = 2 - drop_rows
        success = planned - error_rows
        print(json.dumps({{
            "state": "FINALIZED",
            "counts": {{
                "planned_tasks": planned,
                "status_counts": {{
                    "success": success,
                    "unresolved_error": error_rows,
                }},
            }},
        }}))
    raise SystemExit(0)

os.execv(sys.executable, [sys.executable, *args])
""",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        return fake_python

    def _run_recipe(
        self,
        tmpdir: str,
        *,
        error_rows: int = 0,
        drop_rows: int = 0,
        trailing_newline: bool = True,
        entrypoint: str = "run_qwen_sharegpt_regeneration.sh",
        profile: str = "qwen3.6-27b",
    ):
        directory = Path(tmpdir)
        input_path = directory / "input.jsonl"
        capture_path = directory / "args.jsonl"
        capture_path.write_text("", encoding="utf-8")
        input_text = "".join(
            json.dumps(
                {
                    "id": str(index),
                    "conversations": [{"role": "user", "content": f"question {index}"}],
                }
            )
            + "\n"
            for index in range(2)
        )
        input_path.write_text(
            input_text if trailing_newline else input_text.rstrip("\n"),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "MODEL_PROFILE": profile,
            "PYTHON": str(self._make_fake_python(directory)),
            "INPUT_FILE": str(input_path),
            "ARTIFACT_DIR": str(directory / "artifact"),
            "CAPTURE_ARGS": str(capture_path),
            "FAKE_ERROR_ROWS": str(error_rows),
            "FAKE_DROP_ROWS": str(drop_rows),
        }
        result = subprocess.run(
            ["bash", f"examples/data_regeneration/{entrypoint}"],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
        )
        calls = [
            json.loads(line)
            for line in capture_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        return result, calls

    def _load_recipe(self, name: str):
        import yaml

        with (self.RECIPES / name).open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _cli_call(self, calls, command):
        for args in calls:
            if args[:2] == ["-m", "specforge"] and command in args:
                return args
        raise AssertionError(f"no `specforge data regen {command}` call captured")

    def test_qwen36_profile_uses_reasoning_contract(self):
        with TemporaryDirectory() as tmpdir:
            result, calls = self._run_recipe(tmpdir)

        self.assertEqual(result.returncode, 0, result.stderr)
        run_args = self._cli_call(calls, "run")
        recipe_path = run_args[run_args.index("--config") + 1]
        self.assertTrue(recipe_path.endswith("qwen3.6-27b-sharegpt-reasoning.yaml"))
        self.assertIn("--endpoint", run_args)
        self.assertIn("teacher=http://localhost:30000", run_args)
        self.assertTrue(
            any(arg.startswith("sources.sharegpt.config.path=") for arg in run_args)
        )

        recipe = self._load_recipe("qwen3.6-27b-sharegpt-reasoning.yaml")
        teacher = recipe["generators"]["teacher"]
        self.assertEqual(teacher["model"], "Qwen/Qwen3.6-27B")
        self.assertEqual(teacher["sampling"]["reasoning"], "required")
        self.assertEqual(teacher["sampling"]["max_tokens"], 32768)
        self.assertEqual(teacher["config"]["history_reasoning"], "strip")

    def test_qwen3_8b_compatibility_wrapper_selects_non_reasoning_profile(self):
        with TemporaryDirectory() as tmpdir:
            result, calls = self._run_recipe(
                tmpdir,
                entrypoint="run_qwen3_8b_sharegpt_non_reasoning.sh",
                profile="qwen3-8b",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        run_args = self._cli_call(calls, "run")
        recipe_path = run_args[run_args.index("--config") + 1]
        self.assertTrue(
            recipe_path.endswith("qwen3-8b-sharegpt-non-reasoning.yaml")
        )

        recipe = self._load_recipe("qwen3-8b-sharegpt-non-reasoning.yaml")
        teacher = recipe["generators"]["teacher"]
        self.assertEqual(teacher["model"], "Qwen/Qwen3-8B")
        self.assertEqual(teacher["sampling"]["reasoning"], "disabled")
        self.assertEqual(teacher["sampling"]["max_tokens"], 4096)

    def test_recipes_load_as_valid_pinned_workflows(self):
        from specforge.data.regen.recipe import load_recipe

        for name in (
            "qwen3-8b-sharegpt-non-reasoning.yaml",
            "qwen3.6-27b-sharegpt-reasoning.yaml",
        ):
            recipe = load_recipe(self.RECIPES / name)
            self.assertEqual(
                [stage.operation for stage in recipe.workflow],
                ["replay_assistants"],
            )
            self.assertTrue(recipe.generators["teacher"].revision)

    def test_recipe_runs_full_artifact_lifecycle(self):
        with TemporaryDirectory() as tmpdir:
            result, calls = self._run_recipe(tmpdir)

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [args[4] for args in calls if args[:2] == ["-m", "specforge"]]
        self.assertEqual(commands, ["run", "validate", "finalize", "inspect"])

    def test_recipe_allows_accounted_error_rows(self):
        with TemporaryDirectory() as tmpdir:
            result, _ = self._run_recipe(tmpdir, error_rows=1)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("success fraction: 50.00%", result.stdout)

    def test_recipe_counts_input_without_trailing_newline(self):
        with TemporaryDirectory() as tmpdir:
            result, _ = self._run_recipe(tmpdir, trailing_newline=False)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("input rows: 2", result.stdout)

    def test_recipe_fails_when_rows_are_unaccounted_for(self):
        with TemporaryDirectory() as tmpdir:
            result, _ = self._run_recipe(tmpdir, drop_rows=1)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not account for every input row", result.stderr)
