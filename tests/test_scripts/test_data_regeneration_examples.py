"""The example scripts select recipes and drive `specforge data regen`."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


class TestQwenRegenerationRecipe(TestCase):
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
