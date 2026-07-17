"""CLI surface for the data regeneration artifact lifecycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifact import ArtifactLayout, read_state
from .errors import ArtifactError, ContractError
from .executor import prepare_retry, run_local, run_worker
from .finalize import (
    inspect_artifact,
    publish_artifact,
    resolve_attempts,
    validate_artifact,
)
from .planner import plan_recipe
from .recipe import load_recipe


def _runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        metavar="GENERATOR=URL",
        help="runtime-only inference endpoint; repeat for endpoint pools",
    )
    parser.add_argument(
        "--api-key-env",
        action="append",
        default=[],
        metavar="GENERATOR=ENV",
        help="name of an environment variable holding a runtime API key",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="runtime request timeout in seconds",
    )


def configure_regen_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="regen_command", required=True)

    plan = commands.add_parser("plan", help="resolve and pin an immutable run plan")
    plan.add_argument(
        "--config", required=True, help="YAML or JSON regeneration recipe"
    )
    plan.add_argument("overrides", nargs="*", help="typed dotted recipe overrides")

    run = commands.add_parser("run", help="plan and run every local static shard")
    run.add_argument("--config", required=True, help="YAML or JSON regeneration recipe")
    run.add_argument("overrides", nargs="*", help="typed dotted recipe overrides")
    _runtime_arguments(run)

    worker = commands.add_parser("worker", help="run or resume plan shards")
    worker.add_argument("--plan", required=True, help="path to run-plan.json")
    ownership = worker.add_mutually_exclusive_group(required=True)
    ownership.add_argument(
        "--shard-index", type=int, help="run exactly one static shard"
    )
    ownership.add_argument(
        "--leased",
        action="store_true",
        help="claim shards through the lease store until the pool drains",
    )
    worker.add_argument(
        "--worker-id",
        default=None,
        help="stable worker identity for leased execution",
    )
    worker.add_argument(
        "--lease-seconds",
        type=float,
        default=60.0,
        help="lease duration before an unresponsive worker's shard is reclaimed",
    )
    _runtime_arguments(worker)

    reconcile = commands.add_parser(
        "reconcile",
        help="clean orphan temp files, reclaim expired leases, report progress",
    )
    reconcile.add_argument("--artifact", required=True)

    retry = commands.add_parser("retry", help="retry eligible unresolved attempts")
    retry.add_argument("--artifact", required=True)
    retry.add_argument(
        "--categories",
        nargs="+",
        default=["transport_retryable"],
        choices=["transport_retryable"],
    )
    _runtime_arguments(retry)

    validate = commands.add_parser("validate", help="run publication validators")
    validate.add_argument("--artifact", required=True)

    finalize = commands.add_parser("finalize", help="publish a validated manifest")
    finalize.add_argument("--artifact", required=True)

    inspect = commands.add_parser(
        "inspect", help="inspect provenance and lifecycle state"
    )
    inspect.add_argument("--artifact", required=True)
    inspect.add_argument("--json", action="store_true", dest="as_json")


def _assign(values: list[str], *, label: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ContractError(f"{label} {value!r} must be GENERATOR=VALUE")
        generator, item = value.split("=", 1)
        if not generator or not item:
            raise ContractError(f"{label} {value!r} must be GENERATOR=VALUE")
        result.setdefault(generator, []).append(item)
    return result


def _runtime(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    endpoints = _assign(args.endpoint, label="endpoint")
    api_names = _assign(args.api_key_env, label="api-key-env")
    runtime: dict[str, dict[str, Any]] = {}
    for generator, values in endpoints.items():
        runtime.setdefault(generator, {})["endpoints"] = values
    for generator, values in api_names.items():
        if len(values) != 1:
            raise ContractError(
                f"generator {generator!r} must have exactly one api-key-env"
            )
        runtime.setdefault(generator, {})["api_key_env"] = values[0]
    for value in runtime.values():
        value["timeout"] = args.timeout
    return runtime


def _layout_from_plan(path: str) -> ArtifactLayout:
    plan = Path(path).expanduser().resolve()
    if plan.name != "run-plan.json" or not plan.is_file():
        raise ArtifactError("--plan must name an existing run-plan.json")
    return ArtifactLayout(plan.parent)


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def run_regen_command(args: argparse.Namespace) -> int:
    command = args.regen_command
    if command == "plan":
        layout = plan_recipe(load_recipe(args.config, args.overrides))
        _emit(inspect_artifact(layout))
        return 0
    if command == "run":
        recipe = load_recipe(args.config, args.overrides)
        layout = ArtifactLayout.from_uri(recipe.output.uri)
        if not layout.plan.exists():
            layout = plan_recipe(recipe)
        run_local(layout, runtime=_runtime(args))
        _emit(resolve_attempts(layout))
        return 0
    if command == "worker":
        layout = _layout_from_plan(args.plan)
        if args.leased:
            import os
            import socket

            from .leases import run_leased

            worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
            results = run_leased(
                layout,
                worker_id,
                runtime=_runtime(args),
                lease_seconds=args.lease_seconds,
            )
            _emit({"worker_id": worker_id, "shards": results})
            return 0
        result = run_worker(
            layout,
            args.shard_index,
            runtime=_runtime(args),
        )
        _emit(result)
        return 0
    if command == "reconcile":
        from .maintenance import reconcile_artifact

        _emit(reconcile_artifact(args.artifact))
        return 0
    if command == "retry":
        layout = prepare_retry(args.artifact)
        plan = inspect_artifact(layout)
        results = [
            run_worker(
                layout,
                int(shard["index"]),
                runtime=_runtime(args),
                retry_categories=set(args.categories),
            )
            for shard in plan["shards"]
        ]
        summary = resolve_attempts(layout)
        _emit({"workers": results, "summary": summary})
        return 0
    if command == "validate":
        layout = ArtifactLayout.from_uri(args.artifact)
        if read_state(layout) == "RUNNING":
            resolve_attempts(layout)
        _emit(validate_artifact(layout))
        return 0
    if command == "finalize":
        _emit(publish_artifact(args.artifact))
        return 0
    if command == "inspect":
        # JSON is the stable interface. The default remains concise but valid
        # JSON so it is safe to pipe before callers opt into --json explicitly.
        _emit(inspect_artifact(args.artifact))
        return 0
    raise AssertionError(f"unhandled regeneration command {command!r}")


__all__ = ["configure_regen_parser", "run_regen_command"]
