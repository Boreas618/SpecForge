"""Benchmark regeneration planning, execution, finalization, and validation.

Each stage is measured separately (wall time and peak Python heap via
tracemalloc) against a synthetic source and the deterministic fake backend,
so the numbers reflect pipeline overhead rather than model latency. The
planner, executor, and finalizer are required to stay bounded-memory: peak
heap must not grow linearly with --rows.

Example:
    python benchmarks/bench_regen_scale.py --rows 100000 --shards 64
    python benchmarks/bench_regen_scale.py --rows 50000 --partitioning hash
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

try:
    import specforge  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_source(path: Path, rows: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(
                json.dumps(
                    {
                        "id": f"row-{index}",
                        "messages": [
                            {"role": "user", "content": f"question {index}"},
                            {"role": "assistant", "content": f"stale {index}"},
                        ],
                    }
                )
                + "\n"
            )


def _measure(label: str, callable_):
    tracemalloc.start()
    started = time.perf_counter()
    result = callable_()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"{label:>12}: {elapsed:8.2f}s  peak-heap {peak / (1 << 20):8.1f} MiB",
        flush=True,
    )
    return result, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument(
        "--partitioning", choices=["contiguous", "hash"], default="contiguous"
    )
    parser.add_argument(
        "--workdir", default=None, help="reuse a directory instead of a temp dir"
    )
    args = parser.parse_args()

    from specforge.data.regen.executor import run_local
    from specforge.data.regen.finalize import (
        publish_artifact,
        resolve_attempts,
        validate_artifact,
    )
    from specforge.data.regen.planner import plan_recipe
    from specforge.data.regen.recipe import RegenerationRecipe

    workdir = Path(args.workdir) if args.workdir else None
    context = (
        tempfile.TemporaryDirectory() if workdir is None else _reuse(workdir)
    )
    with context as base:
        base = Path(base)
        source = base / "source.jsonl"
        print(f"synthesizing {args.rows} rows …", flush=True)
        _write_source(source, args.rows)

        recipe = RegenerationRecipe.model_validate(
            {
                "version": 1,
                "seed": 1,
                "sources": {
                    "bench": {
                        "adapter": "jsonl",
                        "record_adapter": "openai_messages",
                        "config": {"path": str(source)},
                    }
                },
                "generators": {
                    "teacher": {
                        "backend": "fake",
                        "model": "bench-teacher",
                        "revision": "bench",
                        "codec": "structured_chat",
                        "config": {"content_template": "answer:{prompt}"},
                    }
                },
                "workflow": [
                    {
                        "id": "regenerate",
                        "operation": "replay_assistants",
                        "generator": "teacher",
                    }
                ],
                "validation": {"profiles": ["baseline"]},
                "output": {
                    "uri": str(base / "artifact"),
                    "shards": args.shards,
                    "partitioning": args.partitioning,
                },
            }
        )

        layout, plan_s = _measure("plan", lambda: plan_recipe(recipe))
        _, generate_s = _measure("generate", lambda: run_local(layout))
        _, resolve_s = _measure("resolve", lambda: resolve_attempts(layout))
        _, validate_s = _measure("validate", lambda: validate_artifact(layout))
        manifest, publish_s = _measure("publish", lambda: publish_artifact(layout))

        rows = manifest["counts"]["output_rows"]
        print(f"\nrows: {rows}  shards: {args.shards} ({args.partitioning})")
        for label, seconds in (
            ("plan", plan_s),
            ("generate", generate_s),
            ("resolve", resolve_s),
            ("validate", validate_s),
        ):
            print(f"{label:>12}: {rows / seconds:12.0f} rows/s")


def _reuse(path: Path):
    from contextlib import contextmanager

    @contextmanager
    def keep():
        path.mkdir(parents=True, exist_ok=True)
        yield str(path)

    return keep()


if __name__ == "__main__":
    main()
