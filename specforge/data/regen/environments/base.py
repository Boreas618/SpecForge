"""Tool-environment contract for sandboxed, evidence-recorded execution.

An environment declares its identity (name, version), the tools it serves,
and its side-effect class. Recipes select environments by registered name;
the planner refuses a stage whose side-effect policy the environment cannot
satisfy. Execution results are plain text plus recorded evidence — never
tensors, and never presented as live execution when they were replayed.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class ToolEnvironment(Protocol):
    #: registered component name and declared version.
    name: str
    version: str
    #: "none" for pure/deterministic environments, "sandboxed" for
    #: environments that touch real resources inside a declared sandbox.
    side_effects: str

    def declared_tools(self) -> tuple[str, ...]:
        """Names this environment can execute; other calls are rejects."""
        ...

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> str:
        """Execute one tool call and return its text result.

        Implementations must honor ``timeout_seconds`` and raise
        ``GenerationError`` (transport/timeout categories) on failure.
        """
        ...


__all__ = ["ToolEnvironment"]
