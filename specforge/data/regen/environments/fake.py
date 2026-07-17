"""Deterministic side-effect-free tool environment for tests and dry runs."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import canonical_json
from ..errors import ContractError, PolicyReject


class DeterministicToolEnvironment:
    name = "deterministic"
    version = "1"
    side_effects = "none"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = dict(config or {})
        results = config.get("results", {})
        if not isinstance(results, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in results.items()
        ):
            raise ContractError(
                "deterministic environment requires config.results as "
                "{tool_name: result_template}"
            )
        self.results = dict(results)

    def declared_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self.results))

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> str:
        template = self.results.get(name)
        if template is None:
            raise PolicyReject(
                f"environment {self.name!r} does not declare tool {name!r}"
            )
        try:
            return template.format(
                arguments=canonical_json(dict(arguments)), **dict(arguments)
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise PolicyReject(
                f"tool {name!r} arguments do not satisfy the result template: {exc}"
            ) from None


def create_deterministic_environment(
    config: Mapping[str, Any] | None = None,
) -> DeterministicToolEnvironment:
    return DeterministicToolEnvironment(config)


__all__ = ["DeterministicToolEnvironment", "create_deterministic_environment"]
