"""Sandboxed tool-loop execution with recorded evidence.

The generator drives the loop: every emitted tool call is executed by the
stage's declared ``ToolEnvironment``, the result is appended as a bound tool
message, and generation continues on the extended history until the
assistant stops calling tools. Every source-authored turn survives as a
prefix of the trajectory, every generated block is kept losslessly, and the
execution evidence (environment identity, argument/result digests) is
recorded in the stage history.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..contracts import (
    RecordEnvelope,
    RecordKey,
    StageEvent,
    canonical_digest,
)
from ..errors import ContractError, PolicyReject
from ..recipe import StageSpec
from ..records.messages import preserved_projection
from ..registry import TOOL_ENVIRONMENTS
from .base import OperationContext

DEFAULT_MAX_TOOL_ROUNDS = 8
DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0


def _environment(stage: StageSpec):
    name = stage.config.get("environment")
    if not isinstance(name, str) or not name:
        raise ContractError(
            f"stage {stage.id!r} requires config.environment naming a "
            "registered tool environment"
        )
    registration = TOOL_ENVIRONMENTS.resolve(
        name, required_capabilities={"tool_execution"}
    )
    environment = registration.factory(stage.config.get("environment_config"))
    policy = stage.config.get("side_effect_policy", "forbid")
    if policy not in {"forbid", "sandboxed"}:
        raise ContractError("side_effect_policy must be 'forbid' or 'sandboxed'")
    if policy == "forbid" and environment.side_effects != "none":
        raise ContractError(
            f"environment {name!r} declares side effects "
            f"({environment.side_effects!r}) but the stage forbids them"
        )
    return environment


class ExecuteToolLoopOperation:
    """Generate → execute tools → observe → generate, until convergence."""

    def run(
        self,
        envelope: RecordEnvelope,
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]:
        source_messages = envelope.payload.get("conversations")
        if not isinstance(source_messages, list) or not source_messages:
            raise ContractError("tool loop requires payload.conversations")
        if any(message.get("role") == "assistant" for message in source_messages):
            raise PolicyReject(
                "tool loop starts from prompt-only rows; replay handles "
                "recorded assistant history"
            )
        tools = list(envelope.payload.get("tools") or [])
        if not tools:
            raise PolicyReject("tool loop requires declared tools on the row")

        environment = _environment(stage)
        declared = set(environment.declared_tools())
        max_rounds = int(stage.config.get("max_tool_rounds", DEFAULT_MAX_TOOL_ROUNDS))
        timeout_seconds = float(
            stage.config.get("tool_timeout_seconds", DEFAULT_TOOL_TIMEOUT_SECONDS)
        )
        if max_rounds <= 0 or timeout_seconds <= 0:
            raise ContractError("tool loop rounds and timeout must be positive")

        variant = envelope.key.variant
        history = deepcopy(source_messages)
        generations: list[dict[str, Any]] = []
        executions: list[dict[str, Any]] = []
        converged = False
        for round_number in range(1, max_rounds + 1):
            result = context.generate(
                envelope,
                stage,
                messages=history,
                tools=tools,
                variant=variant,
                generation_ordinal=round_number,
            )
            assistant = deepcopy(dict(result.message))
            history.append(assistant)
            result_meta = result.to_dict()
            result_meta.pop("message", None)
            generations.append(result_meta)

            calls = assistant.get("tool_calls") or []
            if not calls:
                converged = True
                break
            for call in calls:
                name = call["function"]["name"]
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise PolicyReject(
                        f"tool call {name!r} has no stable id to bind its result"
                    )
                if name not in declared:
                    raise PolicyReject(
                        f"generator called undeclared tool {name!r}"
                    )
                arguments = call["function"].get("arguments", {})
                output = environment.execute(
                    name, arguments, timeout_seconds=timeout_seconds
                )
                if not isinstance(output, str):
                    raise ContractError("tool environments must return text")
                history.append(
                    {
                        "role": "tool",
                        "content": output,
                        "tool_call_id": call_id,
                        "name": name,
                    }
                )
                executions.append(
                    {
                        "round": round_number,
                        "tool": name,
                        "call_id": call_id,
                        "environment": environment.name,
                        "environment_version": environment.version,
                        "arguments_digest": canonical_digest(dict(arguments)),
                        "result_digest": canonical_digest(output),
                        "result_bytes": len(output.encode("utf-8")),
                    }
                )
        if not converged:
            raise PolicyReject(
                f"tool loop did not converge within {max_rounds} rounds"
            )

        payload = dict(envelope.payload)
        payload["conversations"] = history
        source_prefix = preserved_projection(envelope.payload)
        if preserved_projection(payload)[: len(source_prefix)] != source_prefix:
            raise ContractError("tool loop altered source-authored messages")
        key = RecordKey(
            source=envelope.key.source,
            source_id=envelope.key.source_id,
            variant=variant,
        )
        return [
            envelope.with_payload(
                payload,
                key=key,
                event=StageEvent(
                    stage_id=stage.id,
                    operation=stage.operation,
                    generator=stage.generator,
                    variant=variant,
                    generation_count=len(generations),
                    metadata={
                        "generations": generations,
                        "tool_executions": executions,
                        "environment": environment.name,
                        "environment_version": environment.version,
                    },
                ),
            )
        ]


def create_tool_loop_operation() -> ExecuteToolLoopOperation:
    return ExecuteToolLoopOperation()


__all__ = ["ExecuteToolLoopOperation", "create_tool_loop_operation"]
