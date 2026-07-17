"""Critique, revision, and preference-pair operations.

Stage relationships stay structured: critiques live in a typed
``payload["critiques"]`` list, revisions preserve the text they replaced in
``payload["revisions"]``, and preference pairs keep the chosen trajectory as
the canonical conversation with the rejected trajectory alongside it. No
stage flattens its history into an assistant message string.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from ..contracts import RecordEnvelope, RecordKey, StageEvent, canonical_json
from ..errors import ContractError, PolicyReject
from ..recipe import StageSpec
from .base import OperationContext

DEFAULT_CRITIQUE_INSTRUCTION = (
    "Critique the assistant's final answer. Point out factual, logical, "
    "and clarity problems."
)
DEFAULT_REVISE_INSTRUCTION = (
    "Revise your previous answer to address this critique. Reply with the "
    "complete revised answer only."
)
DEFAULT_PAIR_INSTRUCTION = (
    "Compare the candidate answers. Return JSON with the zero-based indexes "
    '{"chosen": best, "rejected": worst}.'
)


def _instruction(stage: StageSpec, key: str, default: str) -> str:
    value = stage.config.get(key, default)
    if not isinstance(value, str) or not value:
        raise ContractError(f"stage {stage.id!r} {key} must be non-empty text")
    return value


def _final_assistant(messages: list[dict[str, Any]], *, stage: StageSpec) -> int:
    if not messages or messages[-1].get("role") != "assistant":
        raise PolicyReject(
            f"stage {stage.id!r} requires a conversation ending with an "
            "assistant turn"
        )
    return len(messages) - 1


class CritiqueOperation:
    """Attach a structured critique of the final assistant turn."""

    def run(
        self,
        envelope: RecordEnvelope,
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]:
        messages = list(envelope.payload.get("conversations") or [])
        _final_assistant(messages, stage=stage)
        instruction = _instruction(stage, "instruction", DEFAULT_CRITIQUE_INSTRUCTION)
        result = context.generate(
            envelope,
            stage,
            messages=[*deepcopy(messages), {"role": "user", "content": instruction}],
            tools=[],
            variant=envelope.key.variant,
            generation_ordinal=1,
        )
        critique = {
            "stage_id": stage.id,
            "generator": stage.generator,
            "content": result.message.get("content", ""),
        }
        reasoning = result.message.get("reasoning_content")
        if isinstance(reasoning, str):
            critique["reasoning_content"] = reasoning

        payload = dict(envelope.payload)
        payload["critiques"] = [*payload.get("critiques", []), critique]
        result_meta = result.to_dict()
        result_meta.pop("message", None)
        return [
            envelope.with_payload(
                payload,
                key=envelope.key,
                event=StageEvent(
                    stage_id=stage.id,
                    operation=stage.operation,
                    generator=stage.generator,
                    variant=envelope.key.variant,
                    generation_count=1,
                    metadata={"generations": [result_meta]},
                ),
            )
        ]


class ReviseOperation:
    """Replace the final assistant turn with a critique-conditioned revision."""

    def run(
        self,
        envelope: RecordEnvelope,
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]:
        messages = list(envelope.payload.get("conversations") or [])
        final_index = _final_assistant(messages, stage=stage)
        critiques = envelope.payload.get("critiques") or []
        if not critiques:
            raise PolicyReject(
                f"stage {stage.id!r} requires an earlier critique stage"
            )
        critique = critiques[-1]
        instruction = _instruction(stage, "instruction", DEFAULT_REVISE_INSTRUCTION)
        result = context.generate(
            envelope,
            stage,
            messages=[
                *deepcopy(messages),
                {
                    "role": "user",
                    "content": f"{instruction}\n\nCritique:\n{critique['content']}",
                },
            ],
            tools=[],
            variant=envelope.key.variant,
            generation_ordinal=1,
        )
        revised = deepcopy(dict(result.message))
        previous = deepcopy(messages[final_index])

        payload = dict(envelope.payload)
        payload["conversations"] = [*messages[:final_index], revised]
        payload["revisions"] = [
            *payload.get("revisions", []),
            {
                "stage_id": stage.id,
                "critique_stage_id": critique["stage_id"],
                "previous": previous,
            },
        ]
        result_meta = result.to_dict()
        result_meta.pop("message", None)
        return [
            envelope.with_payload(
                payload,
                key=envelope.key,
                event=StageEvent(
                    stage_id=stage.id,
                    operation=stage.operation,
                    generator=stage.generator,
                    variant=envelope.key.variant,
                    generation_count=1,
                    metadata={"generations": [result_meta]},
                ),
            )
        ]


class CandidatePairOperation:
    """Join a row's candidate variants into one typed preference record."""

    def run_many(
        self,
        envelopes: list[RecordEnvelope],
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]:
        if not envelopes:
            return []
        if len(envelopes) < 2:
            raise PolicyReject("candidate_pair requires at least two variants")
        if stage.generator is None:
            raise ContractError("candidate_pair requires a judge generator")
        base = envelopes[0]
        identity = (base.key.source, canonical_json(base.key.source_id))
        if any(
            (item.key.source, canonical_json(item.key.source_id)) != identity
            for item in envelopes
        ):
            raise ContractError("candidate_pair cannot join different source rows")

        instruction = _instruction(stage, "instruction", DEFAULT_PAIR_INSTRUCTION)
        candidates = [
            {
                "index": index,
                "record_key": item.key.to_dict(),
                "conversations": list(item.payload.get("conversations") or []),
            }
            for index, item in enumerate(envelopes)
        ]
        result = context.generate(
            base,
            stage,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": canonical_json(candidates)},
            ],
            tools=[],
            variant=f"{base.key.variant}.{stage.id}.judge",
            generation_ordinal=1,
        )
        content = result.message.get("content")
        try:
            verdict = json.loads(content) if isinstance(content, str) else content
            chosen_index = int(verdict["chosen"])
            rejected_index = int(verdict["rejected"])
            if isinstance(verdict["chosen"], bool) or isinstance(
                verdict["rejected"], bool
            ):
                raise ValueError
        except (TypeError, KeyError, ValueError, json.JSONDecodeError):
            raise PolicyReject(
                "judge did not return chosen/rejected candidate indexes"
            ) from None
        valid = range(len(envelopes))
        if (
            chosen_index not in valid
            or rejected_index not in valid
            or chosen_index == rejected_index
        ):
            raise PolicyReject("judge returned an invalid candidate pairing")

        chosen = envelopes[chosen_index]
        rejected = envelopes[rejected_index]
        variant = f"{chosen.key.variant}.{stage.id}.pair"
        payload = dict(chosen.payload)
        payload["rejected_conversations"] = deepcopy(
            list(rejected.payload.get("conversations") or [])
        )
        payload["preference"] = {
            "stage_id": stage.id,
            "chosen_key": chosen.key.to_dict(),
            "rejected_key": rejected.key.to_dict(),
        }
        result_meta = result.to_dict()
        result_meta.pop("message", None)
        return [
            chosen.with_payload(
                payload,
                key=RecordKey(chosen.key.source, chosen.key.source_id, variant),
                event=StageEvent(
                    stage_id=stage.id,
                    operation=stage.operation,
                    generator=stage.generator,
                    variant=variant,
                    generation_count=1,
                    metadata={
                        "candidate_keys": [item.key.to_dict() for item in envelopes],
                        "chosen_index": chosen_index,
                        "rejected_index": rejected_index,
                        "judge_generation": result_meta,
                    },
                ),
            )
        ]


def create_critique_operation() -> CritiqueOperation:
    return CritiqueOperation()


def create_revise_operation() -> ReviseOperation:
    return ReviseOperation()


def create_candidate_pair_operation() -> CandidatePairOperation:
    return CandidatePairOperation()


__all__ = [
    "CandidatePairOperation",
    "CritiqueOperation",
    "ReviseOperation",
    "create_candidate_pair_operation",
    "create_critique_operation",
    "create_revise_operation",
]
