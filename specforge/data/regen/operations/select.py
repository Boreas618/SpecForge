"""Candidate selection operations that preserve the variant graph."""

from __future__ import annotations

import json
from typing import Any

from ..contracts import RecordEnvelope, RecordKey, StageEvent, canonical_json
from ..errors import ContractError, PolicyReject
from ..recipe import StageSpec
from .base import OperationContext


class SelectCandidateOperation:
    def run_many(
        self,
        envelopes: list[RecordEnvelope],
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]:
        if not envelopes:
            return []
        if len(envelopes) < 2:
            raise PolicyReject("select_candidate requires at least two variants")
        if stage.generator is None:
            raise ContractError("select_candidate requires a judge generator")
        base = envelopes[0]
        identity = (base.key.source, canonical_json(base.key.source_id))
        if any(
            (item.key.source, canonical_json(item.key.source_id)) != identity
            for item in envelopes
        ):
            raise ContractError("select_candidate cannot join different source rows")
        candidates = [
            {
                "index": index,
                "record_key": item.key.to_dict(),
                "payload": dict(item.payload),
            }
            for index, item in enumerate(envelopes)
        ]
        instruction = stage.config.get(
            "instruction",
            "Select the best candidate. Return only its zero-based integer index.",
        )
        if not isinstance(instruction, str) or not instruction:
            raise ContractError("select_candidate instruction must be text")
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
            parsed = json.loads(content) if isinstance(content, str) else content
            if isinstance(parsed, dict):
                parsed = parsed.get("index")
            if isinstance(parsed, bool):
                raise ValueError
            selected_index = int(parsed)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise PolicyReject("judge did not return a candidate index") from None
        if not 0 <= selected_index < len(envelopes):
            raise PolicyReject(
                "judge selected a candidate index outside the variant set"
            )
        selected = envelopes[selected_index]
        variant = f"{selected.key.variant}.{stage.id}.selected"
        key = RecordKey(
            source=selected.key.source,
            source_id=selected.key.source_id,
            variant=variant,
        )
        return [
            selected.with_payload(
                selected.payload,
                key=key,
                event=StageEvent(
                    stage_id=stage.id,
                    operation=stage.operation,
                    generator=stage.generator,
                    variant=variant,
                    generation_count=1,
                    metadata={
                        "candidate_keys": [item.key.to_dict() for item in envelopes],
                        "selected_index": selected_index,
                        "judge_generation": result.to_dict(),
                    },
                ),
            )
        ]


class SelectFirstOperation:
    def run_many(
        self,
        envelopes: list[RecordEnvelope],
        stage: StageSpec,
        context: OperationContext,
    ) -> list[RecordEnvelope]:
        if not envelopes:
            return []
        selected = envelopes[0]
        variant = f"{selected.key.variant}.{stage.id}.selected"
        return [
            selected.with_payload(
                selected.payload,
                key=RecordKey(selected.key.source, selected.key.source_id, variant),
                event=StageEvent(
                    stage_id=stage.id,
                    operation=stage.operation,
                    generator=None,
                    variant=variant,
                    metadata={
                        "candidate_keys": [item.key.to_dict() for item in envelopes],
                        "selected_index": 0,
                    },
                ),
            )
        ]


def create_select_candidate_operation() -> SelectCandidateOperation:
    return SelectCandidateOperation()


def create_select_first_operation() -> SelectFirstOperation:
    return SelectFirstOperation()


__all__ = [
    "SelectCandidateOperation",
    "SelectFirstOperation",
    "create_select_candidate_operation",
    "create_select_first_operation",
]
