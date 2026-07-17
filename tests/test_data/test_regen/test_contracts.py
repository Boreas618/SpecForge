from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest

from specforge.data.regen.contracts import (
    GenerationResult,
    RecordEnvelope,
    RecordKey,
    StageEvent,
    assert_text_trajectory_value,
    canonical_digest,
    contiguous_shard_bounds,
    deterministic_generation_seed,
    type_preserving_id,
)
from specforge.data.regen.errors import ContractError


def test_ids_are_type_preserving_and_seeds_are_stage_stable():
    assert type_preserving_id(1) != type_preserving_id("1")
    key = RecordKey("source", 1)
    seed = deterministic_generation_seed(17, key, "answer", "candidate-0", 1)
    assert seed == deterministic_generation_seed(17, key, "answer", "candidate-0", 1)
    assert seed != deterministic_generation_seed(17, key, "answer", "candidate-1", 1)
    assert seed != deterministic_generation_seed(17, key, "judge", "candidate-0", 1)


@pytest.mark.parametrize(
    "field",
    [
        "hidden_states",
        "hidden-state",
        "logits",
        "embedding",
        "kv_cache",
        "past_key_values",
        "tensor_features",
    ],
)
def test_tensor_fields_are_forbidden_at_every_payload_boundary(field):
    with pytest.raises(ContractError, match="tensor/hidden-state"):
        assert_text_trajectory_value({field: [1, 2, 3]})
    with pytest.raises(ContractError, match="tensor/hidden-state"):
        RecordEnvelope(
            key=RecordKey("source", "row"),
            input_position=0,
            payload={field: [1, 2, 3]},
            source_fingerprint="abc",
            source_payload_digest="abc",
            source_preserved_digest="abc",
        )


def test_arbitrary_python_and_nonfinite_values_are_forbidden():
    with pytest.raises(ContractError, match="not JSON"):
        assert_text_trajectory_value({"content": object()})
    with pytest.raises(ContractError, match="non-finite"):
        assert_text_trajectory_value({"score": float("nan")})


def test_envelope_preserves_source_digest_across_derived_stages():
    payload = {
        "id": "row",
        "conversations": [{"role": "user", "content": "hello"}],
    }
    envelope = RecordEnvelope(
        key=RecordKey("source", "row"),
        input_position=4,
        payload=payload,
        source_fingerprint="sha256:source",
        source_payload_digest=canonical_digest(payload),
        source_preserved_digest=canonical_digest(payload["conversations"]),
    )
    derived = envelope.with_payload(
        {
            **payload,
            "conversations": [
                *payload["conversations"],
                {
                    "role": "assistant",
                    "reasoning_content": "reason",
                    "content": "answer",
                },
            ],
        },
        event=StageEvent(
            stage_id="answer",
            operation="replay_assistants",
            generator="teacher",
            variant="base",
            generation_count=1,
        ),
    )
    assert derived.source_payload_digest == envelope.source_payload_digest
    assert derived.stage_history[0].stage_id == "answer"
    assert RecordEnvelope.from_dict(derived.to_dict()) == derived


def test_generation_result_never_serializes_raw_token_ids():
    ids = (10, 20, 30)
    digest = canonical_digest(list(ids))
    result = GenerationResult(
        message={
            "role": "assistant",
            "reasoning_content": "reason",
            "content": "answer",
        },
        finish_reason="stop",
        backend="raw",
        model="model",
        codec="typed",
        exactness="token_exact",
        raw_token_digest=digest,
        raw_token_count=len(ids),
        raw_token_ids=ids,
    )
    serialized = result.to_dict()
    assert "raw_token_ids" not in serialized
    assert serialized["raw_token_digest"] == digest


def test_generation_result_rejects_wrong_raw_evidence():
    with pytest.raises(ContractError, match="raw_token_digest"):
        GenerationResult(
            message={"role": "assistant", "content": "answer"},
            finish_reason="stop",
            backend="raw",
            model="model",
            codec="typed",
            raw_token_digest="wrong",
            raw_token_ids=(1, 2),
        )


def test_contiguous_shards_are_complete_and_balanced():
    bounds = [contiguous_shard_bounds(13, 5, index) for index in range(5)]
    assert bounds == [(0, 3), (3, 6), (6, 9), (9, 11), (11, 13)]
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))


def test_importing_regen_does_not_import_torch_or_transformers():
    script = """
import json, sys
import specforge.data.regen
print(json.dumps({name: name in sys.modules for name in ('torch', 'transformers')}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"torch": False, "transformers": False}


def test_data_package_has_no_eager_aggregate_imports():
    # Concrete APIs live in their owning modules; the package root must stay
    # empty so artifact inspection and regeneration never load model or
    # training dependencies through it.
    script = """
import json, sys
import specforge.data
import specforge.data.artifact
print(json.dumps({name: name in sys.modules for name in ('torch', 'transformers')}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"torch": False, "transformers": False}
