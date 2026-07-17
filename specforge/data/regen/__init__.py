"""First-class text and reasoning trajectory regeneration."""

from .contracts import (
    Finding,
    GenerationRequest,
    GenerationResult,
    RecordEnvelope,
    RecordKey,
    StageEvent,
    assert_text_trajectory_value,
    canonical_digest,
    canonical_json,
    contiguous_shard_bounds,
    deterministic_generation_seed,
)
from .errors import (
    ArtifactError,
    CapabilityError,
    ContractError,
    FailureCategory,
    GenerationError,
    PolicyReject,
    RegenerationError,
)
from .recipe import RegenerationRecipe, load_recipe


def plan_recipe(*args, **kwargs):
    from .planner import plan_recipe as implementation

    return implementation(*args, **kwargs)


def run_worker(*args, **kwargs):
    from .executor import run_worker as implementation

    return implementation(*args, **kwargs)


def finalize_artifact(*args, **kwargs):
    from .finalize import finalize_artifact as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "ArtifactError",
    "CapabilityError",
    "ContractError",
    "FailureCategory",
    "Finding",
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "PolicyReject",
    "RecordEnvelope",
    "RecordKey",
    "RegenerationRecipe",
    "RegenerationError",
    "StageEvent",
    "assert_text_trajectory_value",
    "canonical_digest",
    "canonical_json",
    "contiguous_shard_bounds",
    "deterministic_generation_seed",
    "load_recipe",
    "plan_recipe",
    "run_worker",
    "finalize_artifact",
]
