"""Built-in text trajectory operations."""

from ..registry import OPERATIONS
from .critique import (
    create_candidate_pair_operation,
    create_critique_operation,
    create_revise_operation,
)
from .filter import create_filter_operation
from .identity import create_identity_operation
from .replay import create_complete_operation, create_replay_operation
from .select import (
    create_select_candidate_operation,
    create_select_first_operation,
)
from .tool_loop import create_tool_loop_operation

OPERATIONS.register(
    "identity",
    create_identity_operation,
    version="1",
    capabilities={"deterministic_transform", "text_trajectory"},
)
OPERATIONS.register(
    "filter",
    create_filter_operation,
    version="1",
    capabilities={"deterministic_transform", "text_trajectory"},
)
OPERATIONS.register(
    "select_candidate",
    create_select_candidate_operation,
    version="1",
    capabilities={"chat", "variant_join", "text_trajectory"},
)
OPERATIONS.register(
    "select_first",
    create_select_first_operation,
    version="1",
    capabilities={"variant_join", "deterministic_transform", "text_trajectory"},
)
OPERATIONS.register(
    "replay_assistants",
    create_replay_operation,
    version="1",
    capabilities={
        "chat",
        "on_policy_history",
        "multiple_candidates",
        "tools",
        "recorded_tool_replay",
    },
)
OPERATIONS.register(
    "complete_prompt",
    create_complete_operation,
    version="1",
    capabilities={"chat", "on_policy_history", "multiple_candidates", "tools"},
)
OPERATIONS.register(
    "execute_tool_loop",
    create_tool_loop_operation,
    version="1",
    capabilities={
        "chat",
        "on_policy_history",
        "tools",
        "tool_execution",
        "text_trajectory",
    },
)
OPERATIONS.register(
    "critique",
    create_critique_operation,
    version="1",
    capabilities={"chat", "text_trajectory"},
)
OPERATIONS.register(
    "revise",
    create_revise_operation,
    version="1",
    capabilities={"chat", "text_trajectory"},
)
OPERATIONS.register(
    "candidate_pair",
    create_candidate_pair_operation,
    version="1",
    capabilities={"chat", "variant_join", "text_trajectory"},
)

__all__ = [
    "create_candidate_pair_operation",
    "create_complete_operation",
    "create_critique_operation",
    "create_identity_operation",
    "create_replay_operation",
    "create_revise_operation",
    "create_filter_operation",
    "create_select_candidate_operation",
    "create_select_first_operation",
    "create_tool_loop_operation",
]
