"""Built-in tool environments."""

from ..registry import TOOL_ENVIRONMENTS
from .fake import create_deterministic_environment

TOOL_ENVIRONMENTS.register(
    "deterministic",
    create_deterministic_environment,
    version="1",
    capabilities={"tool_execution", "no_side_effects", "text_trajectory"},
)

__all__ = ["create_deterministic_environment"]
