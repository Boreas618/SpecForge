"""Built-in record schema adapters."""

from ..registry import RECORD_ADAPTERS
from .messages import normalize_openai_record, normalize_sharegpt_record

RECORD_ADAPTERS.register(
    "openai_messages",
    normalize_openai_record,
    version="1",
    capabilities={"chat", "reasoning_split", "tools", "text_trajectory"},
)
RECORD_ADAPTERS.register(
    "sharegpt",
    normalize_sharegpt_record,
    version="1",
    capabilities={"chat", "reasoning_split", "text_trajectory"},
)

__all__ = ["normalize_openai_record", "normalize_sharegpt_record"]
