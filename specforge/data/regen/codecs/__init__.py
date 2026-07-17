"""Built-in serving-format codecs."""

from ..registry import CODECS
from .chat_template import create_chat_template_codec
from .exact_raw import create_exact_raw_codec
from .kimi_k3 import create_kimi_k3_codec
from .split_reasoning import create_split_reasoning_codec
from .structured_chat import create_structured_chat_codec

CODECS.register(
    "structured_chat",
    create_structured_chat_codec,
    version="1",
    capabilities={"chat", "reasoning_split", "tools", "text_trajectory"},
)
CODECS.register(
    "chat_template",
    create_chat_template_codec,
    version="1",
    capabilities={"chat", "reasoning_split", "tools", "text_trajectory"},
)
CODECS.register(
    "split_reasoning",
    create_split_reasoning_codec,
    version="1",
    capabilities={"chat", "reasoning_split", "tools", "text_trajectory"},
)
CODECS.register(
    "kimi_k3",
    create_kimi_k3_codec,
    version="1",
    capabilities={"chat", "reasoning_split", "text_trajectory"},
)
CODECS.register(
    "exact_raw_tokens",
    create_exact_raw_codec,
    version="1",
    capabilities={
        "chat",
        "reasoning_split",
        "raw_token_ids",
        "exact_rebuild",
        "tools",
        "text_trajectory",
    },
)

__all__ = [
    "create_chat_template_codec",
    "create_exact_raw_codec",
    "create_kimi_k3_codec",
    "create_split_reasoning_codec",
    "create_structured_chat_codec",
]
