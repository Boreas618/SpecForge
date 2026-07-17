"""Built-in generation backends."""

from ..registry import BACKENDS
from .fake import create_fake_backend
from .openai_chat import create_openai_chat_backend
from .sglang_raw import create_sglang_raw_backend

BACKENDS.register(
    "fake",
    create_fake_backend,
    version="1",
    capabilities={
        "chat",
        "reasoning_split",
        "request_seed",
        "multiple_candidates",
        "tools",
        "text_trajectory",
    },
)
BACKENDS.register(
    "openai_chat",
    create_openai_chat_backend,
    version="1",
    capabilities={
        "chat",
        "reasoning_split",
        "request_seed",
        "tools",
        "multiple_candidates",
        "text_trajectory",
    },
)
BACKENDS.register(
    "sglang_raw",
    create_sglang_raw_backend,
    version="1",
    capabilities={
        "chat",
        "reasoning_split",
        "request_seed",
        "raw_token_ids",
        "tools",
        "multiple_candidates",
        "text_trajectory",
    },
)

__all__ = [
    "create_fake_backend",
    "create_openai_chat_backend",
    "create_sglang_raw_backend",
]
