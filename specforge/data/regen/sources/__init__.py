"""Built-in source adapters."""

from ..registry import SOURCE_ADAPTERS
from .huggingface import create_huggingface_source
from .jsonl import create_jsonl_source

SOURCE_ADAPTERS.register(
    "jsonl",
    create_jsonl_source,
    version="1",
    capabilities={"streaming", "content_fingerprint", "text_trajectory"},
)
SOURCE_ADAPTERS.register(
    "huggingface",
    create_huggingface_source,
    version="1",
    capabilities={"pinned_revision", "text_trajectory"},
)

__all__ = ["create_huggingface_source", "create_jsonl_source"]
