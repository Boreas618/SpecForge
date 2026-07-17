"""Stable error types and failure categories for text regeneration."""

from __future__ import annotations

import os
import re
from enum import Enum


class FailureCategory(str, Enum):
    SOURCE_INVALID = "source_invalid"
    CAPABILITY_ERROR = "capability_error"
    TRANSPORT_RETRYABLE = "transport_retryable"
    TRANSPORT_TERMINAL = "transport_terminal"
    GENERATION_INVALID = "generation_invalid"
    EXACT_REBUILD = "exact_rebuild"
    POLICY_REJECT = "policy_reject"
    VALIDATION_FAILED = "validation_failed"
    INTERNAL_ERROR = "internal_error"


class RegenerationError(RuntimeError):
    """Base exception for regeneration failures."""


class ContractError(RegenerationError, ValueError):
    """A value violates a public regeneration contract."""


class CapabilityError(ContractError):
    """Selected registered components cannot satisfy a recipe stage."""


class ArtifactError(RegenerationError):
    """Artifact state, identity, coverage, or integrity is invalid."""


class GenerationError(RegenerationError):
    """A backend or codec failed to produce a valid semantic result."""

    def __init__(
        self,
        message: str,
        *,
        category: FailureCategory = FailureCategory.GENERATION_INVALID,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.request_digest: str | None = None
        self.request_seed: int | None = None


class PolicyReject(RegenerationError):
    """A valid result was deliberately rejected by a declared recipe policy."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.category = FailureCategory.POLICY_REJECT


def safe_diagnostic(exc: BaseException, *, limit: int = 1000) -> str:
    """Return a bounded single-line diagnostic suitable for an artifact journal."""

    text = " ".join(str(exc).split()) or type(exc).__name__
    text = re.sub(r"https?://\S+", "<endpoint>", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?i)(authorization|api[_-]?key|password|secret|bearer)(\s*[:=]?\s*)\S+",
        r"\1\2<redacted>",
        text,
    )
    for private_root in {os.getcwd(), os.path.expanduser("~")}:
        if private_root and private_root != "/":
            text = text.replace(private_root, "<path>")
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


__all__ = [
    "ArtifactError",
    "CapabilityError",
    "ContractError",
    "FailureCategory",
    "GenerationError",
    "PolicyReject",
    "RegenerationError",
    "safe_diagnostic",
]
