"""Redacted structured logging and lightweight counters for regeneration.

Log records are key/value events on the ``specforge.data.regen`` logger.
Values are redacted before emission: anything resembling a URL, credential,
or filesystem-absolute private path is replaced, and long values are
truncated, so operational logs stay publishable.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from typing import Any

LOGGER = logging.getLogger("specforge.data.regen")

_URL = re.compile(r"\bhttps?://\S+", re.IGNORECASE)
_SECRETISH = re.compile(
    r"(authorization|api[-_]?key|bearer|token|secret|password)", re.IGNORECASE
)
_MAX_VALUE_LENGTH = 200


def redact_value(key: str, value: Any) -> Any:
    if _SECRETISH.search(key):
        return "[redacted]"
    if isinstance(value, str):
        value = _URL.sub("[endpoint]", value)
        if value.startswith("/"):
            value = f".../{value.rsplit('/', 1)[-1]}"
        if len(value) > _MAX_VALUE_LENGTH:
            value = value[:_MAX_VALUE_LENGTH] + "…"
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_value(key, str(value))


def log_event(event: str, **fields: Any) -> None:
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    safe = {key: redact_value(key, value) for key, value in fields.items()}
    LOGGER.info(
        "%s %s",
        event,
        " ".join(f"{key}={value}" for key, value in sorted(safe.items())),
    )


class Metrics:
    """Process-local thread-safe counters, exposed for tests and operators."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[str] = Counter()

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()


METRICS = Metrics()

__all__ = ["LOGGER", "METRICS", "Metrics", "log_event", "redact_value"]
