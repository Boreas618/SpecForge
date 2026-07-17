"""Endpoint-pool health, circuit breaking, and bounded in-flight budgets.

A pool tracks one generator's runtime endpoints. Selection starts from the
request's seed-stable preference (endpoint placement never enters request
identity) and falls over to the next healthy endpoint when the preferred one
is saturated or its circuit is open. Consecutive transport failures open an
endpoint's circuit; after a cool-down one probe request is admitted and a
success closes the circuit again.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

from .contracts import GenerationRequest
from .errors import FailureCategory, GenerationError


class _EndpointState:
    __slots__ = ("inflight", "consecutive_failures", "open_until", "probing")

    def __init__(self) -> None:
        self.inflight = 0
        self.consecutive_failures = 0
        self.open_until = 0.0
        self.probing = False


class EndpointPool:
    def __init__(
        self,
        endpoints: tuple[str, ...] | list[str],
        *,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        max_inflight: int | None = None,
        acquire_timeout: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not endpoints:
            raise GenerationError(
                "endpoint pool requires at least one endpoint",
                category=FailureCategory.TRANSPORT_TERMINAL,
            )
        if failure_threshold <= 0 or cooldown_seconds < 0 or acquire_timeout <= 0:
            raise GenerationError(
                "invalid endpoint pool budgets",
                category=FailureCategory.TRANSPORT_TERMINAL,
            )
        if max_inflight is not None and max_inflight <= 0:
            raise GenerationError(
                "max_inflight must be positive when set",
                category=FailureCategory.TRANSPORT_TERMINAL,
            )
        self.endpoints = tuple(endpoints)
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = float(cooldown_seconds)
        self.max_inflight = max_inflight
        self.acquire_timeout = float(acquire_timeout)
        self.clock = clock
        self._states = {endpoint: _EndpointState() for endpoint in self.endpoints}
        self._condition = threading.Condition()

    def _preferred_index(self, request: GenerationRequest) -> int:
        selector = (
            request.seed
            if request.seed is not None
            else int(request.record_key.digest[:16], 16)
        )
        return selector % len(self.endpoints)

    def _eligible(self, endpoint: str, now: float) -> bool:
        state = self._states[endpoint]
        if self.max_inflight is not None and state.inflight >= self.max_inflight:
            return False
        if state.consecutive_failures < self.failure_threshold:
            return True
        if now < state.open_until:
            return False
        # Half-open: admit exactly one probe until it resolves.
        return not state.probing

    def _try_acquire(self, request: GenerationRequest) -> str | None:
        now = self.clock()
        start = self._preferred_index(request)
        for offset in range(len(self.endpoints)):
            endpoint = self.endpoints[(start + offset) % len(self.endpoints)]
            if self._eligible(endpoint, now):
                state = self._states[endpoint]
                state.inflight += 1
                if state.consecutive_failures >= self.failure_threshold:
                    state.probing = True
                return endpoint
        return None

    @contextmanager
    def lease(self, request: GenerationRequest) -> Iterator[str]:
        """Yield an endpoint; record the outcome and release the slot."""

        deadline = self.clock() + self.acquire_timeout
        with self._condition:
            endpoint = self._try_acquire(request)
            while endpoint is None:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise GenerationError(
                        "all inference endpoints are saturated or unhealthy",
                        category=FailureCategory.TRANSPORT_RETRYABLE,
                    )
                self._condition.wait(timeout=min(remaining, 1.0))
                endpoint = self._try_acquire(request)
        try:
            yield endpoint
        except GenerationError as exc:
            transport_failure = exc.category in {
                FailureCategory.TRANSPORT_RETRYABLE,
                FailureCategory.TRANSPORT_TERMINAL,
            }
            self._record(endpoint, success=not transport_failure)
            raise
        except BaseException:
            self._record(endpoint, success=False)
            raise
        else:
            self._record(endpoint, success=True)

    def _record(self, endpoint: str, *, success: bool) -> None:
        with self._condition:
            state = self._states[endpoint]
            state.inflight = max(0, state.inflight - 1)
            state.probing = False
            if success:
                state.consecutive_failures = 0
                state.open_until = 0.0
            else:
                state.consecutive_failures += 1
                if state.consecutive_failures >= self.failure_threshold:
                    state.open_until = self.clock() + self.cooldown_seconds
            self._condition.notify_all()

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Health view for metrics/logs; endpoint URLs stay runtime-only."""

        with self._condition:
            now = self.clock()
            return {
                f"endpoint-{index}": {
                    "inflight": state.inflight,
                    "consecutive_failures": state.consecutive_failures,
                    "circuit_open": bool(
                        state.consecutive_failures >= self.failure_threshold
                        and now < state.open_until
                    ),
                }
                for index, (endpoint, state) in enumerate(
                    sorted(self._states.items())
                )
            }


def pool_from_runtime(runtime: dict, endpoints: tuple[str, ...]) -> EndpointPool:
    """Return the shared pool from runtime, creating and caching one if needed."""

    pool = runtime.get("pool")
    if isinstance(pool, EndpointPool):
        return pool
    pool = EndpointPool(
        endpoints,
        failure_threshold=int(runtime.get("failure_threshold", 5)),
        cooldown_seconds=float(runtime.get("cooldown_seconds", 30.0)),
        max_inflight=runtime.get("max_inflight"),
        acquire_timeout=float(runtime.get("acquire_timeout", 60.0)),
    )
    return runtime.setdefault("pool", pool)


__all__ = ["EndpointPool", "pool_from_runtime"]
