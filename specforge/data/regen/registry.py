"""Typed component registries and capability negotiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, TypeVar

from .errors import CapabilityError, ContractError

T = TypeVar("T")


@dataclass(frozen=True)
class Registration(Generic[T]):
    name: str
    factory: Callable[..., T]
    version: str
    capabilities: frozenset[str]


class ComponentRegistry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Registration[T]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        *,
        version: str = "1",
        capabilities: Iterable[str] = (),
    ) -> Registration[T]:
        if name in self._entries:
            raise ContractError(f"duplicate {self.kind} registration {name!r}")
        registration = Registration(
            name=name,
            factory=factory,
            version=version,
            capabilities=frozenset(capabilities),
        )
        self._entries[name] = registration
        return registration

    def resolve(
        self, name: str, *, required_capabilities: Iterable[str] = ()
    ) -> Registration[T]:
        try:
            registration = self._entries[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries)) or "<none>"
            raise CapabilityError(
                f"unknown {self.kind} {name!r}; available: {available}"
            ) from exc
        required = frozenset(required_capabilities)
        missing = required - registration.capabilities
        if missing:
            raise CapabilityError(
                f"{self.kind} {name!r} lacks capabilities: "
                f"{', '.join(sorted(missing))}"
            )
        return registration

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "version": entry.version,
                "capabilities": sorted(entry.capabilities),
            }
            for name, entry in sorted(self._entries.items())
        }

    def clear(self) -> None:
        """Clear registrations. Intended for isolated tests only."""

        self._entries.clear()


SOURCE_ADAPTERS: ComponentRegistry[Any] = ComponentRegistry("source adapter")
RECORD_ADAPTERS: ComponentRegistry[Any] = ComponentRegistry("record adapter")
OPERATIONS: ComponentRegistry[Any] = ComponentRegistry("operation")
BACKENDS: ComponentRegistry[Any] = ComponentRegistry("generation backend")
CODECS: ComponentRegistry[Any] = ComponentRegistry("conversation codec")
VALIDATORS: ComponentRegistry[Any] = ComponentRegistry("validator")
TOOL_ENVIRONMENTS: ComponentRegistry[Any] = ComponentRegistry("tool environment")

_BUILTINS_LOADED = False

PLUGIN_ENTRY_POINT_GROUP = "specforge.data.regen"
_LOADED_PLUGINS: dict[str, dict[str, Any]] = {}


def load_builtin_components() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Imports perform explicit registration; heavyweight optional dependencies stay
    # inside adapter factories, not at module import time.
    from . import backends as _backends  # noqa: F401
    from . import codecs as _codecs  # noqa: F401
    from . import environments as _environments  # noqa: F401
    from . import operations as _operations  # noqa: F401
    from . import records as _records  # noqa: F401
    from . import sources as _sources  # noqa: F401
    from . import validators as _validators  # noqa: F401

    _BUILTINS_LOADED = True


def _iter_entry_points(group: str):
    """Discovery seam; tests substitute synthetic entry points."""

    from importlib import metadata

    return tuple(metadata.entry_points(group=group))


def load_plugin_components(names: Iterable[str]) -> None:
    """Load explicitly named third-party component plugins.

    Discovery is opt-in per recipe: nothing is imported unless the recipe
    names the plugin, and the loaded distribution's identity becomes part of
    the plan snapshot. A plugin entry point resolves to a zero-argument
    callable that registers its components against these registries.
    """

    names = list(names)
    if not names:
        return
    load_builtin_components()
    available = {point.name: point for point in _iter_entry_points(
        PLUGIN_ENTRY_POINT_GROUP
    )}
    for name in names:
        if name in _LOADED_PLUGINS:
            continue
        point = available.get(name)
        if point is None:
            known = ", ".join(sorted(available)) or "<none>"
            raise ContractError(
                f"regeneration plugin {name!r} is not installed; found: {known}"
            )
        register = point.load()
        if not callable(register):
            raise ContractError(f"plugin {name!r} entry point is not callable")
        register()
        distribution = getattr(point, "dist", None)
        _LOADED_PLUGINS[name] = {
            "entry_point": f"{getattr(point, 'value', name)}",
            "distribution": getattr(distribution, "name", None),
            "version": getattr(distribution, "version", None),
        }


def plugin_snapshot() -> dict[str, dict[str, Any]]:
    return {name: dict(value) for name, value in sorted(_LOADED_PLUGINS.items())}


def registry_snapshot() -> dict[str, Any]:
    load_builtin_components()
    return {
        "sources": SOURCE_ADAPTERS.snapshot(),
        "records": RECORD_ADAPTERS.snapshot(),
        "operations": OPERATIONS.snapshot(),
        "backends": BACKENDS.snapshot(),
        "codecs": CODECS.snapshot(),
        "validators": VALIDATORS.snapshot(),
        "environments": TOOL_ENVIRONMENTS.snapshot(),
        "plugins": plugin_snapshot(),
    }


__all__ = [
    "BACKENDS",
    "CODECS",
    "OPERATIONS",
    "PLUGIN_ENTRY_POINT_GROUP",
    "RECORD_ADAPTERS",
    "SOURCE_ADAPTERS",
    "TOOL_ENVIRONMENTS",
    "VALIDATORS",
    "ComponentRegistry",
    "Registration",
    "load_builtin_components",
    "load_plugin_components",
    "plugin_snapshot",
    "registry_snapshot",
]
