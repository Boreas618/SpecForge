"""SpecForge public API.

The historical package imported every model and training dependency eagerly. Keep the
same direct attributes while resolving them lazily so lightweight subpackages such as
``specforge.data.regen`` can be used without optional distributed/model dependencies.
"""

from __future__ import annotations

from importlib import import_module

_CORE_EXPORTS = {
    "OnlineDFlashModel",
    "OnlineDominoModel",
    "OnlineDSparkModel",
    "OnlineEagle3Model",
    "OnlinePEagleModel",
    "QwenVLOnlineEagle3Model",
}
_MODELING_EXPORTS = {
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
    "CustomEagle3TargetEngine",
    "CustomEagle3TargetModel",
    "Eagle3TargetEngine",
    "HFEagle3TargetEngine",
    "HFEagle3TargetModel",
    "LlamaForCausalLMEagle3",
    "PEagleDraftModel",
    "SGLangEagle3TargetEngine",
    "SGLangEagle3TargetModel",
    "TargetEngine",
    "get_eagle3_target_model",
    "get_target_engine",
}

# Preserve the existing star-import surface. Direct historical attributes remain
# available through __getattr__ even though they were not listed in __all__ before.
__all__ = ["modeling", "core"]


def __getattr__(name: str):
    if name in {"core", "modeling"}:
        value = import_module(f".{name}", __name__)
    elif name in _CORE_EXPORTS:
        value = getattr(import_module(".core", __name__), name)
    elif name in _MODELING_EXPORTS:
        value = getattr(import_module(".modeling", __name__), name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _CORE_EXPORTS | _MODELING_EXPORTS)
