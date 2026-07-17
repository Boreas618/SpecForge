"""SpecForge data APIs.

Imports are lazy so text-regeneration planning, validation, and artifact inspection
do not load torch, transformers, or distributed training dependencies.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DatasetArtifact": (".artifact", "DatasetArtifact"),
    "open_dataset_artifact": (".artifact", "open_dataset_artifact"),
    "preprocessing_cache_key": (".artifact", "preprocessing_cache_key"),
    "build_eagle3_dataset": (".preprocessing", "build_eagle3_dataset"),
    "build_offline_eagle3_dataset": (
        ".preprocessing",
        "build_offline_eagle3_dataset",
    ),
    "generate_vocab_mapping_file": (
        ".preprocessing",
        "generate_vocab_mapping_file",
    ),
    "preprocess_conversations": (".preprocessing", "preprocess_conversations"),
    "prepare_dp_dataloaders": (".utils", "prepare_dp_dataloaders"),
    "ChatTemplate": (".template", "ChatTemplate"),
}

__all__ = [
    "DatasetArtifact",
    "open_dataset_artifact",
    "preprocessing_cache_key",
    "build_eagle3_dataset",
    "build_offline_eagle3_dataset",
    "generate_vocab_mapping_file",
    "preprocess_conversations",
    "prepare_dp_dataloaders",
    "ChatTemplate",
]


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
