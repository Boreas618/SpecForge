"""Pinned Hugging Face dataset source adapter.

The optional ``datasets`` dependency is imported only when this adapter is created.
"""

from __future__ import annotations

from typing import Any

from ..contracts import canonical_digest
from ..errors import ContractError
from ..recipe import SourceSpec
from .base import SourceIdentity, SourceRow


class HuggingFaceSource:
    def __init__(self, source_name: str, spec: SourceSpec) -> None:
        dataset_id = spec.config.get("id")
        revision = spec.config.get("revision")
        split = spec.config.get("split", "train")
        name = spec.config.get("name")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ContractError(
                f"source {source_name!r}: huggingface config.id is required"
            )
        if not isinstance(revision, str) or not revision:
            raise ContractError(
                f"source {source_name!r}: huggingface config.revision must be pinned"
            )
        if spec.config.get("trust_remote_code", False):
            raise ContractError(
                "Hugging Face remote code is disabled for regeneration sources"
            )

        from datasets import load_dataset

        self.dataset = load_dataset(
            dataset_id,
            name=name,
            split=split,
            revision=revision,
            trust_remote_code=False,
        )
        fingerprint_payload = {
            "adapter": "huggingface",
            "id": dataset_id,
            "name": name,
            "split": split,
            "revision": revision,
            "dataset_fingerprint": getattr(self.dataset, "_fingerprint", None),
            "rows": len(self.dataset),
        }
        self._identity = SourceIdentity(
            adapter="huggingface",
            locator=f"{dataset_id}:{name or ''}:{split}",
            fingerprint=f"sha256:{canonical_digest(fingerprint_payload)}",
            rows=len(self.dataset),
            revision=revision,
        )

    @property
    def identity(self) -> SourceIdentity:
        return self._identity

    def iter_rows(self):
        for position, value in enumerate(self.dataset):
            if not isinstance(value, dict):
                raise ContractError(
                    f"Hugging Face row {position} is {type(value).__name__}, not object"
                )
            yield SourceRow(position=position, value=value)


def create_huggingface_source(source_name: str, spec: SourceSpec) -> HuggingFaceSource:
    return HuggingFaceSource(source_name, spec)


__all__ = ["HuggingFaceSource", "create_huggingface_source"]
