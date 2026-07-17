"""Verified text-dataset access for finalized regeneration artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .regen.artifact import verify_manifest
from .regen.contracts import RecordEnvelope, canonical_digest, iter_jsonl
from .regen.errors import ArtifactError


@dataclass(frozen=True)
class DatasetArtifact:
    root: Path
    manifest: Mapping[str, Any]

    @property
    def digest(self) -> str:
        return str(self.manifest["artifact_digest"])

    @property
    def recipe_digest(self) -> str:
        return str(self.manifest["recipe_digest"])

    def data_files(self) -> tuple[Path, ...]:
        files = []
        for entry in self.manifest.get("files", []):
            relative = entry.get("path")
            if (
                isinstance(relative, str)
                and relative.startswith("data/")
                and relative.endswith(".jsonl")
            ):
                files.append(self.root / relative)
        if not files:
            raise ArtifactError("finalized artifact manifest has no JSONL data files")
        return tuple(sorted(files))

    def iter_envelopes(self) -> Iterator[RecordEnvelope]:
        rows = 0
        for path in self.data_files():
            for _, value in iter_jsonl(path):
                rows += 1
                yield RecordEnvelope.from_dict(value)
        expected = self.manifest.get("counts", {}).get("output_rows")
        if rows != expected:
            raise ArtifactError(
                f"artifact data row count mismatch ({rows} != {expected})"
            )

    def iter_text_records(self) -> Iterator[dict[str, Any]]:
        """Yield ordinary text records while retaining the artifact externally."""

        for envelope in self.iter_envelopes():
            yield dict(envelope.payload)

    def iter_preference_records(self) -> Iterator[dict[str, Any]]:
        """Yield typed preference pairs from a candidate-pair artifact.

        The chosen trajectory is the row's canonical ``conversations``; the
        rejected trajectory and the judging provenance ride alongside it.
        Rows without a preference structure are rejected rather than
        silently coerced.
        """

        for envelope in self.iter_envelopes():
            payload = dict(envelope.payload)
            preference = payload.get("preference")
            rejected = payload.get("rejected_conversations")
            if not isinstance(preference, Mapping) or not isinstance(
                rejected, list
            ):
                raise ArtifactError(
                    "artifact row is not a preference record; use "
                    "iter_text_records for plain conversation artifacts"
                )
            yield {
                "id": payload.get("id"),
                "chosen_conversations": list(payload.get("conversations") or []),
                "rejected_conversations": rejected,
                "preference": dict(preference),
            }

    def provenance(
        self, *, preprocessing_cache_key: str | None = None
    ) -> dict[str, Any]:
        value = {
            "dataset_artifact_digest": self.digest,
            "dataset_recipe_digest": self.recipe_digest,
        }
        if preprocessing_cache_key is not None:
            value["dataset_preprocessing_cache_key"] = preprocessing_cache_key
        return value


def open_dataset_artifact(path_or_root: str | Path) -> DatasetArtifact:
    path = Path(path_or_root).expanduser().resolve()
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if manifest_path.name != "manifest.json":
        raise ArtifactError(
            "dataset_artifact must name a manifest.json or artifact root"
        )
    manifest = verify_manifest(manifest_path)
    return DatasetArtifact(root=manifest_path.parent, manifest=manifest)


def preprocessing_cache_key(
    artifact_digest: str, semantic_fields: Mapping[str, Any]
) -> str:
    """Content address a processed text view, including every semantic knob."""

    return canonical_digest(
        {
            "strategy": "dataset_artifact_text_preprocessing_v1",
            "artifact_digest": artifact_digest,
            "semantic_fields": dict(semantic_fields),
        }
    )


__all__ = [
    "DatasetArtifact",
    "open_dataset_artifact",
    "preprocessing_cache_key",
]
