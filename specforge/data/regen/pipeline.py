"""Model- and dataset-agnostic execution of ordered regeneration stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import (
    GenerationRequest,
    GenerationResult,
    RecordEnvelope,
    canonical_digest,
    deterministic_generation_seed,
)
from .errors import CapabilityError, GenerationError
from .recipe import RegenerationRecipe, StageSpec
from .registry import BACKENDS, CODECS, OPERATIONS, load_builtin_components


@dataclass(frozen=True)
class _GeneratorRuntime:
    backend: Any
    codec: Any
    backend_name: str
    model: str


class Pipeline:
    """Execute a recipe's linear stages for one source envelope.

    Components are resolved entirely through registries. Runtime endpoint and
    credential values are passed separately and never enter requests, stage
    history, recipes, or artifacts.
    """

    def __init__(
        self,
        recipe: RegenerationRecipe,
        runtime: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        load_builtin_components()
        self.recipe = recipe
        runtime = runtime or {}
        self._generators: dict[str, _GeneratorRuntime] = {}
        for name, spec in recipe.generators.items():
            backend_registration = BACKENDS.resolve(spec.backend)
            codec_registration = CODECS.resolve(spec.codec)
            backend_runtime = runtime.get(name, {})
            backend = backend_registration.factory(spec, backend_runtime)
            codec = codec_registration.factory(spec, backend_runtime.get("codec", {}))
            self._generators[name] = _GeneratorRuntime(
                backend=backend,
                codec=codec,
                backend_name=spec.backend,
                model=spec.model,
            )

    def generate(
        self,
        envelope: RecordEnvelope,
        stage: StageSpec,
        *,
        messages: list[dict],
        tools: list[dict],
        variant: str,
        generation_ordinal: int,
    ) -> GenerationResult:
        if stage.generator is None:
            raise CapabilityError(f"stage {stage.id!r} has no generator")
        try:
            generator = self._generators[stage.generator]
            spec = self.recipe.generators[stage.generator]
        except KeyError as exc:
            raise CapabilityError(
                f"stage {stage.id!r} references unavailable generator "
                f"{stage.generator!r}"
            ) from exc
        seed = deterministic_generation_seed(
            self.recipe.seed,
            envelope.key,
            stage.id,
            variant,
            generation_ordinal,
        )
        sampling = spec.sampling.model_dump(mode="json")
        request = GenerationRequest(
            record_key=envelope.key,
            stage_id=stage.id,
            variant=variant,
            generation_ordinal=generation_ordinal,
            messages=tuple(deepcopy(messages)),
            tools=tuple(deepcopy(tools)),
            sampling=sampling,
            seed=seed,
        )
        prepare_request = getattr(generator.codec, "prepare_request", None)
        if prepare_request is not None:
            request = prepare_request(request)
        try:
            response = generator.backend.generate(request)
            return generator.codec.decode(
                response,
                request,
                backend=generator.backend_name,
                model=generator.model,
            )
        except GenerationError as exc:
            exc.request_digest = canonical_digest(request.to_dict())
            exc.request_seed = request.seed
            raise

    def run(self, envelope: RecordEnvelope) -> list[RecordEnvelope]:
        current = [envelope]
        for stage in self.recipe.workflow:
            operation = OPERATIONS.resolve(stage.operation).factory()
            run_many = getattr(operation, "run_many", None)
            if run_many is not None:
                following = list(run_many(current, stage, self))
            else:
                following = []
                for item in current:
                    following.extend(operation.run(item, stage, self))
            current = following
            if not current:
                break
        return current


__all__ = ["Pipeline"]
