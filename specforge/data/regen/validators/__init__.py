"""Built-in text trajectory validators."""

from ..registry import VALIDATORS
from .baseline import create_baseline_validator
from .loss_mask import create_loss_mask_validator

VALIDATORS.register(
    "baseline",
    create_baseline_validator,
    version="1",
    capabilities={"text_trajectory", "source_preservation"},
)
VALIDATORS.register(
    "specforge_loss_mask",
    create_loss_mask_validator,
    version="1",
    capabilities={"text_trajectory", "render_parity", "loss_mask_parity"},
)

__all__ = ["create_baseline_validator", "create_loss_mask_validator"]
