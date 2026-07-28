"""Model package.

Legacy RSRNeT classes remain available from ``models.bert_model``.  Avoid
importing them eagerly so the independent SAVER path does not require legacy
TorchCRF/torchvision dependencies merely to import its routing components.
"""

from .saver import (
    ModernBertTextEncoder,
    RiskControlledCalibrator,
    SaverConfig,
    SaverForMNER,
    SaverForMRE,
    SaverOutput,
    Siglip2VisionEncoder,
    build_saver_model,
)

__all__ = [
    "ModernBertTextEncoder",
    "RiskControlledCalibrator",
    "SaverConfig",
    "SaverForMNER",
    "SaverForMRE",
    "SaverOutput",
    "Siglip2VisionEncoder",
    "build_saver_model",
]
