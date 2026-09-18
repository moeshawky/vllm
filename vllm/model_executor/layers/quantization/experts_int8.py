# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
)


class ExpertsInt8Config(QuantizationConfig):
    """Online int8 quantization for MoE expert weights.
    Linear layers are left unquantized.

    Backward-compatible config for ``--quantization experts_int8``.
    Prefer ``--quantization int8_per_channel``
    """

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        """Return the name value."""
        return "experts_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        """Return the supported_act_dtypes value."""
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        """Return the min_capability value."""
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        """Return the config_filenames value."""
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExpertsInt8Config":
        """Create instance from configuration dictionary."""
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        """Return the quant_method value."""
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            return Int8OnlineMoEMethod(layer=layer)
        return None
