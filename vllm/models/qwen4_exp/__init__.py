# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp model package."""

from typing import TYPE_CHECKING, Any

from .common.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

if TYPE_CHECKING:
    from .nvidia.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )
    from .nvidia.mtp import Qwen4ExpMTP


def __getattr__(name: str) -> Any:
    """Lazy platform dispatch for Qwen4Exp model classes.

    Resolves Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration, and
    Qwen4ExpMTP to the correct platform-specific implementation at first
    attribute access. XPU and default paths use the NVIDIA implementation.
    TPU uses the common device-agnostic module (with optional MTP). ROCm
    uses the AMD implementation. This avoids importing platform-specific
    dependencies until the class is actually requested.
    """
    if name in {
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
    }:
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            # XPU dispatches to nvidia path (compatible via torch.jit.trace)
            from .nvidia.model import (
                Qwen4ExpForCausalLM,
                Qwen4ExpForConditionalGeneration,
            )
            from .nvidia.mtp import Qwen4ExpMTP
        elif current_platform.is_tpu():
            # TPU/JAX dispatches to common device-agnostic model
            from .common.model import (
                Qwen4ExpForCausalLM,
                Qwen4ExpForConditionalGeneration,
            )
            try:
                from .common.mtp import Qwen4ExpMTP
            except ImportError:
                # MTP device-agnostic module absent in this checkout; disabled on TPU/JAX
                Qwen4ExpMTP = None
        elif current_platform.is_rocm():
            from .amd.model import (
                Qwen4ExpForCausalLM,
                Qwen4ExpForConditionalGeneration,
            )
            from .amd.mtp import Qwen4ExpMTP
        else:
            from .nvidia.model import (
                Qwen4ExpForCausalLM,
                Qwen4ExpForConditionalGeneration,
            )
            from .nvidia.mtp import Qwen4ExpMTP

        return {
            "Qwen4ExpForCausalLM": Qwen4ExpForCausalLM,
            "Qwen4ExpForConditionalGeneration": (Qwen4ExpForConditionalGeneration),
            "Qwen4ExpMTP": Qwen4ExpMTP,
        }[name]
    raise AttributeError(name)


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
]
