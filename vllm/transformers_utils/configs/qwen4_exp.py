# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp config shim.

vLLM model code imports the config classes from this module. The canonical
definitions live in ``transformers.models.qwen4_exp`` (shipped with
``transformers==5.16.1``). Re-export them here so the vLLM import path resolves
without duplicating config definitions.
"""

from transformers.models.qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
    Qwen4ExpVisionConfig,
)

__all__ = [
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
]
