# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 NVIDIA-specific operations package.

Exports the attn_res kernel for attention residual mixing.
"""

from .attn_res import attn_res

__all__ = ["attn_res"]
