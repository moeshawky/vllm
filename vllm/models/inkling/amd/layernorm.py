# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling RMSNorm (no bias, weight-scaled), backed by the vendored Triton kernel."""

from __future__ import annotations

import torch
from torch import nn

from .ops import rmsnorm


class InklingRMSNorm(nn.Module):
    """RMSNorm without bias, backed by vendored Triton kernel."""
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initializes RMSNorm weight parameter and epsilon value."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying RMS normalization using the vendored
        Triton kernel."""
        if x.numel() == 0:
            return x
        original_shape = x.shape
        x_2d = x.contiguous().view(-1, self.hidden_size)
        y = rmsnorm(x_2d, self.weight, self.variance_epsilon)
        return y.view(original_shape)
