# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common pooling types and parameters.

Defines the type aliases for pooling functions and the
PoolingParamsUpdate dataclass which carries parameters for
pooling operations, including whether token IDs are required.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

from vllm.pooling_params import PoolingParams

_T = TypeVar("_T", bound=torch.Tensor | list[torch.Tensor])

ProjectorFn = Callable[[torch.Tensor], torch.Tensor]
ClassifierFn = Callable[[torch.Tensor], torch.Tensor]
ActivationFn = Callable[[_T], _T]


@dataclass(frozen=True)
class PoolingParamsUpdate:
    """Parameters controlling pooling behavior.

    Carries configuration for pooling operations, including
    whether CPU prompt token IDs are required.

    Attributes:
        requires_token_ids: Set to True to enable CPU prompt
            token IDs for the pooler.
    """
    requires_token_ids: bool = False
    """Set this flag to enable CPU prompt token IDs for your pooler."""

    def __or__(self, other: "PoolingParamsUpdate") -> "PoolingParamsUpdate":
        """Combine two PoolingParamsUpdate instances with logical OR.

        Args:
            other: Another PoolingParamsUpdate to combine with.

        Returns:
            A new PoolingParamsUpdate with requires_token_ids
            set to True if either instance requires it.
        """
        return PoolingParamsUpdate(
            requires_token_ids=self.requires_token_ids or other.requires_token_ids,
        )

    def apply(self, params: PoolingParams) -> None:
        """Apply the pooling parameter update to a PoolingParams instance.

        Args:
            params: PoolingParams to update in place.
        """
        params.requires_token_ids = self.requires_token_ids


__all__ = ["ActivationFn", "ClassifierFn", "ProjectorFn", "PoolingParamsUpdate"]
