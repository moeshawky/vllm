# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence pooling methods for aggregating token-level representations.

Implements CLS, LAST, and MEAN pooling strategies for combining
token representations into a single sequence-level vector. Also
provides a factory function for constructing pooling methods
from configuration.
"""

from abc import ABC, abstractmethod
from collections.abc import Set
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.config.pooler import SequencePoolingType
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.pool.metadata import PoolingMetadata

SequencePoolingMethodOutput: TypeAlias = torch.Tensor | list[torch.Tensor]

_MEAN_POOL_ACCUMULATION_CHUNK_BYTES = 16 * 1024 * 1024  # 16MB


class SequencePoolingMethod(nn.Module, ABC):
    """Abstract base class for sequence pooling methods.

    Defines the interface for aggregating token-level hidden
    states into a single sequence-level representation.
    Subclasses must implement forward.
    """

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        raise NotImplementedError


class CLSPool(SequencePoolingMethod):
    """Pools the first token (CLS token) from each sequence.

    Selects the representation at the first token position
    as the pooled output. Does not support partial prefill.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        """Select the CLS token from each sequence.

        Args:
            hidden_states: Hidden states tensor of shape
                [batch_size, seq_len, hidden_size].
            pooling_metadata: Metadata containing pooling cursor
                information.

        Returns:
            Tensor of CLS token representations.

        Raises:
            RuntimeError: If partial prefill is detected.
        """
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        if pooling_cursor.is_partial_prefill():
            raise RuntimeError("partial prefill is not supported with CLS pooling")

        return hidden_states[pooling_cursor.first_token_indices_gpu]


class LastPool(SequencePoolingMethod):
    """Pools the last token from each sequence.

    Selects the representation at the final token position
    as the pooled output.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        """Select the last token from each sequence.

        Args:
            hidden_states: Hidden states tensor of shape
                [batch_size, seq_len, hidden_size].
            pooling_metadata: Metadata containing pooling cursor
                information.

        Returns:
            Tensor of last token representations.
        """
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        return hidden_states[pooling_cursor.last_token_indices_gpu]


class MeanPool(SequencePoolingMethod):
    """Pools by averaging all token representations in each sequence.

    Computes the mean of all token hidden states, optionally
    skipping partial prefill sequences. Uses chunked accumulation
    for memory-efficient processing of long sequences.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        """Compute the mean-pooled representation for each sequence.

        Args:
            hidden_states: Hidden states tensor of shape
                [batch_size, seq_len, hidden_size].
            pooling_metadata: Metadata containing pooling cursor
                and prompt length information.

        Returns:
            Tensor of mean-pooled representations.
        """
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        if pooling_cursor.is_partial_prefill():
            raise RuntimeError("partial prefill is not supported with MEAN pooling")

        prompt_lens_cpu = pooling_cursor.prompt_lens_cpu
        num_seqs = prompt_lens_cpu.numel()
        hidden_size = hidden_states.shape[-1]

        if num_seqs == 0:
            # early return for empty batch
            return hidden_states.new_empty((0, hidden_size), dtype=torch.float32)

        prompt_lens = async_tensor_h2d(
            prompt_lens_cpu, device=hidden_states.device, dtype=torch.int64
        )
        # eg. [2, 1, 3] -> [0, 0, 1, 2, 2, 2]
        segment_ids = torch.repeat_interleave(
            torch.arange(num_seqs, device=hidden_states.device, dtype=torch.long),
            prompt_lens,
            output_size=int(prompt_lens_cpu.sum()),
        )
        segment_sums = torch.zeros(
            (num_seqs, hidden_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )

        bytes_per_token = hidden_size * torch.finfo(torch.float32).bits // 8
        chunk_size = max(1, _MEAN_POOL_ACCUMULATION_CHUNK_BYTES // bytes_per_token)

        # iterate over the batch in chunks
        for start in range(0, hidden_states.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[0])
            # using index_add_ to accumulate for each segment
            segment_sums.index_add_(
                0,
                segment_ids[start:end],
                hidden_states[start:end].to(dtype=torch.float32),
            )

        return segment_sums / prompt_lens.unsqueeze(1)


def get_seq_pooling_method(
    pooling_type: SequencePoolingType | str,
) -> SequencePoolingMethod:
    """Construct a sequence pooling method from a pooling type name.

    Args:
        pooling_type: One of "CLS", "LAST", or "MEAN".

    Returns:
        An instance of the corresponding pooling method.

    Raises:
        NotImplementedError: If pooling_type is not recognized.
    """
    if pooling_type == "CLS":
        return CLSPool()
    if pooling_type == "LAST":
        return LastPool()
    if pooling_type == "MEAN":
        return MeanPool()

    raise NotImplementedError(f"Unknown sequence pooling type: {pooling_type!r}")
