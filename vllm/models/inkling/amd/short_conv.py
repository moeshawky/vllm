# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling short convolution: depthwise causal conv1d (+ residual) over a paged
sliding-window conv-state cache.

Each decoder layer owns one ``InklingConvState`` (``sconv_swa_attn.py``) holding the
manager-allocated paged cache for the layer's 4 sconv streams (K, V, attn-output,
mlp-output), packed head-major into one block. Each ``InklingShortConv`` is a
stateless weight + kernel launcher that, per forward (positions-addressed, the
same path for prefill / decode / mixed), inserts the current tokens' inputs
into their paged slot and convolves each token against the ``W`` taps ending
at its absolute position, reading pre-forward window positions out of the
paged cache via the block table.

Per-forward metadata (``block_table`` / ``slot_mapping`` / ``seq_idx`` /
``query_start``) is built once by ``InklingSconvMetadataBuilder`` and published under
the owner's prefix in the forward context; the absolute ``positions`` are
threaded in from the model. The insert + conv run in a single ``fused_sconv``
launch (same path for prefill / decode / mixed / spec). All inputs are
fixed-address persistent buffers and the grid is fixed, so the conv replays
correctly under eager, PIECEWISE, and FULL cudagraphs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.model_executor.utils import set_weight_attrs

from .ops import fused_sconv
from .sconv_swa_attn import InklingConvState, InklingSconvMetadata


class InklingShortConv(nn.Module):
    """Stateless Inkling short-convolution layer holding a depthwise conv
    weight and an owner reference to the InklingConvState that manages the
    paged conv-state cache. During forward, inserts current token inputs
    into the cache, convolves against the W-tap window ending at each token's
    absolute position, and returns the convolved result. Returns the input
    unchanged when no attention metadata or bound cache is available
    (profiling mode).
    """

    def __init__(
        self, dim: int, kernel_size: int, owner: InklingConvState, stream_idx: int
    ) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.owner = owner
        self.stream_idx = stream_idx
        self.tp_rank = get_tensor_model_parallel_rank()

        # Depthwise conv weight; checkpoint stores (dim, 1, W).
        self.weight = Parameter(torch.empty(dim, 1, kernel_size), requires_grad=False)
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        """Load the depthwise conv weight, sharding along the channel
        dimension for tensor parallelism when the loaded weight's channel
        count exceeds the per-rank parameter size.
        """
        if loaded_weight.shape[0] != param.shape[0]:
            shard = param.shape[0]
            loaded_weight = loaded_weight.narrow(0, self.tp_rank * shard, shard)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Insert current token inputs into the paged conv-state cache and
        convolve each token against the W-tap window ending at its absolute
        position. x has shape (num_tokens, dim) and positions has shape
        (num_tokens,) of absolute positions. Returns x unchanged if attention
        metadata or the bound cache is unavailable (e.g. during profiling
        before KV allocation). The block size used for indexing is taken from
        the owner's cache_block_size, which may be larger than the kernel
        window due to page-size unification.
        """
        # x: (num_tokens, dim); positions: (num_tokens,) absolute positions.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Memory-profiling / no metadata: identity (residual).
            return x
        m = attn_metadata.get(self.owner.prefix)
        if m is None:
            return x
        assert isinstance(m, InklingSconvMetadata)
        cache = self.owner.kv_cache
        if cache.numel() == 0:
            # Cache not yet bound (profiling before KV alloc): identity.
            return x

        off_s, ws = self.owner.stream_ranges[self.stream_idx]
        # The hybrid KV-cache planner can enlarge the logical conv block so
        # that its physical page size matches the attention caches (for
        # example, W=4 becomes 32 when --block-size=128). Metadata slot
        # mappings are built with that enlarged size, so index the bound cache
        # with its runtime token dimension rather than the kernel window size.
        block_size = self.owner.cache_block_size
        x = x.contiguous()
        weight = self.weight.squeeze(1)  # (dim, W)

        return fused_sconv(
            x,
            weight,
            cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            off_s,
            ws,
            block_size,
            activation=None,
            use_residual=True,
        )
