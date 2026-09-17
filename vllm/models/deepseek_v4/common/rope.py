# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV4 rotary embedding initialization."""

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


def build_deepseek_v4_rope(
    config,
    *,
    head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    compress_ratio: int,
) -> RotaryEmbedding:
    """Build a DeepseekV4 RotaryEmbedding from config, adjusting rope_theta
    and rope_type based on compress_ratio.
    """
    rope_parameters = config.rope_parameters
    rope_parameters["rope_theta"] = (  # compress_rope_theta is used when compress_ratio > 1 to adjust the rope theta for the compressed attention pattern.
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if rope_parameters["rope_type"] != "default":  # rope_type is set to deepseek_yarn or deepseek_llama_scaling based on apply_yarn_scaling; defaulting to yarn.
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    rope_parameters["mscale"] = 0  # Disable mscale for DeepseekV4, overriding the default YaRN scaling.
    rope_parameters["mscale_all_dim"] = 0  # Disable mscale
    rope_parameters["is_deepseek_v4"] = True  # signals to the rotary embedding layer that this is a DeepseekV4-specific configuration.
    rope_parameters["rope_dim"] = rope_head_dim  # restricts the RoPE to only the rope dimensions, not the full head.
    return get_rope(
        head_dim,
        max_position=max_position_embeddings,
        rope_parameters=rope_parameters,
        is_neox_style=False,
    )
