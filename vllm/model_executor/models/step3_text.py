# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Jurassic model."""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.step3_vl import Step3TextConfig

from .interfaces import SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class FusedMoEBlock(nn.Module):
    """Fused MoE layer with gate routing and expert computation
    using FusedMoEFactory."""
    def __init__(
        self,
        config: ModelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        """Initialize FusedMoEBlock with FusedMoEFactory experts and a
        ReplicatedLinear gate; validates tensor parallel size against
        expert count."""
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        self.experts = FusedMoEFactory(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply gate routing and expert computation to produce final
        hidden states."""
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        return final_hidden_states.view(orig_shape)


class Step3TextMLP(nn.Module):
    """Gate-up and down projection linear layers with SiluAndMul
    activation for Step3 text decoder."""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize Step3TextMLP with MergedColumnParallelLinear and
        RowParallelLinear; validates silu activation."""
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply gate_up_proj, activation, and down_proj to produce
        output."""
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output


class Step3TextAttention(nn.Module):
    """Attention with ReplicatedLinear QKV projection, inter_norm,
    wq projection, and rotary embeddings."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
        rope_parameters: dict[str, Any],
        share_q_dim: int | None = None,
        max_position_embedding: int = 8192,
        head_dim: int = 256,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        """Initialize Step3TextAttention with ReplicatedLinear QKV,
        RowParallelLinear output, inter_norm, and wq projection layers."""
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        if num_kv_heads != 1:
            raise ValueError(
                f"Step3TextAttention num_kv_heads must be 1, but got {num_kv_heads}."
            )
        self.num_kv_heads = num_kv_heads

        self.head_dim = head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.q_size = share_q_dim if share_q_dim else self.head_dim

        self.qkv_proj = ReplicatedLinear(
            hidden_size,
            self.q_size + self.kv_size * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.inter_norm = RMSNorm(self.q_size, eps=norm_eps)
        self.wq = ColumnParallelLinear(
            self.q_size,
            self.head_dim * self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embedding,
            rope_parameters=rope_parameters,
        )
        scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scaling,
            self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute QKV, apply inter_norm and wq, rotary embeddings,
        and run attention."""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.inter_norm(q)
        q = self.wq(q)[0]
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        residual, _ = self.o_proj(attn_output)
        return residual


class Step3TextDecoderLayer(nn.Module):
    """Single decoder layer with attention, optional MoE, and two
    RMSNorm residual connections."""
    def __init__(
        self,
        config: Step3TextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize Step3TextDecoderLayer with Step3TextAttention,
        optional FusedMoEBlock, and two RMSNorm layers; detects MoE
        layers from config."""
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Step3TextAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            norm_eps=config.rms_norm_eps,
            max_position_embedding=config.max_position_embedding,
            head_dim=config.head_dim,
            share_q_dim=config.share_q_dim,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
        )

        layer_idx = int(prefix.split("layers.")[1].split(".")[0])
        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layers_idx = [int(i) for i in moe_layers_enum.strip().split(",")]
        else:
            # Default to 1dense.
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]

        if layer_idx in moe_layers_idx:
            self.moe = FusedMoEBlock(
                config=config, quant_config=quant_config, prefix=f"{prefix}.moe"
            )
            self.share_expert = Step3TextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.share_expert_dim,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=f"{prefix}.share_expert",
            )
            self.use_moe = True
        else:
            self.mlp = Step3TextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.use_moe = False
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply attention, then optionally MoE with share expert,
        and MLP with residual connections."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.use_moe:
            share_output = self.share_expert(hidden_states)
            moe_output = self.moe(hidden_states)
            hidden_states = share_output + moe_output
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class Step3TextModel(nn.Module):
    """Full text transformer with embedding, decoder layers, and
    final norm; supports pipeline parallelism and MoE layers."""
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize Step3TextModel with embedding tokens, decoder layers
        via make_layers, and final norm; creates empty intermediate tensors
        factory."""
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.config = config

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step3TextDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map input IDs to embeddings using the vocabulary parallel
        embedding layer."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process hidden states through decoder layers with pipeline
        parallelism support."""
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HF weights with complex mapping for QKV split, MoE stacked
        params, and expert params."""
        qkv_params_mapping = [
            # (param_name, shard_name, relative_start_idx, relative_end_idx)
            (
                ".qkv_proj",
                ".q_proj",
                0,
                self.config.share_q_dim
                / (self.config.share_q_dim + self.config.head_dim * 2),
            ),
            (
                ".qkv_proj",
                ".k_proj",
                self.config.share_q_dim
                / (self.config.share_q_dim + self.config.head_dim * 2),
                (self.config.share_q_dim + self.config.head_dim)
                / (self.config.share_q_dim + self.config.head_dim * 2),
            ),
            (
                ".qkv_proj",
                ".v_proj",
                (self.config.share_q_dim + self.config.head_dim)
                / (self.config.share_q_dim + self.config.head_dim * 2),
                (self.config.share_q_dim + self.config.head_dim * 2)
                / (self.config.share_q_dim + self.config.head_dim * 2),
            ),
        ]
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )

        expert_params_mapping = [
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight",
                ".moe.gate_proj.weight",
                "w1",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight",
                ".moe.up_proj.weight",
                "w3",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w2_weight",
                ".moe.down_proj.weight",
                "w2",
            ),
        ]

        disable_moe_stacked_params = [data[1] for data in expert_params_mapping]

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if any(
                    disable_moe_stacked_param in name
                    for disable_moe_stacked_param in disable_moe_stacked_params
                ):
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    for expert_id in range(loaded_weight.shape[0]):
                        loaded_weight_expert = loaded_weight[expert_id]
                        weight_loader(
                            param,
                            loaded_weight_expert,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(name)
                    break
                else:
                    for (
                        param_name,
                        weight_name,
                        start_idx,
                        end_idx,
                    ) in qkv_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        dim = param.shape[param.output_dim]
                        begin_idx = int(start_idx * dim)
                        end_idx = int(end_idx * dim)
                        param_slice = param.narrow(
                            param.output_dim, begin_idx, end_idx - begin_idx
                        )
                        param_slice.copy_(loaded_weight)
                        loaded_params.add(name)
                        break
                    else:
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
        return loaded_params


class Step3TextForCausalLM(nn.Module, SupportsPP):
    """Causal language model head with PP support; delegates to
    Step3TextModel for hidden states."""
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        """Initialize Step3TextForCausalLM with Step3TextModel, ParallelLMHead,
        and LogitsProcessor."""
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.vllm_config = vllm_config

        self.model = Step3TextModel(vllm_config=vllm_config, prefix=prefix)

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Delegate to the inner model's embed_input_ids method."""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        """Pass inputs through the inner model and return hidden states."""
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states using the logits processor."""
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HF weights using AutoWeightsLoader."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
