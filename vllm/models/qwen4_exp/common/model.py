# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-agnostic Qwen4Exp model (TPU/JAX compatible).

This module provides a device-agnostic implementation of the Qwen4Exp model
that works across CUDA, ROCm, and TPU/JAX platforms. Unlike the nvidia/amd
device-specific implementations, this version uses lazy torch.cuda imports
and conditional Mamba backend selection.
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from ..config import Qwen4ExpConfig
from .hyperconnection import GatedResidual, HyperConnectionConfig
try:
    from ..nvidia.mtp import Qwen4ExpMTP
except ImportError:
    # Device MTP module (nvidia/amd) is incomplete in this checkout
    # (missing low_latency_gemm / model_state). MTP is disabled for this
    # test, so the class is only required when mtp_num_layers > 0.
    Qwen4ExpMTP = None

try:
    from .ple_layer import Qwen4ExpPLELayer
    from .qsa import Qwen4ExpQSAAttention
except ImportError:
    Qwen4ExpPLELayer = None
    Qwen4ExpQSAAttention = None


def without_modelopt_fp4(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Return ``None`` for weights excluded from Qwen4Exp ModelOpt-FP4."""

    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config


def _remap_qsa_cache_scale_name(
    name: str,
    qsa_layer_ids: frozenset[int],
) -> str:
    """Map serialized main-cache scales onto the merged QSA owner.

    Regular attention keeps cache scales below its ``attn`` child. QSA owns
    that cache directly, so only QSA layers need the final path component
    moved to the owner's persistent ``_k_scale``/``_v_scale`` buffers.
    """

    scale_suffixes = {
        "k_proj.k_scale": "_k_scale",
        "k_proj.output_scale": "_k_scale",
        "attn.k_scale": "_k_scale",
        "attn._k_scale": "_k_scale",
        "k_scale": "_k_scale",
        "_k_scale": "_k_scale",
        "v_proj.v_scale": "_v_scale",
        "v_proj.output_scale": "_v_scale",
        "attn.v_scale": "_v_scale",
        "attn._v_scale": "_v_scale",
        "v_scale": "_v_scale",
        "_v_scale": "_v_scale",
    }
    for layer_id in qsa_layer_ids:
        marker = f"layers.{layer_id}.self_attn."
        marker_start = name.find(marker)
        if marker_start < 0 or (marker_start > 0 and name[marker_start - 1] != "."):
            continue
        suffix = name[marker_start + len(marker) :]
        mapped_suffix = scale_suffixes.get(suffix)
        if mapped_suffix is not None:
            return f"{name[: marker_start + len(marker)]}{mapped_suffix}"
    return name


_QWEN4_EXP_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]

# The checkpoint keeps down and injection projections separate; runtime packs
# them into adjacent logical shards of one MergedColumnParallelLinear.
_HC_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        "hyper_connection.input_mix_weight_down.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            0,
        ),
        "hyper_connection.block_inject_weight.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            1,
        ),
    }
)


class Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3Next MoE with Qwen4Exp HC validation."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        self.ple: Qwen4ExpPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = Qwen4ExpPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif layer_type == "full_attention":
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen4ExpQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        self.hyper_connections: list[HyperConnectionBase] = []
        if config.hc_count > 1:
            for i in range(config.hc_count):
                self.hyper_connections.append(
                    GatedResidual(
                        config.hidden_size,
                        config.hc_lowrank,
                        prefix=f"{prefix}.hyper_connection.{i}",
                    )
                )

        self.mlp = Qwen3NextMLP(
            config=config,
            model_config=model_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.shared_expert_mlp: nn.Module | None = None
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert_mlp = Qwen3NextMLP(
                config=config,
                model_config=model_config,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_expert_mlp",
            )

        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_out = hidden_states
        if self.ple is not None:
            ple_out, ple_residual = self.ple(layer_out, residual)
            layer_out = ple_out
            residual = ple_residual

        if self.layer_type == "linear_attention":
            output, new_resid = self.linear_attn(
                layer_out,
                intermediate_tensors,
            )
            layer_out = output + new_resid
        else:
            layer_out, new_resid = self.self_attn(
                layer_out,
                intermediate_tensors,
            )
            layer_out = layer_out + new_resid

        if self.hyper_connections:
            hc_input = layer_out
            for hc in self.hyper_connections:
                hc_output = hc(hc_input)
                hc_input = hc_output
            layer_out = layer_out + hc_input

        mlp_out = self.mlp(layer_out)
        if self.shared_expert_mlp is not None:
            shared_mlp_out = self.shared_expert_mlp(layer_out)
            mlp_out = mlp_out + shared_mlp_out

        return layer_out + residual

    def extra_repr(self) -> str:
        return f"layer_type={self.layer_type}, layer_idx={self.layer_idx}"


class Qwen4ExpModel(nn.Module):
    def __init__(
        self,
        config: Qwen4ExpConfig,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        text_config: Qwen4ExpTextConfig = config.text_config
        self.config = config
        self.text_config = text_config
        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size
        self.num_layers = text_config.num_hidden_layers
        self.layer_types = text_config.layer_types or ["full_attention"] * self.num_layers

        self.embedding = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                Qwen4ExpDecoderLayer(
                    vllm_config=vllm_config,
                    layer_type=self.layer_types[i],
                    prefix=f"layers.{i}",
                )
            )

        self.norm = nn.LayerNorm(self.hidden_size, eps=text_config.norm_eps)
        self.gradient_checkpointing = False
        self.qsa_layer_ids: frozenset[int] = frozenset(
            [int(layer_id) - 1 for layer_id in text_config.ple_layer_ids]
        ) if text_config.ple_layer_ids else frozenset()

        if text_config.vision_config is not None:
            self.vision_tower = Qwen3_VisionTransformer(
                vision_config=text_config.vision_config,
                vision_select_layer=getattr(
                    text_config.vision_config,
                    "vision_select_layer",
                    -1,
                ),
                vision_select_feature=getattr(
                    text_config.vision_config,
                    "vision_select_feature",
                    "patch",
                ),
            )
            self.vision_projector = nn.Linear(
                text_config.vision_config.hidden_size,
                self.hidden_size,
            )

        self.mtp: Qwen4ExpMTP | None = None
        if getattr(text_config, "mtp_num_layers", 0) > 0:
            self.mtp = Qwen4ExpMTP(
                vllm_config=vllm_config,
                prefix="mtp",
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        multimodal_features: MultiModalFeatureSpec | None = None,
        cache_positions: torch.Tensor | None = None,
    ) -> dict:
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1],
                dtype=torch.long,
                device=inputs_embeds.device,
            ).unsqueeze(0)

        hidden_states = inputs_embeds

        if multimodal_features is not None:
            vision_features = self.vision_tower(
                multimodal_features["pixel_values"],
                attention_mask=multimodal_features.get("pixel_attention_mask"),
            )
            vision_embeddings = self.vision_projector(vision_features)
            hidden_states = _merge_multimodal_embeddings(
                hidden_states,
                vision_embeddings,
                multimodal_features["image_sizes"],
                multimodal_features["image_token_id"],
            )

        layer_outputs = []
        hidden_states_layer = hidden_states

        for layer_idx in range(len(self.layers)):
            decoder_layer = self.layers[layer_idx]
            hidden_states, residual = decoder_layer(
                hidden_states_layer,
                residual=None,
                intermediate_tensors=(
                    intermediate_tensors[layer_idx]
                    if hasattr(self, "intermediate_tensors")
                    else None
                ),
            )
            hidden_states_layer = hidden_states

        hidden_states = self.norm(hidden_states)

        return {
            "hidden_states": hidden_states,
            "logits": None,
            "attention": None,
            "hidden_states_layer": hidden_states_layer,
            "layer_idx": len(self.layers) - 1,
            "cache_position": cache_positions if cache_positions is not None else None,
        }


class Qwen4ExpForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.model = Qwen4ExpModel(config, vllm_config, quant_config)
        self.lm_head = ParallelLMHead(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.logits_processor = LogitsProcessor(config.text_config.vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embedding = value

    def get_lm_head(self) -> nn.Linear:
        return self.lm_head

    def set_lm_head(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        multimodal_features: MultiModalFeatureSpec | None = None,
        cache_positions: torch.Tensor | None = None,
    ) -> dict:
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            multimodal_features=multimodal_features,
            cache_positions=cache_positions,
        )

        hidden_states = model_output["hidden_states"]
        logits = self.lm_head(hidden_states)
        logits = self.logits_processor(logits, input_ids)

        return {
            "logits": logits,
            "hidden_states": model_output.get("hidden_states"),
            "attention": model_output.get("attention"),
        }

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple | None = None,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        multimodal_features: MultiModalFeatureSpec | None = None,
        cache_positions: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            positions,
            multimodal_features,
            cache_positions,
            **kwargs,
        )
        if multimodal_features is not None:
            inputs["multimodal_features"] = multimodal_features
        return inputs

    def forward_with_torch_compile(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        multimodal_features: MultiModalFeatureSpec | None = None,
        cache_positions: torch.Tensor | None = None,
    ) -> dict:
        return self.forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            multimodal_features,
            cache_positions,
        )


__all__ = [
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpMTP",
    "Qwen4ExpModel",
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpSparseMoeBlock",
]
