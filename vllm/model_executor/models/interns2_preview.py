# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preview version of InternS2 with time-series support.

Provides the InternS2PreviewProcessingInfo and
InternS2PreviewForConditionalGeneration with
additional weight mapping for time_series projections.
"""

from collections.abc import Iterable

import torch
from transformers import AutoProcessor

from vllm.multimodal import MULTIMODAL_REGISTRY

from .qwen3_5 import Qwen3_5MoeForConditionalGeneration
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import AutoWeightsLoader, WeightsMapper


class InternS2PreviewProcessingInfo(Qwen3VLProcessingInfo):
    """ProcessingInfo for InternS2Preview, delegating to context."""

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> AutoProcessor:
        return self.ctx.get_hf_processor(**kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=InternS2PreviewProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class InternS2PreviewForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    """Multimodal model with time-series weight mapping."""

    # `mtp.` is already dropped by `Qwen3_5ForConditionalGeneration`.
    hf_to_vllm_mapper = Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper | (
        WeightsMapper(
            orig_to_new_prefix={"model.time_series.": None, "time_series.": None}
        )
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with combined mapper including time_series mapping."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
