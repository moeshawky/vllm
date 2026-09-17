# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SmolVLM model implementation for compact multimodal vision-language
inference following the Idefics3 design pattern."""

from transformers import SmolVLMProcessor

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from .idefics3 import Idefics3DummyInputsBuilder as SmolVLMDummyInputsBuilder
from .idefics3 import Idefics3ForConditionalGeneration, Idefics3ProcessingInfo
from .idefics3 import Idefics3MultiModalProcessor as SmolVLMMultiModalProcessor


class SmolVLMProcessingInfo(Idefics3ProcessingInfo):
    """Extends Idefics3ProcessingInfo to return
    SmolVLMProcessor for SmolVLM models."""
    def get_hf_processor(self, **kwargs: object) -> SmolVLMProcessor:
        """Returns a SmolVLMProcessor from the
        processing context."""
        return self.ctx.get_hf_processor(SmolVLMProcessor, **kwargs)

    def _get_image_token(self, processor: SmolVLMProcessor) -> tuple[str, str, str]:
        """Extracts image_token, fake_image_token,
        and global_image_token from the processor."""
        image_token = processor.image_token
        fake_image_token = processor.fake_image_token
        global_image_token = processor.global_image_token
        return image_token, fake_image_token, global_image_token


@MULTIMODAL_REGISTRY.register_processor(
    SmolVLMMultiModalProcessor,
    info=SmolVLMProcessingInfo,
    dummy_inputs=SmolVLMDummyInputsBuilder,
)
class SmolVLMForConditionalGeneration(Idefics3ForConditionalGeneration):
    """Implements a compact vision-language model for
    multimodal text generation, registered as a SmolVLM processor."""
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """Initializes SmolVLMForConditionalGeneration by
        delegating to Idefics3ForConditionalGeneration
        with the vllm config and prefix."""
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
        )
