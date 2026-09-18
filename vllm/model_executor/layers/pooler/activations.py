# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooling activation functions and factory functions.

Provides the PoolerActivation abstract base class and concrete
implementations including identity, normalize, sigmoid, softmax,
and multi-label classification activations. Also provides factory
functions for constructing activation functions from model
configuration objects.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


def get_act_fn(
    config: PretrainedConfig,
    static_num_labels: bool = True,
) -> "PoolerActivation":
    """Construct a PoolerActivation from a PretrainedConfig.

    Determines the activation function based on the model's
    problem type and configuration fields. Classification
    problems use PoolerClassify, regression uses PoolerIdentity,
    and multi-label uses PoolerMultiLabelClassify. For cross-encoder
    models, resolves the activation function by qualified name.

    Args:
        config: PretrainedConfig containing model configuration.
        static_num_labels: If True, uses static num_labels from
            config for classification.

    Returns:
        A PoolerActivation instance configured from the model.
    """
    # Implement alignment with transformers ForSequenceClassificationLoss
    # https://github.com/huggingface/transformers/blob/57bb6db6ee4cfaccc45b8d474dfad5a17811ca60/src/transformers/loss/loss_utils.py#L92
    num_labels: int | None = None
    if static_num_labels:
        num_labels = getattr(config, "num_labels", 0)

    problem_type = getattr(config, "problem_type", "")
    if problem_type == "regression":
        return PoolerIdentity()
    if problem_type == "single_label_classification":
        return PoolerClassify(num_labels=num_labels)
    if problem_type == "multi_label_classification":
        return PoolerMultiLabelClassify()

    # get cross_encoder act_fn
    function_name: str | None = None
    if (
        hasattr(config, "sentence_transformers")
        and "activation_fn" in config.sentence_transformers
    ):
        function_name = config.sentence_transformers["activation_fn"]
    elif (
        hasattr(config, "sbert_ce_default_activation_function")
        and config.sbert_ce_default_activation_function is not None
    ):
        function_name = config.sbert_ce_default_activation_function

    if function_name is not None:
        if not function_name.startswith("torch.nn.modules."):
            raise ValueError(
                "Loading of activation functions is restricted to "
                "torch.nn.modules for security reasons"
            )
        fn = resolve_obj_by_qualname(function_name)()
        return PoolerActivation.wraps(fn)

    return PoolerClassify(num_labels=num_labels)


def resolve_classifier_act_fn(
    model_config: ModelConfig,
    static_num_labels: bool = True,
    act_fn: "PoolerActivation | None" = None,
) -> "PoolerActivation":
    """Resolve the classifier activation function to use.

    Returns the provided act_fn if not None, otherwise constructs
    one from the model configuration. Validates that act_fn is
    callable if provided.

    Args:
        model_config: ModelConfig containing model configuration.
        static_num_labels: If True, uses static num_labels.
        act_fn: Pre-constructed activation function, or None.

    Returns:
        A PoolerActivation instance.

    Raises:
        TypeError: If act_fn is provided but not callable.
    """
    if act_fn is None:
        return get_act_fn(model_config.hf_config, static_num_labels)

    if not callable(act_fn):
        raise TypeError(f"Expected a callable activation function, got {type(act_fn)}")
    return act_fn


_T = TypeVar("_T", torch.Tensor, list[torch.Tensor])


class PoolerActivation(nn.Module, ABC):
    """Abstract base class for pooling activation functions.

    Defines the interface for applying activation functions
    to pooled data, with support for both single and list
    inputs.
    """

    @staticmethod
    def wraps(module: nn.Module) -> "PoolerActivation":
        """Wrap a PyTorch module as a PoolerActivation.

        Dispatches to PoolerIdentity for nn.Identity,
        PoolerClassify for Sigmoid/Softmax, and
        LambdaPoolerActivation for other modules.

        Args:
            module: PyTorch module to wrap.

        Returns:
            A PoolerActivation instance wrapping the module.
        """
        if isinstance(module, nn.Identity):
            return PoolerIdentity()
        if isinstance(module, (nn.Sigmoid, nn.Softmax)):
            return PoolerClassify()

        return LambdaPoolerActivation(module)

    @abstractmethod
    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        """Apply activation to a single chunk of pooled data.

        Args:
            pooled_data: Input tensor to activate.

        Returns:
            Activated tensor.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError

    def forward(self, pooled_data: _T) -> _T:
        """Apply the activation function to pooled data.

        Handles both single tensors and lists of tensors
        by dispatching to forward_chunk for each element.

        Args:
            pooled_data: Tensor or list of tensors to activate.

        Returns:
            Activated tensor or list of activated tensors.
        """
        # shape:
        # classify -> (batch_size, num_classes)
        # embed -> (batch_size, embedding_size) or list(embedding_size)
        if isinstance(pooled_data, list):
            return [self.forward_chunk(data) for data in pooled_data]

        return self.forward_chunk(pooled_data)


class PoolerIdentity(PoolerActivation):
    """Identity activation that returns the input unchanged.

    Passes the pooled data through without any transformation.
    """

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged.

        Args:
            pooled_data: Input tensor.

        Returns:
            The same tensor as input.
        """
        return pooled_data


class PoolerNormalize(PoolerActivation):
    """Normalizes pooled data using L2 normalization.

    Applies L2 normalization along the last dimension
    to produce unit-norm embeddings.
    """

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        """Apply L2 normalization to the pooled data.

        Args:
            pooled_data: Input tensor to normalize.

        Returns:
            L2-normalized tensor.
        """
        return F.normalize(pooled_data, p=2, dim=-1)


class PoolerMultiLabelClassify(PoolerActivation):
    """Applies sigmoid activation for multi-label classification.

    Sigmoid is applied elementwise to produce independent
    probability scores for each class.
    """

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid activation for multi-label classification.

        Args:
            pooled_data: Input logits tensor.

        Returns:
            Sigmoid-activated probabilities in [0, 1].
        """
        return F.sigmoid(pooled_data)


class PoolerClassify(PoolerActivation):
    """Applies softmax or sigmoid activation for classification.

    For single-label classification, applies softmax over
    the class dimension. For binary or multi-label with
    fewer than 2 labels, applies sigmoid instead.
    """

    def __init__(self, *, num_labels: int | None = None) -> None:
        """Initialize the classification activation.

        Args:
            num_labels: Number of classification labels.
                If 0, a warning is logged and sigmoid is used
                as a fallback.
        """
        super().__init__()

        if num_labels == 0:
            logger.warning(
                "num_labels should be > 0 for classification "
                "models, falling back to sigmoid. "
                "Please check if the configuration is correct."
            )

        self.num_labels = num_labels

    def extra_repr(self) -> str:
        """Return a string representation of the classifier.

        Returns:
            String listing the number of labels.
        """
        return f"num_labels={self.num_labels}"

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        """Apply softmax or sigmoid based on number of labels.

        If num_labels is None, uses the last dimension of
        the input as the number of labels. For fewer than 2
        labels, uses sigmoid; otherwise uses softmax.

        Args:
            pooled_data: Input tensor of logits.

        Returns:
            Softmax or sigmoid activated tensor.
        """
        num_labels = self.num_labels
        if num_labels is None:
            num_labels = pooled_data.shape[-1]

        if num_labels < 2:
            return F.sigmoid(pooled_data)

        return F.softmax(pooled_data, dim=-1)


class LambdaPoolerActivation(PoolerActivation):
    """Wraps a callable as a PoolerActivation.

    Allows arbitrary PyTorch-compatible callables to be
    used as pooling activation functions through a
    common interface.
    """

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Initialize with a callable activation function.

        Args:
            fn: Callable that takes a tensor and returns
                a tensor, to be applied as the activation.
        """
        super().__init__()

        self.fn = fn

    def extra_repr(self) -> str:
        """Return a string representation of the lambda activation.

        Returns:
            String with the function name or class name.
        """
        name = getattr(self.fn, "__name__", None)
        if name is None:
            name = self.fn.__class__.__name__
        return f"fn={name}"

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped callable activation function.

        Args:
            pooled_data: Input tensor.

        Returns:
            Output of the wrapped activation function.
        """
        return self.fn(pooled_data)
