# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 quantization utilities.

Provides helpers for CUTLASS FP8 kernel support checks,
per-tensor dequantization, channelwise scale conversion,
and FP8-to-e4m3fnuz normalization.
"""


import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform


def cutlass_fp8_supported() -> bool:
    """Check if the current platform supports CUTLASS FP8 kernels.

    Returns:
        True if CUTLASS scaled MM supports FP8 on this platform.
    """
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_scaled_mm_supports_fp8(capability)


def cutlass_block_fp8_supported() -> bool:
    """Check if the current platform supports CUTLASS block FP8 kernels.

    Returns:
        True if CUTLASS supports block FP8 on this platform.
    """
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_scaled_mm_supports_block_fp8(capability)


def cutlass_group_gemm_supported() -> bool:
    """Check if the current platform supports CUTLASS grouped GEMM.

    Returns:
        True if CUTLASS group GEMM is supported on this platform.
    """
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_group_gemm_supported(capability)


CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()


def per_tensor_dequantize(
    tensor: torch.Tensor, inv_scale: float | torch.Tensor
) -> torch.Tensor:
    """Dequantize a per-tensor quantized weight tensor.

    Args:
        tensor: Quantized weight tensor in FP8.
        inv_scale: Inverse scale factor for dequantization.

    Returns:
        Dequantized tensor in float16.
    """
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def all_close_1d(x: torch.Tensor) -> bool:
    """Check if all elements in a 1D tensor are close to the first.

    Args:
        x: 1D tensor to check.

    Returns:
        True if all elements are approximately equal to x[0].
    """
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def convert_to_channelwise(
    weight_scale: torch.Tensor, logical_widths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a uniform weight scale to per-channel scales.

    Args:
        weight_scale: Uniform scale tensor.
        logical_widths: List of logical widths per channel.

    Returns:
        Tuple of (channelwise_scale, expanded_scale).
    """
    # Create channelwise buffer
    weight_scale_channel = torch.empty(
        (sum(logical_widths), 1), dtype=torch.float32, device=weight_scale.device
    )

    # Expand each scale to match the size of each logical matrix.
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        weight_scale_channel[start:end, :] = weight_scale[idx]
        start = end

    return weight_scale_channel


def requantize_with_max_scale(
    weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize weights using the maximum scale across channels.

    Args:
        weight: Quantized weight tensor.
        weight_scale: Per-channel weight scales.
        logical_widths: Logical widths for each shard.

    Returns:
        Tuple of (max_scale, requantized_weight).
    """
    # Max scale to be used for requanitzation.
    max_w_scale = weight_scale.max()

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    #
    # Extra note: upon weight reloading weight_scale.ndim == 0
    unfused_module_in_checkpoint = (
        weight_scale.ndim != 0
        and weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min
    )

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            # Skip any component with zero width.
            if logical_width == 0:
                continue
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :], weight_scale[idx])
            weight[start:end, :], _ = ops.scaled_fp8_quant(weight_dq, max_w_scale)
            start = end

    return max_w_scale, weight


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert FP8 e4m3fn values to e4m3fnuz representation.

    The NaN value in e4m3fnuz at bit pattern 128 is replaced
    with zero to match the e4m3fn zero-point convention.
    Scaling factors are doubled to preserve the effective values.

    Args:
        weight: FP8 e4m3fn weight tensor.
        weight_scale: Weight scale tensor.
        input_scale: Optional input scale tensor.

    Returns:
        Tuple of (converted weight, converted weight scale, converted input scale).
    """
    assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale
