# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-independent MiniMax M3 model code.

Shared modules used by both the AMD/ROCm and NVIDIA/CUDA backends:
indexer (lightning block scoring and top-k selection), sparse attention
(backend, metadata, and Triton attend kernel), multimodal preprocessing
(image/video tokenization and dummy inputs), and platform-specific ops
routed through the current_platform checks in each consumer.
"""