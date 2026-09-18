# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Intel Neural Compressor (INC) quantization integration.

Exports INCConfig for loading INC-quantized checkpoints.
"""

from .inc import INCConfig

__all__ = ["INCConfig"]
