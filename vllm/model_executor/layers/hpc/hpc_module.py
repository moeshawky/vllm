# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn


class HpcModule(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def support(cls, *args, **kwargs):
        """Class method returning True to indicate HPC module support."""
        return True

    def process_weights_after_loading(self, model):
        """No-op weight processing after loading."""
        pass

    def forward(self, *args, **kwargs):
        """No-op forward pass."""
        pass
