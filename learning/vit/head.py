from __future__ import annotations

import torch
import torch.nn as nn

from .config import VitConfig
from .layer import LayerTorchCompatible
from .module import Module


class Head(Module):
    def __init__(self, config: VitConfig | None = None):
        super().__init__()
        self.config = config if config is not None else VitConfig()

        self.key = LayerTorchCompatible(
            input_dim=self.config.embedding_dimension,
            output_dim=self.config.head_size,
        )
        self.query = LayerTorchCompatible(
            input_dim=self.config.embedding_dimension,
            output_dim=self.config.head_size,
        )
        self.value = LayerTorchCompatible(
            input_dim=self.config.embedding_dimension,
            output_dim=self.config.head_size,
        )

        self.add_module(self.key)
        self.add_module(self.query)
        self.add_module(self.value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        # query shape: (batch_size, number_of_patches, head_size)
        # key.transpose(-2, -1) shape: (batch_size, head_size, number_of_patches)
        # scores shape: (batch_size, number_of_patches, number_of_patches)
        scores = query @ key.transpose(-2, -1)
        scores = scores / (self.config.head_size ** 0.5)

        # dim=-1 means each token creates a probability distribution over all tokens.
        weights = torch.softmax(scores, dim=-1)

        # output shape: (batch_size, number_of_patches, head_size)
        output = weights @ value
        return output


class TorchHead(nn.Module):
    def __init__(self, config: VitConfig | None = None):
        super().__init__()
        self.config = config if config is not None else VitConfig()

        self.key = nn.Linear(self.config.embedding_dimension, self.config.head_size)
        self.query = nn.Linear(self.config.embedding_dimension, self.config.head_size)
        self.value = nn.Linear(self.config.embedding_dimension, self.config.head_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        scores = query @ key.transpose(-2, -1)
        scores = scores / (self.config.head_size ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ value
        return output
