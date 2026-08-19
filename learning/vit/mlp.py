from __future__ import annotations

import torch

from .layer import LayerTorchCompatible
from .module import Module


class ReLUTorchCompatible(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class MLP(Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.layer_1 = LayerTorchCompatible(
            number_of_neurons=hidden_dim,
            number_of_inputs=input_dim,
        )
        self.relu = ReLUTorchCompatible()
        self.layer_2 = LayerTorchCompatible(
            number_of_neurons=output_dim,
            number_of_inputs=hidden_dim,
        )

        self.layers = [self.layer_1, self.relu, self.layer_2]

        for layer in self.layers:
            self.add_module(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
