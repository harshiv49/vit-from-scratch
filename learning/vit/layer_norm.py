from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Module


class LayerNormTorchCompatible(Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps

        self.gamma = torch.ones(embedding_dim, requires_grad=True)
        self.beta = torch.zeros(embedding_dim, requires_grad=True)

        self.add_parameter(self.gamma)
        self.add_parameter(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * x_normalized + self.beta


class TorchLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.embedding_dim,),
            weight=self.gamma,
            bias=self.beta,
            eps=self.eps,
        )
