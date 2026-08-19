"""Small Vision Transformer block shape demo.

Run from the repository root:
    uv run python learning/cifar10_vit/vit_block_demo.py

This starts after patch projection. It assumes we already have tokens shaped:
    (batch, number_of_patches, embedding_dimension)

For CIFAR with 4x4 patches and a 192-dimensional embedding:
    (8, 64, 192)
"""

import torch
from torch import nn


class TokenMLP(nn.Module):
    """Feature mixer applied independently to every patch token.

    Input shape:  (B, N, D)
    Output shape: (B, N, D)

    It does not mix patch positions. It only transforms the D features inside
    each token. In image terms, this is similar in spirit to a shared 1x1
    operation over all patch locations.
    """

    def __init__(self, embedding_dimension: int, expansion_factor: int = 4) -> None:
        super().__init__()
        hidden_dimension = expansion_factor * embedding_dimension
        self.layers = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, embedding_dimension),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.layers(tokens)


class VisionTransformerBlock(nn.Module):
    """One pre-norm ViT encoder block.

    The block structure is:
        x = x + self_attention(layer_norm_1(x))
        x = x + mlp(layer_norm_2(x))

    Shape stays the same from input to output:
        (B, N, D) -> (B, N, D)
    """

    def __init__(self, embedding_dimension: int, number_of_heads: int) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=number_of_heads,
            batch_first=True,
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.mlp = TokenMLP(embedding_dimension)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        print("input tokens:                 ", tuple(tokens.shape))

        attention_input = self.layer_norm_1(tokens)
        print("after layer_norm_1:           ", tuple(attention_input.shape))

        attention_output, attention_weights = self.self_attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=True,
        )
        print("after self_attention output:  ", tuple(attention_output.shape))
        print("attention weights:            ", tuple(attention_weights.shape))

        tokens = tokens + attention_output
        print("after first residual add:     ", tuple(tokens.shape))

        mlp_input = self.layer_norm_2(tokens)
        print("after layer_norm_2:           ", tuple(mlp_input.shape))

        mlp_output = self.mlp(mlp_input)
        print("after mlp output:             ", tuple(mlp_output.shape))

        tokens = tokens + mlp_output
        print("after second residual add:    ", tuple(tokens.shape))

        return tokens


def main() -> None:
    torch.manual_seed(42)

    batch_size = 8
    number_of_patches = 64
    embedding_dimension = 192
    number_of_heads = 3

    # Pretend these came from:
    # images -> nn.Unfold -> transpose -> nn.Linear
    tokens = torch.randn(batch_size, number_of_patches, embedding_dimension)

    block = VisionTransformerBlock(
        embedding_dimension=embedding_dimension,
        number_of_heads=number_of_heads,
    )

    output_tokens = block(tokens)

    assert output_tokens.shape == tokens.shape
    print("\nShape check passed: output shape equals input shape.")


if __name__ == "__main__":
    main()
