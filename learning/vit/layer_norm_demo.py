"""Small, explicit LayerNorm and BatchNorm comparison.

Run from the repository root:
    uv run python learning/vit/layer_norm_demo.py

The example uses tokens shaped (batch, patches, features) = (2, 3, 4).
It computes both normalization methods directly, then checks each result
against the corresponding PyTorch module.
"""

import torch
from torch import nn


def main() -> None:
    torch.set_printoptions(precision=3, sci_mode=False)

    # Two images, three patch tokens per image, four features per token.
    tokens = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)
    batch_size, number_of_patches, number_of_features = tokens.shape
    epsilon = 1e-5

    print("Input tokens:")
    print(tokens)
    print("Shape:", tuple(tokens.shape))
    print()

    # ------------------------------------------------------------------
    # LayerNorm: normalize each row/token across its feature columns.
    # Each (batch, patch) pair gets its own mean and variance.
    # ------------------------------------------------------------------
    layer_mean = tokens.mean(dim=-1, keepdim=True)
    layer_variance = tokens.var(dim=-1, unbiased=False, keepdim=True)
    layer_normalized = (tokens - layer_mean) / torch.sqrt(
        layer_variance + epsilon
    )

    layer_norm = nn.LayerNorm(
        number_of_features,
        eps=epsilon,
        elementwise_affine=False,
    )
    layer_module_output = layer_norm(tokens)

    print("LayerNorm: normalize across the final feature dimension")
    print("Mean shape:", tuple(layer_mean.shape))
    print("Variance shape:", tuple(layer_variance.shape))
    print("Each token uses its own mean and variance.")
    print("Manual output:")
    print(layer_normalized)
    print("Manual == PyTorch:", torch.allclose(layer_normalized, layer_module_output))
    print("Means after normalization, per token:")
    print(layer_normalized.mean(dim=-1))
    print("Variances after normalization, per token:")
    print(layer_normalized.var(dim=-1, unbiased=False))
    print()

    # ------------------------------------------------------------------
    # BatchNorm: flatten batch and patch rows, then normalize each
    # feature column using all rows. This is training-mode batch math.
    # ------------------------------------------------------------------
    flat_tokens = tokens.reshape(-1, number_of_features)
    batch_mean = flat_tokens.mean(dim=0, keepdim=True)
    batch_variance = flat_tokens.var(dim=0, unbiased=False, keepdim=True)
    batch_normalized_flat = (flat_tokens - batch_mean) / torch.sqrt(
        batch_variance + epsilon
    )
    batch_normalized = batch_normalized_flat.reshape(
        batch_size,
        number_of_patches,
        number_of_features,
    )

    batch_norm = nn.BatchNorm1d(
        number_of_features,
        eps=epsilon,
        affine=False,
        track_running_stats=False,
    )
    batch_module_output = batch_norm(flat_tokens).reshape_as(tokens)

    print("BatchNorm: normalize each feature column across all rows")
    print("Flattened input shape:", tuple(flat_tokens.shape))
    print("Mean shape:", tuple(batch_mean.shape))
    print("Variance shape:", tuple(batch_variance.shape))
    print("The statistics use all batch * patch rows.")
    print("Manual output:")
    print(batch_normalized)
    print("Manual == PyTorch:", torch.allclose(batch_normalized, batch_module_output))
    print("Means after normalization, per feature:")
    print(batch_normalized_flat.mean(dim=0))
    print("Variances after normalization, per feature:")
    print(batch_normalized_flat.var(dim=0, unbiased=False))


if __name__ == "__main__":
    main()
