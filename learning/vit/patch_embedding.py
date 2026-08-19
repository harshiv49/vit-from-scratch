from __future__ import annotations

import torch
import torch.nn as nn

from .config import VitConfig
from .layer import LayerTorchCompatible
from .module import Module


def validate_patch_shape(images: torch.Tensor, config: VitConfig) -> tuple[int, int, int, int]:
    batch_size, channels, height, width = images.shape

    if channels != config.channels:
        raise ValueError(f"expected {config.channels} channels, got {channels}")

    if height % config.patch_height != 0 or width % config.patch_width != 0:
        raise ValueError("image height/width must divide evenly by patch height/width")

    return batch_size, channels, height, width


def extract_patches_manual(images: torch.Tensor, config: VitConfig) -> torch.Tensor:
    all_patches: list[torch.Tensor] = []
    _batch_size, _channels, height, width = validate_patch_shape(images, config)

    for row in range(0, height, config.patch_height):
        for col in range(0, width, config.patch_width):
            # patch (batch_size, channel, patch_height, patch_width)
            patch = images[
                :,
                :,
                row : row + config.patch_height,
                col : col + config.patch_width,
            ]

            # we flatten from the channel dimension onward
            # example with CIFAR-10 and 4x4 patches: (8, 3, 4, 4) -> (8, 48)
            patch = torch.flatten(patch, start_dim=1)
            all_patches.append(patch)

    # For a square patch size P, assuming the image divides evenly:
    # N = (H / P) × (W / P) = H × W / P².
    # With H = W = 32 and P = 4, there are 8 × 8 = 64 patches.
    # torch.stack combines multiple tensors by adding a new dimension.
    # patches shape: (batch_size, number_of_patches, patch_dim)
    return torch.stack(all_patches, dim=1)


def extract_patches_tensor_unfold(images: torch.Tensor, config: VitConfig) -> torch.Tensor:
    batch_size, channels, height, width = validate_patch_shape(images, config)

    # First unfold over image height:
    # (B, C, H, W) -> (B, C, patch_rows, W, patch_height)
    # Then unfold over image width:
    # (B, C, patch_rows, W, patch_height) ->
    # (B, C, patch_rows, patch_cols, patch_height, patch_width)
    patches = images.unfold(2, config.patch_height, config.patch_height).unfold(
        3,
        config.patch_width,
        config.patch_width,
    )

    patch_rows = height // config.patch_height
    patch_cols = width // config.patch_width

    # Move patch grid before the patch contents:
    # (B, C, patch_rows, patch_cols, patch_height, patch_width) ->
    # (B, patch_rows, patch_cols, C, patch_height, patch_width)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

    # Flatten each patch:
    # (B, patch_rows, patch_cols, C, patch_height, patch_width) ->
    # (B, number_of_patches, C * patch_height * patch_width)
    return patches.view(batch_size, patch_rows * patch_cols, channels * config.patch_height * config.patch_width)


class ImagePatchEmbeddingTorchCompatible(Module):
    def __init__(self, config: VitConfig | None = None):
        super().__init__()
        self.config = config if config is not None else VitConfig()

        self.patch_projection = LayerTorchCompatible(
            number_of_neurons=self.config.embedding_dimension,
            number_of_inputs=self.config.patch_dim,
        )
        self.add_module(self.patch_projection)

    # image into image embeddings
    def convert_image_patches_to_embeddings_manual(self, images: torch.Tensor) -> torch.Tensor:
        patches = extract_patches_manual(images, self.config)

        # image_tokens shape: (batch_size, number_of_patches, embedding_dimension)
        image_tokens = self.patch_projection(patches)
        return image_tokens

    def convert_image_patches_to_embeddings_tensor_unfold(self, images: torch.Tensor) -> torch.Tensor:
        patches = extract_patches_tensor_unfold(images, self.config)
        image_tokens = self.patch_projection(patches)
        return image_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.convert_image_patches_to_embeddings_tensor_unfold(images)


class TorchImagePatchEmbedding(nn.Module):
    def __init__(self, config: VitConfig | None = None):
        super().__init__()
        self.config = config if config is not None else VitConfig()

        self.patch_projection = nn.Linear(
            in_features=self.config.patch_dim,
            out_features=self.config.embedding_dimension,
        )
        self.unfolder = nn.Unfold(
            kernel_size=(self.config.patch_height, self.config.patch_width),
            stride=(self.config.patch_height, self.config.patch_width),
        )

    # image into image embeddings
    def convert_image_patches_to_embeddings_manual(self, images: torch.Tensor) -> torch.Tensor:
        patches = extract_patches_manual(images, self.config)
        image_tokens = self.patch_projection(patches)
        return image_tokens

    def convert_image_patches_to_embeddings_tensor_unfold(self, images: torch.Tensor) -> torch.Tensor:
        patches = extract_patches_tensor_unfold(images, self.config)
        image_tokens = self.patch_projection(patches)
        return image_tokens

    def convert_image_patches_to_embeddings_torch_nn(self, images: torch.Tensor) -> torch.Tensor:
        # nn.Unfold returns: (batch_size, patch_dim, number_of_patches)
        patches = self.unfolder(images)

        # transformer code wants: (batch_size, number_of_patches, patch_dim)
        patches = patches.transpose(1, 2).contiguous()

        image_tokens = self.patch_projection(patches)
        return image_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.convert_image_patches_to_embeddings_torch_nn(images)
