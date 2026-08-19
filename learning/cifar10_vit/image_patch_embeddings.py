from __future__ import annotations

import torch
import torch.nn as nn

from .config import VitConfig
from .layer import LayerTorchCompatible
from .module import Module


def extract_patches_manual(images: torch.Tensor, config: VitConfig) -> torch.Tensor:
    all_patches: list[torch.Tensor] = []
    _batch_size, _channels, height, width = images.shape

    if height % config.patch_height != 0 or width % config.patch_width != 0:
        raise ValueError("image height/width must divide evenly by patch height/width")

    for row in range(0, height, config.patch_height):
        for col in range(0, width, config.patch_width):
            # patch shape: (batch_size, channels, patch_height, patch_width)
            patch = images[
                :,
                :,
                row : row + config.patch_height,
                col : col + config.patch_width,
            ]

            # flattened patch shape: (batch_size, channels * patch_height * patch_width)
            patch = torch.flatten(patch, start_dim=1)
            all_patches.append(patch)

    # patches shape: (batch_size, number_of_patches, patch_dim)
    return torch.stack(all_patches, dim=1)


class ImagePatchEmbeddingTorchCompatible(Module):
    def __init__(self, config: VitConfig | None = None):
        super().__init__()
        self.config = config if config is not None else VitConfig()

        self.patch_projection = LayerTorchCompatible(
            number_of_neurons=self.config.embedding_dimension,
            number_of_inputs=self.config.patch_dim,
        )
        self.add_module(self.patch_projection)

    def convert_image_patches_to_embeddings_manual(self, images: torch.Tensor) -> torch.Tensor:
        patches = extract_patches_manual(images, self.config)

        # image_tokens shape: (batch_size, number_of_patches, embedding_dimension)
        image_tokens = self.patch_projection(patches)
        return image_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.convert_image_patches_to_embeddings_manual(images)


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

    def convert_image_patches_to_embeddings_manual(self, images: torch.Tensor) -> torch.Tensor:
        patches = extract_patches_manual(images, self.config)
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
