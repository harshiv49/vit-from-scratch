from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VitConfig:
    channels: int = 3
    image_height: int = 32
    image_width: int = 32
    patch_height: int = 4
    patch_width: int = 4
    embedding_dimension: int = 192
    head_size: int = 16
    number_of_heads: int = 12
    number_of_classes: int = 10

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch_height * self.patch_width

    @property
    def number_of_patches(self) -> int:
        return (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
