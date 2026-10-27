from dataclasses import asdict, dataclass

import torch
from torch import nn


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert [B, C, H, W] images into row-major [B, N, C*P*P] patches."""
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    batch, channels, height, width = images.shape
    if patch_size < 1 or height % patch_size or width % patch_size:
        raise ValueError("patch_size must divide image height and width")
    grid_height = height // patch_size
    grid_width = width // patch_size
    patches = images.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    return patches.permute(0, 2, 3, 1, 4, 5).reshape(
        batch, grid_height * grid_width, channels * patch_size * patch_size
    )


def unpatchify(
    patches: torch.Tensor, channels: int, height: int, width: int, patch_size: int
) -> torch.Tensor:
    """Reverse patchify, useful for checking that patch order matches the image."""
    if height % patch_size or width % patch_size:
        raise ValueError("patch_size must divide image height and width")
    batch, count, values = patches.shape
    grid_height, grid_width = height // patch_size, width // patch_size
    if count != grid_height * grid_width or values != channels * patch_size**2:
        raise ValueError("patch tensor does not match the requested image shape")
    return (
        patches.reshape(batch, grid_height, grid_width, channels, patch_size, patch_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, height, width)
    )


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 28
    channels: int = 1
    patch_size: int = 4
    embed_dim: int = 64
    heads: int = 4
    depth: int = 2
    mlp_dim: int = 128
    classes: int = 10

    def __post_init__(self):
        if min(asdict(self).values()) < 1:
            raise ValueError("all configuration values must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.embed_dim % self.heads:
            raise ValueError("embed_dim must be divisible by heads")


class PatchEmbedding(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.projection = nn.Linear(
            config.channels * config.patch_size**2, config.embed_dim
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(patchify(images, self.patch_size))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor, return_attention: bool = False):
        batch, count, width = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, count, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)
        scores = (queries @ keys.transpose(-2, -1)) * self.scale
        weights = scores.softmax(dim=-1)
        attended = weights @ values
        attended = attended.transpose(1, 2).reshape(batch, count, width)
        output = self.projection(attended)
        if return_attention:
            return output, weights
        return output


class EncoderBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embed_dim)
        self.attention = MultiHeadSelfAttention(config.embed_dim, config.heads)
        self.mlp_norm = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.embed_dim),
        )

    def forward(self, tokens: torch.Tensor, return_attention: bool = False):
        if return_attention:
            update, weights = self.attention(
                self.attention_norm(tokens), return_attention=True
            )
        else:
            update = self.attention(self.attention_norm(tokens))
        tokens = tokens + update
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        if return_attention:
            return tokens, weights
        return tokens


class TinyViT(nn.Module):
    def __init__(self, config: ViTConfig = ViTConfig()):
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(config)
        patch_count = (config.image_size // config.patch_size) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, patch_count + 1, config.embed_dim)
        )
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.embed_dim)
        self.classifier = nn.Linear(config.embed_dim, config.classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor, return_attention: bool = False):
        if images.shape[1:] != (
            self.config.channels,
            self.config.image_size,
            self.config.image_size,
        ):
            raise ValueError("image shape does not match ViTConfig")
        patches = self.patch_embedding(images)
        class_token = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_token, patches), dim=1) + self.position_embedding
        all_weights = []
        for block in self.blocks:
            if return_attention:
                tokens, weights = block(tokens, return_attention=True)
                all_weights.append(weights)
            else:
                tokens = block(tokens)
        logits = self.classifier(self.norm(tokens[:, 0]))
        if return_attention:
            return logits, torch.stack(all_weights, dim=1)
        return logits
