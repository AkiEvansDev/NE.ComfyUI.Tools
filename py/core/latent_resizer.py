from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def normalization(channels: int):
    return nn.GroupNorm(32, channels)


def zero_module(module: nn.Module):
    for parameter in module.parameters():
        parameter.detach().zero_()

    return module


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.norm = normalization(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        batch, channels, height, width = q.shape

        q, k, v = (
            rearrange(tensor, "b c h w -> b 1 (h w) c").contiguous()
            for tensor in (q, k, v)
        )

        x = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(
            x,
            "b 1 (h w) c -> b c h w",
            h=height,
            w=width,
            c=channels,
            b=batch,
        )

    def forward(self, x, **kwargs):
        return x + self.proj_out(self.attention(x))


def make_attn(in_channels: int, attn_kwargs=None):
    return AttnBlock(in_channels)


class ResBlockEmb(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float = 0,
        out_channels: int | None = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        kernel_size: int = 3,
        exchange_temb_dims: bool = False,
        skip_t_emb: bool = False,
    ):
        super().__init__()

        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims
        self.skip_t_emb = skip_t_emb

        padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.emb_out_channels = 2 * self.out_channels if use_scale_shift_norm else self.out_channels

        if self.skip_t_emb:
            if self.use_scale_shift_norm:
                raise ValueError("skip_t_emb cannot be used with use_scale_shift_norm.")

            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, self.emb_out_channels),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, kernel_size, padding=padding)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = torch.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm = self.out_layers[0]
            out_rest = self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)

            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")

            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class LatentResizer(nn.Module):
    def __init__(
        self,
        in_blocks: int = 10,
        out_blocks: int = 10,
        channels: int = 128,
        dropout: float = 0,
        attn: bool = True,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(4, channels, 3, padding=1)
        self.channels = channels

        embed_dim = 32
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.in_blocks = nn.ModuleList([])
        for block_index in range(in_blocks):
            if block_index in (1, in_blocks - 1) and attn:
                self.in_blocks.append(make_attn(channels))

            self.in_blocks.append(ResBlockEmb(channels, embed_dim, dropout))

        self.out_blocks = nn.ModuleList([])
        for block_index in range(out_blocks):
            if block_index in (1, out_blocks - 1) and attn:
                self.out_blocks.append(make_attn(channels))

            self.out_blocks.append(ResBlockEmb(channels, embed_dim, dropout))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, 4, 3, padding=1)

    @classmethod
    def load_model(cls, filename, device="cpu", dtype=torch.float32, dropout=0):
        if "weights_only" in torch.load.__code__.co_varnames:
            weights = torch.load(filename, map_location=torch.device("cpu"), weights_only=True)
        else:
            weights = torch.load(filename, map_location=torch.device("cpu"))

        in_blocks = 0
        out_blocks = 0
        in_tfs = 0
        out_tfs = 0
        channels = weights["conv_in.bias"].shape[0]

        for key in weights.keys():
            parts = key.split(".")

            if parts[0] == "in_blocks":
                in_blocks = max(in_blocks, int(parts[1]))
                if parts[2] == "q" and parts[3] == "weight":
                    in_tfs += 1

            if parts[0] == "out_blocks":
                out_blocks = max(out_blocks, int(parts[1]))
                if parts[2] == "q" and parts[3] == "weight":
                    out_tfs += 1

        in_blocks = in_blocks + 1 - in_tfs
        out_blocks = out_blocks + 1 - out_tfs

        resizer = cls(
            in_blocks=in_blocks,
            out_blocks=out_blocks,
            channels=channels,
            dropout=dropout,
            attn=out_tfs != 0,
        )

        resizer.load_state_dict(weights)
        resizer.eval()
        resizer.to(device, dtype=dtype)

        return resizer

    def forward(self, x, scale=None, size=None):
        if scale is None and size is None:
            raise ValueError("Either scale or size must be provided.")

        if scale is not None and size is not None:
            raise ValueError("Only one of scale or size can be provided.")

        if scale is not None:
            size = tuple(int(round(value)) for value in (x.shape[-2] * scale, x.shape[-1] * scale))
        else:
            scale = size[-1] / x.shape[-1]

        if size == x.shape[-2:]:
            return x

        scale_tensor = torch.tensor([scale - 1], dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_tensor)

        x = self.conv_in(x)

        for block in self.in_blocks:
            x = block(x, emb) if isinstance(block, ResBlockEmb) else block(x)

        x = F.interpolate(x, size=size, mode="bilinear")

        for block in self.out_blocks:
            x = block(x, emb) if isinstance(block, ResBlockEmb) else block(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x