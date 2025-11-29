# models/kvae3d_loader.py
"""
KVAE-3D loader & architecture (standalone, no dependency on third_party/kvae-1).

This file re-implements a Cached Causal 3D VAE that is compatible with the
KVAE-3D 1.0 weights from kandinskylab (MIT-licensed original implementation).

Main entrypoint:

    from models.kvae3d_loader import load_kvae3d

    vae = load_kvae3d()  # loads default config + vae/kvae_3d_2025.safetensors
    latents, split_list, _ = vae.encode(video)   # video: [B, 3, T, H, W], m11
    rec = vae.decode(latents, split_list)

Where:
  - T is the number of frames
  - H, W are spatial dimensions
  - m11 normalization means values in [-1, 1]
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file as load_safetensors_file


# ---------------------------------------------------------------------------
# Default config (from KVAE-3D HF config.json)
# ---------------------------------------------------------------------------

DEFAULT_KVAE3D_CONFIG: Dict[str, Any] = {
    "common": {
        "experiment_name": "vae_ch16_488",
    },
    "data": {
        # Input is expected in [-1, 1] range ("m11" normalization)
        "input_norm": "m11",
    },
    "model": {
        "encoder_params": {
            "ch": 128,
            "in_channels": 3,
            "num_res_blocks": 2,
            "temporal_compress_times": 4,
            "z_channels": 16,
            # the rest use defaults from CachedEncoder3D
        },
        "decoder_params": {
            "ch": 128,
            "num_res_blocks": 2,
            "out_ch": 3,
            "temporal_compress_times": 4,
            "z_channels": 16,
            # the rest use defaults from CachedDecoder3D
        },
    },
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def cast_tuple(t: Union[int, Tuple[int, ...]], length: int = 1) -> Tuple[int, ...]:
    """Ensure `t` is a tuple of length `length`."""
    if isinstance(t, tuple):
        return t
    return (t,) * length


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish activation, as in the original implementation."""
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Core low-level 3D convolution building blocks
# ---------------------------------------------------------------------------

class SafeConv3d(nn.Conv3d):
    """
    Memory-aware Conv3d that chunks the temporal dimension if the input
    is too large, to avoid OOM. Behavior is compatible with the KVAE-3D code.
    """

    def forward(
        self,
        x: torch.Tensor,
        write_to: Optional[torch.Tensor] = None,
        transform: Optional[Any] = None,
    ) -> torch.Tensor:
        if transform is None:
            transform = lambda v: v

        memory_gb = x.numel() * x.element_size() / (10 ** 9)
        if memory_gb > 3:
            kernel_size_t = self.kernel_size[0]
            part_num = math.ceil(memory_gb / 2)
            input_chunks = torch.chunk(x, part_num, dim=2)  # N, C, T, H, W

            # Special tiny-temporal case
            if input_chunks[0].size(2) < 3 and kernel_size_t > 1:
                assert write_to is not None, "write_to must be provided in tiny-temporal mode"
                for i in range(x.size(2) - 2):
                    torch.cuda.empty_cache()
                    time.sleep(0.2)
                    chunk = transform(x[:, :, i : i + 3])
                    write_to[:, :, i : i + 1] = super(SafeConv3d, self).forward(chunk)
                return write_to

            if write_to is None:
                outputs = []
                z = None
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size_t == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size_t + 1 :], chunk], dim=2)
                    outputs.append(super(SafeConv3d, self).forward(transform(z)))
                return torch.cat(outputs, dim=2)
            else:
                time_offset = 0
                z = None
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size_t == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size_t + 1 :], chunk], dim=2)
                    z_time = z.size(2) - (kernel_size_t - 1)
                    write_to[:, :, time_offset : time_offset + z_time] = super(SafeConv3d, self).forward(
                        transform(z)
                    )
                    time_offset += z_time
                return write_to
        else:
            if write_to is None:
                return super(SafeConv3d, self).forward(transform(x))
            else:
                write_to[...] = super(SafeConv3d, self).forward(transform(x))
                return write_to


class CausalConv3d(nn.Module):
    """
    Simple causal 3D convolution: pads along time so that the kernel
    never sees future frames. Spatial dimensions are padded symmetrically.
    """

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Tuple[int, int, int] = (1, 1, 1),
        dilation: Tuple[int, int, int] = (1, 1, 1),
        **kwargs: Any,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_ks, height_ks, width_ks = kernel_size

        assert (height_ks % 2) == 1 and (width_ks % 2) == 1

        self.height_pad = height_ks // 2
        self.width_pad = width_ks // 2
        self.time_pad = time_ks - 1
        self.time_kernel_size = time_ks
        self.stride = stride

        self.conv = SafeConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad: (W_left, W_right, H_left, H_right, T_left, T_right)
        padding_3d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, self.time_pad, 0)
        x_padded = F.pad(x, padding_3d, mode="replicate")
        return self.conv(x_padded)


# ---------------------------------------------------------------------------
# Normalization (GroupNorm / RMSNorm variants)
# ---------------------------------------------------------------------------

def RMSNorm(in_channels: int, *args: Any, **kwargs: Any) -> nn.Module:
    return WanRMSNorm(n_ch=in_channels, bias=False)


class WanRMSNorm(nn.Module):
    """
    RMS normalization layer (used when norm_type == "rms_norm").
    """

    def __init__(self, n_ch: int, bias: bool = False) -> None:
        super().__init__()
        shape = (n_ch, 1, 1, 1)
        self.scale = n_ch ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # F.normalize along channels, then scale and shift.
        x_norm = F.normalize(x, dim=1)
        return x_norm * self.scale * self.gamma + self.bias


class CachedGroupNorm(nn.GroupNorm):
    """
    GroupNorm with a "cache" parameter in forward, to match the KVAE-3D API.
    The cache is only used as a flag for "first vs later" calls.
    """

    def group_forward(
        self,
        x_input: torch.Tensor,
        expectation: Optional[Sequence[torch.Tensor]] = None,
        variance: Optional[Sequence[torch.Tensor]] = None,
        return_stat: bool = False,
    ):
        # Not used in current forward path, but kept for compatibility.
        input_dtype = x_input.dtype
        x = x_input.to(torch.float32)
        chunks = torch.chunk(x, self.num_groups, dim=1)
        if expectation is None:
            ch_mean = [torch.mean(chunk, dim=(1, 2, 3, 4), keepdim=True) for chunk in chunks]
        else:
            ch_mean = expectation

        if variance is None:
            ch_var = [torch.var(chunk, dim=(1, 2, 3, 4), keepdim=True, unbiased=False) for chunk in chunks]
        else:
            ch_var = variance

        x_norm = [(chunk - m) / torch.sqrt(v + self.eps) for chunk, m, v in zip(chunks, ch_mean, ch_var)]
        x_norm = torch.cat(x_norm, dim=1)

        x_norm.mul_(self.weight.data.view(1, -1, 1, 1, 1))
        x_norm.add_(self.bias.data.view(1, -1, 1, 1, 1))

        x_out = x_norm.to(input_dtype)
        if return_stat:
            return x_out, ch_mean, ch_var
        return x_out

    def forward(self, x: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
        out = super().forward(x)
        # We only use the cache as a "first call" flag.
        if cache.get("mean") is None and cache.get("var") is None:
            cache["mean"] = 1
            cache["var"] = 1
        return out


def Normalize(in_channels: int, gather: bool = False, **kwargs: Any) -> nn.Module:
    # gather is unused, but we keep the signature for compatibility.
    return CachedGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# ---------------------------------------------------------------------------
# Cached causal conv and residual blocks
# ---------------------------------------------------------------------------

class CachedCausalConv3d(CausalConv3d):
    """
    Causal 3D convolution with explicit temporal padding cache, so that
    segments can be processed sequentially without losing temporal context.
    """

    def forward(self, x: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
        t_stride = self.stride[0]
        # pad only spatial dims here (temporal pad is handled via cache)
        padding_3d = (self.height_pad, self.height_pad, self.width_pad, self.width_pad, 0, 0)
        x_parallel = F.pad(x, padding_3d, mode="replicate")

        if cache["padding"] is None:
            first_frame = x_parallel[:, :, :1]
            time_pad_shape = list(first_frame.shape)
            time_pad_shape[2] = self.time_pad
            padding = first_frame.expand(time_pad_shape)
        else:
            padding = cache["padding"]

        out_size = list(x.shape)
        out_size[1] = self.conv.out_channels
        if t_stride == 2:
            out_size[2] = (x.size(2) + 1) // 2
        output = torch.empty(tuple(out_size), dtype=x.dtype, device=x.device)

        # How many output frames come from the "padding-poisoned" part
        offset_out = math.ceil(padding.size(2) / t_stride)
        offset_in = offset_out * t_stride - padding.size(2)

        if offset_out > 0:
            padding_poisoned = torch.cat(
                [padding, x_parallel[:, :, : offset_in + self.time_kernel_size - t_stride]],
                dim=2,
            )
            output[:, :, :offset_out] = self.conv(padding_poisoned)

        if offset_out < output.size(2):
            output[:, :, offset_out:] = self.conv(x_parallel[:, :, offset_in:])

        # compute what part of x_parallel becomes "padding" for the next call
        pad_offset = (
            offset_in
            + t_stride * math.trunc((x_parallel.size(2) - offset_in - self.time_kernel_size) / t_stride)
            + t_stride
        )
        cache["padding"] = torch.clone(x_parallel[:, :, pad_offset:])
        return output


class CachedCausalResnetBlock3D(nn.Module):
    """
    3D ResNet block with optional conditional norm on latent z (zq),
    causal convolutions, and cache-aware streaming.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
        gather_norm: bool = False,
        normalization: Any = Normalize,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalization(in_channels, zq_ch=zq_ch, add_conv=add_conv)
        self.conv1 = CachedCausalConv3d(
            chan_in=in_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = normalization(out_channels, zq_ch=zq_ch, add_conv=add_conv)
        self.conv2 = CachedCausalConv3d(
            chan_in=out_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CachedCausalConv3d(
                    chan_in=in_channels,
                    chan_out=out_channels,
                    kernel_size=3,
                )
            else:
                self.nin_shortcut = SafeConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor],
        layer_cache: Dict[str, Any],
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Special-case hooks for large 17x1080 inputs (as in original code)
        def maybe_clear_cache():
            if x.size(2) == 17 and x.size(3) == 1080 and zq is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        h = x

        if zq is None:
            h = self.norm1(h, cache=layer_cache["norm1"])
        else:
            h = self.norm1(h, zq, cache=layer_cache["norm1"])

        maybe_clear_cache()

        # They use SiLU instead of swish here
        h = F.silu(h, inplace=True)
        maybe_clear_cache()

        h = self.conv1(h, cache=layer_cache["conv1"])
        maybe_clear_cache()

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        if zq is None:
            h = self.norm2(h, cache=layer_cache["norm2"])
        else:
            h = self.norm2(h, zq, cache=layer_cache["norm2"])

        h = F.silu(h, inplace=True)
        h = self.conv2(h, cache=layer_cache["conv2"])

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, cache=layer_cache["conv_shortcut"])
            else:
                x = self.nin_shortcut(x)

        return x + h


# ---------------------------------------------------------------------------
# PixelShuffle-like temporal+spatial (down/up) sampling with caching
# ---------------------------------------------------------------------------

class CachedPXSDownsample(nn.Module):
    """
    Downsample spatially via pixel unshuffle + conv, and optionally
    temporally via causal conv + average pooling, with caching.
    """

    def __init__(self, in_channels: int, compress_time: bool, factor: int = 2):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)

        out_channels = in_channels

        self.spatial_conv = SafeConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            padding_mode="reflect",
        )

        if self.temporal_compress:
            self.temporal_conv = CachedCausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                dilation=(1, 1, 1),
            )

        self.linear = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )

    def spatial_downsample(self, x: torch.Tensor) -> torch.Tensor:
        # Pixel unshuffle + averaging across blocks
        pxs_input = rearrange(x, "b c t h w -> (b t) c h w")
        pxs_interm = self.unshuffle(pxs_input)
        b, c, h, w = pxs_interm.shape
        pxs_interm_view = pxs_interm.view(b, c // self.factor**2, self.factor**2, h, w)
        pxs_out = torch.mean(pxs_interm_view, dim=2)
        pxs_out = rearrange(pxs_out, "(b t) c h w -> b c t h w", t=x.size(2))

        conv_out = self.spatial_conv(x)
        return conv_out + pxs_out

    def temporal_downsample(self, x: torch.Tensor, cache: Sequence[Dict[str, Any]]) -> torch.Tensor:
        # Interpolation part (avg_pool) + causal conv with cache
        permuted = rearrange(x, "b c t h w -> (b h w) c t")
        if cache[0]["padding"] is None:
            first, rest = permuted[..., :1], permuted[..., 1:]
            if rest.size(-1) > 0:
                rest_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)
                full_interp = torch.cat([first, rest_interp], dim=-1)
            else:
                full_interp = first
        else:
            rest = permuted
            if rest.size(-1) > 0:
                full_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)

        full_interp = rearrange(full_interp, "(b h w) c t -> b c t h w", h=x.size(-2), w=x.size(-1))
        conv_out = self.temporal_conv(x, cache[0])
        return conv_out + full_interp

    def forward(self, x: torch.Tensor, cache: Sequence[Dict[str, Any]]) -> torch.Tensor:
        out = self.spatial_downsample(x)
        if self.temporal_compress:
            out = self.temporal_downsample(out, cache=cache)
        return self.linear(out)


class CachedSpatialNorm3D(nn.Module):
    """
    Conditional spatial norm in 3D, modulated by latent zq.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        freeze_norm_layer: bool = False,
        add_conv: bool = False,
        pad_mode: str = "constant",
        normalization: Any = Normalize,
        **norm_layer_params: Any,
    ):
        super().__init__()
        self.norm_layer = normalization(in_channels=f_channels, **norm_layer_params)
        self.add_conv = add_conv
        if add_conv:
            self.conv = CachedCausalConv3d(
                chan_in=zq_channels,
                chan_out=zq_channels,
                kernel_size=3,
            )

        self.conv_y = SafeConv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
        )
        self.conv_b = SafeConv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
        )

    def forward(self, f: torch.Tensor, zq: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
        if cache["norm"]["mean"] is None and cache["norm"]["var"] is None:
            # First call: special handling for f/zq split
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]

            zq_first = F.interpolate(zq_first, size=f_first_size, mode="nearest")

            if zq.size(2) > 1:
                zq_rest_splits = torch.split(zq_rest, 32, dim=1)
                interpolated_splits = [
                    F.interpolate(split, size=f_rest_size, mode="nearest") for split in zq_rest_splits
                ]
                zq_rest = torch.cat(interpolated_splits, dim=1)
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                zq = zq_first
        else:
            f_size = f.shape[-3:]
            zq_splits = torch.split(zq, 32, dim=1)
            interpolated_splits = [F.interpolate(split, size=f_size, mode="nearest") for split in zq_splits]
            zq = torch.cat(interpolated_splits, dim=1)

        if self.add_conv:
            zq = self.conv(zq, cache["add_conv"])

        norm_f = self.norm_layer(f, cache["norm"])
        norm_f.mul_(self.conv_y(zq))
        norm_f.add_(self.conv_b(zq))

        if cache["norm"]["mean"] is None and cache["norm"]["var"] is None:
            cache["norm"]["mean"] = 1
            cache["norm"]["var"] = 1

        return norm_f


def Normalize3D(
    in_channels: int,
    zq_ch: int,
    add_conv: bool,
    normalization: Any = Normalize,
) -> nn.Module:
    return CachedSpatialNorm3D(
        f_channels=in_channels,
        zq_channels=zq_ch,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True,
        normalization=normalization,
    )


class PXSUpsample(nn.Module):
    """
    Upsample in space (via pixel shuffle + conv) and optionally in time.
    Base (non-cached) implementation.
    """

    def __init__(self, in_channels: int, compress_time: bool, factor: int = 2):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = SafeConv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="reflect",
        )

        if self.temporal_compress:
            self.temporal_conv = SafeConv3d(
                in_channels,
                in_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
            )

        self.linear = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
        )

    def spatial_upsample(self, x: torch.Tensor) -> torch.Tensor:
        def conv_part(v: torch.Tensor) -> torch.Tensor:
            to = torch.empty_like(v)
            return self.spatial_conv(v, write_to=to)

        b, t, c, h, w = x.shape  # semantics: actually x is [B, C, T, H, W], but t*c == C*T
        x_view = x.view(b, t * c, h, w)

        x_interp = F.interpolate(x_view, scale_factor=2, mode="nearest")
        x_interp = x_interp.view(b, t, c, 2 * h, 2 * w)
        x_interp.add_(conv_part(x_interp))
        return x_interp

    def temporal_upsample(self, x: torch.Tensor) -> torch.Tensor:
        time_factor = 1.0 + 1.0 * (x.size(2) > 1)
        if isinstance(time_factor, torch.Tensor):
            time_factor = time_factor.item()

        repeated = x.repeat_interleave(int(time_factor), dim=2)
        tail = repeated[..., int(time_factor - 1) :, :, :]

        padding_3d = (0, 0, 0, 0, 2, 0)
        tail_pad = F.pad(tail, padding_3d, mode="replicate")
        conv_out = self.temporal_conv(tail_pad)
        return conv_out + tail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_compress:
            x = self.temporal_upsample(x)
        s_out = self.spatial_upsample(x)
        return self.linear(s_out)


class CachedPXSUpsample(PXSUpsample):
    """
    Upsample with temporal caching compatible with CachedCausalConv3d.
    """

    def __init__(self, in_channels: int, compress_time: bool, factor: int = 2):
        super().__init__(in_channels, compress_time, factor)
        self.temporal_compress = compress_time
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = SafeConv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="reflect",
        )

        if self.temporal_compress:
            self.temporal_conv = CachedCausalConv3d(
                in_channels,
                in_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
            )

        self.linear = SafeConv3d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
        )

    def temporal_upsample(self, x: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
        time_factor = 1.0 + 1.0 * (x.size(2) > 1)
        if isinstance(time_factor, torch.Tensor):
            time_factor = time_factor.item()

        repeated = x.repeat_interleave(int(time_factor), dim=2)

        if cache["padding"] is None:
            tail = repeated[..., int(time_factor - 1) :, :, :]
        else:
            tail = repeated

        conv_out = self.temporal_conv(tail, cache)
        return conv_out + tail

    def forward(self, x: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
        if self.temporal_compress:
            x = self.temporal_upsample(x, cache)

        s_out = self.spatial_upsample(x)
        to = torch.empty_like(s_out)
        return self.linear(s_out, write_to=to)


# ---------------------------------------------------------------------------
# Encoder & decoder
# ---------------------------------------------------------------------------

class CachedEncoder3D(nn.Module):
    """
    KVAE-3D encoder with temporal compression and cached causal convolutions.
    Input shape: [B, C_in, T, H, W]
    Output shape: [B, 2 * z_channels, T', H', W'] (mean + logvar packed)
    """

    def __init__(
        self,
        *,
        ch: int = 128,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        in_channels: int = 3,
        resolution: int = 0,
        z_channels: int = 16,
        double_z: bool = True,
        temporal_compress_times: int = 4,
        gather_norm: bool = False,
        norm_type: str = "group_norm",
        **ignore_kwargs: Any,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # log2 of temporal_compress_times
        self.temporal_compress_level = int(math.log2(temporal_compress_times))

        self.conv_in = CachedCausalConv3d(
            chan_in=in_channels,
            chan_out=self.ch,
            kernel_size=3,
        )

        normalization = Normalize if norm_type == "group_norm" else RMSNorm

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()  # reserved for future attention blocks

            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        temb_channels=self.temb_ch,
                        gather_norm=gather_norm,
                        normalization=normalization,
                    )
                )
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level < self.temporal_compress_level:
                    down.downsample = CachedPXSDownsample(block_in, compress_time=True)
                else:
                    down.downsample = CachedPXSDownsample(block_in, compress_time=False)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            gather_norm=gather_norm,
            normalization=normalization,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            gather_norm=gather_norm,
            normalization=normalization,
        )

        # end
        self.norm_out = normalization(block_in, gather=gather_norm)
        self.conv_out = CachedCausalConv3d(
            chan_in=block_in,
            chan_out=2 * z_channels if double_z else z_channels,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor, cache_dict: Dict[str, Any], use_cp: bool = True) -> torch.Tensor:
        temb = None

        h = self.conv_in(x, cache=cache_dict["conv_in"])
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb, layer_cache=cache_dict[i_level][i_block])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h, cache=cache_dict[i_level]["down"])

        h = self.mid.block_1(h, temb, layer_cache=cache_dict["mid_1"])
        h = self.mid.block_2(h, temb, layer_cache=cache_dict["mid_2"])

        h = self.norm_out(h, cache=cache_dict["norm_out"])
        h = nonlinearity(h)
        h = self.conv_out(h, cache=cache_dict["conv_out"])
        return h

    def get_last_layer(self) -> torch.Tensor:
        return self.conv_out.conv.weight


class CachedDecoder3D(nn.Module):
    """
    KVAE-3D decoder with temporal upsampling and cached causal convolutions.
    Input shape: [B, z_channels, T_latent, H_latent, W_latent]
    Output shape: [B, out_ch, T, H, W]
    """

    def __init__(
        self,
        ch: int = 128,
        out_ch: Optional[int] = None,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        resolution: int = 0,
        z_channels: int = 16,
        give_pre_end: bool = False,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
        pad_mode: str = "first",
        temporal_compress_times: int = 4,
        gather_norm: bool = False,
        norm_type: str = "group_norm",
        **kwargs: Any,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end

        self.temporal_compress_level = int(math.log2(temporal_compress_times))
        if zq_ch is None:
            zq_ch = z_channels

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = CachedCausalConv3d(
            chan_in=z_channels,
            chan_out=block_in,
            kernel_size=3,
        )

        base_norm = Normalize if norm_type == "group_norm" else RMSNorm
        modulated_norm = lambda *a, **kw: Normalize3D(*a, normalization=base_norm, **kw)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=modulated_norm,
            gather_norm=gather_norm,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=modulated_norm,
            gather_norm=gather_norm,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        normalization=modulated_norm,
                        gather_norm=gather_norm,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = CachedPXSUpsample(block_in, compress_time=False)
                else:
                    up.upsample = CachedPXSUpsample(block_in, compress_time=True)
            self.up.insert(0, up)

        self.norm_out = modulated_norm(block_in, zq_ch, add_conv=add_conv)
        self.conv_out = CachedCausalConv3d(
            chan_in=block_in,
            chan_out=out_ch,
            kernel_size=3,
        )

    def forward(self, z: torch.Tensor, cache_dict: Dict[str, Any]) -> torch.Tensor:
        self.last_z_shape = z.shape
        temb = None

        zq = z
        h = self.conv_in(z, cache_dict["conv_in"])

        h = self.mid.block_1(h, temb, layer_cache=cache_dict["mid_1"], zq=zq)
        h = self.mid.block_2(h, temb, layer_cache=cache_dict["mid_2"], zq=zq)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    h,
                    temb,
                    layer_cache=cache_dict[i_level][i_block],
                    zq=zq,
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h, cache_dict[i_level]["up"])

        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq, cache_dict["norm_out"])
        h = nonlinearity(h)
        h = self.conv_out(h, cache_dict["conv_out"])
        return h

    def get_last_layer(self) -> torch.Tensor:
        return self.conv_out.conv.weight


# ---------------------------------------------------------------------------
# High-level VAE wrapper (KVAE-3D)
# ---------------------------------------------------------------------------

class KVAE3D(nn.Module):
    """
    KVAE-3D VAE: cached causal encoder & decoder wiring plus
    helper encode/decode methods.

    Input / output:
      - encode(x, seg_len): x [B, 3, T, H, W] -> (z, split_list, params)
      - decode(z, split_list): -> recon [B, 3, T, H, W]
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config: Dict[str, Any] = dict(config)
        self.conf = {
            "enc": dict(config["model"]["encoder_params"]),
            "dec": dict(config["model"]["decoder_params"]),
        }
        self.encoder = CachedEncoder3D(**self.conf["enc"])
        self.decoder = CachedDecoder3D(**self.conf["dec"])

    # -- cache creation -----------------------------------------------------

    def _make_empty_cache(self, block: str) -> Dict[str, Any]:
        conf = self.conf[block]

        def make_dict(name: str, p: Optional[int] = None) -> Any:
            if name == "conv":
                return {"padding": None}

            layer, module = name.split("_", 1)
            if layer == "norm":
                if module == "enc":
                    return {"mean": None, "var": None}
                else:
                    return {"norm": make_dict("norm_enc"), "add_conv": make_dict("conv")}
            elif layer == "resblock":
                return {
                    "norm1": make_dict(f"norm_{module}"),
                    "norm2": make_dict(f"norm_{module}"),
                    "conv1": make_dict("conv"),
                    "conv2": make_dict("conv"),
                    "conv_shortcut": make_dict("conv"),
                }
            elif layer.isdigit():
                out_dict = {"down": [make_dict("conv"), make_dict("conv")], "up": make_dict("conv")}
                assert p is not None
                for i in range(p):
                    out_dict[i] = make_dict(f"resblock_{module}")
                return out_dict
            else:
                raise ValueError(f"Unexpected cache name: {name}")

        cache: Dict[str, Any] = {
            "conv_in": make_dict("conv"),
            "mid_1": make_dict(f"resblock_{block}"),
            "mid_2": make_dict(f"resblock_{block}"),
            "norm_out": make_dict(f"norm_{block}"),
            "conv_out": make_dict("conv"),
        }
        for i in range(len(conf.get("ch_mult", (1, 2, 4, 8)))):
            cache[i] = make_dict(f"{i}_block", p=conf["num_res_blocks"] + 1)
        return cache

    # -- encode / decode ----------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        seg_len: int = 16,
    ) -> Tuple[torch.Tensor, Sequence[int], Sequence[torch.Tensor]]:
        """
        Encode a video batch.

        Parameters
        ----------
        x : [B, 3, T, H, W] tensor in the config's `input_norm` space (m11 by default)
        seg_len : segment length for streaming encode

        Returns
        -------
        latent : [B, z_channels, T_latent, H_latent, W_latent]
        split_list : list of segment lengths on the *input* time axis
        params : list of raw encoder outputs (mean/logvar packed) per segment
        """
        cache = self._make_empty_cache("enc")

        # Build segment sizes in time
        split_list: list[int] = [seg_len + 1]
        n_frames = x.size(2) - (seg_len + 1)
        while n_frames > 0:
            split_list.append(seg_len)
            n_frames -= seg_len
        split_list[-1] += n_frames  # adjust last if n_frames is negative

        latent_chunks: list[torch.Tensor] = []
        params_chunks: list[torch.Tensor] = []

        for chunk in torch.split(x, split_list, dim=2):
            l = self.encoder(chunk, cache)
            sample, _ = torch.chunk(l, 2, dim=1)
            latent_chunks.append(sample)
            params_chunks.append(l)

        latent = torch.cat(latent_chunks, dim=2)
        return latent, split_list, params_chunks

    @torch.no_grad()
    def decode(self, z: torch.Tensor, split_list: Sequence[int]) -> torch.Tensor:
        """
        Decode latent video representations back to pixel space.

        Parameters
        ----------
        z : [B, z_channels, T_latent, H_latent, W_latent]
        split_list : original temporal segment lengths from encode()

        Returns
        -------
        recs : [B, 3, T, H, W]
        """
        cache = self._make_empty_cache("dec")

        # Adjust segment sizes for latent time axis
        temporal_factor = self.conf["enc"]["temporal_compress_times"]
        latent_split = [math.ceil(size / temporal_factor) for size in split_list]

        rec_chunks: list[torch.Tensor] = []
        for chunk in torch.split(z, latent_split, dim=2):
            out = self.decoder(chunk, cache)
            rec_chunks.append(out)

        recs = torch.cat(rec_chunks, dim=2)
        return recs

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seg_len: int = 16) -> torch.Tensor:
        latent, split_list, _ = self.encode(x, seg_len=seg_len)
        return self.decode(latent, split_list)


# ---------------------------------------------------------------------------
# Loading from local safetensors
# ---------------------------------------------------------------------------

def _load_state_dict_flex(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """
    Try to load a state dict, optionally stripping common prefixes
    like 'model.' or 'module.' if necessary.
    """
    try:
        model.load_state_dict(state_dict, strict=strict)
        return
    except RuntimeError:
        prefixes = ["model.", "module.", "vae."]
        for prefix in prefixes:
            fixed = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in state_dict.items()
            }
            try:
                model.load_state_dict(fixed, strict=strict)
                return
            except RuntimeError:
                continue
        # If we reach here, re-raise the original error
        model.load_state_dict(state_dict, strict=strict)


def _normalize_config(config: Union[Mapping[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(config)
    cfg.setdefault("common", {})
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg["model"].setdefault("encoder_params", {})
    cfg["model"].setdefault("decoder_params", {})

    # Fill in defaults if missing
    enc = cfg["model"]["encoder_params"]
    dec = cfg["model"]["decoder_params"]

    enc.setdefault("ch", 128)
    enc.setdefault("in_channels", 3)
    enc.setdefault("num_res_blocks", 2)
    enc.setdefault("temporal_compress_times", 4)
    enc.setdefault("z_channels", 16)

    dec.setdefault("ch", 128)
    dec.setdefault("num_res_blocks", 2)
    dec.setdefault("out_ch", 3)
    dec.setdefault("temporal_compress_times", 4)
    dec.setdefault("z_channels", 16)

    return cfg


def load_kvae3d_config(config: Optional[Union[str, Path, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """
    Load KVAE-3D config from:
      - a dict-like object,
      - a JSON file path,
      - or fall back to DEFAULT_KVAE3D_CONFIG.
    """
    if config is None:
        return _normalize_config(DEFAULT_KVAE3D_CONFIG)

    if isinstance(config, (str, Path)):
        config_path = Path(config)
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return _normalize_config(cfg)

    return _normalize_config(config)


def load_kvae3d(
    weights_path: Optional[Union[str, Path]] = None,
    config: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    strict: bool = True,
) -> KVAE3D:
    """
    Build and load a KVAE-3D model from local safetensors weights.

    Parameters
    ----------
    weights_path:
        Path to the .safetensors file. If None, defaults to:
        <repo_root>/vae/kvae_3d_2025.safetensors
        where repo_root is the parent of the 'models' directory.
    config:
        KVAE-3D config as dict, path to JSON, or None to use DEFAULT_KVAE3D_CONFIG.
    device:
        Torch device (e.g. "cuda", "cuda:0", "cpu"). If None, chooses
        "cuda" if available else "cpu".
    dtype:
        Torch dtype for model weights. If None, defaults to torch.bfloat16
        on CUDA and torch.float32 on CPU.
    strict:
        Whether to enforce strict matching of the state dict.

    Returns
    -------
    KVAE3D
        Model ready for encode/decode calls.
    """
    cfg = load_kvae3d_config(config)

    if weights_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        weights_path = repo_root / "vae" / "kvae_3d_2025.safetensors"
    weights_path = Path(weights_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if dtype is None:
        if device.type == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

    # Build model on CPU first to avoid unnecessary GPU allocation
    model = KVAE3D(cfg)
    state_dict = load_safetensors_file(str(weights_path), device="cpu")
    _load_state_dict_flex(model, state_dict, strict=strict)

    model.to(device=device, dtype=dtype)
    model.eval()
    return model
