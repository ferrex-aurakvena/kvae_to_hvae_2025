# latent_io.py

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save as safetensors_save, load as safetensors_load
import zstandard as zstd


def save_latent_pair_zstd(path: str | Path, z_h: torch.Tensor, z_k: torch.Tensor) -> None:
    """
    Save a pair of latents (Hunyuan + KVAE) into a single .safetensors.zst file.

    Expects tensors shaped [C, T', H', W'].
    Stores them as FP16 on CPU, packed via safetensors, then zstd-compressed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "z_h": z_h.detach().to("cpu", dtype=torch.float16).contiguous(),
        "z_k": z_k.detach().to("cpu", dtype=torch.float16).contiguous(),
    }

    # safetensors -> bytes in memory
    st_bytes: bytes = safetensors_save(state)

    # zstd compress
    cctx = zstd.ZstdCompressor(level=6)
    compressed = cctx.compress(st_bytes)

    with path.open("wb") as f:
        f.write(compressed)


def load_latent_pair_zstd(path: str | Path, device: str | torch.device = "cpu"):
    """
    Load a .safetensors.zst latent pair and return (z_h, z_k) as float32 on device.

    Shapes: [C, T', H', W'] each.
    """
    path = Path(path)
    with path.open("rb") as f:
        compressed = f.read()

    dctx = zstd.ZstdDecompressor()
    st_bytes = dctx.decompress(compressed)

    state = safetensors_load(st_bytes)  # CPU tensors
    z_h = state["z_h"].to(device=device, dtype=torch.float32, non_blocking=True)
    z_k = state["z_k"].to(device=device, dtype=torch.float32, non_blocking=True)
    return z_h, z_k

