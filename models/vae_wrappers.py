# models/vae_wrappers.py

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import nn
from diffusers import AutoencoderKLHunyuanVideo
from safetensors.torch import load_file as load_safetensors

from .kvae3d_loader import load_kvae3d as _load_kvae3d_core


def load_hunyuan_vae_from_local(
    vae_dir: str | Path = "vae",
    filename: str = "hvae_2025.safetensors",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """
    Load HunyuanVideo 3D VAE.

    - Architecture: diffusers.AutoencoderKLHunyuanVideo
    - Weights:
        1) Start from HF pretrained "hunyuanvideo-community/HunyuanVideo" (subfolder="vae")
        2) If vae/<filename> exists, load those weights on top (strict=False).

    Inputs for encode():
        video: [B, 3, T, H, W] in [-1, 1] ("m11" normalization).

    Latents:
        encode(video) -> object with `.latent_dist`; we typically use
        unscaled latents: z_h = latent_dist.sample().
    """
    device_t = torch.device(device)

    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae.to(device_t)
    vae.eval()

    local_path = Path(vae_dir) / filename
    if local_path.exists():
        print(f"[Hunyuan VAE] Loading local weights from {local_path}")
        state = load_safetensors(str(local_path))
        missing, unexpected = vae.load_state_dict(state, strict=False)
        if missing:
            print(f"[Hunyuan VAE] Missing keys in local state_dict: {missing}")
        if unexpected:
            print(f"[Hunyuan VAE] Unexpected keys in local state_dict: {unexpected}")
    else:
        print(
            f"[Hunyuan VAE] Local weights {local_path} not found. "
            "Using HF pretrained weights."
        )

    vae.requires_grad_(False)
    return vae


def load_kvae3d_from_local(
    vae_dir: str | Path = "vae",
    filename: str = "kvae_3d_2025.safetensors",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """
    Load KVAE-3D VAE using our local implementation in models.kvae3d_loader.

    - Architecture: models.kvae3d_loader.KVAE3D
    - Weights: vae/kvae_3d_2025.safetensors (your local copy of KVAE-3D-1.0)

    Inputs for encode():
        video: [B, 3, T, H, W] in [-1, 1] ("m11" normalization).

    Latents:
        encode(video, seg_len=16) -> (z_k, split_list, params)
        where z_k has shape [B, 16, T', H', W'].
    """
    device_t = torch.device(device)
    weights_path = Path(vae_dir) / filename

    if not weights_path.exists():
        print(
            f"[KVAE-3D] WARNING: {weights_path} not found. "
            "Falling back to kvae3d_loader default path (repo_root/vae/kvae_3d_2025.safetensors)."
        )
        weights_path = None  # let load_kvae3d() resolve its default

    print(
        "[KVAE-3D] Loading KVAE-3D model from "
        f"{weights_path if weights_path is not None else 'default weights path'}"
    )

    model = _load_kvae3d_core(
        weights_path=weights_path,
        config=None,
        device=device_t,
        dtype=dtype,
        strict=True,
    )
    model.requires_grad_(False)
    return model

