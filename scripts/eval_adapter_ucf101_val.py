#!/usr/bin/env python
# scripts/eval_adapter_ucf101_val.py

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F_torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import decord
from torchvision.transforms import functional as TF
from safetensors.torch import load_file as load_safetensors

# -------------------------------------------------------------------------
# Make repo root importable
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from latent_io import load_latent_pair_zstd
from models.adapter import HunyuanToKVAEAdapter
from models.vae_wrappers import (
    load_hunyuan_vae_from_local,
    load_kvae3d_from_local,
)

VIDEO_EXTS = [".avi", ".mp4", ".mkv", ".mov", ".webm"]


# -------------------------------------------------------------------------
# Dataset over .safetensors.zst pairs
# -------------------------------------------------------------------------
class ZstdLatentPairDataset(Dataset):
    """
    Dataset over .safetensors.zst latent pairs: each item returns (z_h, z_k)
    shaped [C, T', H', W'] as float32 on CPU.
    """

    def __init__(self, latent_root: str | Path):
        self.latent_root = Path(latent_root)
        self.files: List[Path] = sorted(self.latent_root.rglob("*.safetensors.zst"))
        if not self.files:
            raise RuntimeError(f"No .safetensors.zst files under {latent_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        # returns float32 on CPU
        z_h, z_k = load_latent_pair_zstd(path, device="cpu")
        return z_h, z_k


# -------------------------------------------------------------------------
# Video loader (same semantics as extraction script)
# -------------------------------------------------------------------------
def load_video_tensor(
    path: Path,
    num_frames: int | None,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load a video with decord and return [1, 3, T, H, W] in float32 on device,
    normalized to [-1, 1].

    This mirrors the behavior used in latent extraction:
      - uniform sampling of `num_frames`
      - resizing to (height, width)
      - [-1,1] normalization
    """
    vr = decord.VideoReader(str(path))
    total = len(vr)
    if total == 0:
        raise ValueError(f"Empty video: {path}")

    if (num_frames is None) or (total <= num_frames):
        indices = np.arange(total)
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=np.int64)

    batch = vr.get_batch(indices)  # [T, H, W, 3], uint8
    frames_np = batch.asnumpy()
    frames = torch.from_numpy(frames_np)  # [T, H, W, 3]

    # [T, 3, H, W]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0

    # resize to target size
    frames = TF.resize(frames, [height, width], antialias=True)

    # [-1, 1]
    frames = frames * 2.0 - 1.0

    # [1, 3, T, H, W]
    video = frames.permute(1, 0, 2, 3).unsqueeze(0)
    return video.to(device=device, dtype=torch.float32, non_blocking=True)


# -------------------------------------------------------------------------
# PSNR for [-1,1] ("m11") range
# -------------------------------------------------------------------------
def psnr_m11(src: torch.Tensor, dist: torch.Tensor) -> float:
    """
    PSNR in dB for tensors in [-1, 1].

    Both src and dist: [B, C, T, H, W]
    """
    src01 = (src + 1.0) / 2.0
    dist01 = (dist + 1.0) / 2.0

    src01 = src01.clamp(0.0, 1.0)
    dist01 = dist01.clamp(0.0, 1.0)

    mse = F_torch.mse_loss(src01, dist01)
    if mse <= 0:
        return float("inf")
    return float(-10.0 * torch.log10(mse).item())


# -------------------------------------------------------------------------
# Latent-space evaluation
# -------------------------------------------------------------------------
@torch.no_grad()
def evaluate_latent_metrics(
    dataloader: DataLoader,
    adapter_h2k: torch.nn.Module,
    adapter_k2h: torch.nn.Module,
    device: torch.device,
    max_samples: int,
) -> dict:
    adapter_h2k.eval()
    adapter_k2h.eval()

    total = 0
    sum_h2k = 0.0
    sum_k2h = 0.0
    sum_cycle_h = 0.0
    sum_cycle_k = 0.0

    for z_h, z_k in tqdm(dataloader, desc="Latent eval", leave=False):
        b = z_h.size(0)
        if total >= max_samples:
            break
        if total + b > max_samples:
            # trim last batch
            trim = max_samples - total
            z_h = z_h[:trim]
            z_k = z_k[:trim]
            b = trim

        z_h = z_h.to(device=device, dtype=torch.float32, non_blocking=True)  # [B, C, T, H, W]
        z_k = z_k.to(device=device, dtype=torch.float32, non_blocking=True)

        # direct mappings
        z_h2k = adapter_h2k(z_h)
        z_k2h = adapter_k2h(z_k)

        # direct MSEs
        mse_h2k = F_torch.mse_loss(z_h2k, z_k, reduction="mean").item()
        mse_k2h = F_torch.mse_loss(z_k2h, z_h, reduction="mean").item()

        # cycles
        z_h_cycle = adapter_k2h(z_h2k)  # H -> K -> H
        z_k_cycle = adapter_h2k(z_k2h)  # K -> H -> K

        mse_cycle_h = F_torch.mse_loss(z_h_cycle, z_h, reduction="mean").item()
        mse_cycle_k = F_torch.mse_loss(z_k_cycle, z_k, reduction="mean").item()

        total += b
        sum_h2k += mse_h2k * b
        sum_k2h += mse_k2h * b
        sum_cycle_h += mse_cycle_h * b
        sum_cycle_k += mse_cycle_k * b

    if total == 0:
        raise RuntimeError("No samples were evaluated in latent metrics.")

    return {
        "num_samples": total,
        "mse_h2k": sum_h2k / total,
        "mse_k2h": sum_k2h / total,
        "mse_cycle_h": sum_cycle_h / total,
        "mse_cycle_k": sum_cycle_k / total,
    }


# -------------------------------------------------------------------------
# Pixel-space evaluation
# -------------------------------------------------------------------------
@torch.no_grad()
def evaluate_pixel_metrics(
    latent_root: Path,
    videos_root: Path,
    adapter_h2k: torch.nn.Module,
    adapter_k2h: torch.nn.Module,
    hvae: torch.nn.Module,
    kvae: torch.nn.Module,
    num_pixel_samples: int,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> dict:
    """
    For a subset of samples, compare decodes against original-preprocessed video:

      - PSNR(orig, Hunyuan(z_h))
      - PSNR(orig, KVAE(z_k))
      - PSNR(orig, KVAE(H→K(z_h)))
      - PSNR(orig, Hunyuan(K→H(z_k)))

    Assumes the latent directory mirrors the video directory structure, i.e.:

      videos_root / rel.with_suffix(".avi")
      latent_root / rel.with_suffix(".safetensors.zst")
    """
    adapter_h2k.eval()
    adapter_k2h.eval()
    hvae.eval()
    kvae.eval()

    ds = ZstdLatentPairDataset(latent_root)
    files = ds.files

    # Get model dtypes for later
    h_dtype = next(hvae.parameters()).dtype
    k_dtype = next(kvae.parameters()).dtype

    # KVAE decode uses split_list; for 33-frame clips and seg_len=16 we know it is [17, 16].
    # (This matches KVAE3D.encode's splitting for T=33, seg_len=16.)
    kvae_split_list: List[int] = [17, 16]

    n = 0
    sum_psnr_orig_h = 0.0
    sum_psnr_orig_k = 0.0
    sum_psnr_orig_h2k2k = 0.0
    sum_psnr_orig_k2h2h = 0.0

    for idx, latent_path in enumerate(tqdm(files, desc="Pixel eval", leave=False)):
        if n >= num_pixel_samples:
            break

        # map latent path -> video path
        rel = latent_path.relative_to(latent_root)
        video_path: Path | None = None

        for ext in VIDEO_EXTS:
            candidate = videos_root / rel.with_suffix(ext)
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            print(f"[WARN] No video file found for latent {latent_path}, skipping.")
            continue

        # load latents
        z_h, z_k = load_latent_pair_zstd(latent_path, device=device)  # [C, T', H', W'], float32
        z_h = z_h.unsqueeze(0)  # [1, C, T', H', W']
        z_k = z_k.unsqueeze(0)

        # load original preprocessed clip: [1, 3, T, H, W] in [-1, 1]
        try:
            x_orig = load_video_tensor(
                video_path,
                num_frames=num_frames,
                height=height,
                width=width,
                device=device,
            )
        except Exception as e:
            print(f"[WARN] Failed to load video {video_path}: {e}")
            continue

        # ensure float32 for PSNR later
        x_orig_f32 = x_orig.to(dtype=torch.float32)

        # --- decode paths ---

        # base Hunyuan decode from z_h
        z_h_for_h = z_h.to(dtype=h_dtype)
        out_h = hvae.decode(z_h_for_h)
        x_h = out_h.sample if hasattr(out_h, "sample") else out_h
        x_h_f32 = x_h.to(dtype=torch.float32)

        # base KVAE decode from z_k
        z_k_for_k = z_k.to(dtype=k_dtype)
        x_k = kvae.decode(z_k_for_k, kvae_split_list)
        x_k_f32 = x_k.to(dtype=torch.float32)

        # Hunyuan -> KVAE path: z_h → adapter_h2k → KVAE decode
        z_h32 = z_h.to(dtype=torch.float32)
        z_h2k = adapter_h2k(z_h32)  # [1, 16, T', H', W']
        z_h2k_for_k = z_h2k.to(dtype=k_dtype)
        x_h2k2k = kvae.decode(z_h2k_for_k, kvae_split_list)
        x_h2k2k_f32 = x_h2k2k.to(dtype=torch.float32)

        # KVAE -> Hunyuan path: z_k → adapter_k2h → Hunyuan decode
        z_k32 = z_k.to(dtype=torch.float32)
        z_k2h = adapter_k2h(z_k32)  # [1, 16, T', H', W']
        z_k2h_for_h = z_k2h.to(dtype=h_dtype)
        out_k2h = hvae.decode(z_k2h_for_h)
        x_k2h2h = out_k2h.sample if hasattr(out_k2h, "sample") else out_k2h
        x_k2h2h_f32 = x_k2h2h.to(dtype=torch.float32)

        # --- PSNRs ---
        psnr_orig_h = psnr_m11(x_orig_f32, x_h_f32)
        psnr_orig_k = psnr_m11(x_orig_f32, x_k_f32)
        psnr_orig_h2k2k = psnr_m11(x_orig_f32, x_h2k2k_f32)
        psnr_orig_k2h2h = psnr_m11(x_orig_f32, x_k2h2h_f32)

        n += 1
        sum_psnr_orig_h += psnr_orig_h
        sum_psnr_orig_k += psnr_orig_k
        sum_psnr_orig_h2k2k += psnr_orig_h2k2k
        sum_psnr_orig_k2h2h += psnr_orig_k2h2h

    if n == 0:
        print("[WARN] No pixel samples evaluated (no matching videos found?).")
        return {
            "num_samples": 0,
            "psnr_orig_h": float("nan"),
            "psnr_orig_k": float("nan"),
            "psnr_orig_h2k2k": float("nan"),
            "psnr_orig_k2h2h": float("nan"),
        }

    return {
        "num_samples": n,
        "psnr_orig_h": sum_psnr_orig_h / n,
        "psnr_orig_k": sum_psnr_orig_k / n,
        "psnr_orig_h2k2k": sum_psnr_orig_h2k2k / n,
        "psnr_orig_k2h2h": sum_psnr_orig_k2h2h / n,
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hunyuan↔KVAE latent adapters on UCF101 val latents."
    )
    parser.add_argument(
        "--latent-root",
        type=str,
        default="datasets/UCF101_val_latents_33",
        help="Root directory with *.safetensors.zst latent pairs (val).",
    )
    parser.add_argument(
        "--videos-root",
        type=str,
        default="datasets/UCF101/val",
        help="Root directory with original UCF101 val videos. "
        "If not set or missing, pixel metrics are skipped.",
    )
    parser.add_argument(
        "--adapter-h2k",
        type=str,
        default="adapter_hunyuan_to_kvae3d.safetensors",
        help="Path to Hunyuan→KVAE adapter weights.",
    )
    parser.add_argument(
        "--adapter-k2h",
        type=str,
        default="adapter_kvae3d_to_hunyuan.safetensors",
        help="Path to KVAE→Hunyuan adapter weights.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for latent-space evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max number of latent samples for latent-space metrics.",
    )
    parser.add_argument(
        "--num-pixel-samples",
        type=int,
        default=128,
        help="Number of samples for pixel-space metrics (will be a subset).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=33,
        help="Number of frames to sample per video for pixel eval (must match extraction).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Target height for video preprocessing (must match extraction).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Target width for video preprocessing (must match extraction).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for evaluation (e.g. cuda, cuda:0, cpu).",
    )
    args = parser.parse_args()

    latent_root = Path(args.latent_root)
    videos_root = Path(args.videos_root)
    device = torch.device(args.device)

    # ------------------ load dataset & adapters ------------------
    ds = ZstdLatentPairDataset(latent_root)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Hunyuan -> KVAE adapter
    adapter_h2k = HunyuanToKVAEAdapter(channels=16, num_blocks=3).to(device)
    state_h2k = load_safetensors(args.adapter_h2k)
    adapter_h2k.load_state_dict(state_h2k)
    adapter_h2k.eval()

    # KVAE -> Hunyuan adapter
    adapter_k2h = HunyuanToKVAEAdapter(channels=16, num_blocks=3).to(device)
    state_k2h = load_safetensors(args.adapter_k2h)
    adapter_k2h.load_state_dict(state_k2h)
    adapter_k2h.eval()

    print(f"Loaded {len(ds)} latent pairs from {latent_root}")
    print("Evaluating latent-space metrics...")

    latent_stats = evaluate_latent_metrics(
        dl,
        adapter_h2k,
        adapter_k2h,
        device=device,
        max_samples=args.max_samples,
    )

    print("\n=== Latent-space metrics ===")
    print(f"  Samples:                 {latent_stats['num_samples']}")
    print(f"  H→K MSE:                 {latent_stats['mse_h2k']:.6f}")
    print(f"  K→H MSE:                 {latent_stats['mse_k2h']:.6f}")
    print(f"  H cycle (H→K→H) MSE:     {latent_stats['mse_cycle_h']:.6f}")
    print(f"  K cycle (K→H→K) MSE:     {latent_stats['mse_cycle_k']:.6f}")

    # ------------------ pixel-space metrics (optional) ------------------
    do_pixel = videos_root.exists()
    if not do_pixel:
        print(f"\n[Info] videos_root={videos_root} does not exist; skipping pixel metrics.")
        return

    print("\nLoading VAEs for pixel-space evaluation...")
    hvae = load_hunyuan_vae_from_local(device=device)
    kvae = load_kvae3d_from_local(device=device)

    print(
        f"Evaluating pixel-space PSNR on up to {args.num_pixel_samples} samples "
        f"from {videos_root}..."
    )

    pixel_stats = evaluate_pixel_metrics(
        latent_root=latent_root,
        videos_root=videos_root,
        adapter_h2k=adapter_h2k,
        adapter_k2h=adapter_k2h,
        hvae=hvae,
        kvae=kvae,
        num_pixel_samples=args.num_pixel_samples,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        device=device,
    )

    print("\n=== Pixel-space metrics (PSNR, dB; m11→[0,1]) ===")
    print(f"  Samples:                         {pixel_stats['num_samples']}")
    if pixel_stats["num_samples"] > 0:
        print(f"  orig vs Hunyuan(z_h):            {pixel_stats['psnr_orig_h']:.2f}")
        print(f"  orig vs KVAE(z_k):               {pixel_stats['psnr_orig_k']:.2f}")
        print(f"  orig vs KVAE(H→K(z_h)):          {pixel_stats['psnr_orig_h2k2k']:.2f}")
        print(f"  orig vs Hunyuan(K→H(z_k)):       {pixel_stats['psnr_orig_k2h2h']:.2f}")
    else:
        print("  (no pixel samples evaluated)")


if __name__ == "__main__":
    main()
