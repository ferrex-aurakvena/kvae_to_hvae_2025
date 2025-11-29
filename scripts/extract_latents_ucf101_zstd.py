# scripts/extract_latents_ucf101_zstd.py

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
from typing import List

import numpy as np
import torch
from torchvision.transforms import functional as F
import decord
from tqdm import tqdm

# ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from latent_io import save_latent_pair_zstd
from models.vae_wrappers import load_hunyuan_vae_from_local, load_kvae3d_from_local

VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}


def find_video_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in VIDEO_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def load_video_tensor(
    path: Path,
    num_frames: int | None,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    temporal_divider: int | None = None,
) -> torch.Tensor:
    """
    Load a video with decord and return [1, 3, T, H, W] on device.

    - Frames are resized to (height, width).
    - Pixel range is normalized to [-1, 1] ("m11").
    - If temporal_divider is not None, we pad in time so that:
        T_out % temporal_divider == 1
      by duplicating the last frame.
    """
    vr = decord.VideoReader(str(path))
    total = len(vr)
    if total == 0:
        raise ValueError(f"Empty video: {path}")

    # Treat non-positive num_frames as "use all frames"
    if num_frames is not None and num_frames <= 0:
        num_frames = None

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
    frames = F.resize(frames, [height, width], antialias=True)

    # [-1, 1]
    frames = frames * 2.0 - 1.0  # "m11" normalization

    # Optionally ensure T % temporal_divider == 1 for KVAE-3D streaming
    if temporal_divider is not None and temporal_divider > 0:
        T = frames.shape[0]
        while T % temporal_divider != 1:
            frames = torch.cat([frames, frames[-1:].clone()], dim=0)
            T = frames.shape[0]

    # [1, 3, T, H, W]
    video = frames.permute(1, 0, 2, 3).unsqueeze(0)
    return video.to(device=device, dtype=dtype, non_blocking=True)


@torch.no_grad()
def encode_hunyuan(vae: torch.nn.Module, video: torch.Tensor) -> torch.Tensor:
    """
    Hunyuan encode to *unscaled* latents [1, 16, T', H', W'].
    We do NOT multiply by scaling_factor here.
    """
    out = vae.encode(video)
    if hasattr(out, "latent_dist"):
        z = out.latent_dist.sample()
    elif hasattr(out, "sample"):
        z = out.sample
    else:
        z = out
    return z  # [B, C, T', H', W']


@torch.no_grad()
def encode_kvae3d(kvae: torch.nn.Module, video: torch.Tensor, seg_len: int = 16) -> torch.Tensor:
    """
    KVAE-3D encode to latents [1, 16, T', H', W'] using our local KVAE3D API.

    The loader returns:
        latent, split_list, params

    We only need the latent (first element).
    """
    latent, split_list, _ = kvae.encode(video, seg_len=seg_len)
    return latent  # [B, C, T', H', W']


def main():
    parser = argparse.ArgumentParser(description="Extract Hunyuan + KVAE-3D latents from UCF101 videos.")
    parser.add_argument(
        "--videos-root",
        type=str,
        default="datasets/UCF101",
        help="Root with UCF101 videos (train/val/test subfolders).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="datasets/UCF101_latents_32",
        help="Where to store .safetensors.zst latent files.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=1000,
        help="Maximum number of videos to process (default: 1000).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames to sample per video (uniform). "
             "If <=0, use all frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Target height (UCF101 is 240x320; keep 240 by default).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Target width (UCF101 is 240x320; keep 320 by default).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for VAE inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .safetensors.zst files if present.",
    )
    args = parser.parse_args()

    videos_root = Path(args.videos_root)
    output_root = Path(args.output_root)
    device = torch.device(args.device)

    all_videos = find_video_files(videos_root)
    if not all_videos:
        print(f"No video files found under {videos_root} with extensions {sorted(VIDEO_EXTS)}")
        sys.exit(1)

    if args.max_clips is not None and args.max_clips < len(all_videos):
        videos = all_videos[: args.max_clips]
    else:
        videos = all_videos

    print(f"Found {len(all_videos)} videos; will process {len(videos)} clips.")
    print(f"Writing latents to: {output_root}")

    # ---------------------------------------------------------------------
    # Load VAEs from your local vae/ directory
    # ---------------------------------------------------------------------
    print("Loading Hunyuan VAE from vae/hvae_2025.safetensors (or HF fallback)...")
    hvae = load_hunyuan_vae_from_local(device=device)

    print("Loading KVAE-3D from vae/kvae_3d_2025.safetensors (local KVAE-3D loader)...")
    kvae = load_kvae3d_from_local(device=device)

    # Dtypes & temporal divider
    hvae_dtype = next(hvae.parameters()).dtype
    kvae_dtype = next(kvae.parameters()).dtype

    # KVAE-3D expects T % (temporal_compress_times * 2) == 1
    temporal_divider = int(kvae.conf["enc"].get("temporal_compress_times", 4) * 2)
    print(f"[Info] Using temporal divider={temporal_divider} for padding (T % divider == 1).")

    failures = 0

    for vid_path in tqdm(videos, desc="Extracting latents"):
        try:
            rel = vid_path.relative_to(videos_root)
        except ValueError:
            rel = Path(vid_path.name)

        out_rel = rel.with_suffix(".safetensors.zst")
        out_path = output_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            continue

        # -----------------------------------------------------------------
        # Load and normalize video to [-1,1], pad T so that T % divider == 1
        # -----------------------------------------------------------------
        try:
            video = load_video_tensor(
                vid_path,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                device=device,
                dtype=torch.float32,          # load as fp32 first
                temporal_divider=temporal_divider,
            )
        except Exception as e:
            print(f"[WARN] Failed to load {vid_path}: {e}")
            failures += 1
            continue

        # -----------------------------------------------------------------
        # Encode with both VAEs, using their native dtypes
        # -----------------------------------------------------------------
        try:
            # Hunyuan expects same dtype as its weights
            video_h = video.to(dtype=hvae_dtype)
            z_h = encode_hunyuan(hvae, video_h)  # [1,16,T',H',W']

            # KVAE-3D expects same dtype as its weights (bfloat16 on CUDA)
            video_k = video.to(dtype=kvae_dtype)
            z_k = encode_kvae3d(kvae, video_k, seg_len=16)  # [1,16,T',H',W']
        except Exception as e:
            print(f"[WARN] Encoding failed for {vid_path}: {e}")
            traceback.print_exc()
            failures += 1
            continue

        if z_h.shape != z_k.shape:
            print(
                f"[WARN] Latent shape mismatch for {vid_path}: "
                f"Hunyuan {tuple(z_h.shape)} vs KVAE {tuple(z_k.shape)}"
            )

        # squeeze batch dim and save as [C, T', H', W']
        if z_h.shape[0] != 1 or z_k.shape[0] != 1:
            print(f"[WARN] Non-batch-1 latents for {vid_path}, taking first element.")
        z_h_s = z_h[0]
        z_k_s = z_k[0]

        try:
            save_latent_pair_zstd(out_path, z_h_s, z_k_s)
        except Exception as e:
            print(f"[WARN] Failed to save latents for {vid_path}: {e}")
            failures += 1
            continue

    print(f"Done. Processed {len(videos)} videos, with {failures} failures.")
    print(f"Latents stored under: {output_root}")


if __name__ == "__main__":
    main()

