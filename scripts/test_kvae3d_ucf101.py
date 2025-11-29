#!/usr/bin/env python
import argparse
from pathlib import Path

# -------------------------------------------------------------------------
# Make repo root importable so we can do `from models...` cleanly
# -------------------------------------------------------------------------
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torchvision.io import read_video
from torchvision.utils import save_image

from models.kvae3d_loader import load_kvae3d


def load_ucf_clip(
    video_path: str,
    frames: int | None = 65,
    input_norm: str = "m11",
    temporal_divider: int = 8,
) -> torch.Tensor:
    """
    Load a clip from a UCF101 .avi and prepare it for KVAE-3D.

    Returns:
        x: [1, 3, T, H, W] tensor in the normalization space specified
           by `input_norm` (typically 'm11' => [-1, 1]).
    """
    video_path = str(video_path)
    # vframes: [T, H, W, C], uint8
    vframes, _, _ = read_video(video_path, pts_unit="sec")

    # Optional: limit / pad to a target length
    if frames is not None:
        if vframes.shape[0] >= frames:
            vframes = vframes[:frames]
        else:
            pad_len = frames - vframes.shape[0]
            last = vframes[-1:].expand(pad_len, -1, -1, -1)
            vframes = torch.cat([vframes, last], dim=0)

    # If no fixed frames were given, we still want len % divider == 1
    if frames is None:
        T = vframes.shape[0]
        while T % temporal_divider != 1:
            vframes = torch.cat([vframes, vframes[-1:]], dim=0)
            T = vframes.shape[0]

    # [T, H, W, C] uint8 -> [T, C, H, W] float32
    vframes = vframes.permute(0, 3, 1, 2).float()  # [T, C, H, W]

    if input_norm == "01":
        vframes = vframes / 255.0
    elif input_norm == "m11":
        vframes = vframes / 128.0 - 1.0
    else:
        raise ValueError(f"Unsupported input_norm: {input_norm}")

    # Add batch and time dims to match [B, C, T, H, W]
    x = vframes.unsqueeze(0)  # [1, T, C, H, W]
    x = x.transpose(1, 2)     # -> [1, C, T, H, W]
    return x


def psnr_m11(src: torch.Tensor, dist: torch.Tensor) -> float:
    """
    Simple PSNR in dB for tensors in [-1, 1] ("m11" space).
    """
    # Map from [-1,1] to [0,1]
    src01 = (src + 1) / 2
    dist01 = (dist + 1) / 2

    src01 = src01.clamp(0.0, 1.0)
    dist01 = dist01.clamp(0.0, 1.0)

    mse = F.mse_loss(src01, dist01)
    if mse <= 0:
        return float("inf")
    return float(-10.0 * torch.log10(mse).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-path",
        type=str,
        default="datasets/UCF101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi",
        help="Path to a UCF101 video file (.avi)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=65,
        help=(
            "Target number of frames (will pad or trim). "
            "Use 0 to keep all and pad to T % 8 == 1."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. cuda, cuda:0, cpu). Default: auto",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="kvae3d_test_output",
        help="Directory to save example frames.",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    print("Loading KVAE-3D...")
    vae = load_kvae3d(device=device)
    input_norm = vae.config.get("data", {}).get("input_norm", "m11")

    # Load clip
    frames = None if args.frames == 0 else args.frames
    print(f"Loading video: {args.video_path}")
    x = load_ucf_clip(
        args.video_path,
        frames=frames,
        input_norm=input_norm,
        temporal_divider=vae.conf["enc"]["temporal_compress_times"] * 2,
    ).to(device)

    # Match model dtype (e.g., bfloat16 on CUDA) to avoid conv3d dtype mismatch
    model_dtype = next(vae.parameters()).dtype
    if x.dtype != model_dtype:
        x = x.to(dtype=model_dtype)

    print(f"Input shape: {tuple(x.shape)} (B, C, T, H, W)")

    # Encode / decode
    with torch.no_grad():
        latent, split_list, _ = vae.encode(x, seg_len=16)
        print(f"Latent shape: {tuple(latent.shape)}; split_list={split_list}")

        rec = vae.decode(latent, split_list)
        print(f"Reconstruction shape: {tuple(rec.shape)}")

    # Sanity: crop reconstruction to original time length
    T_in = x.shape[2]
    rec = rec[:, :, :T_in]

    # Compute PSNR
    psnr_val = psnr_m11(x, rec)
    print(f"PSNR (m11 → [0,1] range): {psnr_val:.2f} dB")

    # Save some frames for visual inspection
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Take first, middle, last frame for quick check
    frame_indices = [0, T_in // 2, T_in - 1]
    for idx in frame_indices:
        # [B,C,T,H,W] -> [C,H,W] for this frame
        orig = x[0, :, idx]
        recon = rec[0, :, idx]

        # Back to [0,1] for saving
        o_01 = ((orig + 1) / 2).clamp(0, 1)
        r_01 = ((recon + 1) / 2).clamp(0, 1)

        grid = torch.stack([o_01, r_01], dim=0)  # [2, C, H, W]
        save_image(grid, out_dir / f"frame_{idx:03d}_orig_recon.png", nrow=2)

    print(f"Saved example frames to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

