# scripts/train_adapter.py

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from safetensors.torch import save_file as save_safetensors

# ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from latent_io import load_latent_pair_zstd
from models.adapter import HunyuanToKVAEAdapter


class ZstdLatentPairDataset(Dataset):
    """
    Dataset over .safetensors.zst latent pairs: each item returns (z_h, z_k) shaped [C, T', H', W'].
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
        z_h, z_k = load_latent_pair_zstd(path, device="cpu")  # [C, T', H', W']
        return z_h, z_k


def train_adapter(
    latent_root: str | Path,
    out_path: str | Path = "adapter_hunyuan_to_kvae3d.safetensors",
    num_epochs: int = 5,
    batch_size: int = 1,
    lr: float = 1e-3,
    device: str = "cuda",
):
    device_t = torch.device(device)

    ds = ZstdLatentPairDataset(latent_root)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    adapter = HunyuanToKVAEAdapter(channels=16, num_blocks=3).to(device_t)
    adapter.train()

    optim = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Starting training on {len(ds)} latent pairs...")

    for epoch in range(num_epochs):
        running_loss = 0.0
        adapter.train()

        for z_h, z_k in tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # z_h, z_k: [B, C, T', H', W']
            z_h = z_h.to(device_t, non_blocking=True)
            z_k = z_k.to(device_t, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=device_t.type == "cuda", dtype=torch.bfloat16
            ):
                z_k_hat = adapter(z_h)
                loss = F.mse_loss(z_k_hat, z_k)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item() * z_h.size(0)

        avg_loss = running_loss / len(ds)
        print(f"  Epoch {epoch+1}/{num_epochs} - latent MSE: {avg_loss:.6f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(adapter.state_dict(), str(out_path))
    print(f"Saved adapter weights to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Hunyuan→KVAE-3D latent adapter on UCF101 latents."
    )
    parser.add_argument(
        "--latent-root",
        type=str,
        default="datasets/UCF101_latents",
        help="Root directory with *.safetensors.zst latent pairs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="adapter_hunyuan_to_kvae3d.safetensors",
        help="Output safetensors file for adapter weights.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for training adapter."
    )
    args = parser.parse_args()

    train_adapter(
        latent_root=args.latent_root,
        out_path=args.out,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()

