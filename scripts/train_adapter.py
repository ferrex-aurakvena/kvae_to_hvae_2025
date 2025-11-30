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
from models.adapter import HunyuanToKVAEAdapter, KVAEToHunyuanAdapter


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
    out_h2k: str | Path = "adapter_hunyuan_to_kvae3d.safetensors",
    out_k2h: str | Path = "adapter_kvae3d_to_hunyuan.safetensors",
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

    # Forward adapter: Hunyuan -> KVAE
    adapter_h2k = HunyuanToKVAEAdapter(channels=16, num_blocks=3).to(device_t)
    # Reverse adapter: KVAE -> Hunyuan
    adapter_k2h = KVAEToHunyuanAdapter(channels=16, num_blocks=3).to(device_t)

    adapter_h2k.train()
    adapter_k2h.train()

    # Single optimizer over both adapters
    optim = torch.optim.AdamW(
        list(adapter_h2k.parameters()) + list(adapter_k2h.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler()

    print(f"Starting training on {len(ds)} latent pairs...")
    print("Training both directions:")
    print(f"  Hunyuan -> KVAE   → {out_h2k}")
    print(f"  KVAE   -> Hunyuan → {out_k2h}")

    for epoch in range(num_epochs):
        running_loss_fwd = 0.0  # Hunyuan -> KVAE
        running_loss_rev = 0.0  # KVAE -> Hunyuan
        adapter_h2k.train()
        adapter_k2h.train()

        for z_h, z_k in tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # z_h, z_k: [B, C, T', H', W']
            z_h = z_h.to(device_t, non_blocking=True)
            z_k = z_k.to(device_t, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=device_t.type == "cuda", dtype=torch.bfloat16
            ):
                # forward direction: Hunyuan -> KVAE
                z_k_hat = adapter_h2k(z_h)
                loss_fwd = F.mse_loss(z_k_hat, z_k)

                # reverse direction: KVAE -> Hunyuan
                z_h_hat = adapter_k2h(z_k)
                loss_rev = F.mse_loss(z_h_hat, z_h)

                loss = loss_fwd + loss_rev

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss_fwd += loss_fwd.item() * z_h.size(0)
            running_loss_rev += loss_rev.item() * z_h.size(0)

        avg_loss_fwd = running_loss_fwd / len(ds)
        avg_loss_rev = running_loss_rev / len(ds)
        print(
            f"  Epoch {epoch+1}/{num_epochs} - "
            f"H→K MSE: {avg_loss_fwd:.6f} | K→H MSE: {avg_loss_rev:.6f}"
        )

    out_h2k = Path(out_h2k)
    out_k2h = Path(out_k2h)
    out_h2k.parent.mkdir(parents=True, exist_ok=True)
    out_k2h.parent.mkdir(parents=True, exist_ok=True)

    save_safetensors(adapter_h2k.state_dict(), str(out_h2k))
    save_safetensors(adapter_k2h.state_dict(), str(out_k2h))

    print(f"Saved Hunyuan→KVAE adapter to:   {out_h2k}")
    print(f"Saved KVAE→Hunyuan adapter to:   {out_k2h}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Hunyuan↔KVAE-3D latent adapters on UCF101 latents."
    )
    parser.add_argument(
        "--latent-root",
        type=str,
        default="datasets/UCF101_latents_32",
        help="Root directory with *.safetensors.zst latent pairs.",
    )
    parser.add_argument(
        "--out-h2k",
        type=str,
        default="adapter_hunyuan_to_kvae3d.safetensors",
        help="Output safetensors file for Hunyuan→KVAE adapter.",
    )
    parser.add_argument(
        "--out-k2h",
        type=str,
        default="adapter_kvae3d_to_hunyuan.safetensors",
        help="Output safetensors file for KVAE→Hunyuan adapter.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for training adapters."
    )
    args = parser.parse_args()

    train_adapter(
        latent_root=args.latent_root,
        out_h2k=args.out_h2k,
        out_k2h=args.out_k2h,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()

