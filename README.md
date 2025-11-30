# KVAE-3D ↔ Hunyuan Video VAE Adapters

This repository trains **latent-to-latent adapters** between:

- **Hunyuan Video VAE** (the 3D VAE used by Hunyuan Video and Kandinsky 5), and  
- **KVAE-3D** (an improved 3D VAE with causal / cached temporal structure).

The goal is to make **KVAE-3D a drop-in decoder (and optionally encoder)** for any model that speaks **Hunyuan Video VAE latents** – including **Kandinsky 5** and **Hunyuan Video** itself.

We learn *bidirectional* mappings:

- **Hunyuan → KVAE**: `adapter_hunyuan_to_kvae3d.safetensors`  
- **KVAE → Hunyuan**: `adapter_kvae3d_to_hunyuan.safetensors`  

Both are small 3D ResNet-style adapters operating directly in the latent space.

---

## High-Level Idea

### Hunyuan → KVAE (decode side)

In the standard Hunyuan/Kandinsky pipeline:

1. Pixels → Hunyuan VAE encode → **Hunyuan latents**  
2. DiT operates in that latent space  
3. Hunyuan VAE decodes the final latents to pixels

We want to **replace Hunyuan’s decoder with KVAE-3D** while keeping the DiT untouched.

We do this by training an adapter:

> `z_k ≈ F(z_h_raw)`  

where:

- `z_h_raw` is the **unscaled** Hunyuan latent (before VAE scaling_factor),  
- `z_k` is the KVAE-3D latent,  
- both are 3D latent grids of shape `[C=16, T', H', W']`.

At inference time, for any model that outputs Hunyuan latents:

1. Take its DiT latents `z_scaled`
2. Convert back to unscaled:  
   `z_h_raw = z_scaled / scaling_factor`
3. Apply adapter:  
   `z_k = adapter_hunyuan_to_kvae3d(z_h_raw)`
4. Decode with KVAE-3D instead of Hunyuan Video VAE

### KVAE → Hunyuan (encode side / low-VRAM mode)

For some users, VRAM is tight and they’d prefer to load **only KVAE-3D**.

To still use Hunyuan-trained DiTs, we also train the reverse mapping:

> `z_h_raw ≈ G(z_k)`

This enables:

- Encode with KVAE-3D  
- Map KVAE latents → Hunyuan latent space with `adapter_kvae3d_to_hunyuan`  
- Feed Hunyuan-style latents into any Hunyuan-based DiT  
- Optionally decode again with KVAE-3D

In other words, a user could, in principle, run a **Hunyuan or Kandinsky pipeline with KVAE-3D as the only VAE** (plus the adapters).

---

## Repository Overview

```text
kvae_to_hvae_2025/
  datasets/
    UCF101/                 # UCF101 videos (train/test split)
    UCF101_latents_33/      # Extracted latent pairs (Hunyuan + KVAE), .safetensors.zst
  vae/
    hvae_2025.safetensors   # Hunyuan Video VAE weights (local)
    kvae_3d_2025.safetensors# KVAE-3D VAE weights (local)
  models/
    adapter.py              # Bidirectional latent adapters (H→K and K→H)
    kvae3d_loader.py        # Local KVAE-3D implementation + loader
    vae_wrappers.py         # Helpers to load Hunyuan VAE + KVAE-3D
  scripts/
    extract_latents_ucf101_zstd.py  # UCF101 → latent pairs extraction
    test_kvae3d_ucf101.py          # KVAE-3D reconstruction sanity check
    train_adapter.py               # Train both adapters jointly
  latent_io.py             # zstd+safetensors I/O for latent pairs
  requirements.txt
  README.md
````

---

## Latent Extraction

We use **UCF101** as a paired training corpus:

* Input videos: `datasets/UCF101/.../*.avi`
* Resolution: rescaled to `240×320`
* Temporal: 33 frames per clip (with a small padding rule to fit KVAE-3D’s streaming requirements)
* Normalization: pixel range `[0, 1] → [-1, 1]` (m11)

For each clip:

1. We encode with **Hunyuan Video VAE** → `z_h_raw ∈ ℝ^{16×T'×H'×W'}`
2. We encode with **KVAE-3D** → `z_k ∈ ℝ^{16×T'×H'×W'}`
3. We store both latents (batch dim squeezed) as:

   * `z_h` (Hunyuan),
   * `z_k` (KVAE),
     inside a `.safetensors.zst` file.

Extraction command used for the 10k latent run:

```bash
python scripts/extract_latents_ucf101_zstd.py \
  --videos-root datasets/UCF101 \
  --output-root datasets/UCF101_latents_33 \
  --max-clips 10000 \
  --num-frames 33 \
  --height 240 \
  --width 320 \
  --device cuda
```

Result:

* **10,000** clips processed
* **0** failures
* Latents under `datasets/UCF101_latents_33`

---

## Adapter Architecture

Both adapters share the same structure:

* Operate on latents `[B, 16, T', H', W']`
* Fully convolutional `Conv3d + GroupNorm + SiLU` stack
* Purely 1×1×1 kernels (channel mixing only; no spatial/temporal scaling)
* 3 residual blocks

This makes them:

* resolution-agnostic (any `T', H', W'`),
* lightweight, and
* easy to plug into existing pipelines.

Adapters:

* `HunyuanToKVAEAdapter` – Hunyuan → KVAE-3D
* `KVAEToHunyuanAdapter` – KVAE-3D → Hunyuan

---

## Training the Adapters

We train **both directions jointly** on the same latent pairs.

Loss:

* Forward: `L_fwd = MSE( H→K(z_h), z_k )`
* Reverse: `L_rev = MSE( K→H(z_k), z_h )`
* Total: `L = L_fwd + L_rev`

This creates a lightweight “cycle-consistent” pressure where both mappings have to meet in the middle instead of one overfitting to quirks of one space.

Example training command used for the main run:

```bash
python scripts/train_adapter.py \
  --latent-root datasets/UCF101_latents_33 \
  --out-h2k adapter_hunyuan_to_kvae3d.safetensors \
  --out-k2h adapter_kvae3d_to_hunyuan.safetensors \
  --epochs 250 \
  --batch-size 512 \
  --lr 1e-4 \
  --device cuda
```

Final reported training metrics (250 epochs, 10k clips, batch size 512):

```text
Epoch 250/250 - H→K MSE: 0.328768 | K→H MSE: 0.380549
Saved Hunyuan→KVAE adapter to:   adapter_hunyuan_to_kvae3d.safetensors
Saved KVAE→Hunyuan adapter to:   adapter_kvae3d_to_hunyuan.safetensors
```

These are average *latent-space* MSEs over the full training set.

---

## KVAE-3D Loader

`models/kvae3d_loader.py` implements a **local KVAE-3D architecture**:

* Causal 3D convolutions
* Temporal + spatial down/upsampling with pixel shuffle/unshuffle
* Cached streaming encode/decode with segmenting in time
* Configured to match KVAE-3D 1.0 (16 latent channels, 4×8×8 compression)

Weights are loaded from `vae/kvae_3d_2025.safetensors` with no runtime dependency on the original `kvae-1` repo.

A sanity test on UCF101:

```bash
python scripts/test_kvae3d_ucf101.py \
  --video-path datasets/UCF101/test/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
```

This encodes then decodes a UCF101 clip and prints PSNR + a few frame comparisons.

---

## Intended Usage (Future integration outside this repository)

This repo currently focuses on **extracting paired latents** and **training the adapters**.

Target downstream uses include:

1. **Kandinsky 5 Pro T2V SFT 10s**

   * Replace Hunyuan decode with `KVAE-3D decode ∘ (Hunyuan→KVAE adapter)`.
2. **Hunyuan Video**

   * Same idea: use KVAE-3D as a drop-in decoder for better recon/temporal behavior.
3. **Single-VAE / low-VRAM mode**

   * Encode/Decode with KVAE-3D only
   * Use KVAE→Hunyuan adapter to communicate with Hunyuan-trained DiTs.

The actual pipeline glue (e.g., a `HybridVAE` class) will live in whichever project consumes these adapters (Kandinsky / Hunyuan forks, custom UIs, etc.).

---

## Dependencies

See `requirements.txt` for exact versions, but at a high level:

* `torch`, `torchvision`
* `diffusers` (for `AutoencoderKLHunyuanVideo`)
* `safetensors`
* `zstandard`
* `decord`
* `tqdm`
* `einops`

Install with:

```bash
pip install -r requirements.txt
```

---

## Status

* ✅ KVAE-3D loader implemented locally
* ✅ Paired latent extraction from UCF101 (~10k clips)
* ✅ Bidirectional adapters trained:

  * Hunyuan → KVAE-3D
  * KVAE-3D → Hunyuan

