"""Verify that our image_embeds + deepstack actually match what the standard
pipeline produces. Critical sanity check before chasing other splice bugs.

For the same PIL image:
  A. Standard path: processor -> model.visual(pixel_values, grid_thw) -> (image_embeds, deepstack)
  B. Our path: encode_image() -> _merge() -> our image_embeds + cached deepstack

Prints shape, dtype, norm, max-abs-diff, and cosine similarity between them.
If they're not byte-identical, that's the root of all the splice divergence.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.qwen_wm_model import QwenWMModel


def _resize_center_crop(arr, target=448):
    h, w = arr.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    cropped = arr[top : top + s, left : left + s]
    return Image.fromarray(cropped).resize((target, target), Image.BILINEAR)


def compare(tag, a, b):
    print(f"\n  {tag}:")
    if isinstance(a, list):
        print(f"    standard: list[{len(a)}] (per-deepstack-layer)")
        print(f"    ours:     list[{len(b)}]")
        for i, (x, y) in enumerate(zip(a, b)):
            _compare_tensor(f"      layer {i}", x, y)
    else:
        _compare_tensor("   ", a, b)


def _compare_tensor(prefix, a, b):
    a_f = a.float()
    b_f = b.float()
    if a.shape != b.shape:
        print(f"{prefix}  SHAPE MISMATCH: standard={tuple(a.shape)} ours={tuple(b.shape)}")
        return
    diff = (a_f - b_f).abs()
    cos = torch.nn.functional.cosine_similarity(a_f.flatten().unsqueeze(0), b_f.flatten().unsqueeze(0)).item()
    print(f"{prefix}  shape={tuple(a.shape)}  dtype=({a.dtype} vs {b.dtype})")
    print(f"{prefix}  norms: standard={a_f.norm():.3f}  ours={b_f.norm():.3f}")
    print(f"{prefix}  max-abs-diff={diff.max().item():.6f}  mean-abs-diff={diff.mean().item():.6f}")
    print(f"{prefix}  cosine-sim(flattened)={cos:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    print("loading QwenWMModel...")
    wm = QwenWMModel(
        predictor_ckpt_path=args.ckpt,
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
        precision=torch.bfloat16,
        image_size=448,
        obs_horizon=2,
        max_action_horizon=16,
    )
    model = wm.full_model
    processor = wm.processor

    # Pick the demo we know standard pipeline can answer (red_pentagon -> blue_moon)
    pkl = Path(args.data_root) / "lt/red_pentagon_blue_moon/demos/blocktoblock_0_success.pkl"
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    final = _resize_center_crop(np.asarray(data["frames"][-1], dtype=np.uint8))
    print(f"image: {final.size}")

    # --- Path A: STANDARD processor -> visual ---
    print("\n=== Path A: standard processor + model.visual ===")
    inputs = processor(text=["<image>"], images=[final], return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    print(f"  processor pixel_values shape: {tuple(inputs['pixel_values'].shape)}")
    print(f"  processor image_grid_thw: {inputs['image_grid_thw'].tolist()}")
    with torch.no_grad():
        std_img_embeds_list, std_deepstack = model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
    std_img_embeds = torch.cat(std_img_embeds_list, dim=0)
    print(f"  standard image_embeds: shape={tuple(std_img_embeds.shape)}  dtype={std_img_embeds.dtype}")
    print(f"  standard deepstack: list[{len(std_deepstack)}]; layer-0 shape {tuple(std_deepstack[0].shape)}")

    # --- Path B: our encode_image + _merge ---
    print("\n=== Path B: our encode_image + _merge ===")
    z = wm.encode_image(final)  # also caches wm._last_deepstack
    ours_img_embeds = wm._merge(z.unsqueeze(0))[0]  # (P_out, D)
    ours_deepstack = wm._last_deepstack
    print(f"  ours image_embeds: shape={tuple(ours_img_embeds.shape)}  dtype={ours_img_embeds.dtype}")
    print(f"  ours deepstack: list[{len(ours_deepstack)}]; layer-0 shape {tuple(ours_deepstack[0].shape)}")

    # --- Compare ---
    compare("image_embeds (post-merger)", std_img_embeds, ours_img_embeds)
    compare("deepstack", std_deepstack, ours_deepstack)

    # Also inspect the pixel_values themselves -- maybe our manual patchify differs
    print("\n=== pixel_values shape & stats comparison ===")
    pv_std = inputs["pixel_values"]
    print(f"  standard pixel_values: shape={tuple(pv_std.shape)} dtype={pv_std.dtype}")
    print(f"  standard pixel_values: min={pv_std.float().min():.3f} max={pv_std.float().max():.3f} mean={pv_std.float().mean():.3f}")

    # Replicate our manual patchify on the same image
    from torch import as_tensor
    arr = np.asarray(final.convert("RGB"), dtype=np.uint8).copy()
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    t = t.to(wm.device).to(wm.precision)
    t = (t.unsqueeze(0) - wm._mean_b) / wm._std_b
    B, C, H, W = t.shape
    T = int(model.config.vision_config.temporal_patch_size)
    ps = int(model.config.vision_config.patch_size)
    gh, gw = H // ps, W // ps
    t = t.unsqueeze(1).expand(B, T, C, H, W)
    t = t.reshape(B, T, C, gh, ps, gw, ps)
    t = t.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
    pv_ours = t.reshape(B * gh * gw, T * C * ps * ps)
    print(f"  ours pixel_values: shape={tuple(pv_ours.shape)} dtype={pv_ours.dtype}")
    print(f"  ours pixel_values: min={pv_ours.float().min():.3f} max={pv_ours.float().max():.3f} mean={pv_ours.float().mean():.3f}")
    if pv_std.shape == pv_ours.shape:
        diff = (pv_std.float() - pv_ours.float()).abs()
        print(f"  pv max-abs-diff: {diff.max().item():.6f}  mean-abs-diff: {diff.mean().item():.6f}")
        cos = torch.nn.functional.cosine_similarity(pv_std.float().flatten().unsqueeze(0), pv_ours.float().flatten().unsqueeze(0)).item()
        print(f"  pv cosine-sim: {cos:.6f}")
    else:
        print(f"  PV SHAPE MISMATCH!")

    print("\nDONE")


if __name__ == "__main__":
    main()
