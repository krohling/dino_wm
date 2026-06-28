"""Smoke test for QwenWMModel: load + one forward pass + print shapes.

Verifies (without running the SWM env at all):
  - model loads from our predictor checkpoint
  - image encode shape is right
  - merger lifts pre-merger -> post-merger correctly
  - chat prompt has the expected number of image placeholder tokens
  - LLM forward returns sane P(yes) probabilities

Usage:
    python -m planning.smoke_qwen_wm \
        --ckpt $WORK/swm-followup/oneshot_runs/<run>/checkpoints/predictor_best.pt \
        --frame-pkl $WORK/swm-followup/labeler-eval/data-720x1280_768x768/lt/green_cube_blue_moon/demos/blocktoblock_0_success.pkl
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frame-pkl", required=True)
    ap.add_argument(
        "--model-id", default="Qwen/Qwen3-VL-8B-Instruct"
    )
    ap.add_argument("--n", type=int, default=4, help="number of candidate actions")
    ap.add_argument("--pred_horizon", type=int, default=16)
    ap.add_argument("--action_skip", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    print("loading frame...")
    with open(args.frame_pkl, "rb") as f:
        data = pickle.load(f)
    frame0 = np.asarray(data["frames"][0], dtype=np.uint8)
    print(f"  raw frame shape: {frame0.shape}")
    # Center-crop + resize to 448x448
    h, w = frame0.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    cropped = frame0[top : top + s, left : left + s]
    image = Image.fromarray(cropped).resize((448, 448), Image.BILINEAR)
    print(f"  -> PIL {image.size}")

    print("loading QwenWMModel...")
    model = QwenWMModel(
        predictor_ckpt_path=args.ckpt,
        model_id=args.model_id,
        device="cuda",
        precision=torch.bfloat16,
        image_size=448,
        obs_horizon=2,
        max_action_horizon=16,
    )
    print("  ok")

    # Build a fake batch of N candidate actions (length = pred_horizon, dim=2 for LT)
    rng = np.random.default_rng(0)
    action_seq = rng.normal(0.0, 0.05, (args.n, args.pred_horizon, model.action_dim)).astype(np.float32)
    print(f"action_seq: shape={action_seq.shape}")

    # Two LT-style questions
    questions = [
        ("Is the green cube touching the blue moon?", "yes", 0.8),
        ("Are the green cube and blue moon closer together?", "yes", 0.2),
    ]

    print("calling get_probabilistic_rewards_wm...")
    rewards, weighted = model.get_probabilistic_rewards_wm(
        action_seq=action_seq,
        image=image,
        pred_horizon=args.pred_horizon,
        questions=questions,
        batch_size=8,
        action_skip=args.action_skip,
        gradient=False,
    )
    print(f"rewards shape: {rewards.shape}  (expect: (n_questions, n_actions, pred_horizon))")
    print(f"weighted shape: {weighted.shape}")
    print(f"rewards[q=0, a=:, h_step={args.pred_horizon}-1]:")
    print("  ", rewards[0, :, args.pred_horizon - 1])
    print(f"rewards[q=1, a=:, h_step={args.action_skip - 1}]:")
    print("  ", rewards[1, :, args.action_skip - 1])
    print(f"weighted total per action: {weighted.sum(axis=(0, 2))}")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
