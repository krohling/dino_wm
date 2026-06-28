"""Calibration test: can our merger+LLM+VQA pipeline correctly read real frames?

Loads N successful LT demos, scores both the initial and final frame with the
goal question for that demo's block combo, and reports:
  - Mean P(yes) at initial frame (objects apart -> should be LOW)
  - Mean P(yes) at final  frame (objects touching -> should be HIGH)
  - Per-frame accuracy at threshold 0.5
  - Calibration: does P(yes) cleanly separate the two distributions?

Optional add-on (--with-predictor): for each demo, also predict the final
frame's latent from the first 16 actions and score that. Comparing
"real final frame" vs "predicted final frame" tells us how much VQA accuracy
we lose by going through the predictor.

Prior context (Kevin): Qwen3-VL zero-shot VQA on LT/OGB was measured at ~80%
across both envs in labeler-eval; SWM's fine-tuned PaliGemma was ~97%. So
~80% accuracy here is "pipeline working as well as the VLM allows."
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.qwen_wm_model import QwenWMModel


def _resize_center_crop(arr: np.ndarray, target: int) -> Image.Image:
    h, w = arr.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    cropped = arr[top : top + s, left : left + s]
    return Image.fromarray(cropped).resize((target, target), Image.BILINEAR)


def score_frame(wm: QwenWMModel, frame_pil: Image.Image, question_text: str) -> float:
    """Encode the frame, apply merger, run VQA, return P(yes)."""
    z = wm.encode_image(frame_pil)             # (784, 1152) pre-merger
    img_embeds = wm._merge(z.unsqueeze(0))     # (1, 196, 4096) post-merger
    prompt = wm._build_prompt_info((question_text, "yes", 1.0))
    p_yes = wm._llm_yes_no_probs(prompt, img_embeds, gradient=False)
    return float(p_yes.item())


def score_predicted_frame(
    wm: QwenWMModel,
    history_frames: list[Image.Image],   # length obs_horizon (e.g., 2)
    actions: np.ndarray,                  # shape (H, action_dim), unnormalized
    question_text: str,
) -> float:
    """Same as score_frame but uses our predictor's output instead of a real frame."""
    H = actions.shape[0]
    assert H <= wm.max_action_horizon, (H, wm.max_action_horizon)
    # Encode history frames -> (obs_horizon, P, D) pre-merger
    z_hist = torch.stack([wm.encode_image(f) for f in history_frames], dim=0)
    # Normalize + pad actions
    a_norm = ((torch.from_numpy(actions).to(wm.device, dtype=wm.precision)
               - wm.action_mean.view(1, -1))
              / wm.action_std.view(1, -1))
    a_padded = torch.zeros(1, wm.max_action_horizon, wm.action_dim,
                           dtype=wm.precision, device=wm.device)
    a_padded[0, :H] = a_norm
    mask = torch.zeros(1, wm.max_action_horizon, dtype=torch.bool, device=wm.device)
    mask[0, :H] = True
    with torch.no_grad():
        z_pred = wm._predict_with_mask(z_hist, a_padded, mask)  # (1, P, D)
        img_embeds = wm._merge(z_pred)                          # (1, 196, 4096)
        prompt = wm._build_prompt_info((question_text, "yes", 1.0))
        p_yes = wm._llm_yes_no_probs(prompt, img_embeds, gradient=False)
    return float(p_yes.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="predictor_best.pt")
    ap.add_argument("--data-root", required=True,
                    help="$WORK/swm-followup/labeler-eval/data-720x1280_768x768")
    ap.add_argument("--env", default="lt", choices=("lt",),
                    help="(only lt for now)")
    ap.add_argument("--n-demos", type=int, default=30)
    ap.add_argument("--with-predictor", action="store_true",
                    help="Also score predictor's prediction of the final frame "
                         "(using the first 16 demo actions); off by default")
    ap.add_argument("--out", default=None,
                    help="Optional path to write JSON results")
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
    print("  ok")

    # Find all *_success.pkl demos from each combo
    env_root = Path(args.data_root) / args.env
    sub = "demos"  # LT-specific
    demo_paths = []
    for combo_dir in sorted(env_root.iterdir()):
        if not combo_dir.is_dir() or combo_dir.name.startswith("ood_"):
            continue
        for p in sorted((combo_dir / sub).glob("*_success.pkl")):
            demo_paths.append((combo_dir.name, p))
    print(f"found {len(demo_paths)} success demos across in-distribution combos")
    if len(demo_paths) == 0:
        sys.exit("no demos found")

    rng = np.random.default_rng(0)
    rng.shuffle(demo_paths)
    demo_paths = demo_paths[: args.n_demos]
    print(f"using {len(demo_paths)} demos for calibration")

    results = []
    for i, (combo, pkl_path) in enumerate(demo_paths):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            md = data.get("metadata", {})
            start_block = md.get("start_block", "").replace("_", " ")
            target_block = md.get("oracle_target_block", "").replace("_", " ")
            if not start_block or not target_block:
                print(f"  SKIP {pkl_path.name}: missing metadata"); continue
            q_text = f"Is the {start_block} touching the {target_block}?"

            frames = data["frames"]
            n_frames = len(frames)
            if n_frames < 3:
                continue
            initial_arr = np.asarray(frames[0], dtype=np.uint8)
            final_arr = np.asarray(frames[-1], dtype=np.uint8)
            initial_img = _resize_center_crop(initial_arr, target=448)
            final_img = _resize_center_crop(final_arr, target=448)

            p_initial = score_frame(wm, initial_img, q_text)
            p_final = score_frame(wm, final_img, q_text)

            entry = {
                "demo": str(pkl_path.name),
                "combo": combo,
                "question": q_text,
                "p_yes_initial": p_initial,
                "p_yes_final": p_final,
                "gap": p_final - p_initial,
            }

            if args.with_predictor and n_frames > wm.max_action_horizon + 1:
                # Predict the frame at t = max_action_horizon = 16 (NOT the success
                # frame, which is at t = n_frames - 1, well beyond our predictor's
                # horizon). Score VQA on real vs predicted at t=16.
                t_target = wm.max_action_horizon
                history = [
                    _resize_center_crop(np.asarray(frames[0], dtype=np.uint8), 448),
                    _resize_center_crop(np.asarray(frames[0], dtype=np.uint8), 448),
                ]  # repeat initial frame as both history slots
                actions_seq = np.stack(
                    [np.asarray(a, dtype=np.float32) for a in data["actions"][:t_target]],
                    axis=0,
                )
                real_t16_img = _resize_center_crop(
                    np.asarray(frames[t_target], dtype=np.uint8), 448
                )
                p_real_t16 = score_frame(wm, real_t16_img, q_text)
                p_pred_t16 = score_predicted_frame(wm, history, actions_seq, q_text)
                entry["p_yes_real_t16"] = p_real_t16
                entry["p_yes_pred_t16"] = p_pred_t16
                entry["pipeline_drop_t16"] = p_real_t16 - p_pred_t16

            results.append(entry)
            print(
                f"  [{i+1}/{len(demo_paths)}] {combo:32s}  "
                f"p_initial={p_initial:.3f}  p_final={p_final:.3f}  gap={entry['gap']:+.3f}"
            )
        except Exception as e:
            print(f"  ERR {pkl_path.name}: {e}")
            continue

    if not results:
        sys.exit("no successful evaluations")

    p_initial = np.array([r["p_yes_initial"] for r in results])
    p_final = np.array([r["p_yes_final"] for r in results])
    gap = p_final - p_initial
    acc_initial = float((p_initial < 0.5).mean())  # initial frames -> "no"
    acc_final = float((p_final >= 0.5).mean())     # final frames -> "yes"

    print("\n" + "=" * 80)
    print(f"CALIBRATION (n={len(results)} demos)")
    print("-" * 80)
    print(f"  mean P(yes) initial: {p_initial.mean():.3f}  (std {p_initial.std():.3f})")
    print(f"  mean P(yes) final  : {p_final.mean():.3f}  (std {p_final.std():.3f})")
    print(f"  mean gap (final - initial): {gap.mean():+.3f}  (std {gap.std():.3f})")
    print(f"  per-frame accuracy at 0.5: initial = {acc_initial:.2%}, "
          f"final = {acc_final:.2%}, mean = {(acc_initial + acc_final)/2:.2%}")
    print("=" * 80)

    if args.with_predictor:
        p_real_t16 = np.array([r["p_yes_real_t16"] for r in results if "p_yes_real_t16" in r])
        p_pred_t16 = np.array([r["p_yes_pred_t16"] for r in results if "p_yes_pred_t16" in r])
        if len(p_real_t16) > 0:
            print()
            print(f"PREDICTOR PIPELINE DROP (n={len(p_real_t16)} demos with t>=17)")
            print("-" * 80)
            print(f"  mean P(yes) real frame @t=16     : {p_real_t16.mean():.3f}")
            print(f"  mean P(yes) predicted frame @t=16: {p_pred_t16.mean():.3f}")
            print(f"  mean drop: {(p_real_t16 - p_pred_t16).mean():+.3f}")
            print("=" * 80)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                {
                    "n_demos": len(results),
                    "summary": {
                        "mean_p_initial": float(p_initial.mean()),
                        "mean_p_final": float(p_final.mean()),
                        "mean_gap": float(gap.mean()),
                        "acc_initial": acc_initial,
                        "acc_final": acc_final,
                    },
                    "per_demo": results,
                },
                f, indent=2,
            )
        print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
