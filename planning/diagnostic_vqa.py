"""Diagnostic: figure out why our merger+LLM+VQA pipeline says P(yes)~0.002 everywhere.

Runs three independent paths on a handful of real LT success frames:

  A. STANDARD Qwen3-VL pipeline (image -> processor -> model.generate)
     -> Shows what the model actually generates (text). Establishes whether
        the model can answer at all and which tokens it uses.
  B. STANDARD pipeline + logit inspection
     -> P(token) for all yes/no candidate tokens at the assistant-turn start.
        Tells us which token Qwen actually puts mass on (lower-case vs capital).
  C. OUR SPLICING pipeline (encode -> merger -> splice into input_embeds)
     -> Same logit inspection. If A/B disagree with C, the bug is in the
        splice. If A/B/C all agree but our QwenWMModel saw ~0, the bug is in
        which token IDs we chose in QwenWMModel._build_prompt_info.

Runs on 5 demos (cheap). Prints a side-by-side comparison.
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


# Candidate tokens we want to check P(...) for at the assistant-turn-start position.
YES_VARIANTS = [" Yes", " yes", "Yes", "yes"]
NO_VARIANTS = [" No", " no", "No", "no"]


def _resize_center_crop(arr, target=448):
    h, w = arr.shape[:2]
    s = min(h, w); top, left = (h - s) // 2, (w - s) // 2
    cropped = arr[top : top + s, left : left + s]
    return Image.fromarray(cropped).resize((target, target), Image.BILINEAR)


def candidate_token_ids(tokenizer, variants):
    """Return dict {variant_str: token_id_int} for variants that tokenize to a single token."""
    out = {}
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out[v] = ids[0]
        else:
            out[v] = None  # multi-token; skip
    return out


@torch.no_grad()
def run_standard(model, processor, image, question, max_new_tokens=8):
    """A & B: standard Qwen3-VL forward + generation, plus logit inspection."""
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"{question} Answer with one word: yes or no."},
        ]}
    ]
    chat_str = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[chat_str], images=[image], return_tensors="pt", padding=False).to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # (A) generate text
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = gen_ids[0, inputs["input_ids"].shape[1] :]
    answer_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)

    # (B) logits at the position that would predict the first generated token
    out = model(**inputs, output_hidden_states=False)
    last_logits = out.logits[0, -1, :].float()  # (V,)
    return answer_text, last_logits


@torch.no_grad()
def run_splice(wm: QwenWMModel, image, question):
    """C: our splicing pipeline via the outer Qwen3VLModel.forward (M-RoPE +
    deepstack handled automatically by monkey-patching get_image_features)."""
    z = wm.encode_image(image)
    img_embeds = wm._merge(z.unsqueeze(0))      # (1, P_out, D_out)
    deepstack = wm._last_deepstack
    prompt = wm._build_prompt_info((question, "yes", 1.0))

    device = wm.device
    input_ids = prompt.input_ids.unsqueeze(0).to(device)
    attn = prompt.attention_mask.unsqueeze(0).to(device)

    ps = int(wm.full_model.config.vision_config.patch_size)
    grid_h = wm.image_size // ps
    grid_w = wm.image_size // ps
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long, device=device)

    per_image = [img_embeds[0].to(wm.precision)]
    deepstack_batched = None
    if deepstack is not None:
        deepstack_batched = [layer_feat.to(wm.precision) for layer_feat in deepstack]

    def patched(pixel_values, image_grid_thw_):
        return tuple(per_image), deepstack_batched

    orig = wm.full_model.model.get_image_features
    wm.full_model.model.get_image_features = patched
    try:
        out = wm.full_model.model(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=torch.zeros(1, dtype=wm.precision, device=device),
            image_grid_thw=image_grid_thw,
        )
    finally:
        wm.full_model.model.get_image_features = orig

    last_hidden = out.last_hidden_state[:, -1, :]
    last_logits = wm.full_model.lm_head(last_hidden)[0].float()
    return last_logits


def report_token_probs(label, logits, yes_ids, no_ids):
    """Print P(token) for each candidate, AND the binary softmax sum-over-cases."""
    probs = torch.softmax(logits, dim=-1)
    print(f"  {label}:")
    print(f"    raw P(token):")
    for tok, tid in yes_ids.items():
        if tid is not None:
            print(f"      P({tok!r:<8s}) = {probs[tid]:.5f}  (logit={logits[tid]:+.2f})")
    for tok, tid in no_ids.items():
        if tid is not None:
            print(f"      P({tok!r:<8s}) = {probs[tid]:.5f}  (logit={logits[tid]:+.2f})")
    # Sum-over-cases binary
    yes_mass = sum(probs[t] for t in yes_ids.values() if t is not None)
    no_mass = sum(probs[t] for t in no_ids.values() if t is not None)
    total = yes_mass + no_mass
    if total > 0:
        print(f"    yes-vs-no binary: P(yes-side)={yes_mass/total:.3f}  P(no-side)={no_mass/total:.3f}")
    # Top-5 tokens overall (text decode helper not loaded here -- just print ids)
    top = torch.topk(probs, 5)
    print(f"    top-5 token ids: {top.indices.cpu().tolist()}  probs: {[f'{p:.3f}' for p in top.values.cpu().tolist()]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    print("loading QwenWMModel (provides standard + splicing access)...")
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
    tokenizer = wm.tokenizer

    # Inspect candidate yes/no token IDs
    yes_ids = candidate_token_ids(tokenizer, YES_VARIANTS)
    no_ids = candidate_token_ids(tokenizer, NO_VARIANTS)
    print("\n=== candidate yes/no token IDs ===")
    for tok, tid in yes_ids.items():
        print(f"  {tok!r:<10s} -> {tid}")
    for tok, tid in no_ids.items():
        print(f"  {tok!r:<10s} -> {tid}")

    # Compare what QwenWMModel._first_single_token would pick
    print("\n=== QwenWMModel._first_single_token would pick ===")
    print(f"  yes-id = {wm._first_single_token([' yes', 'yes', ' Yes', 'Yes'])}")
    print(f"  no-id  = {wm._first_single_token([' no', 'no', ' No', 'No'])}")

    # Pick demos
    env_root = Path(args.data_root) / "lt"
    demo_paths = []
    for combo_dir in sorted(env_root.iterdir()):
        if not combo_dir.is_dir() or combo_dir.name.startswith("ood_"):
            continue
        for p in sorted((combo_dir / "demos").glob("*_success.pkl")):
            demo_paths.append((combo_dir.name, p))
            break  # just one per combo
        if len(demo_paths) >= args.n:
            break
    print(f"\n=== running diagnostic on {len(demo_paths)} demos ===")

    for combo, pkl_path in demo_paths:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        md = data["metadata"]
        start_block = md["start_block"].replace("_", " ")
        target_block = md["oracle_target_block"].replace("_", " ")
        question = f"Is the {start_block} touching the {target_block}?"
        final_img = _resize_center_crop(np.asarray(data["frames"][-1], dtype=np.uint8))

        print(f"\n--- {combo} :: {pkl_path.name} ---")
        print(f"  question: {question}  (final frame should be YES)")

        # A & B: standard pipeline
        answer_text, std_logits = run_standard(model, processor, final_img, question)
        print(f"\n  (A) model.generate() answered: {answer_text!r}")
        report_token_probs("(B) standard-pipeline logits", std_logits, yes_ids, no_ids)

        # C: splicing pipeline
        splice_logits = run_splice(wm, final_img, question)
        report_token_probs("(C) splicing-pipeline logits", splice_logits, yes_ids, no_ids)

        # Diagnostic: do A and C agree on the top token?
        std_top = int(torch.argmax(std_logits).item())
        spl_top = int(torch.argmax(splice_logits).item())
        print(f"  argmax token id: standard={std_top}  splice={spl_top}  match={std_top == spl_top}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
