"""Precompute VLM teacher supervision for distillation training.

For every (episode, frame, question) in an env's preprocessed dataset:
    real frame -> frozen Qwen3-VL encoder -> merger -> LLM VQA -> P(yes)

The result is stored as one sidecar JSON per episode next to the .pt files:
    <data_path>/teacher/<episode_id>.json
    [ per frame: [ {"q": str, "type": str, "oracle": bool, "p_yes": float}, ... ] ]

Teacher outputs are a pure function of (frame, question) because everything in
the path is frozen -- compute once, reuse across all training runs/losses.

Questions come from a qa_pairs JSON extracted from the raw labeler-eval pkls
(see the extraction snippet in the session notes): {episode_id: [frame_idx ->
[{"q","oracle","type"}]]}.

Usage:
    python scripts/precompute_teacher.py \
        --data-path $BASE/world-model-data/ogb \
        --qa-json $BASE/ogb_qa_pairs.json \
        --predictor-ckpt $BASE/predictor_best.pt   # loaded but unused; QwenWMModel needs one
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.qwen_wm_model import QwenWMModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--qa-json", required=True)
    ap.add_argument("--predictor-ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="frames scored per LLM batch (same question)")
    ap.add_argument("--limit-episodes", type=int, default=None)
    args = ap.parse_args()

    data_path = Path(args.data_path)
    out_dir = data_path / "teacher"
    out_dir.mkdir(exist_ok=True)

    with open(args.qa_json) as f:
        qa_all = json.load(f)
    with open(data_path / "manifest.json") as f:
        manifest = json.load(f)
    episodes = manifest["episodes"]
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]

    print(f"episodes: {len(episodes)}  qa episodes: {len(qa_all)}")

    wm = QwenWMModel(
        predictor_ckpt_path=args.predictor_ckpt,
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
        precision=torch.bfloat16,
        image_size=448,
        obs_horizon=2,
        max_action_horizon=16,
    )

    t_start = time.time()
    total_scored = 0
    for ei, ep in enumerate(episodes):
        ep_id = ep["id"]
        out_path = out_dir / f"{ep_id}.json"
        if out_path.exists():
            continue
        if ep_id not in qa_all:
            print(f"  WARN no qa for {ep_id}; skipping")
            continue
        qa_frames = qa_all[ep_id]
        d = torch.load(data_path / ep["filename"], map_location="cpu", weights_only=True)
        frames_u8 = d["frames"]  # (T, 3, S, S) uint8
        T = frames_u8.shape[0]

        # 1. Encode all frames once (encoder+merger), keep post-merger embeds + deepstack
        embeds = []   # (P_out, D) per frame
        deepstacks = []
        for t in range(T):
            arr = frames_u8[t].permute(1, 2, 0).numpy()  # HWC uint8
            img = Image.fromarray(arr)
            z = wm.encode_image(img)                     # also caches deepstack
            embeds.append(wm._merge(z.unsqueeze(0))[0])  # (P_out, D)
            deepstacks.append(wm._last_deepstack)

        # 2. Group scoring by question text so each batch shares one prompt
        #    Build tasks: (frame_idx, entry_idx, question, oracle, type)
        tasks = []
        for t in range(min(T, len(qa_frames))):
            for j, entry in enumerate(qa_frames[t]):
                tasks.append((t, j, entry["q"], entry.get("oracle"), entry.get("type", "")))

        results = [[None] * len(qa_frames[t]) if t < len(qa_frames) else [] for t in range(T)]
        by_q: dict[str, list] = {}
        for task in tasks:
            by_q.setdefault(task[2], []).append(task)

        for q_text, q_tasks in by_q.items():
            prompt = wm._build_prompt_info((q_text, "yes", 1.0))
            for s in range(0, len(q_tasks), args.batch_size):
                chunk = q_tasks[s : s + args.batch_size]
                # Score each frame in the chunk individually if deepstack differs;
                # deepstack is per-frame so batch=1 per frame for correctness.
                for (t, j, _q, oracle, q_type) in chunk:
                    p_yes = wm._llm_yes_no_probs(
                        prompt,
                        embeds[t].unsqueeze(0),
                        gradient=False,
                        deepstack_override=deepstacks[t],
                    )
                    results[t][j] = {
                        "q": q_text,
                        "type": q_type,
                        "oracle": oracle,
                        "p_yes": float(p_yes.item()),
                    }
                    total_scored += 1

        with open(out_path, "w") as f:
            json.dump(results, f)
        elapsed = time.time() - t_start
        print(f"[{ei+1}/{len(episodes)}] {ep_id}: {len(tasks)} questions scored  "
              f"(total {total_scored}, {elapsed:.0f}s elapsed)", flush=True)

    # Summary of teacher-vs-oracle agreement (sanity metric)
    agree, n = 0, 0
    for ep in episodes:
        p = out_dir / f"{ep['id']}.json"
        if not p.exists():
            continue
        with open(p) as f:
            frames = json.load(f)
        for entries in frames:
            for e in entries:
                if e and e.get("oracle") is not None:
                    pred = e["p_yes"] >= 0.5
                    agree += int(pred == bool(e["oracle"]))
                    n += 1
    if n:
        print(f"\nTEACHER vs ORACLE agreement: {agree}/{n} = {agree/n:.1%}")
    print("DONE")


if __name__ == "__main__":
    main()
