"""Teacher precompute v2: subsampled questions + batched LLM scoring.

v1 scored every question individually (~27 q/s => 39 GPU-hours for the
600-episode SWM-recipe dataset). v2:
  1. Subsamples --questions-per-frame (default 4), stratified by question
     type (one per type where available) -- matches how the distillation
     sampler consumes them (one question per training sample).
  2. Batches frames that share a question text into one LLM forward.
     Per-frame deepstack features concatenate along the token axis, which
     is exactly the layout Qwen3VL's _deepstack_process expects.

Output schema identical to v1 (per-episode sidecar JSONs under
<data-path>/teacher/), so TeacherStore/DistillDataset work unchanged.

Usage:
    python scripts/precompute_teacher_v2.py \
        --data-path $BASE/world-model-data/ogb-recipe \
        --qa-json  $BASE/world-model-data/ogb-recipe/qa_pairs.json \
        --predictor-ckpt $BASE/predictor_best.pt \
        --questions-per-frame 4 --batch-size 24
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.qwen_wm_model import QwenWMModel


def _frame_to_pil(frame) -> Image.Image:
    if isinstance(frame, bytes):
        return Image.open(io.BytesIO(frame)).convert("RGB")
    # uint8 tensor (3, S, S)
    return Image.fromarray(frame.permute(1, 2, 0).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--qa-json", required=True)
    ap.add_argument("--predictor-ckpt", required=True)
    ap.add_argument("--questions-per-frame", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
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
    print(f"episodes: {len(episodes)}", flush=True)

    wm = QwenWMModel(
        predictor_ckpt_path=args.predictor_ckpt,
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda", precision=torch.bfloat16,
        image_size=448, obs_horizon=2, max_action_horizon=16,
    )
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    total = 0
    for ei, ep in enumerate(episodes):
        ep_id = ep["id"]
        out_path = out_dir / f"{ep_id}.json"
        if out_path.exists():
            continue
        if ep_id not in qa_all:
            print(f"  WARN no qa for {ep_id}", flush=True)
            continue
        qa_frames = qa_all[ep_id]
        d = torch.load(data_path / ep["filename"], map_location="cpu", weights_only=False)
        frames = d["frames"]
        T = len(frames) if isinstance(frames, list) else frames.shape[0]

        # 1. Encode all frames once, keep post-merger embeds + per-frame deepstack
        embeds, deepstacks = [], []
        for t in range(T):
            img = _frame_to_pil(frames[t])
            z = wm.encode_image(img)
            embeds.append(wm._merge(z.unsqueeze(0))[0])
            deepstacks.append(wm._last_deepstack)

        # 2. Subsample: one question per type per frame (up to questions_per_frame)
        results = [[] for _ in range(T)]
        tasks_by_q = defaultdict(list)  # question text -> [(frame_idx, entry)]
        for t in range(min(T, len(qa_frames))):
            by_type = defaultdict(list)
            for e in qa_frames[t]:
                by_type[e.get("type", "")].append(e)
            chosen = []
            for q_type in sorted(by_type.keys()):
                pool = by_type[q_type]
                chosen.append(pool[rng.integers(len(pool))])
            rng.shuffle(chosen)
            for e in chosen[: args.questions_per_frame]:
                tasks_by_q[e["q"]].append((t, e))

        # 3. Batched scoring per question
        for q_text, items in tasks_by_q.items():
            prompt = wm._build_prompt_info((q_text, "yes", 1.0))
            for s in range(0, len(items), args.batch_size):
                chunk = items[s : s + args.batch_size]
                frame_idxs = [t for (t, _e) in chunk]
                batch_embeds = torch.stack([embeds[t] for t in frame_idxs], dim=0)
                # Concatenate per-frame deepstacks along the token axis
                n_layers = len(deepstacks[frame_idxs[0]])
                ds = [
                    torch.cat([deepstacks[t][l] for t in frame_idxs], dim=0)
                    for l in range(n_layers)
                ]
                p_yes = wm._llm_yes_no_probs(
                    prompt, batch_embeds, gradient=False, deepstack_override=ds,
                )
                for k, (t, e) in enumerate(chunk):
                    results[t].append({
                        "q": q_text, "type": e.get("type", ""),
                        "oracle": e.get("oracle"), "p_yes": float(p_yes[k].item()),
                    })
                    total += 1

        with open(out_path, "w") as f:
            json.dump(results, f)
        el = time.time() - t0
        rate = total / max(el, 1)
        print(f"[{ei+1}/{len(episodes)}] {ep_id}: scored (total {total}, {rate:.1f} q/s)", flush=True)

    # Teacher-vs-oracle agreement
    agree = n = 0
    for ep in episodes:
        p = out_dir / f"{ep['id']}.json"
        if not p.exists():
            continue
        with open(p) as f:
            for entries in json.load(f):
                for e in entries:
                    if e and e.get("oracle") is not None:
                        agree += int((e["p_yes"] >= 0.5) == bool(e["oracle"]))
                        n += 1
    if n:
        print(f"\nTEACHER vs ORACLE agreement: {agree}/{n} = {agree/n:.1%}")
    print("DONE")


if __name__ == "__main__":
    main()
