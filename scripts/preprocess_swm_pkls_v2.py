"""Preprocessor v2: labeler-eval / collect_demos pkls -> per-episode .pt with
JPEG-compressed frames (~10x smaller than raw uint8 tensors) + qa_pairs JSON.

Differences from v1 (preprocess_swm_pkls.py):
  - frames stored as list[bytes] (JPEG q=92 at target resolution) instead of
    a uint8 tensor; SWMOneShotDataset decodes lazily
  - also emits <out-root>/<env>/qa_pairs.json for the processed episodes
    (same schema as the Vista extraction: {ep_id: [frame -> [{q,oracle,type}]]})
  - layout-agnostic: --sources takes one or more <combo>=<pkl_dir> pairs so
    collect_demos output (flat trajectories/) and labeler-eval layouts both work

Usage (SWM-recipe collection):
    python preprocess_swm_pkls_v2.py \
        --out-root $WORK/swm-followup/ogb-recipe-448 --env ogb \
        --sources noisy=$WORK/swm-followup/ogb-recipe-data/noisy/trajectories \
                  play=$WORK/swm-followup/ogb-recipe-data/play/trajectories \
        --target-size 448
"""
import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _to_jpeg(arr: np.ndarray, target: int, quality: int = 92) -> bytes:
    h, w = arr.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    img = Image.fromarray(arr[top : top + s, left : left + s])
    if img.size != (target, target):
        img = img.resize((target, target), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--sources", nargs="+", required=True,
                    help="<combo>=<pkl_dir> pairs")
    ap.add_argument("--target-size", type=int, default=448)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.env
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    qa_all = {}
    skipped = 0
    for spec in args.sources:
        combo, _, pkl_dir = spec.partition("=")
        pkls = sorted(Path(pkl_dir).glob("*.pkl"))
        if args.limit:
            pkls = pkls[: args.limit]
        print(f"source {combo}: {len(pkls)} pkls", flush=True)
        for i, p in enumerate(pkls):
            ep_id = f"{combo}__{p.stem}"
            out_path = out_dir / f"{ep_id}.pt"
            try:
                with open(p, "rb") as f:
                    d = pickle.load(f)
                frames = d["frames"]
                actions = d["actions"]
                if len(actions) != len(frames) - 1:
                    skipped += 1
                    continue
                T = len(actions)
                if not out_path.exists():
                    jpegs = [
                        _to_jpeg(np.asarray(frames[t], dtype=np.uint8),
                                 args.target_size, args.quality)
                        for t in range(T)
                    ]
                    actions_f32 = torch.from_numpy(
                        np.stack([np.asarray(a, dtype=np.float32) for a in actions])
                    )
                    proprio = torch.zeros(T, 1)
                    torch.save(
                        {"frames": jpegs, "actions": actions_f32, "proprio": proprio},
                        out_path,
                    )
                # qa extraction (frames 0..T-1)
                qa = d.get("qa_pairs", [])
                frames_out = []
                for t in range(min(T, len(qa))):
                    entries = []
                    for item in qa[t]:
                        if isinstance(item, (list, tuple)) and len(item) == 3:
                            q_text, oracle, q_type = item
                            entries.append(
                                {"q": str(q_text), "oracle": bool(oracle), "type": str(q_type)}
                            )
                    frames_out.append(entries)
                qa_all[ep_id] = frames_out
                a_dim = int(np.asarray(actions[0]).shape[0])
                manifest.append({
                    "id": ep_id, "combo": combo, "ood": False, "length": T,
                    "action_dim": a_dim, "proprio_dim": 1,
                    "filename": out_path.name,
                })
            except Exception as e:
                print(f"  SKIP {ep_id}: {e}", file=sys.stderr)
                skipped += 1
            if (i + 1) % 50 == 0:
                print(f"  {combo}: {i+1}/{len(pkls)}", flush=True)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump({"env": args.env, "target_size": args.target_size,
                   "format": "jpeg", "episodes": manifest}, f, indent=2)
    with open(out_dir / "qa_pairs.json", "w") as f:
        json.dump(qa_all, f)
    n_q = sum(len(e) for frames in qa_all.values() for e in frames)
    print(f"DONE episodes={len(manifest)} skipped={skipped} questions={n_q}")


if __name__ == "__main__":
    main()
