"""One-time preprocessor: labeler-eval pkls -> per-episode tensor files.

Reads:
    <data-root>/{lt,ogb}/<combo>/(demos|trajectories)/*.pkl

Writes:
    <out-root>/{lt,ogb}/<combo>__<stem>.pt   (each is a dict)
    <out-root>/{lt,ogb}/manifest.json        (index of episodes)

The pkls were produced by the labeler-eval pipeline and may reference
tf_agents types (LT 'observations'); set TF_USE_LEGACY_KERAS=1 and have
language-table on PYTHONPATH when running this. The resulting .pt files
have no third-party dependencies.

Per-episode layout (frames and actions aligned 1:1 over t=0..T-1):
    frames:  uint8 (T, 3, S, S)        - center-cropped + resized to S=target_size
    actions: float32 (T, action_dim)
    proprio: float32 (T, proprio_dim)  - end-effector pose (LT) or robot state slice (OGB)
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import torch
from PIL import Image

TARGET_SIZE_DEFAULT = 448


def _resize_frames(frames_list, target):
    out = []
    for arr in frames_list:
        arr = np.asarray(arr, dtype=np.uint8)
        h, w = arr.shape[:2]
        s = min(h, w)
        top = (h - s) // 2
        left = (w - s) // 2
        cropped = arr[top : top + s, left : left + s]
        if cropped.shape[0] != target:
            img = Image.fromarray(cropped).resize((target, target), Image.BILINEAR)
            cropped = np.asarray(img, dtype=np.uint8)
        out.append(cropped.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out, axis=0))


def _lt_proprio(observations):
    rows = []
    for ts in observations:
        v = None
        try:
            obs = ts.observation
            for key in ("effector_target_translation", "effector_translation"):
                if hasattr(obs, key):
                    v = getattr(obs, key)
                    break
                if isinstance(obs, dict) and key in obs:
                    v = obs[key]
                    break
        except Exception:
            pass
        if v is None:
            v = np.zeros(2, dtype=np.float32)
        rows.append(np.asarray(v, dtype=np.float32).flatten())
    return torch.from_numpy(np.stack(rows, axis=0).astype(np.float32))


def _ogb_proprio(block_states):
    """Return end-effector proprio per frame.

    OGB block_states[i] is the env state at frame i. We pull `agent_xy` /
    `ee_*` if present; otherwise fall back to the first 3 dims of a flat
    state vector. Output is always (T, P) float32 with consistent P across
    frames in one episode.
    """
    rows = []
    chosen_dim = None
    for st in block_states:
        v = None
        try:
            if isinstance(st, dict):
                for k in ("ee_pos", "ee_xyz", "robot_pose", "agent_xy"):
                    if k in st:
                        v = np.asarray(st[k], dtype=np.float32).flatten()
                        break
                if v is None and "qpos" in st:
                    v = np.asarray(st["qpos"], dtype=np.float32).flatten()[:3]
            elif isinstance(st, np.ndarray):
                v = st.flatten().astype(np.float32)[:3]
        except Exception:
            v = None
        if v is None:
            v = np.zeros(2, dtype=np.float32)
        if chosen_dim is None:
            chosen_dim = v.shape[0]
        else:
            # Truncate or pad to match the first row's dim.
            if v.shape[0] > chosen_dim:
                v = v[:chosen_dim]
            elif v.shape[0] < chosen_dim:
                v = np.pad(v, (0, chosen_dim - v.shape[0]))
        rows.append(v)
    return torch.from_numpy(np.stack(rows, axis=0).astype(np.float32))


def _process(pkl_path: Path, env: str, target: int):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    frames = data["frames"]
    actions = data["actions"]
    n_f, n_a = len(frames), len(actions)
    if n_a != n_f - 1:
        return None
    T = n_a
    frames_u8 = _resize_frames(frames[:T], target=target)
    actions_f32 = torch.from_numpy(
        np.stack([np.asarray(a, dtype=np.float32) for a in actions], axis=0)
    )
    if env == "lt":
        proprio = _lt_proprio(data.get("observations", [])[:T])
    elif env == "ogb":
        proprio = _ogb_proprio(data.get("block_states", [])[:T])
    else:
        raise ValueError(env)
    if proprio.shape[0] != T:
        proprio = torch.zeros(T, max(1, proprio.shape[1] if proprio.ndim > 1 else 1))
    return {"frames": frames_u8, "actions": actions_f32, "proprio": proprio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--env", choices=("lt", "ogb"), required=True)
    ap.add_argument("--target-size", type=int, default=TARGET_SIZE_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude-ood", action="store_true",
                    help="Skip combos whose name starts with 'ood_'")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root) / args.env
    out_root.mkdir(parents=True, exist_ok=True)

    env_root = data_root / args.env
    sub = "demos" if args.env == "lt" else "trajectories"

    pkls = []
    for combo_dir in sorted(env_root.iterdir()):
        if not combo_dir.is_dir():
            continue
        if args.exclude_ood and combo_dir.name.startswith("ood_"):
            continue
        pdir = combo_dir / sub
        if not pdir.is_dir():
            continue
        for p in sorted(pdir.glob("*.pkl")):
            pkls.append((combo_dir.name, p))
    print(f"found {len(pkls)} pkls for env={args.env}", flush=True)
    if args.limit is not None:
        pkls = pkls[: args.limit]
        print(f"limit -> {len(pkls)}", flush=True)

    manifest = []
    skipped = 0
    for i, (combo, pkl_path) in enumerate(pkls):
        ep_id = f"{combo}__{pkl_path.stem}"
        out_path = out_root / f"{ep_id}.pt"
        try:
            if out_path.exists():
                d = torch.load(out_path, map_location="cpu", weights_only=True)
            else:
                d = _process(pkl_path, args.env, target=args.target_size)
                if d is None:
                    skipped += 1
                    continue
                torch.save(d, out_path)
        except Exception as e:
            print(f"  SKIP {ep_id}: {e}", file=sys.stderr)
            skipped += 1
            continue
        manifest.append({
            "id": ep_id,
            "combo": combo,
            "ood": combo.startswith("ood_"),
            "length": int(d["frames"].shape[0]),
            "action_dim": int(d["actions"].shape[1]),
            "proprio_dim": int(d["proprio"].shape[1]) if d["proprio"].ndim > 1 else 1,
            "filename": out_path.name,
        })
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(pkls)} (skipped {skipped})", flush=True)

    with open(out_root / "manifest.json", "w") as f:
        json.dump(
            {"env": args.env, "target_size": args.target_size, "episodes": manifest},
            f, indent=2,
        )
    print(f"DONE env={args.env}  episodes={len(manifest)}  skipped={skipped}")


if __name__ == "__main__":
    main()
