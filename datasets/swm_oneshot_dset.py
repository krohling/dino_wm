"""Variable-horizon dataset for one-shot conditional world-model training.

Mirrors SWM's training pattern: each sample is (initial obs window, length-H
action prefix, target frame at horizon H). H varies across samples — uniform
in {1..max_action_horizon} for train, fixed sweep over `eval_horizons` for val.

Reads per-episode .pt files produced by scripts/preprocess_swm_pkls.py:
    frames:  uint8 (T, 3, S, S)
    actions: float32 (T, action_dim)
    proprio: float32 (T, proprio_dim)   # ignored here
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import io

import numpy as np
import torch
from torch.utils.data import Dataset


def _decode_jpeg(buf: bytes) -> torch.Tensor:
    """JPEG bytes -> (3, H, W) float tensor in [0, 1]."""
    from PIL import Image
    with Image.open(io.BytesIO(buf)) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


class SWMOneShotDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_action_horizon: int = 16,
        obs_horizon: int = 2,
        eval_horizons: Optional[Sequence[int]] = None,
        n_rollout: Optional[int] = None,
        include_ood: bool = False,
        normalize_action: bool = True,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
        episode_ids: Optional[List[str]] = None,
        seed: int = 0,
    ):
        self.data_path = Path(data_path)
        self.max_action_horizon = max_action_horizon
        self.obs_horizon = obs_horizon
        self.eval_horizons = tuple(eval_horizons) if eval_horizons else None
        self.seed = seed

        with open(self.data_path / "manifest.json") as f:
            manifest = json.load(f)
        eps = manifest["episodes"]
        if not include_ood:
            eps = [e for e in eps if not e.get("ood", False)]
        if episode_ids is not None:
            keep = set(episode_ids)
            eps = [e for e in eps if e["id"] in keep]
        if n_rollout is not None:
            eps = eps[: n_rollout]
        if not eps:
            raise RuntimeError(f"no episodes in {self.data_path}")
        self.episodes = eps
        self.action_dim = int(eps[0]["action_dim"])

        # Load everything to RAM. Two on-disk frame formats are supported:
        #   - raw:  frames = uint8 tensor (T, 3, S, S)      (original format)
        #   - jpeg: frames = list[bytes], each a JPEG image (compressed format,
        #           ~10x smaller; decoded lazily in _get_frame)
        self._cache: list[dict] = []
        for e in eps:
            d = torch.load(self.data_path / e["filename"], map_location="cpu", weights_only=False)
            frames = d["frames"]
            length = len(frames) if isinstance(frames, list) else int(frames.shape[0])
            self._cache.append({
                "frames": frames,
                "actions": d["actions"], # f32  (T, A)
                "length": length,
            })

        # Build sample index: list of (ep_idx, t_start) valid starts.
        # A valid start needs obs_horizon frames behind it and at least 1 action+frame ahead.
        starts = []
        for ep_idx, c in enumerate(self._cache):
            T = c["length"]
            t_min = self.obs_horizon - 1
            t_max = T - 2  # need actions[t_start] AND frames[t_start+1]
            if t_max < t_min:
                continue
            for t in range(t_min, t_max + 1):
                starts.append((ep_idx, t))
        self.starts = starts
        if not self.starts:
            raise RuntimeError(f"no valid sample starts in {self.data_path}")

        # Action normalization: fit from data if not provided.
        if normalize_action:
            if action_mean is None or action_std is None:
                s = torch.zeros(self.action_dim, dtype=torch.float64)
                s2 = torch.zeros(self.action_dim, dtype=torch.float64)
                n = 0
                for c in self._cache:
                    a = c["actions"].double()
                    s += a.sum(dim=0)
                    s2 += (a * a).sum(dim=0)
                    n += a.shape[0]
                mean = (s / n).float()
                var = (s2 / n).float() - mean * mean
                self.action_mean = mean
                self.action_std = var.clamp_min(1e-8).sqrt()
            else:
                self.action_mean = action_mean.float()
                self.action_std = action_std.float()
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)

        # In eval mode, multiply samples by len(eval_horizons) so each (start, H) is its own item.
        if self.eval_horizons is not None:
            expanded = []
            for s_idx, (ep_idx, t) in enumerate(self.starts):
                T = self._cache[ep_idx]["length"]
                max_avail = T - 1 - t  # frames available after t
                for H in self.eval_horizons:
                    if H <= max_avail and H <= self.max_action_horizon:
                        expanded.append((ep_idx, t, H))
            self.eval_samples = expanded

    def __len__(self) -> int:
        if self.eval_horizons is not None:
            return len(self.eval_samples)
        return len(self.starts)

    def __getitem__(self, idx: int):
        if self.eval_horizons is not None:
            ep_idx, t_start, H = self.eval_samples[idx]
        else:
            ep_idx, t_start = self.starts[idx]
            c = self._cache[ep_idx]
            max_avail = c["length"] - 1 - t_start
            cap = min(self.max_action_horizon, max_avail)
            # Per-sample RNG keyed by (seed, idx) — different across epochs because
            # the DataLoader's worker seeding mutates each epoch, but Python's
            # default rng_for_idx is reproducible per call.
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
            H = int(rng.integers(low=1, high=cap + 1))  # high is exclusive

        c = self._cache[ep_idx]
        frames_u8 = c["frames"]   # uint8 tensor (T, 3, S, S) OR list of JPEG bytes
        actions = c["actions"]    # f32  (T, A)

        # History: frames[t_start - obs_horizon + 1 .. t_start] inclusive
        hist_start = t_start - self.obs_horizon + 1
        if isinstance(frames_u8, list):
            # JPEG-compressed storage: decode only the frames this sample needs.
            history = torch.stack(
                [_decode_jpeg(frames_u8[t]) for t in range(hist_start, t_start + 1)], dim=0
            )
            target = _decode_jpeg(frames_u8[t_start + H])
        else:
            history = frames_u8[hist_start : t_start + 1].float() / 255.0  # (obs_horizon, 3, S, S)
            target = frames_u8[t_start + H].float() / 255.0  # (3, S, S)

        # Actions: actions[t_start .. t_start + H - 1], normalized, padded to max
        act_slice = actions[t_start : t_start + H]
        act_norm = (act_slice - self.action_mean) / self.action_std
        actions_padded = torch.zeros(self.max_action_horizon, self.action_dim, dtype=torch.float32)
        actions_padded[:H] = act_norm
        action_mask = torch.zeros(self.max_action_horizon, dtype=torch.bool)
        action_mask[:H] = True

        return {
            "history": history,
            "actions": actions_padded,
            "action_mask": action_mask,
            "target": target,
            "horizon": torch.tensor(H, dtype=torch.long),
        }


def load_swm_oneshot_train_val(
    data_path: str,
    max_action_horizon: int = 16,
    obs_horizon: int = 2,
    eval_horizons: Sequence[int] = (1, 2, 4, 8, 16),
    split_ratio: float = 0.9,
    n_rollout: Optional[int] = None,
    include_ood: bool = False,
    normalize_action: bool = True,
    seed: int = 42,
):
    """Split episodes by id (not by sample) so val episodes are unseen during train."""
    with open(Path(data_path) / "manifest.json") as f:
        manifest = json.load(f)
    eps = manifest["episodes"]
    if not include_ood:
        eps = [e for e in eps if not e.get("ood", False)]
    if n_rollout is not None:
        eps = eps[: n_rollout]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eps))
    n_train = max(1, int(round(split_ratio * len(eps))))
    train_ids = [eps[i]["id"] for i in perm[:n_train]]
    val_ids = [eps[i]["id"] for i in perm[n_train:]]
    if not val_ids:
        # tiny dataset — fall back to taking 1 val episode
        train_ids = [eps[i]["id"] for i in perm[: max(1, len(eps) - 1)]]
        val_ids = [eps[i]["id"] for i in perm[max(1, len(eps) - 1) :]]

    train = SWMOneShotDataset(
        data_path=data_path,
        max_action_horizon=max_action_horizon,
        obs_horizon=obs_horizon,
        eval_horizons=None,
        include_ood=include_ood,
        normalize_action=normalize_action,
        episode_ids=train_ids,
        seed=seed,
    )
    # Reuse train's action stats for val
    val = SWMOneShotDataset(
        data_path=data_path,
        max_action_horizon=max_action_horizon,
        obs_horizon=obs_horizon,
        eval_horizons=eval_horizons,
        include_ood=include_ood,
        normalize_action=normalize_action,
        action_mean=train.action_mean,
        action_std=train.action_std,
        episode_ids=val_ids,
        seed=seed + 1,
    )
    return train, val
