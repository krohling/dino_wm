"""DINO-WM dataset for preprocessed SWM-Next trajectories (LT and OGB).

Reads per-episode .pt files produced by scripts/preprocess_swm_pkls.py.
Each file is a dict with keys (T = episode length):
    frames:  uint8   (T, 3, S, S)
    actions: float32 (T, action_dim)
    proprio: float32 (T, proprio_dim)
"""
import json
from pathlib import Path
from typing import Callable, Optional

import torch

from .traj_dset import TrajDataset, TrajSlicerDataset, split_traj_datasets


class SWMTrajDataset(TrajDataset):
    """Lazy per-episode loader for SWM-Next preprocessed data."""

    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        include_ood: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action

        with open(self.data_path / "manifest.json") as f:
            manifest = json.load(f)
        episodes = manifest["episodes"]
        if not include_ood:
            episodes = [e for e in episodes if not e.get("ood", False)]
        if n_rollout is not None:
            episodes = episodes[: n_rollout]
        if not episodes:
            raise RuntimeError(f"no episodes after filtering in {self.data_path}")

        self.episodes = episodes
        self.seq_lengths = [int(e["length"]) for e in episodes]
        self.action_dim = int(episodes[0]["action_dim"])
        self.proprio_dim = int(episodes[0]["proprio_dim"])
        self.state_dim = self.proprio_dim  # no separate state channel

        if normalize_action:
            self._fit_action_stats()
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
        self.proprio_mean = torch.zeros(self.proprio_dim)
        self.proprio_std = torch.ones(self.proprio_dim)

    def _ep_path(self, ep) -> Path:
        return self.data_path / ep["filename"]

    def _fit_action_stats(self):
        n = 0
        s = torch.zeros(self.action_dim, dtype=torch.float64)
        s2 = torch.zeros(self.action_dim, dtype=torch.float64)
        for ep in self.episodes:
            d = torch.load(self._ep_path(ep), map_location="cpu", weights_only=True)
            a = d["actions"].double()
            s += a.sum(dim=0)
            s2 += (a * a).sum(dim=0)
            n += a.shape[0]
        mean = (s / n).float()
        var = (s2 / n).float() - mean * mean
        self.action_mean = mean
        self.action_std = var.clamp_min(1e-8).sqrt()

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        out = []
        for ep in self.episodes:
            d = torch.load(self._ep_path(ep), map_location="cpu", weights_only=True)
            out.append((d["actions"] - self.action_mean) / self.action_std)
        return torch.cat(out, dim=0)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        d = torch.load(self._ep_path(ep), map_location="cpu", weights_only=True)
        image = d["frames"].float() / 255.0  # (T, 3, S, S) in [0, 1]
        if self.transform is not None:
            image = self.transform(image)
        action = (d["actions"] - self.action_mean) / self.action_std
        proprio = (d["proprio"] - self.proprio_mean) / self.proprio_std
        state = proprio.clone()
        return {"visual": image, "proprio": proprio}, action, state, {}

    def __len__(self):
        return len(self.episodes)


def _load_split(
    transform,
    data_path: str,
    normalize_action: bool = True,
    n_rollout: Optional[int] = None,
    split_ratio: float = 0.9,
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 1,
    include_ood: bool = False,
):
    full = SWMTrajDataset(
        data_path=data_path,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        include_ood=include_ood,
    )
    n_total = len(full)
    if n_total < 2:
        raise RuntimeError(
            f"need >=2 episodes for a train/val split, got {n_total} in {data_path}"
        )
    train, val = split_traj_datasets(full, train_fraction=split_ratio)
    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val, num_frames, frameskip)
    return (
        {"train": train_slices, "valid": val_slices},
        {"train": train, "valid": val},
    )


def load_langtable_slice_train_val(**kwargs):
    return _load_split(**kwargs)


def load_ogbench_slice_train_val(**kwargs):
    return _load_split(**kwargs)
