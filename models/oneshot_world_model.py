"""One-shot conditional world model: frozen encoder + OneShotPredictor.

Forward I/O is deliberately different from DINO-WM's VWorldModel because the
training pattern is different (variable-horizon target frame, not autoregressive
next-frame). The trainer (train_oneshot.py) drives this directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class OneShotWorldModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, H, W) in [0, 1] -> (B, T, num_patches, emb_dim)"""
        B, T = frames.shape[:2]
        flat = rearrange(frames, "b t c h w -> (b t) c h w")
        z = self.encoder(flat)  # (B*T, P, D)
        return rearrange(z, "(b t) p d -> b t p d", b=B)

    def forward(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        history:     (B, obs_horizon, 3, H, W) in [0, 1]
        actions:     (B, max_action_horizon, action_dim) — pad past valid H with zeros
        action_mask: (B, max_action_horizon) bool — True at valid action positions
        target:      (B, 3, H, W) in [0, 1] — frame at the horizon implied by action_mask

        Returns dict with:
            loss          scalar (1 - mean cosine_sim)
            cos_sim       scalar (mean over batch and patches)
            z_pred        (B, P, D)
            z_target      (B, P, D) — detached
            horizons      (B,) long — action_mask.sum(-1)
        """
        z_obs = self.encode(history)  # (B, obs_horizon, P, D)
        # Encode target with the same frozen encoder; detach so gradient does not
        # try to update encoder parameters that already have requires_grad=False.
        with torch.no_grad():
            z_target = self.encoder(target).detach()  # (B, P, D)

        z_pred = self.predictor(z_obs, actions, action_mask=action_mask)  # (B, P, D)

        # Patch-level cosine similarity, averaged over patches and batch.
        cos = F.cosine_similarity(z_pred, z_target, dim=-1)  # (B, P)
        cos_sim = cos.mean()
        loss = 1.0 - cos_sim

        # Baseline: identity prediction (copy last history frame's latent).
        # If our predictor's cos_sim is not meaningfully higher than this, it's
        # learned to do nothing — useful diagnostic when frames are very similar.
        z_identity = z_obs[:, -1, :, :].detach()  # (B, P, D)
        cos_identity = F.cosine_similarity(z_identity, z_target, dim=-1)  # (B, P)

        return {
            "loss": loss,
            "cos_sim": cos_sim.detach(),
            "cos_sim_identity": cos_identity.mean().detach(),
            "z_pred": z_pred,
            "z_target": z_target,
            "horizons": action_mask.sum(dim=-1).long(),
            "cos_per_sample": cos.mean(dim=-1).detach(),  # (B,) for per-horizon bucketing
            "cos_per_sample_identity": cos_identity.mean(dim=-1).detach(),
        }
