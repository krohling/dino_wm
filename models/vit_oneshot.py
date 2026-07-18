"""One-shot conditional predictor for SWM-Next.

Replaces DINO-WM's autoregressive ViTPredictor with a single-forward-pass
model that takes:
    z_obs:       (B, obs_horizon, num_patches, emb_dim) — encoded history
    actions:     (B, max_action_horizon, action_dim)    — padded action seq
    action_mask: (B, max_action_horizon) bool           — True where valid

and emits the predicted latent of the frame at horizon H = action_mask.sum(-1):
    z_pred:      (B, num_patches, emb_dim)

H is implicit in the action_mask — the predictor's attention is constrained
to attend only to valid action tokens, so the same model handles any horizon
from 1 to max_action_horizon with no architecture change.

Mirrors SWM's "predict answer conditioned on (image, variable-length action
prefix)" pattern, but with a latent target instead of a VQA answer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class OneShotPredictor(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1152,
        action_dim: int = 2,
        num_patches: int = 784,
        obs_horizon: int = 2,
        max_action_horizon: int = 16,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        io_dim: int | None = None,
    ):
        """io_dim: when the latent space (e.g. post-merger LLM embeddings,
        ~2048-4096 dim) is wider than the transformer we can afford, set
        io_dim to the latent dim and emb_dim to the internal width; linear
        in/out projections bridge the two. io_dim=None means io == emb_dim
        (the original behavior)."""
        super().__init__()
        assert emb_dim % heads == 0, f"emb_dim {emb_dim} not divisible by heads {heads}"
        self.emb_dim = emb_dim
        self.io_dim = io_dim if io_dim is not None else emb_dim
        self.action_dim = action_dim
        self.num_patches = num_patches
        self.obs_horizon = obs_horizon
        self.max_action_horizon = max_action_horizon

        if self.io_dim != emb_dim:
            self.in_proj = nn.Linear(self.io_dim, emb_dim)
            self.out_proj = nn.Linear(emb_dim, self.io_dim)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        # Action -> emb_dim (LayerNorm to stabilize across tasks with different action scales)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        # Learnable readout tokens (the model writes its prediction into these positions)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Position embeddings — separate per role so the model can tell visual
        # vs action vs query positions apart even before any frame/time info.
        self.visual_pos = nn.Parameter(torch.zeros(1, obs_horizon * num_patches, emb_dim))
        self.action_pos = nn.Parameter(torch.zeros(1, max_action_horizon, emb_dim))
        self.query_pos = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        nn.init.trunc_normal_(self.visual_pos, std=0.02)
        nn.init.trunc_normal_(self.action_pos, std=0.02)
        nn.init.trunc_normal_(self.query_pos, std=0.02)

        # Bidirectional transformer (no causal mask).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(emb_dim)

    @property
    def n_visual_tokens(self) -> int:
        return self.obs_horizon * self.num_patches

    def forward(
        self,
        z_obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        z_obs:       (B, obs_horizon, num_patches, emb_dim)
        actions:     (B, max_action_horizon, action_dim) — pad past valid H with zeros
        action_mask: (B, max_action_horizon) bool, True at valid action positions

        Returns: z_pred (B, num_patches, emb_dim)
        """
        B = z_obs.shape[0]
        assert z_obs.shape[1] == self.obs_horizon, (z_obs.shape, self.obs_horizon)
        assert z_obs.shape[2] == self.num_patches
        assert z_obs.shape[3] == self.io_dim, (z_obs.shape, self.io_dim)
        assert actions.shape == (B, self.max_action_horizon, self.action_dim), (
            actions.shape, (B, self.max_action_horizon, self.action_dim)
        )

        z_obs = self.in_proj(z_obs)
        vis = z_obs.reshape(B, self.n_visual_tokens, self.emb_dim) + self.visual_pos
        act = self.action_proj(actions) + self.action_pos
        qry = self.query_tokens.expand(B, -1, -1) + self.query_pos
        seq = torch.cat([vis, act, qry], dim=1)  # (B, V + A + Q, D)

        # nn.TransformerEncoder convention: True = position to IGNORE in attention.
        if action_mask is not None:
            assert action_mask.dtype == torch.bool
            assert action_mask.shape == (B, self.max_action_horizon)
            pad = torch.zeros(B, seq.shape[1], dtype=torch.bool, device=seq.device)
            pad[:, self.n_visual_tokens : self.n_visual_tokens + self.max_action_horizon] = ~action_mask
        else:
            pad = None

        out = self.transformer(seq, src_key_padding_mask=pad)
        out = self.final_norm(out)
        return self.out_proj(out[:, -self.num_patches :, :])  # query positions
