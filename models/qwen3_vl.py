"""Qwen3-VL vision encoder wrapper for DINO-WM.

Exposes pre-merger features (28x28=784 tokens at 1152-dim from a 448x448 input),
matching the DINO-WM encoder API used by VWorldModel.
"""
import torch
import torch.nn as nn
from einops import rearrange


# OpenAI CLIP / SigLIP-style normalization used by Qwen3-VL's image processor.
QWEN_MEAN = (0.48145466, 0.4578275, 0.40821073)
QWEN_STD = (0.26862954, 0.26130258, 0.27577711)


class Qwen3VLViTEncoder(nn.Module):
    """Frozen Qwen3-VL ViT, returning pre-merger patch features.

    Forward I/O matches the rest of DINO-WM:
        input:  x (B, 3, image_size, image_size), values in [0, 1]
        output: (B, num_patches, emb_dim)
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        image_size: int = 448,
        freeze: bool = True,
    ):
        super().__init__()
        # DO NOT put "dino" in the name -- it triggers a Resize in VWorldModel.
        self.name = "qwen3_vl"
        self.image_size = image_size

        from transformers import Qwen3VLForConditionalGeneration, AutoConfig

        cfg = AutoConfig.from_pretrained(model_id)
        vc = cfg.vision_config
        self.patch_size = int(vc.patch_size)
        self.temporal_patch_size = int(getattr(vc, "temporal_patch_size", 2))
        # Pre-merger hidden size (1152 for Qwen3-VL-8B).
        self.emb_dim = int(vc.hidden_size)

        assert image_size % self.patch_size == 0, (
            f"image_size {image_size} must be divisible by patch_size {self.patch_size}"
        )
        self.grid_h = image_size // self.patch_size
        self.grid_w = image_size // self.patch_size
        self.num_patches = self.grid_h * self.grid_w  # 784 at 448x448 patch=16

        # latent_ndim = 2 means VWorldModel treats output as (B, P, D).
        self.latent_ndim = 2

        # Load full model and keep just the visual branch.
        full = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.visual = full.visual
        del full

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad = False
            self.visual.eval()

        self.register_buffer(
            "_mean", torch.tensor(QWEN_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(QWEN_STD).view(1, 3, 1, 1), persistent=False
        )

        # Forward pre-hook on merger captures (B*num_patches, emb_dim).
        self._captured: torch.Tensor | None = None
        self.visual.merger.register_forward_pre_hook(self._capture_pre_merger)

    def _capture_pre_merger(self, module, inputs):
        # inputs is a tuple; the first arg is the post-blocks, pre-merger features.
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        self._captured = x

    def _to_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert (B, 3, H, W) -> (B*P, patch_dim), and produce grid_thw."""
        B, C, H, W = x.shape
        assert C == 3, f"expected 3 channels, got {C}"
        assert H == self.image_size and W == self.image_size, (
            f"expected {self.image_size}x{self.image_size}, got {H}x{W}"
        )
        # Normalize from [0,1] to CLIP-style stats.
        x = (x - self._mean) / self._std
        # Cast to visual dtype (bf16).
        target_dtype = next(self.visual.parameters()).dtype
        x = x.to(target_dtype)
        # Replicate temporally to match temporal_patch_size.
        T = self.temporal_patch_size
        x = x.unsqueeze(1).expand(B, T, C, H, W)
        # Split into spatial patches:
        # (B, T, C, gh, ps, gw, ps) -> (B, gh, gw, T, C, ps, ps) -> (B*gh*gw, T*C*ps*ps)
        ps = self.patch_size
        gh, gw = self.grid_h, self.grid_w
        x = x.reshape(B, T, C, gh, ps, gw, ps)
        x = x.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        patch_dim = T * C * ps * ps
        x = x.reshape(B * gh * gw, patch_dim)
        thw = torch.tensor(
            [[1, gh, gw]], dtype=torch.long, device=x.device
        ).repeat(B, 1)
        return x, thw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1] -> (B, num_patches, emb_dim)."""
        B = x.shape[0]
        pv, thw = self._to_patches(x)
        self._captured = None
        # Frozen path: no autograd through visual. We still emit a non-leaf
        # tensor so DataParallel/AMP wrappers behave.
        if not any(p.requires_grad for p in self.visual.parameters()):
            with torch.no_grad():
                _ = self.visual(pv, grid_thw=thw)
        else:
            _ = self.visual(pv, grid_thw=thw)
        pre = self._captured
        self._captured = None
        assert pre is not None, "merger pre-hook did not fire"
        pre = rearrange(pre, "(b p) d -> b p d", b=B)
        # Return fp32 for downstream stability.
        return pre.float()
