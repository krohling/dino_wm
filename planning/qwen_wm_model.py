"""Qwen3-VL + OneShotPredictor as a drop-in for SWM's planning interface.

Implements the same `get_probabilistic_rewards_wm(action_seq, image, pred_horizon,
questions, batch_size, action_skip, gradient=False)` API that
`swm.semantic_world_model.SWMGradModel` provides, so we can re-use SWM's planner
(`swm.planning_algos.get_plan`) and eval harness (`swm.evaluation.eval`)
unchanged.

Compute flow for one candidate action sequence at one horizon point:
    1. Encode current image with Qwen3-VL ViT -> pre-merger latent z_0 (784, 1152)
    2. Roll OneShotPredictor with (z_obs=[z_{-1}, z_0], action_prefix) -> z_pred (784, 1152)
    3. Pass z_pred through Qwen3-VL's merger -> post-merger image embeds (196, 4096)
    4. Build a chat prompt "<image>\n<question>" and tokenize
    5. Splice the post-merger image embeds into the input embedding sequence at
       the image-token positions
    6. Forward through Qwen3-VL's LLM -> logits at the assistant-turn start
    7. Softmax over the final vocab, read off P("yes") and P("no")

Returned rewards have shape (num_questions, num_actions, pred_horizon); like SWM,
the horizon axis only has nonzero entries at the action_skip-strided positions
(other slots filled with the nearest evaluated point).
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

log = logging.getLogger(__name__)


# Qwen3-VL uses SigLIP-style normalization (image_mean=image_std=0.5).
# Verified via AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct").image_processor.
_QWEN_MEAN = (0.5, 0.5, 0.5)
_QWEN_STD = (0.5, 0.5, 0.5)


@dataclasses.dataclass
class _PromptInfo:
    """Cached per-question prompt tokens to avoid re-tokenizing in the inner loop."""
    text: str
    input_ids: torch.Tensor          # (L,)
    attention_mask: torch.Tensor     # (L,)
    image_token_positions: torch.Tensor  # (n_image_tokens,) long
    desired_token_ids: list          # all single-token variants of the desired answer (e.g. yes / Yes / " yes" / " Yes")
    other_token_ids: list            # all single-token variants of the other answer
    weight: float
    mm_token_type_ids: torch.Tensor | None = None  # (L,) -- required by transformers>=5 for M-RoPE


class QwenWMModel:
    """One-shot Qwen3-VL world model -> probabilistic-reward adapter."""

    def __init__(
        self,
        predictor_ckpt_path: str | Path,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        precision: torch.dtype = torch.bfloat16,
        image_size: int = 448,
        obs_horizon: int = 2,
        max_action_horizon: int = 16,
        chat_template_question_format: str = "{question}",
        use_deepstack_for_predictions: bool = True,
    ):
        # use_deepstack_for_predictions: when scoring PREDICTED latents, pass
        # the current frame's deepstack (True; approximation used by the cosine
        # model) or none (False; matches how the distill student was trained --
        # its readout never saw deepstack on predicted latents).
        self.use_deepstack_for_predictions = use_deepstack_for_predictions
        self.device = device
        self.precision = precision
        self.image_size = image_size
        self.obs_horizon = obs_horizon
        self.max_action_horizon = max_action_horizon
        self.chat_template_question_format = chat_template_question_format

        # 1. Load Qwen3-VL VLM (full model -- we need both visual and LLM)
        log.info(f"loading Qwen3-VL VLM: {model_id}")
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=precision, low_cpu_mem_usage=True,
        ).to(device).eval()
        for p in self.full_model.parameters():
            p.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        # transformers 5.x moved the visual encoder to `.model.visual`; 4.x had `.visual`.
        _m = self.full_model
        self.visual = _m.model.visual if hasattr(_m, "model") and hasattr(_m.model, "visual") else _m.visual
        # Discover the structural pieces we need to call directly:
        #   - vision blocks + patch_embed (run once on initial image)
        #   - merger (run on every rolled-out latent to lift 1152 -> 4096)
        self.merger = self.visual.merger
        self.image_token_id = self.full_model.config.image_token_id
        # Common Qwen3-VL: there are also vision-start/-end tokens; we only need
        # to find image_token_id positions in the input_ids to splice features.
        log.info(
            f"  image_token_id={self.image_token_id}  "
            f"text hidden_size={self.full_model.config.text_config.hidden_size}"
        )

        # 2. Load our predictor checkpoint
        log.info(f"loading predictor from {predictor_ckpt_path}")
        ckpt = torch.load(predictor_ckpt_path, map_location="cpu", weights_only=False)
        from models.vit_oneshot import OneShotPredictor
        cfg = ckpt["cfg"]
        action_dim = int(ckpt["action_mean"].shape[0])
        vc = self.full_model.config.vision_config
        grid = image_size // int(vc.patch_size)
        merge = int(getattr(vc, "spatial_merge_size", 2))
        # Infer the prediction space from the checkpoint weights themselves:
        #   query_tokens: (1, num_patches, internal_dim); in_proj present => io projections.
        sd = ckpt["predictor"]
        ck_patches = int(sd["query_tokens"].shape[1])
        ck_internal = int(sd["query_tokens"].shape[2])
        ck_io = int(sd["in_proj.weight"].shape[1]) if "in_proj.weight" in sd else None
        if ck_patches == (grid // merge) ** 2:
            self.predict_space = "post_merger"
            encoder_emb_dim = int(vc.out_hidden_size)   # latent dim the adapter moves around
            num_patches = ck_patches
        else:
            self.predict_space = "pre_merger"
            encoder_emb_dim = int(vc.hidden_size)
            num_patches = grid ** 2
        log.info(f"  predict_space={self.predict_space}  latent=({num_patches}x{encoder_emb_dim})  "
                 f"predictor internal={ck_internal} io={ck_io}")
        self.predictor = OneShotPredictor(
            emb_dim=ck_internal,
            io_dim=ck_io,
            action_dim=action_dim,
            num_patches=num_patches,
            obs_horizon=obs_horizon,
            max_action_horizon=max_action_horizon,
            depth=cfg["predictor"]["depth"],
            heads=cfg["predictor"]["heads"],
            mlp_dim=cfg["predictor"]["mlp_dim"],
            dropout=0.0,  # eval -> no dropout
        )
        missing, unexpected = self.predictor.load_state_dict(ckpt["predictor"], strict=False)
        if missing or unexpected:
            log.warning(f"predictor load: missing={missing[:3]} unexpected={unexpected[:3]}")
        # Keep the predictor in fp32: it was TRAINED in fp32 (encoder wrapper
        # returned .float()), and its task-relevant signal (deviation from
        # copy-last-frame, ~0.5% in cosine terms) is the same order as bf16's
        # ~0.4% relative resolution. bf16 here would quantize away the signal.
        self.predictor = self.predictor.to(device).float().eval()
        for p in self.predictor.parameters():
            p.requires_grad = False

        self.action_dim = action_dim
        self.num_patches = num_patches
        self.encoder_emb_dim = encoder_emb_dim
        self.action_mean = ckpt["action_mean"].to(device, precision)
        self.action_std = ckpt["action_std"].to(device, precision)

        # Normalize-from-[0,1] buffers for our predictor's encode step
        self._mean_b = torch.tensor(_QWEN_MEAN, device=device, dtype=precision).view(1, 3, 1, 1)
        self._std_b = torch.tensor(_QWEN_STD, device=device, dtype=precision).view(1, 3, 1, 1)

        # Hook the merger to capture pre-merger features when we run visual end-to-end.
        self._pre_merger: torch.Tensor | None = None
        self.visual.merger.register_forward_pre_hook(self._capture_pre_merger)

        # Cache for the most recent encoded image (the planner queries multiple
        # questions / horizons on the same starting image). Keyed on image
        # CONTENT (not id()!) -- Python recycles object addresses, so id()-based
        # caching silently returns stale encodings across MPC steps.
        self._last_image_key: int | None = None
        self._last_z_obs: torch.Tensor | None = None  # (obs_horizon, P, D), pre-merger
        self._last_deepstack: list | None = None      # captured from visual.forward; reused for predicted frames
        # Rolling previous-frame latent for real 2-frame history within an
        # episode (predictor trained with (frame_{t-1}, frame_t), never dupes).
        self._prev_z0: torch.Tensor | None = None

    # ------------------------------------------------------------------ hooks
    def _capture_pre_merger(self, module, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        self._pre_merger = x

    # ------------------------------------------------------------------ image preprocessing
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """PIL RGB -> (3, image_size, image_size) float [0, 1].

        Center-crops to a square first to match training preprocessing
        (preprocessor did the same on 720x1280 / 768x768 source frames).
        Without this, env frames at 320x180 get STRETCHED 1.78:1 -> 1:1 and
        the encoder sees geometrically distorted input the predictor never
        saw during training.
        """
        w, h = image.size
        if w != h:
            s = min(w, h)
            left = (w - s) // 2
            top = (h - s) // 2
            image = image.crop((left, top, left + s, top + s))
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Returns (num_patches, emb_dim) pre-merger latent for one image.

        Side effect: also caches the deepstack features captured during this
        visual.forward in self._last_deepstack so the LLM can be given the
        complete visual context downstream.
        """
        x = self._pil_to_tensor(image).to(self.device).to(self.precision)
        x = (x.unsqueeze(0) - self._mean_b) / self._std_b  # (1, 3, H, W)
        B, C, H, W = x.shape
        T = int(self.full_model.config.vision_config.temporal_patch_size)
        ps = int(self.full_model.config.vision_config.patch_size)
        m = int(self.full_model.config.vision_config.spatial_merge_size)
        gh, gw = H // ps, W // ps
        # Match Qwen's image processor patchify order exactly
        x = x.unsqueeze(1).expand(B, T, C, H, W)
        x = x.reshape(B, T, C, gh // m, m, ps, gw // m, m, ps)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
        x = x.reshape(B * gh * gw, T * C * ps * ps)
        thw = torch.tensor([[1, gh, gw]], dtype=torch.long, device=self.device)
        self._pre_merger = None
        # Capture pre_merger via the hook and deepstack via the return value.
        # transformers 4.x: visual() returns a tuple (embeds, deepstack_list).
        # transformers 5.x: returns BaseModelOutputWithDeepstackFeatures
        # (.pooler_output / .deepstack_features). The old isinstance(tuple)
        # check silently dropped deepstack on 5.x.
        out = self.visual(x, grid_thw=thw)
        if isinstance(out, tuple):
            deepstack = out[1] if len(out) >= 2 else None
        elif hasattr(out, "deepstack_features"):
            deepstack = out.deepstack_features
        else:
            deepstack = None
        self._last_deepstack = deepstack
        if self.predict_space == "post_merger":
            post = out[0] if isinstance(out, tuple) else getattr(out, "pooler_output", None)
            if post is None:
                post = out.last_hidden_state
            if isinstance(post, (tuple, list)):
                post = torch.cat(list(post), dim=0)
            self._pre_merger = None
            return post.view((gh // m) * (gw // m), -1)  # (196, 4096)
        pre = self._pre_merger
        self._pre_merger = None
        assert pre is not None
        return pre.view(gh * gw, -1)  # (num_patches, 1152)

    # ------------------------------------------------------------------ predictor rollout
    def _predict_latent(
        self, z_history: torch.Tensor, action_seq: torch.Tensor
    ) -> torch.Tensor:
        """z_history: (obs_horizon, P, D)
        action_seq: (B, H, action_dim) (already normalized, padded to H_max with zeros)
        action_mask is derived from H per batch row.
        Returns (B, P, D).
        """
        B, H_max, _ = action_seq.shape
        # Broadcast history across batch
        z_obs = z_history.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        # Build action mask from the row-wise count of non-padded actions
        # (caller is responsible for placing valid actions at the head and
        # zeros at the tail; we pass mask from outside since "all-zero" is a
        # valid action value).
        # Convention: caller passes pre-built mask via _predict_with_mask
        raise NotImplementedError  # use _predict_with_mask instead

    def _predict_with_mask(
        self,
        z_history: torch.Tensor,         # (obs_horizon, P, D)
        actions_padded: torch.Tensor,    # (B, H_max, action_dim) normalized + zero-padded
        action_mask: torch.Tensor,       # (B, H_max) bool
    ) -> torch.Tensor:
        B = actions_padded.shape[0]
        # Predictor runs in fp32 (see __init__); cast inputs accordingly.
        z_obs = z_history.float().unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        return self.predictor(z_obs, actions_padded.float(), action_mask=action_mask)

    # ------------------------------------------------------------------ merger
    def _merge(self, pre_merger_batch: torch.Tensor) -> torch.Tensor:
        """Apply Qwen3-VL's merger to a batch of pre-merger latents.

        pre_merger_batch: (B, P_in, D_in)   D_in=1152, P_in=784 for 448
        returns:         (B, P_out, D_out) D_out=4096, P_out=196 for 448
        """
        B, P_in, D_in = pre_merger_batch.shape
        flat = pre_merger_batch.reshape(B * P_in, D_in)
        out = self.merger(flat)
        # merger output is (B * P_out, D_out). Infer P_out from total size.
        total_out = out.shape[0]
        P_out = total_out // B
        D_out = out.shape[-1]
        return out.reshape(B, P_out, D_out)

    # ------------------------------------------------------------------ prompts
    def _build_prompt_info(self, question_tuple: Tuple[str, str, float]) -> _PromptInfo:
        text, desired_token_str, weight = question_tuple
        # Chat-template the question with a single image. We use a chat with one
        # user turn that contains an image + the question, and ask for a short
        # yes/no answer. add_generation_prompt=True leaves the cursor right
        # where the assistant's next token would appear.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{text} Answer with one word: yes or no."},
                ],
            }
        ]
        # We tokenize without supplying real images; the processor still emits
        # the image placeholder tokens. We supply features later via inputs_embeds.
        # Use a dummy 448x448 image so the processor knows the image grid shape.
        dummy_img = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        chat_str = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # transformers>=5 needs mm_token_type_ids for M-RoPE; ask the processor
        # for them (older processors ignore the kwarg or omit the key).
        try:
            inputs = self.processor(
                text=[chat_str], images=[dummy_img], return_tensors="pt", padding=False,
                return_mm_token_type_ids=True,
            )
        except TypeError:
            inputs = self.processor(
                text=[chat_str], images=[dummy_img], return_tensors="pt", padding=False,
            )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        mm_tti = inputs.get("mm_token_type_ids")
        mm_token_type_ids = mm_tti[0] if mm_tti is not None else None
        image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=False).squeeze(-1)

        # Yes/No token IDs -- gather ALL single-token variants of each side and
        # sum probability mass at evaluation time. Diagnostic showed Qwen3-VL
        # actually generates 'yes'/'no' (no leading space, lowercase) so picking
        # a single ' yes' variant lost the prob mass.
        yes_variants = [" Yes", " yes", "Yes", "yes"]
        no_variants = [" No", " no", "No", "no"]
        yes_ids = self._all_single_tokens(yes_variants)
        no_ids = self._all_single_tokens(no_variants)
        desired_ids = yes_ids if desired_token_str.lower() == "yes" else no_ids
        other_ids = no_ids if desired_token_str.lower() == "yes" else yes_ids

        return _PromptInfo(
            text=text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_positions=image_positions,
            desired_token_ids=desired_ids,
            other_token_ids=other_ids,
            weight=float(weight),
            mm_token_type_ids=mm_token_type_ids,
        )

    def _first_single_token(self, candidates: List[str]) -> int:
        for c in candidates:
            ids = self.tokenizer(c, add_special_tokens=False).input_ids
            if len(ids) == 1:
                return ids[0]
        return self.tokenizer(candidates[0], add_special_tokens=False).input_ids[0]

    def _all_single_tokens(self, candidates: List[str]) -> list[int]:
        """Return token ids for every candidate that tokenizes to a single token."""
        out = []
        for c in candidates:
            ids = self.tokenizer(c, add_special_tokens=False).input_ids
            if len(ids) == 1:
                out.append(ids[0])
        return out

    # ------------------------------------------------------------------ LLM forward
    def _llm_yes_no_probs(
        self,
        prompt: _PromptInfo,
        image_embeds_batch: torch.Tensor,  # (B, P_out, D_out)
        gradient: bool,
        deepstack_override: list | None = None,
    ) -> torch.Tensor:
        """Returns (B,) probability the model would emit one of prompt.desired_token_ids.

        Goes through the OUTER Qwen3VLModel.forward (not the inner text model)
        so that M-RoPE position_ids are computed correctly for image tokens and
        deepstack features are injected at the right LLM layers.

        For each forward, we monkey-patch model.get_image_features to return our
        pre-computed (image_embeds_batch, deepstack) instead of having it
        recompute from pixel_values. The outer forward still handles the
        masked_scatter splice + get_rope_index + deepstack_process internally.
        """
        B = image_embeds_batch.shape[0]
        device = self.device

        input_ids = prompt.input_ids.unsqueeze(0).expand(B, -1).contiguous().to(device)
        attn = prompt.attention_mask.unsqueeze(0).expand(B, -1).contiguous().to(device)

        P_out = image_embeds_batch.shape[1]
        n_img = prompt.image_token_positions.shape[0]
        if n_img != P_out:
            raise ValueError(
                f"prompt has {n_img} image tokens but features have {P_out} patches"
            )

        deepstack = deepstack_override if deepstack_override is not None else self._last_deepstack
        # Deepstack per layer must end up (B*P_out, D). Two accepted inputs:
        #   (P_out, D)    -- single-frame features, tile across the batch
        #   (B*P_out, D)  -- caller already concatenated per-frame features
        deepstack_batched = None
        if deepstack is not None:
            deepstack_batched = []
            for layer_feat in deepstack:
                if layer_feat.dim() != 2:
                    raise ValueError(f"unexpected deepstack feature shape {layer_feat.shape}")
                if layer_feat.shape[0] == P_out:
                    feat = layer_feat.unsqueeze(0).expand(B, -1, -1).reshape(-1, layer_feat.shape[-1])
                elif layer_feat.shape[0] == B * P_out:
                    feat = layer_feat
                else:
                    raise ValueError(
                        f"deepstack rows {layer_feat.shape[0]} matches neither "
                        f"P_out={P_out} nor B*P_out={B * P_out}"
                    )
                deepstack_batched.append(feat.to(self.precision))

        # Convert (B, P_out, D) -> list of B tensors of (P_out, D) for the
        # patched get_image_features return shape.
        per_image = [image_embeds_batch[b].to(self.precision) for b in range(B)]

        # The image_grid_thw the outer model expects: (B, 3) with [t=1, h=14, w=14] for our 448x448 with merge=2.
        merge = int(self.full_model.config.vision_config.spatial_merge_size)
        ps = int(self.full_model.config.vision_config.patch_size)
        gh_post = self.image_size // ps  # 28 pre-merger
        # post-merger grid count must satisfy product/merge^2 == P_out
        # i.e. (gh_post * gh_post) / 4 = 196 -> gh_post=28, post-merge grid 14x14
        grid_h = gh_post
        grid_w = gh_post
        image_grid_thw = torch.tensor(
            [[1, grid_h, grid_w]], dtype=torch.long, device=device
        ).expand(B, 3).contiguous()
        # Dummy pixel_values just so the `if pixel_values is not None` branch fires.
        dummy_pixels = torch.zeros(1, dtype=self.precision, device=device)

        def patched_get_image_features(*args, **kwargs):
            # transformers 5.x calls with return_dict=True and reads
            # .pooler_output / .deepstack_features; 4.x expects a tuple.
            if kwargs.get("return_dict", False):
                import types
                return types.SimpleNamespace(
                    pooler_output=tuple(per_image),
                    deepstack_features=deepstack_batched,
                )
            return tuple(per_image), deepstack_batched

        # transformers>=5 needs mm_token_type_ids for M-RoPE
        extra_kwargs = {}
        if prompt.mm_token_type_ids is not None:
            extra_kwargs["mm_token_type_ids"] = (
                prompt.mm_token_type_ids.unsqueeze(0).expand(B, -1).contiguous().to(device)
            )

        # Monkey-patch + call + restore
        orig = self.full_model.model.get_image_features
        self.full_model.model.get_image_features = patched_get_image_features
        try:
            with torch.enable_grad() if gradient else torch.no_grad():
                outputs = self.full_model.model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    pixel_values=dummy_pixels,
                    image_grid_thw=image_grid_thw,
                    **extra_kwargs,
                )
                hidden = outputs.last_hidden_state
                last_logits = self.full_model.lm_head(hidden[:, -1, :]).float()
        finally:
            self.full_model.model.get_image_features = orig

        probs = torch.softmax(last_logits, dim=-1)
        yes_mass = probs[:, prompt.desired_token_ids].sum(dim=-1)
        no_mass = probs[:, prompt.other_token_ids].sum(dim=-1)
        total = yes_mass + no_mass
        return yes_mass / total.clamp_min(1e-12)

    # ------------------------------------------------------------------ public API
    def reset_episode(self):
        """Clear cross-frame state. Call between episodes/seeds so the
        previous episode's final frame doesn't leak into the new episode's
        history slot."""
        self._last_image_key = None
        self._last_z_obs = None
        self._prev_z0 = None
        self._last_deepstack = None

    def get_probabilistic_rewards_wm(
        self,
        action_seq: torch.Tensor | np.ndarray,
        image: Image.Image,
        pred_horizon: int,
        questions: List[Tuple[str, str, float]],
        batch_size: int = 32,
        action_skip: int = 1,
        gradient: bool = False,
    ):
        """Implements the SWM SWMModel API.

        action_seq: (N, pred_horizon, action_dim)  raw env-action units
        image:      PIL.Image (current frame)
        pred_horizon: max horizon to evaluate (in raw env-action units)
        questions:  list of (text, desired_token, weight) tuples
        action_skip: horizon stride; queries are at h_step in
                     {action_skip, 2*action_skip, ..., pred_horizon}
        gradient:   if True, leave torch.enable_grad() on so the planner can
                    backprop through to action_seq.
        """
        if isinstance(action_seq, np.ndarray):
            action_seq = torch.from_numpy(action_seq)
        # fp32 all the way to the predictor (which runs fp32); .to() keeps the
        # autograd connection to the planner's CPU leaf tensor.
        action_seq = action_seq.to(self.device, dtype=torch.float32)
        N, T_full, A = action_seq.shape
        assert A == self.action_dim, (A, self.action_dim)
        assert T_full <= self.max_action_horizon, (
            f"action_seq horizon {T_full} > max trained horizon "
            f"{self.max_action_horizon}; planner config and predictor must agree"
        )

        # --- 1. Encode current image, build 2-frame history ---
        # Cache key = image CONTENT hash. id(image) is unsafe: the MPC loop
        # allocates a fresh PIL Image every outer step and CPython recycles
        # the freed address, so id() collides and returns stale features.
        image_key = hash(image.tobytes())
        if image_key != self._last_image_key:
            z_0 = self.encode_image(image)  # (P, D), pre-merger
            # Use the PREVIOUS frame's latent as history slot 0 when we have
            # one (matches training: history is (frame_{t-1}, frame_t)).
            # Fall back to duplication only at episode start.
            prev = self._prev_z0 if self._prev_z0 is not None else z_0
            self._last_z_obs = torch.stack([prev, z_0], dim=0)  # (2, P, D)
            self._prev_z0 = z_0
            self._last_image_key = image_key
        z_history = self._last_z_obs  # (2, P, D)

        # --- 2. Normalize actions and build padded tensors for each h_step ---
        h_steps = list(range(action_skip, pred_horizon + action_skip, action_skip))
        # Build the full set of (a_idx, h_step) tasks
        H_max = self.max_action_horizon
        rewards = np.zeros((len(questions), N, pred_horizon), dtype=np.float32)
        # Pre-cache per-question prompts
        prompt_infos = [self._build_prompt_info(q) for q in questions]

        action_mean = self.action_mean.view(1, 1, -1).float()
        action_std = self.action_std.view(1, 1, -1).float()
        action_norm_full = (action_seq - action_mean) / action_std  # (N, T_full, A) fp32

        # Build a single big batch of (a_idx, h_step) -> z_pred
        # Each row is one action prefix of length h_step, padded to H_max.
        big_a: List[torch.Tensor] = []
        big_mask: List[torch.Tensor] = []
        big_meta: List[Tuple[int, int]] = []
        for h_step in h_steps:
            for a_idx in range(N):
                pad = torch.zeros(H_max, A, dtype=torch.float32, device=self.device)
                prefix = action_norm_full[a_idx, :h_step]
                pad[:h_step] = prefix
                mask = torch.zeros(H_max, dtype=torch.bool, device=self.device)
                mask[:h_step] = True
                big_a.append(pad)
                big_mask.append(mask)
                big_meta.append((a_idx, h_step))
        big_a_t = torch.stack(big_a, dim=0)                  # (M, H_max, A)
        big_mask_t = torch.stack(big_mask, dim=0)            # (M, H_max)
        M = big_a_t.shape[0]
        log.debug(f"predictor batch M={M}  N={N}  h_steps={h_steps}")

        rewards_with_grad_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        ctx = contextlib.nullcontext() if gradient else torch.no_grad()
        # --- 3. Predict latents in chunks (fp32 predictor) ---
        z_preds_all = []
        with ctx:
            for s in range(0, M, batch_size):
                e = min(s + batch_size, M)
                z_pred = self._predict_with_mask(z_history, big_a_t[s:e], big_mask_t[s:e])
                z_preds_all.append(z_pred)
            z_preds = torch.cat(z_preds_all, dim=0)  # (M, P, D_in) fp32
            # --- 4. Project to post-merger image embeddings (merger is bf16) ---
            if self.predict_space == "post_merger":
                img_embeds = z_preds.to(self.precision)  # already in LLM space
            else:
                img_embeds = self._merge(z_preds.to(self.precision))  # (M, P_out, D_out)

            # --- 5. Score each prompt for each (a_idx, h_step) ---
            for q_idx, prompt in enumerate(prompt_infos):
                for s in range(0, M, batch_size):
                    e = min(s + batch_size, M)
                    if self.use_deepstack_for_predictions:
                        p_yes = self._llm_yes_no_probs(prompt, img_embeds[s:e], gradient)
                    else:
                        # Distill-trained student: score predicted latents
                        # exactly as during training (no deepstack).
                        p_yes = self._llm_yes_no_probs(
                            prompt, img_embeds[s:e], gradient, deepstack_override=[],
                        )
                    # Distribute to (q_idx, a_idx, h_step-action_skip:h_step)
                    for k, (a_idx, h_step) in enumerate(big_meta[s:e]):
                        # Fill the band ending at h_step
                        rewards[q_idx, a_idx, h_step - action_skip : h_step] = (
                            p_yes[k].detach().float().cpu().numpy()
                        )
                        if gradient:
                            rewards_with_grad_sum = (
                                rewards_with_grad_sum + p_yes[k] * prompt.weight
                            )

        # Apply weights to rewards (same convention as SWMGradModel)
        weighted_rewards = rewards.copy()
        for q_idx, prompt in enumerate(prompt_infos):
            weighted_rewards[q_idx] *= prompt.weight

        if gradient:
            return rewards, weighted_rewards, rewards_with_grad_sum
        return rewards, weighted_rewards

    @torch.no_grad()
    def get_scores(self, images, actions=None, questions=None):
        """VQA on REAL frames (no prediction) -- used by SWM's OGB goal
        generators for subgoal tracking (e.g., 'is the robot grasping X?'
        decides when to switch from the grasp subgoal to the stack subgoal).

        Matches SWMGradModel.get_scores contract: returns a tuple
        (P(yes) tensor, P(no) tensor), one entry per image. `actions` is
        ignored (the callers pass None).
        """
        if isinstance(questions, str):
            questions = [questions] * len(images)
        p_yes_parts = []
        for img, q in zip(images, questions):
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img, dtype=np.uint8))
            z = self.encode_image(img)  # caches this frame's deepstack
            if self.predict_space == "post_merger":
                emb = z.unsqueeze(0).to(self.precision)
            else:
                emb = self._merge(z.unsqueeze(0))
            prompt = self._build_prompt_info((str(q), "yes", 1.0))
            p = self._llm_yes_no_probs(prompt, emb, gradient=False)
            p_yes_parts.append(p)
        p_yes = torch.cat(p_yes_parts, dim=0)
        return (p_yes, 1.0 - p_yes)
