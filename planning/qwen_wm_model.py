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


# OpenAI-CLIP normalization (matches Qwen3-VL's image processor).
_QWEN_MEAN = (0.48145466, 0.4578275, 0.40821073)
_QWEN_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclasses.dataclass
class _PromptInfo:
    """Cached per-question prompt tokens to avoid re-tokenizing in the inner loop."""
    text: str
    input_ids: torch.Tensor          # (L,)
    attention_mask: torch.Tensor     # (L,)
    image_token_positions: torch.Tensor  # (n_image_tokens,) long
    desired_token_id: int
    other_token_id: int
    weight: float


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
    ):
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
        self.visual = self.full_model.visual
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
        encoder_emb_dim = int(self.full_model.config.vision_config.hidden_size)
        action_dim = int(ckpt["action_mean"].shape[0])
        num_patches = (image_size // int(self.full_model.config.vision_config.patch_size)) ** 2
        self.predictor = OneShotPredictor(
            emb_dim=encoder_emb_dim,
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
        self.predictor = self.predictor.to(device).to(precision).eval()
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
        # questions / horizons on the same starting image)
        self._last_image_id: int | None = None
        self._last_z_obs: torch.Tensor | None = None  # (obs_horizon, P, D), pre-merger

    # ------------------------------------------------------------------ hooks
    def _capture_pre_merger(self, module, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        self._pre_merger = x

    # ------------------------------------------------------------------ image preprocessing
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """PIL RGB -> (3, image_size, image_size) float [0, 1]."""
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Returns (num_patches, emb_dim) pre-merger latent for one image."""
        x = self._pil_to_tensor(image).to(self.device).to(self.precision)
        # Reuse the Qwen3VLViTEncoder patchify logic
        from models.qwen3_vl import Qwen3VLViTEncoder
        # We don't have a Qwen3VLViTEncoder instance, but the steps are simple --
        # inline the patchify so we don't need to load duplicate weights.
        x = (x.unsqueeze(0) - self._mean_b) / self._std_b  # (1, 3, H, W)
        B, C, H, W = x.shape
        T = int(self.full_model.config.vision_config.temporal_patch_size)
        ps = int(self.full_model.config.vision_config.patch_size)
        gh, gw = H // ps, W // ps
        x = x.unsqueeze(1).expand(B, T, C, H, W)
        x = x.reshape(B, T, C, gh, ps, gw, ps)
        x = x.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        x = x.reshape(B * gh * gw, T * C * ps * ps)
        thw = torch.tensor([[1, gh, gw]], dtype=torch.long, device=self.device)
        self._pre_merger = None
        _ = self.visual(x, grid_thw=thw)
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
        z_obs = z_history.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        return self.predictor(z_obs, actions_padded, action_mask=action_mask)

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
        inputs = self.processor(
            text=[chat_str], images=[dummy_img], return_tensors="pt", padding=False,
        )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=False).squeeze(-1)

        # Yes / No token IDs. We try lowercase first since SWM uses lowercase.
        # Qwen tokenizers usually have a leading space; "yes" vs " yes" vs "Yes".
        yes_candidates = [" yes", "yes", " Yes", "Yes"]
        no_candidates = [" no", "no", " No", "No"]
        yes_id = self._first_single_token(yes_candidates)
        no_id = self._first_single_token(no_candidates)
        desired_id = yes_id if desired_token_str.lower() == "yes" else no_id
        other_id = no_id if desired_token_str.lower() == "yes" else yes_id

        return _PromptInfo(
            text=text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_positions=image_positions,
            desired_token_id=desired_id,
            other_token_id=other_id,
            weight=float(weight),
        )

    def _first_single_token(self, candidates: List[str]) -> int:
        for c in candidates:
            ids = self.tokenizer(c, add_special_tokens=False).input_ids
            if len(ids) == 1:
                return ids[0]
        # Fall back: use the first id of the first candidate
        return self.tokenizer(candidates[0], add_special_tokens=False).input_ids[0]

    # ------------------------------------------------------------------ LLM forward
    def _llm_yes_no_probs(
        self,
        prompt: _PromptInfo,
        image_embeds_batch: torch.Tensor,  # (B, P_out, D_out)
        gradient: bool,
    ) -> torch.Tensor:
        """Returns (B,) probability the model would emit prompt.desired_token_id."""
        B = image_embeds_batch.shape[0]
        L = prompt.input_ids.shape[0]
        device = self.device

        # Build batched input_ids
        input_ids = prompt.input_ids.unsqueeze(0).expand(B, -1).to(device)
        attn = prompt.attention_mask.unsqueeze(0).expand(B, -1).to(device)
        # Get text-side embeddings
        embed_layer = self.full_model.get_input_embeddings()
        with torch.enable_grad() if gradient else torch.no_grad():
            inputs_embeds = embed_layer(input_ids).to(self.precision)
            # Splice our image embeddings into the image-token positions
            img_pos = prompt.image_token_positions.to(device)
            n_img = img_pos.shape[0]
            P_out = image_embeds_batch.shape[1]
            if n_img != P_out:
                raise ValueError(
                    f"prompt has {n_img} image tokens but predicted features "
                    f"have {P_out} patches. Adjust the chat template / image size."
                )
            # inputs_embeds shape: (B, L, D)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[:, img_pos, :] = image_embeds_batch.to(self.precision)

            # Forward through LLM; Qwen3-VL exposes the language model as
            # `self.full_model.model.language_model` or `self.full_model.language_model`.
            lm = getattr(self.full_model.model, "language_model", None) or self.full_model.language_model
            outputs = lm(inputs_embeds=inputs_embeds, attention_mask=attn)
            hidden = outputs.last_hidden_state  # (B, L, D)
            # Logits at the last position predict the FIRST generated token.
            last_logits = self.full_model.lm_head(hidden[:, -1, :]).float()  # (B, V)
            # Binary softmax over just yes/no
            yes_logit = last_logits[:, prompt.desired_token_id]
            no_logit = last_logits[:, prompt.other_token_id]
            p_yes = torch.softmax(torch.stack([yes_logit, no_logit], dim=-1), dim=-1)[:, 0]
        return p_yes

    # ------------------------------------------------------------------ public API
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
        action_seq = action_seq.to(self.device, dtype=self.precision)
        if gradient:
            action_seq.requires_grad_(True)
        N, T_full, A = action_seq.shape
        assert A == self.action_dim, (A, self.action_dim)
        assert T_full <= self.max_action_horizon, (
            f"action_seq horizon {T_full} > max trained horizon "
            f"{self.max_action_horizon}; planner config and predictor must agree"
        )

        # --- 1. Encode current image, build 2-frame history (frame_{-1} = frame_0) ---
        image_id = id(image)
        if image_id != self._last_image_id:
            z_0 = self.encode_image(image)  # (P, D), pre-merger
            # No real past frame available -> repeat current as the "previous" obs.
            # This matches the labeler-eval convention and is what SWM does too
            # at the start of a plan.
            z_hist = torch.stack([z_0, z_0], dim=0)  # (obs_horizon=2, P, D)
            self._last_z_obs = z_hist
            self._last_image_id = image_id
        z_history = self._last_z_obs  # (2, P, D)

        # --- 2. Normalize actions and build padded tensors for each h_step ---
        h_steps = list(range(action_skip, pred_horizon + action_skip, action_skip))
        # Build the full set of (a_idx, h_step) tasks
        H_max = self.max_action_horizon
        rewards = np.zeros((len(questions), N, pred_horizon), dtype=np.float32)
        # Pre-cache per-question prompts
        prompt_infos = [self._build_prompt_info(q) for q in questions]

        action_mean = self.action_mean.view(1, 1, -1).to(self.precision)
        action_std = self.action_std.view(1, 1, -1).to(self.precision)
        action_norm_full = (action_seq - action_mean) / action_std  # (N, T_full, A)

        # Build a single big batch of (a_idx, h_step) -> z_pred
        # Each row is one action prefix of length h_step, padded to H_max.
        big_a: List[torch.Tensor] = []
        big_mask: List[torch.Tensor] = []
        big_meta: List[Tuple[int, int]] = []
        for h_step in h_steps:
            for a_idx in range(N):
                pad = torch.zeros(H_max, A, dtype=self.precision, device=self.device)
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

        rewards_with_grad_sum = torch.zeros((), device=self.device, dtype=self.precision)
        ctx = contextlib.nullcontext() if gradient else torch.no_grad()
        # --- 3. Predict latents in chunks ---
        z_preds_all = []
        with ctx:
            for s in range(0, M, batch_size):
                e = min(s + batch_size, M)
                z_pred = self._predict_with_mask(z_history, big_a_t[s:e], big_mask_t[s:e])
                z_preds_all.append(z_pred)
            z_preds = torch.cat(z_preds_all, dim=0)  # (M, P, D_in)
            # --- 4. Project to post-merger image embeddings ---
            img_embeds = self._merge(z_preds)  # (M, P_out, D_out)

            # --- 5. Score each prompt for each (a_idx, h_step) ---
            for q_idx, prompt in enumerate(prompt_infos):
                for s in range(0, M, batch_size):
                    e = min(s + batch_size, M)
                    p_yes = self._llm_yes_no_probs(prompt, img_embeds[s:e], gradient)
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

    # Compatibility no-op: SWM's interface also defines get_scores; we only need
    # the rewards method for planning. Raise if anyone calls get_scores so it's
    # obvious this path isn't supported by our adapter.
    def get_scores(self, *args, **kwargs):
        raise NotImplementedError("QwenWMModel only implements get_probabilistic_rewards_wm")
