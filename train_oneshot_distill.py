"""One-shot conditional world-model trainer with VLM-distillation loss.

Extends the cosine-only recipe (train_oneshot.py): in addition to matching
the encoder's future latent in cosine space, the predictor is trained so that
Qwen3-VL's VQA answer on the PREDICTED latent matches its answer on the REAL
future frame (precomputed by scripts/precompute_teacher.py).

    loss = cos_weight * (1 - cos_sim(z_pred, z_target))
         + kl_weight  * BCE(p_yes_student, p_yes_teacher)

The student path (z_pred -> merger -> LLM -> p_yes) runs with gradients
through the frozen merger+LLM into the predictor; gradient checkpointing on
the LLM keeps memory manageable. Teacher values come from sidecar JSONs --
no teacher forward at train time.

Sampling: each training sample draws (start t, horizon H) as usual; its
distillation target is a randomly chosen question attached to frame t+H.
Samples whose target frame has no questions fall back to cosine-only.

Usage:
    python train_oneshot_distill.py \
        data_path=$BASE/world-model-data/ogb \
        training.epochs=20 training.batch_size=8 \
        distill.kl_weight=1.0 distill.cos_weight=1.0
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from datasets.swm_oneshot_dset import SWMOneShotDataset, load_swm_oneshot_train_val
from models.oneshot_world_model import OneShotWorldModel
from models.qwen3_vl import Qwen3VLViTEncoder
from models.vit_oneshot import OneShotPredictor

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- helpers

class TeacherStore:
    """Loads teacher sidecar JSONs; maps (episode_id, frame_idx) -> entries."""

    def __init__(self, data_path: str):
        self.dir = Path(data_path) / "teacher"
        self._cache: dict[str, list] = {}
        if not self.dir.is_dir():
            raise RuntimeError(
                f"{self.dir} missing -- run scripts/precompute_teacher.py first"
            )

    def entries(self, episode_id: str, frame_idx: int) -> list:
        if episode_id not in self._cache:
            p = self.dir / f"{episode_id}.json"
            if not p.exists():
                self._cache[episode_id] = []
            else:
                with open(p) as f:
                    self._cache[episode_id] = json.load(f)
        frames = self._cache[episode_id]
        if not frames or frame_idx >= len(frames):
            return []
        return [e for e in frames[frame_idx] if e]


class DistillDataset(SWMOneShotDataset):
    """SWMOneShotDataset + a randomly-drawn question for the target frame.

    label_source:
      "teacher" -- soft target = frozen VLM's P(yes) on the real future frame
                   (student learns to be VQA-equivalent to the truth as READ
                   by the judge)
      "oracle"  -- hard target = simulator ground truth (SWM's actual label
                   source; student learns to make the judge emit TRUE answers
                   on predicted latents, even where the judge misreads reality)
    """

    def __init__(self, *args, teacher_store: TeacherStore = None,
                 label_source: str = "teacher", **kwargs):
        super().__init__(*args, **kwargs)
        assert label_source in ("teacher", "oracle"), label_source
        self.teacher = teacher_store
        self.label_source = label_source
        self._rng = np.random.default_rng(kwargs.get("seed", 0) + 7)

    def _target_of(self, e) -> float:
        if self.label_source == "teacher":
            return float(e["p_yes"])
        return 1.0 if e.get("oracle") else 0.0

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # Recover which episode / target frame this sample refers to
        if self.eval_horizons is not None:
            ep_idx, t_start, H = self.eval_samples[idx]
        else:
            ep_idx, t_start = self.starts[idx]
            H = int(item["horizon"])
        ep_id = self.episodes[ep_idx]["id"]
        target_frame = t_start + H
        entries = self.teacher.entries(ep_id, target_frame) if self.teacher else []
        if self.label_source == "oracle":
            entries = [e for e in entries if e.get("oracle") is not None]
        if entries:
            # Labels are ~77% "no"; unbalanced sampling collapses the student
            # to constant-no. Draw from the yes pool (per the active label
            # source) half the time when possible.
            yes_pool = [e for e in entries if self._target_of(e) >= 0.5]
            no_pool = [e for e in entries if self._target_of(e) < 0.5]
            if yes_pool and (not no_pool or self._rng.random() < 0.5):
                pool = yes_pool
            else:
                pool = no_pool or yes_pool
            e = pool[self._rng.integers(len(pool))]
            item["question"] = e["q"]
            item["teacher_p_yes"] = torch.tensor(self._target_of(e), dtype=torch.float32)
            item["has_teacher"] = torch.tensor(True)
        else:
            item["question"] = ""
            item["teacher_p_yes"] = torch.tensor(0.0)
            item["has_teacher"] = torch.tensor(False)
        return item


def collate_keep_strings(batch):
    """Default collate for tensors; keep 'question' as a list of strings."""
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if k == "question":
            out[k] = vals
        else:
            out[k] = torch.stack(vals, dim=0)
    return out


# --------------------------------------------------------------------- student VQA head

class StudentVQAHead:
    """Wraps the frozen Qwen3-VL merger+LLM for scoring predicted latents
    with gradients enabled (for distillation)."""

    def __init__(self, model_id: str, device, precision=torch.bfloat16, image_size=448):
        from planning.qwen_wm_model import QwenWMModel  # reuse plumbing
        # Build a minimal QwenWMModel without a predictor by loading with the
        # dummy LT checkpoint is awkward; instead we lift the pieces we need.
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.device = device
        self.precision = precision
        self.image_size = image_size
        self.full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=precision, low_cpu_mem_usage=True
        ).to(device).eval()
        for p in self.full_model.parameters():
            p.requires_grad = False
        # Gradient checkpointing keeps activation memory manageable when we
        # backprop through the (frozen) LLM into the predictor.
        self.full_model.gradient_checkpointing_enable()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        _m = self.full_model
        self.visual = _m.model.visual if hasattr(_m, "model") and hasattr(_m.model, "visual") else _m.visual
        self.merger = self.visual.merger
        self.image_token_id = self.full_model.config.image_token_id

        # Borrow prompt-building + LLM-forward machinery from QwenWMModel via
        # small standalone reimplementations (kept in sync with planning code).
        self._prompt_cache: dict[str, object] = {}
        self._qwm_proto = None  # lazy: reuse QwenWMModel methods unbound

    # -- prompt building (mirrors QwenWMModel._build_prompt_info) --
    def _prompt(self, question: str):
        if question in self._prompt_cache:
            return self._prompt_cache[question]
        from planning.qwen_wm_model import _PromptInfo
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question} Answer with one word: yes or no."},
            ],
        }]
        dummy_img = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        chat_str = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        try:
            inputs = self.processor(text=[chat_str], images=[dummy_img], return_tensors="pt",
                                    padding=False, return_mm_token_type_ids=True)
        except TypeError:
            inputs = self.processor(text=[chat_str], images=[dummy_img], return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"][0]
        mm_tti = inputs.get("mm_token_type_ids")
        def singles(cands):
            out = []
            for c in cands:
                ids = self.tokenizer(c, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    out.append(ids[0])
            return out
        info = _PromptInfo(
            text=question,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"][0],
            image_token_positions=(input_ids == self.image_token_id).nonzero(as_tuple=False).squeeze(-1),
            desired_token_ids=singles([" Yes", " yes", "Yes", "yes"]),
            other_token_ids=singles([" No", " no", "No", "no"]),
            weight=1.0,
            mm_token_type_ids=mm_tti[0] if mm_tti is not None else None,
        )
        self._prompt_cache[question] = info
        return info

    def merge(self, z_pred: torch.Tensor) -> torch.Tensor:
        """(B, P_in, D_in) fp32 -> (B, P_out, D_out) via frozen merger (bf16)."""
        B, P_in, D_in = z_pred.shape
        flat = z_pred.to(self.precision).reshape(B * P_in, D_in)
        out = self.merger(flat)
        P_out = out.shape[0] // B
        return out.reshape(B, P_out, out.shape[-1])

    def p_yes(self, prompt_info, img_embeds: torch.Tensor) -> torch.Tensor:
        """Batched LLM forward (with grad) -> (B,) P(yes). No deepstack for
        predicted latents (predictor has no deepstack equivalent); zeros are
        used, which matches what the planner does when deepstack is absent."""
        B = img_embeds.shape[0]
        device = self.device
        input_ids = prompt_info.input_ids.unsqueeze(0).expand(B, -1).contiguous().to(device)
        attn = prompt_info.attention_mask.unsqueeze(0).expand(B, -1).contiguous().to(device)
        per_image = [img_embeds[b] for b in range(B)]
        ps = int(self.full_model.config.vision_config.patch_size)
        grid = self.image_size // ps
        image_grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.long, device=device).expand(B, 3).contiguous()
        dummy_pixels = torch.zeros(1, dtype=self.precision, device=device)
        extra = {}
        if prompt_info.mm_token_type_ids is not None:
            extra["mm_token_type_ids"] = prompt_info.mm_token_type_ids.unsqueeze(0).expand(B, -1).contiguous().to(device)

        def patched(*args, **kwargs):
            if kwargs.get("return_dict", False):
                import types
                return types.SimpleNamespace(pooler_output=tuple(per_image), deepstack_features=None)
            return tuple(per_image), None

        orig = self.full_model.model.get_image_features
        self.full_model.model.get_image_features = patched
        try:
            outputs = self.full_model.model(
                input_ids=input_ids, attention_mask=attn,
                pixel_values=dummy_pixels, image_grid_thw=image_grid_thw, **extra,
            )
            hidden = outputs.last_hidden_state
            logits = self.full_model.lm_head(hidden[:, -1, :]).float()
        finally:
            self.full_model.model.get_image_features = orig
        probs = torch.softmax(logits, dim=-1)
        yes = probs[:, prompt_info.desired_token_ids].sum(dim=-1)
        no = probs[:, prompt_info.other_token_ids].sum(dim=-1)
        return yes / (yes + no).clamp_min(1e-12)


# --------------------------------------------------------------------- main

@hydra.main(config_path="conf", config_name="train_oneshot_distill", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    run_dir = Path(os.getcwd())
    log.info(f"run dir: {run_dir}")
    log.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda")

    teacher = TeacherStore(cfg.data_path)

    # --- datasets (episode split identical to the cosine recipe) ---
    # Re-use load_swm_oneshot_train_val's split logic by rebuilding with
    # DistillDataset. We duplicate the split code here to inject the class.
    with open(Path(cfg.data_path) / "manifest.json") as f:
        manifest = json.load(f)
    eps = [e for e in manifest["episodes"] if not e.get("ood", False)]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(eps))
    n_train = max(1, int(round(cfg.split_ratio * len(eps))))
    train_ids = [eps[i]["id"] for i in perm[:n_train]]
    val_ids = [eps[i]["id"] for i in perm[n_train:]] or [eps[perm[-1]]["id"]]

    label_source = str(cfg.distill.get("label_source", "teacher"))
    train_ds = DistillDataset(
        data_path=cfg.data_path, max_action_horizon=cfg.max_action_horizon,
        obs_horizon=cfg.obs_horizon, eval_horizons=None,
        normalize_action=cfg.normalize_action, episode_ids=train_ids,
        seed=cfg.seed, teacher_store=teacher, label_source=label_source,
    )
    val_ds = DistillDataset(
        data_path=cfg.data_path, max_action_horizon=cfg.max_action_horizon,
        obs_horizon=cfg.obs_horizon, eval_horizons=tuple(cfg.eval_horizons),
        normalize_action=cfg.normalize_action,
        action_mean=train_ds.action_mean, action_std=train_ds.action_std,
        episode_ids=val_ids, seed=cfg.seed + 1, teacher_store=teacher,
        label_source=label_source,
    )
    log.info(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}  action_dim: {train_ds.action_dim}")

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.training.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate_keep_strings,
                              persistent_workers=cfg.training.num_workers > 0)
    # Cap val (LLM forward per sample -- unbounded val would take hours on
    # the expanded dataset's 5-horizon grid).
    val_cap = int(cfg.training.get("val_max_samples", 3000))
    if len(val_ds) > val_cap:
        g = torch.Generator().manual_seed(cfg.seed)
        idx = torch.randperm(len(val_ds), generator=g)[:val_cap].tolist()
        val_ds = torch.utils.data.Subset(val_ds, idx)
        log.info(f"val capped to {val_cap} samples")
    val_loader = DataLoader(val_ds, batch_size=cfg.training.val_batch_size, shuffle=False,
                            num_workers=cfg.training.num_workers, pin_memory=True,
                            collate_fn=collate_keep_strings,
                            persistent_workers=cfg.training.num_workers > 0)

    # --- models ---
    log.info("loading student VQA head (frozen Qwen3-VL)")
    head = StudentVQAHead(cfg.encoder.model_id, device=device, image_size=cfg.img_size)

    log.info("building encoder wrapper (shares weights conceptually but separate load)")
    encoder = Qwen3VLViTEncoder(model_id=cfg.encoder.model_id, image_size=cfg.img_size, freeze=True).to(device)
    # NOTE: this loads the ViT a second time (~1.2 GB bf16) -- acceptable.

    predictor = OneShotPredictor(
        emb_dim=encoder.emb_dim, action_dim=train_ds.action_dim,
        num_patches=encoder.num_patches, obs_horizon=cfg.obs_horizon,
        max_action_horizon=cfg.max_action_horizon,
        depth=cfg.predictor.depth, heads=cfg.predictor.heads,
        mlp_dim=cfg.predictor.mlp_dim, dropout=cfg.predictor.dropout,
    ).to(device)  # fp32
    if cfg.get("init_from", None):
        ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        predictor.load_state_dict(ckpt["predictor"])
        log.info(f"initialized predictor from {cfg.init_from}")

    model = OneShotWorldModel(encoder=encoder, predictor=predictor).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)

    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.get("entity"),
                     name=cfg.wandb.get("name", run_dir.name),
                     config=OmegaConf.to_container(cfg, resolve=True),
                     dir=str(run_dir), mode=os.environ.get("WANDB_MODE", "offline"))

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"action_mean": train_ds.action_mean, "action_std": train_ds.action_std},
               run_dir / "action_stats.pt")

    cos_w = float(cfg.distill.cos_weight)
    kl_w = float(cfg.distill.kl_weight)
    global_step = 0
    best_val = -1.0

    for epoch in range(1, cfg.training.epochs + 1):
        model.train(); model.encoder.eval()
        t0 = time.time()
        run_loss = run_cos = run_bce = n_seen = n_teacher = 0

        for batch in train_loader:
            history = batch["history"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            action_mask = batch["action_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            teacher_p = batch["teacher_p_yes"].to(device)
            has_t = batch["has_teacher"].to(device)
            questions = batch["question"]

            out = model(history, actions, action_mask, target)
            cos_loss = out["loss"]

            bce_loss = torch.zeros((), device=device)
            if has_t.any() and kl_w > 0:
                # group by question so each LLM batch shares a prompt
                z_pred = out["z_pred"]  # (B, P, D) fp32, grads intact
                img_embeds = head.merge(z_pred)
                idx_by_q = defaultdict(list)
                for i, q in enumerate(questions):
                    if has_t[i]:
                        idx_by_q[q].append(i)
                # Class-weighted BCE: the expanded (play/noisy-heavy) dataset has
                # sparse teacher-yes questions -- per-frame balanced sampling
                # alone still yields ~25% yes and the student collapses to
                # constant-no. Weight yes-target samples up.
                yes_w = float(cfg.distill.get("yes_weight", 3.0))
                bce_terms = []
                w_total = 0.0
                for q, idxs in idx_by_q.items():
                    pi = head._prompt(q)
                    p_yes = head.p_yes(pi, img_embeds[idxs])
                    t_p = teacher_p[idxs]
                    w = torch.where(t_p >= 0.5, torch.full_like(t_p, yes_w), torch.ones_like(t_p))
                    bce_terms.append(
                        F.binary_cross_entropy(p_yes.clamp(1e-6, 1 - 1e-6), t_p, weight=w, reduction="sum")
                    )
                    w_total += float(w.sum())
                bce_loss = torch.stack(bce_terms).sum() / max(1.0, w_total)

            loss = cos_w * cos_loss + kl_w * bce_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), cfg.training.grad_clip)
            optimizer.step()

            bs = history.shape[0]
            run_loss += loss.item() * bs; run_cos += cos_loss.item() * bs
            run_bce += float(bce_loss) * bs; n_seen += bs
            n_teacher += int(has_t.sum())
            global_step += 1
            if global_step % cfg.training.log_every == 0:
                wandb.log({"train/loss": loss.item(), "train/cos_loss": cos_loss.item(),
                           "train/bce_loss": float(bce_loss), "train/grad_norm": float(gnorm),
                           "epoch": epoch}, step=global_step)

        # ---- validation: cosine metrics + teacher agreement on real-frame logits
        # Teacher answers are ~77% "no" (blocks mostly aren't touching/stacked),
        # so raw agreement is skew-blind: constant-NO scores 77.5%. We report
        # per-side agreement and select best on the BALANCED mean.
        model.eval()
        v_cos = v_n = 0
        agree_yes = n_yes = agree_no = n_no = 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(device); actions = batch["actions"].to(device)
                action_mask = batch["action_mask"].to(device); target = batch["target"].to(device)
                teacher_p = batch["teacher_p_yes"].to(device); has_t = batch["has_teacher"].to(device)
                questions = batch["question"]
                out = model(history, actions, action_mask, target)
                v_cos += float(out["cos_sim"]) * history.shape[0]; v_n += history.shape[0]
                if has_t.any():
                    img_embeds = head.merge(out["z_pred"])
                    idx_by_q = defaultdict(list)
                    for i, q in enumerate(questions):
                        if has_t[i]:
                            idx_by_q[q].append(i)
                    for q, idxs in idx_by_q.items():
                        p_yes = head.p_yes(head._prompt(q), img_embeds[idxs])
                        t_yes = teacher_p[idxs] >= 0.5
                        s_yes = p_yes >= 0.5
                        agree_yes += int(((s_yes == t_yes) & t_yes).sum())
                        n_yes += int(t_yes.sum())
                        agree_no += int(((s_yes == t_yes) & ~t_yes).sum())
                        n_no += int((~t_yes).sum())

        val_cos = v_cos / max(1, v_n)
        acc_yes = agree_yes / max(1, n_yes)
        acc_no = agree_no / max(1, n_no)
        val_agree_bal = (acc_yes + acc_no) / 2
        val_agree_raw = (agree_yes + agree_no) / max(1, n_yes + n_no)
        dt = time.time() - t0
        log.info(f"Epoch {epoch}  loss={run_loss/max(1,n_seen):.4f}  cos={run_cos/max(1,n_seen):.4f}  "
                 f"bce={run_bce/max(1,n_seen):.4f}  val_cos={val_cos:.4f}  "
                 f"agree_bal={val_agree_bal:.3f} (yes={acc_yes:.3f} n={n_yes}, no={acc_no:.3f} n={n_no})  "
                 f"raw={val_agree_raw:.3f}  dt={dt:.0f}s")
        wandb.log({"epoch_summary/val_cos_sim": val_cos,
                   "epoch_summary/val_teacher_agree_balanced": val_agree_bal,
                   "epoch_summary/val_teacher_agree_raw": val_agree_raw,
                   "epoch_summary/val_agree_yes": acc_yes,
                   "epoch_summary/val_agree_no": acc_no,
                   "epoch_summary/epoch_time_s": dt, "epoch": epoch}, step=global_step)
        val_agree = val_agree_bal  # best-model selection keys on balanced

        ckpt = {"epoch": epoch, "predictor": predictor.state_dict(),
                "action_mean": train_ds.action_mean, "action_std": train_ds.action_std,
                "cfg": OmegaConf.to_container(cfg, resolve=True)}
        if epoch % cfg.training.save_every == 0 or epoch == cfg.training.epochs:
            torch.save(ckpt, run_dir / f"checkpoints/predictor_epoch{epoch}.pt")
        torch.save(ckpt, run_dir / "checkpoints/predictor_latest.pt")
        if val_agree > best_val:
            best_val = val_agree
            torch.save(ckpt, run_dir / "checkpoints/predictor_best.pt")

    log.info(f"done. best val_teacher_agree={best_val:.3f}")
    run.finish()


if __name__ == "__main__":
    main()
