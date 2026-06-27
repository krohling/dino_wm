"""One-shot conditional world-model trainer for SWM-Next.

Trains a OneShotPredictor (no autoregression) on top of a frozen Qwen3-VL ViT
encoder, using cosine distance as the predictor loss and variable-length
action horizons sampled per training example.

Logged to wandb (offline) every step:
    train_loss, train_cos_sim, grad_norm, lr, throughput_samples_per_s,
    horizon_hist (per epoch)

Logged each validation pass (held-out episodes):
    val_loss, val_cos_sim,
    val_cos_sim_h{H}  for H in eval_horizons,
    val_loss_h{H}     for H in eval_horizons,
    gpu_mem_peak_gb

Usage:
    python train_oneshot.py \
        env_data_path=$DATASET_DIR/lt \
        training.epochs=50 training.batch_size=8

Single-GPU only (no accelerate / DDP) — keeps the training loop simple and
predictable for a research project.
"""
from __future__ import annotations

import argparse
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
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from datasets.swm_oneshot_dset import load_swm_oneshot_train_val
from models.oneshot_world_model import OneShotWorldModel
from models.qwen3_vl import Qwen3VLViTEncoder
from models.vit_oneshot import OneShotPredictor

log = logging.getLogger(__name__)


def build_model(cfg: DictConfig, action_dim: int) -> OneShotWorldModel:
    encoder = Qwen3VLViTEncoder(
        model_id=cfg.encoder.model_id,
        image_size=cfg.img_size,
        freeze=True,
    )
    predictor = OneShotPredictor(
        emb_dim=encoder.emb_dim,
        action_dim=action_dim,
        num_patches=encoder.num_patches,
        obs_horizon=cfg.obs_horizon,
        max_action_horizon=cfg.max_action_horizon,
        depth=cfg.predictor.depth,
        heads=cfg.predictor.heads,
        mlp_dim=cfg.predictor.mlp_dim,
        dropout=cfg.predictor.dropout,
    )
    return OneShotWorldModel(encoder=encoder, predictor=predictor)


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / 1e9


@torch.no_grad()
def run_validation(model, loader, device, eval_horizons: Sequence[int]) -> dict:
    model.eval()
    per_h_cos = defaultdict(list)
    per_h_cos_ident = defaultdict(list)
    per_h_loss = defaultdict(list)
    total_loss, total_cos, total_cos_ident, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        history = batch["history"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        horizons = batch["horizon"].to(device, non_blocking=True)

        out = model(history, actions, action_mask, target)
        cos = out["cos_per_sample"]  # (B,)
        cos_id = out["cos_per_sample_identity"]  # (B,)
        for h in eval_horizons:
            mask = (horizons == h)
            if mask.any():
                per_h_cos[h].append(cos[mask].mean().item())
                per_h_cos_ident[h].append(cos_id[mask].mean().item())
                per_h_loss[h].append((1.0 - cos[mask]).mean().item())
        bs = history.shape[0]
        total_loss += out["loss"].item() * bs
        total_cos += out["cos_sim"].item() * bs
        total_cos_ident += out["cos_sim_identity"].item() * bs
        n += bs

    out_dict = {
        "val_loss": total_loss / max(1, n),
        "val_cos_sim": total_cos / max(1, n),
        "val_cos_sim_identity": total_cos_ident / max(1, n),
        "val_cos_sim_gain": (total_cos - total_cos_ident) / max(1, n),
        "val_n_samples": n,
    }
    for h in eval_horizons:
        if per_h_cos[h]:
            out_dict[f"val_cos_sim_h{h}"] = float(np.mean(per_h_cos[h]))
            out_dict[f"val_cos_sim_identity_h{h}"] = float(np.mean(per_h_cos_ident[h]))
            out_dict[f"val_cos_sim_gain_h{h}"] = float(
                np.mean(per_h_cos[h]) - np.mean(per_h_cos_ident[h])
            )
            out_dict[f"val_loss_h{h}"] = float(np.mean(per_h_loss[h]))
    return out_dict


@hydra.main(config_path="conf", config_name="train_oneshot", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    run_dir = Path(os.getcwd())  # hydra cwd
    log.info(f"run dir: {run_dir}")
    log.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    # Datasets
    log.info(f"loading datasets from {cfg.data_path}")
    train_ds, val_ds = load_swm_oneshot_train_val(
        data_path=cfg.data_path,
        max_action_horizon=cfg.max_action_horizon,
        obs_horizon=cfg.obs_horizon,
        eval_horizons=tuple(cfg.eval_horizons),
        split_ratio=cfg.split_ratio,
        n_rollout=cfg.get("n_rollout", None),
        include_ood=cfg.get("include_ood", False),
        normalize_action=cfg.normalize_action,
        seed=cfg.seed,
    )
    action_dim = train_ds.action_dim
    log.info(
        f"train samples: {len(train_ds)}  val samples: {len(val_ds)}  "
        f"action_dim: {action_dim}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.val_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
    )

    # Model
    log.info("building model")
    model = build_model(cfg, action_dim=action_dim).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(f"params: trainable={n_trainable/1e6:.2f}M  total={n_total/1e6:.2f}M")

    # Optimizer (only on predictor — encoder is frozen)
    optimizer = torch.optim.AdamW(
        model.predictor.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Wandb
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity", None),
        name=cfg.wandb.get("name", run_dir.name),
        config=wandb_config,
        dir=str(run_dir),
        mode=os.environ.get("WANDB_MODE", "offline"),
    )

    # Sanity-check tensor shapes once.
    sample = next(iter(train_loader))
    log.info(
        f"sample shapes: history={sample['history'].shape}  "
        f"actions={sample['actions'].shape}  target={sample['target'].shape}  "
        f"horizon range = [{int(sample['horizon'].min())}, {int(sample['horizon'].max())}]"
    )

    # Create the checkpoints dir up-front (avoids race between best-val save
    # and regular save on epochs where only one of them fires).
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Save action stats so the planner can re-normalize at eval.
    torch.save(
        {"action_mean": train_ds.action_mean, "action_std": train_ds.action_std},
        run_dir / "action_stats.pt",
    )

    global_step = 0
    best_val_cos = -1.0
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        # Encoder must stay in eval mode (frozen BN/LN/dropout; predictor is the only thing learning).
        model.encoder.eval()
        for p in model.encoder.parameters():
            p.requires_grad = False

        epoch_start = time.time()
        reset_peak_mem()
        epoch_horizon_hist = np.zeros(cfg.max_action_horizon + 1, dtype=np.int64)
        running_loss, running_cos, n_seen = 0.0, 0.0, 0

        for step_idx, batch in enumerate(train_loader):
            t0 = time.time()
            history = batch["history"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            action_mask = batch["action_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            horizons_cpu = batch["horizon"].numpy()
            for h in horizons_cpu:
                epoch_horizon_hist[h] += 1

            out = model(history, actions, action_mask, target)
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.predictor.parameters(), max_norm=cfg.training.grad_clip
            )
            optimizer.step()

            bs = history.shape[0]
            running_loss += loss.item() * bs
            running_cos += out["cos_sim"].item() * bs
            n_seen += bs
            global_step += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_dt = time.time() - t0
            samples_per_s = bs / max(step_dt, 1e-9)

            if global_step % cfg.training.log_every == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/cos_sim": out["cos_sim"].item(),
                        "train/grad_norm": float(grad_norm),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/samples_per_s": samples_per_s,
                        "epoch": epoch + step_idx / max(1, len(train_loader)),
                    },
                    step=global_step,
                )

        epoch_dt = time.time() - epoch_start
        train_loss = running_loss / max(1, n_seen)
        train_cos = running_cos / max(1, n_seen)

        # Horizon histogram (one log per epoch).
        h_bins = np.where(epoch_horizon_hist > 0)[0]
        h_table = wandb.Table(
            columns=["horizon", "count"],
            data=[[int(h), int(epoch_horizon_hist[h])] for h in h_bins],
        )

        log.info(
            f"Epoch {epoch}  train_loss={train_loss:.4f}  train_cos={train_cos:.4f}  "
            f"dt={epoch_dt:.1f}s  peak_mem={peak_mem_gb():.2f}GB"
        )

        # Validation
        val_metrics = run_validation(model, val_loader, device, cfg.eval_horizons)
        msg_parts = [
            f"val_loss={val_metrics['val_loss']:.4f}",
            f"val_cos={val_metrics['val_cos_sim']:.4f}",
            f"id_cos={val_metrics['val_cos_sim_identity']:.4f}",
            f"gain={val_metrics['val_cos_sim_gain']:+.4f}",
        ]
        for h in cfg.eval_horizons:
            k = f"val_cos_sim_gain_h{h}"
            if k in val_metrics:
                msg_parts.append(f"gain_h{h}={val_metrics[k]:+.4f}")
        log.info("  " + "  ".join(msg_parts))

        wandb.log(
            {
                "epoch_summary/train_loss": train_loss,
                "epoch_summary/train_cos_sim": train_cos,
                "epoch_summary/peak_mem_gb": peak_mem_gb(),
                "epoch_summary/epoch_time_s": epoch_dt,
                "epoch_summary/horizon_hist": h_table,
                **{f"epoch_summary/{k}": v for k, v in val_metrics.items()},
                "epoch": epoch,
            },
            step=global_step,
        )

        # Checkpoint
        if epoch % cfg.training.save_every == 0 or epoch == cfg.training.epochs:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "predictor": model.predictor.state_dict(),
                "action_mean": train_ds.action_mean,
                "action_std": train_ds.action_std,
                "cfg": wandb_config,
            }
            ckpt_path = run_dir / f"checkpoints/predictor_epoch{epoch}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, run_dir / "checkpoints/predictor_latest.pt")
            log.info(f"  saved {ckpt_path.name}")

        # Track best val
        if val_metrics["val_cos_sim"] > best_val_cos:
            best_val_cos = val_metrics["val_cos_sim"]
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "predictor": model.predictor.state_dict(),
                    "action_mean": train_ds.action_mean,
                    "action_std": train_ds.action_std,
                    "val_cos_sim": best_val_cos,
                    "cfg": wandb_config,
                },
                run_dir / "checkpoints/predictor_best.pt",
            )

    log.info(f"training done. best val_cos_sim={best_val_cos:.4f}")
    run.finish()


if __name__ == "__main__":
    main()
