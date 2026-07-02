"""Run SWM planning eval on LangTable with our QwenWMModel as the reward source.

Mirrors `swms/scripts/evaluate_swm_hydra.py` but swaps `SWMGradModel` for
`planning.qwen_wm_model.QwenWMModel`. Everything downstream (env, diffusion
policy, MPC loop, goal generator, success criterion, video logging) is the
SWM eval pipeline unchanged.

Usage:
    python scripts/evaluate_qwen_lt.py --config-name eval_qwen_lt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# CUDA workspace config -- must be set before any CUDA op runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Make both this repo and the swms repo importable.
THIS_REPO = Path(__file__).resolve().parent.parent
SWMS_REPO = Path(os.environ.get("SWMS_REPO", THIS_REPO.parent / "swms")).resolve()
sys.path.insert(0, str(THIS_REPO))
sys.path.insert(0, str(SWMS_REPO))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

# swms imports (eval loop + planner + env wrappers)
from swm.evaluation import eval as swm_eval

# Our adapter
from planning.qwen_wm_model import QwenWMModel


def _resolve_tasks(cfg: DictConfig):
    raw_tasks = OmegaConf.select(cfg, "tasks", default=None)
    if raw_tasks is None or len(raw_tasks) == 0:
        raise ValueError("cfg.tasks must be provided as a non-empty list")
    out = []
    for i, t in enumerate(raw_tasks):
        if "block_combo" not in t or "diffusion_path" not in t:
            raise ValueError(
                f"tasks[{i}] needs both 'block_combo' and 'diffusion_path', got {t}"
            )
        out.append(
            {
                "block_combo": tuple(t.block_combo),
                "diffusion_path": t.diffusion_path,
            }
        )
    return out


@hydra.main(
    config_path="../conf",
    config_name="eval_qwen_lt",
    version_base=None,
)
def run(cfg: DictConfig):
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    root_save_path = cfg.paths.root_save_path
    os.makedirs(root_save_path, exist_ok=True)

    planning_cfg = cfg.planning
    tasks = _resolve_tasks(cfg)
    # Optional: pick a single task by index (used by SLURM array runs).
    if cfg.get("task_idx", None) is not None:
        tasks = [tasks[int(cfg.task_idx)]]
        print(f"task_idx={cfg.task_idx}: running only {tasks[0]['block_combo']}")

    device = cfg.get("device", "cuda")

    print(f"loading QwenWMModel from {cfg.paths.predictor_ckpt_path}")
    model = QwenWMModel(
        predictor_ckpt_path=cfg.paths.predictor_ckpt_path,
        model_id=cfg.paths.qwen_model_id,
        device=device,
        precision=torch.bfloat16,
        image_size=cfg.image_size,
        obs_horizon=cfg.obs_horizon,
        max_action_horizon=cfg.max_action_horizon,
    )

    overall_success = 0
    overall_total = 0
    overall_time = 0.0

    for task_idx, task in enumerate(tasks):
        block_combo = task["block_combo"]
        diffusion_path = task["diffusion_path"]

        task_desc = f"task {task_idx + 1}/{len(tasks)}: {block_combo[0]} -> {block_combo[1]}"
        success_count = 0
        total_time = 0.0

        for seed in tqdm(
            range(cfg.seed_start, cfg.seed_start + cfg.num_seeds),
            desc=f"Running {task_desc}",
        ):
            uid = f"{cfg.name}/{planning_cfg.planning_name}/{block_combo[0]}_{block_combo[1]}"
            run_name = f"{uid}/{seed}"
            output_dir = f"{root_save_path}/{run_name}/"

            reward_kwargs = {
                "block_combo": block_combo,
                "ood": cfg.get("ood", False),
            }

            # Clear the adapter's cross-frame state (frame cache + rolling
            # 2-frame history) so nothing leaks between episodes.
            model.reset_episode()

            success, time_taken = swm_eval(
                seed=seed,
                reward_type=cfg.goal_type,
                env_type=cfg.env_type,
                device=device,
                output_dir=output_dir,
                # SWMGradModel-only paths -- ignored when model is passed in
                ckpt_path="",
                processor_path="",
                model=model,
                diffusion_path=diffusion_path,
                diffusion=planning_cfg.diffusion,
                mppi=planning_cfg.mppi,
                gradient=planning_cfg.gradient,
                expert_diffusion=planning_cfg.expert_diffusion,
                precision=torch.bfloat16,
                action_skip=planning_cfg.action_skip,
                model_batch_size=cfg.model_batch_size,
                reward_kwargs=reward_kwargs,
                action_dim=cfg.action_dim,
                num_steps=planning_cfg.num_steps,
                num_actions_executed=planning_cfg.num_actions_executed,
                pred_horizon=planning_cfg.pred_horizon,
                num_samples=planning_cfg.num_samples,
                num_planning_iters=planning_cfg.num_planning_iters,
                gradient_lr=planning_cfg.gradient_lr,
                gradient_clipping_value=planning_cfg.gradient_clipping_value,
                mppi_temperature=planning_cfg.mppi_temperature,
                intermediate_hm=cfg.intermediate_hm,
            )

            success_count += int(success)
            total_time += time_taken
            print(
                f"\n[{block_combo[0]}->{block_combo[1]}] seed {seed}: "
                f"success={success}  time={time_taken:.2f} min"
            )

        sr = success_count / max(1, cfg.num_seeds)
        print(f"\n{'='*80}")
        print(f"TASK SUMMARY: {block_combo[0]} -> {block_combo[1]}")
        print(f"  success_rate = {success_count}/{cfg.num_seeds} ({sr:.2%})")
        print(f"  avg_time     = {total_time / max(1, cfg.num_seeds):.2f} min")
        print(f"{'='*80}")

        with open(os.path.join(root_save_path, "results.txt"), "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"task: {block_combo[0]} -> {block_combo[1]}\n")
            f.write(f"diffusion_path: {diffusion_path}\n")
            f.write(f"success: {success_count}/{cfg.num_seeds} ({sr:.2%})\n")
            f.write(f"avg_time: {total_time / max(1, cfg.num_seeds):.2f} min\n")

        overall_success += success_count
        overall_total += cfg.num_seeds
        overall_time += total_time

    print(f"\n{'#'*80}")
    print(f"OVERALL: {overall_success}/{overall_total} ({overall_success/max(1, overall_total):.2%})")
    print(f"avg time per run: {overall_time/max(1, overall_total):.2f} min")
    print(f"{'#'*80}")

    with open(os.path.join(root_save_path, "results.txt"), "a") as f:
        f.write(f"\n{'#'*80}\nOVERALL: {overall_success}/{overall_total} "
                f"({overall_success/max(1, overall_total):.2%})\n")


if __name__ == "__main__":
    run()
