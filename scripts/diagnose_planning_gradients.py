"""Instrument gradient-based planning: how much do we actually move the
diffusion proposal, and what do the gradients look like?

Monkey-patches swm.planning_algos.plan_model_gradient with a faithful copy
that records, per replan call:
  - per-iteration action-gradient norm (pre-clip; post-clip = min(pre, clip))
  - per-iteration weighted reward sum (does the optimizer climb its objective?)
  - total action displacement ||a_final - a_init|| (L2, mean|.|, max|.|),
    plus the initial proposal's norm for scale

Works for BOTH our QwenWMModel (--model qwen) and SWM's PaliGemmaWM
(--model swm; swm_eval constructs SWMGradModel internally from --swm-ckpt).
Identical planning hyperparams => numbers are directly comparable.

Usage (ours):
    python scripts/diagnose_planning_gradients.py --model qwen \
        --predictor-ckpt $B/oneshot_runs/bce_teacher_balanced_v1/checkpoints/predictor_epoch3.pt \
        --num-seeds 5 --out $B/runs/grad_diag/bce_teacher_bal_e3.json

Usage (SWM, on the host holding paligemma ckpts):
    python scripts/diagnose_planning_gradients.py --model swm \
        --swm-ckpt $SWMS_REPO/ckpts/paligemma_wm_ogbench \
        --num-seeds 5 --out .../swm_paligemma.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

THIS_REPO = Path(__file__).resolve().parent.parent
SWMS_REPO = Path(os.environ.get("SWMS_REPO", THIS_REPO.parent / "swms")).resolve()
sys.path.insert(0, str(THIS_REPO))
sys.path.insert(0, str(SWMS_REPO))

import torch

try:
    import ogbench  # noqa: F401  (gym env registration)
except ImportError:
    pass

import swm.planning_algos as PA
from swm.evaluation import eval as swm_eval

RECORDS = []  # one entry per plan_model_gradient call


def instrumented_plan_model_gradient(diffusion_model, get_rewards_fn, pln_cfg,
                                     ret_intermediate=False, action_samp=None):
    """Faithful copy of swm.planning_algos.plan_model_gradient + logging."""
    if action_samp is None:
        action_samp = PA.sample_initial_actions(diffusion_model=diffusion_model, pln_cfg=pln_cfg)
    original_t = action_samp.clone()
    original = original_t.numpy()
    action_samp.requires_grad = True
    action_hist, weighted_rewards_hist = [], []
    action_optimizer = torch.optim.SGD([action_samp], lr=pln_cfg.gradient_lr)

    rec = {"grad_pre": [], "grad_post": [], "reward": []}
    for i in range(pln_cfg.n_planning_itrs):
        action_optimizer.zero_grad()
        rewards, weighted_rewards, rewards_with_grad_sum = get_rewards_fn(
            action_samp, action_skip=pln_cfg.action_skip, gradient=True
        )
        if ret_intermediate:
            action_hist.append(action_samp.cpu().detach().numpy().copy())
            weighted_rewards_hist.append(weighted_rewards.copy())

        loss = -rewards_with_grad_sum
        loss.backward()
        pre = torch.nn.utils.clip_grad_norm_(
            [action_samp], max_norm=pln_cfg.gradient_clipping_value
        )
        action_optimizer.step()
        with torch.no_grad():
            action_samp.clamp_(-pln_cfg.max_action_value, pln_cfg.max_action_value)

        rec["grad_pre"].append(float(pre))
        rec["grad_post"].append(float(min(float(pre), float(pln_cfg.gradient_clipping_value))))
        rec["reward"].append(float(rewards_with_grad_sum))

    final_t = action_samp.detach()
    delta = final_t - original_t
    rec.update({
        "disp_l2": float(delta.norm()),
        "disp_mean_abs": float(delta.abs().mean()),
        "disp_max_abs": float(delta.abs().max()),
        "init_l2": float(original_t.norm()),
        "n_iters": int(pln_cfg.n_planning_itrs),
        "lr": float(pln_cfg.gradient_lr),
        "clip": float(pln_cfg.gradient_clipping_value),
        "max_action_value": float(pln_cfg.max_action_value),
    })
    RECORDS.append(rec)

    action_np = final_t.numpy()
    if ret_intermediate:
        return action_np, rewards, weighted_rewards, action_hist, weighted_rewards_hist
    return action_np, rewards, weighted_rewards


# install the patch (get_plan resolves the name from module globals at call time)
PA.plan_model_gradient = instrumented_plan_model_gradient

OGB_TASKS = [
    ("blue_cube", "green_cube"),
    ("blue_cube", "yellow_cube"),
    ("yellow_cube", "red_cube"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "swm"), required=True)
    ap.add_argument("--predictor-ckpt", default=None, help="qwen mode")
    ap.add_argument("--swm-ckpt", default=None, help="swm mode: paligemma ckpt dir")
    ap.add_argument("--use-deepstack", action="store_true", help="qwen mode: deepstack for predictions")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--seed-start", type=int, default=6000)
    ap.add_argument("--num-seeds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    block_combo = OGB_TASKS[args.task_idx]
    diffusion_path = str(SWMS_REPO / "ckpts" / "ogbench_base_diffusion" /
                         f"{block_combo[0]}_{block_combo[1]}.pt")

    model = None
    ckpt_path, processor_path = "", ""
    if args.model == "qwen":
        assert args.predictor_ckpt, "--predictor-ckpt required for qwen"
        from planning.qwen_wm_model import QwenWMModel
        model = QwenWMModel(
            predictor_ckpt_path=args.predictor_ckpt,
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            device=device, precision=torch.bfloat16,
            image_size=448, obs_horizon=2, max_action_horizon=16,
            use_deepstack_for_predictions=args.use_deepstack,
        )
    else:
        assert args.swm_ckpt, "--swm-ckpt required for swm"
        ckpt_path = args.swm_ckpt
        processor_path = args.swm_ckpt

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        RECORDS.clear()
        if model is not None and hasattr(model, "reset_episode"):
            model.reset_episode()
        success, time_taken = swm_eval(
            seed=seed, reward_type="stack_blocks", env_type="ogbench",
            device=device, output_dir=str(out.parent / f"ep_{seed}"),
            ckpt_path=ckpt_path, processor_path=processor_path, model=model,
            diffusion_path=diffusion_path,
            diffusion=True, mppi=False, gradient=True, expert_diffusion=False,
            precision=torch.bfloat16, action_skip=8, model_batch_size=16,
            reward_kwargs={"block_combo": block_combo, "ood": False},
            action_dim=5, num_steps=50, num_actions_executed=4,
            pred_horizon=16, num_samples=1, num_planning_iters=20,
            gradient_lr=0.2, gradient_clipping_value=10.0,
            mppi_temperature=1.0, intermediate_hm=False,
        )
        results.append({"seed": seed, "success": bool(success),
                        "time_min": float(time_taken), "replans": list(RECORDS)})
        n = len(RECORDS)
        if n:
            import statistics as st
            d = [r["disp_l2"] for r in RECORDS]
            g0 = [r["grad_pre"][0] for r in RECORDS]
            gl = [r["grad_pre"][-1] for r in RECORDS]
            rdelta = [r["reward"][-1] - r["reward"][0] for r in RECORDS]
            print(f"seed {seed}: success={success} replans={n} "
                  f"disp_l2 mean={st.mean(d):.4f} "
                  f"grad_pre iter0 mean={st.mean(g0):.4g} last={st.mean(gl):.4g} "
                  f"reward climb mean={st.mean(rdelta):+.4f}", flush=True)
        with open(out, "w") as f:
            json.dump({"config": vars(args), "results": results}, f)
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
