#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard NSF simulation-loss fine-tuning.

This script starts from nsf_field_regularized_iter_199.pth, continues simulation
loss training, and then writes a standard inference.pkl without Open3D video.
"""

import os
import sys
import json
import pickle
import random
from pathlib import Path
from argparse import ArgumentParser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_SILENT"] = "true"
os.environ["SKIP_OPEN3D_VIDEO"] = "1"

import numpy as np
import torch
import warp as wp
from tqdm import tqdm

from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_cfg(base_path, case_name):
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    optimal_path = PROJECT_ROOT / "experiments_optimization" / case_name / "optimal_params.pkl"
    assert optimal_path.exists(), optimal_path
    with open(optimal_path, "rb") as f:
        cfg.set_optimal_params(pickle.load(f))

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c2w) for c2w in c2ws])

    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"


def load_ckpt(trainer, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    spring_Y = ckpt["spring_Y"]
    assert len(spring_Y) == trainer.simulator.n_springs, (
        f"spring_Y length mismatch: {len(spring_Y)} vs {trainer.simulator.n_springs}"
    )
    trainer.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
    trainer.simulator.set_collide(
        ckpt["collide_elas"].detach().clone(),
        ckpt["collide_fric"].detach().clone(),
    )
    trainer.simulator.set_collide_object(
        ckpt["collide_object_elas"].detach().clone(),
        ckpt["collide_object_fric"].detach().clone(),
    )
    print("[loaded ckpt]", checkpoint_path)
    print("spring_Y:", tuple(spring_Y.shape), float(spring_Y.mean()), float(spring_Y.std()))


def save_standard_inference(trainer, save_path):
    frame_len = trainer.dataset.frame_len
    trainer.simulator.set_init_state(
        trainer.simulator.wp_init_vertices,
        trainer.simulator.wp_init_velocities,
    )
    vertices = [
        wp.to_torch(trainer.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
    ]

    with wp.ScopedTimer("simulate"):
        for i in tqdm(range(1, frame_len)):
            if cfg.data_type == "real":
                trainer.simulator.set_controller_target(i, pure_inference=True)
            if trainer.simulator.object_collision_flag:
                trainer.simulator.update_collision_graph()
            if cfg.use_graph:
                wp.capture_launch(trainer.simulator.forward_graph)
            else:
                trainer.simulator.step()
            x = wp.to_torch(trainer.simulator.wp_states[-1].wp_x, requires_grad=False)
            vertices.append(x.cpu())
            trainer.simulator.set_init_state(
                trainer.simulator.wp_states[-1].wp_x,
                trainer.simulator.wp_states[-1].wp_v,
            )

    arr = torch.stack(vertices, dim=0).cpu().numpy()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(arr, f)
    print("[saved inference]", save_path, arr.shape, arr.dtype)


def main():
    ap = ArgumentParser()
    ap.add_argument("--base_path", required=True)
    ap.add_argument("--case_name", required=True)
    ap.add_argument("--train_frame", type=int, required=True)
    ap.add_argument("--init_checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--finetune_steps", type=int, default=200)
    args = ap.parse_args()

    set_seed(42)
    setup_cfg(args.base_path, args.case_name)
    cfg.iterations = int(args.finetune_steps)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.set_log_file(path=str(out), name="nsf_simft_train_log")

    trainer = InvPhyTrainerWarp(
        data_path=f"{args.base_path}/{args.case_name}/final_data.pkl",
        base_dir=str(out),
        train_frame=args.train_frame,
    )
    load_ckpt(trainer, args.init_checkpoint)

    trainer.visualize_sim = lambda *a, **kw: print("[skip visualize during train]")
    print("[start sim-ft]", cfg.iterations)
    trainer.train()

    ckpts = sorted((out / "train").glob("best_*.pth")) or sorted((out / "train").glob("iter_*.pth"))
    assert ckpts, "No checkpoint generated by sim-ft"
    best = ckpts[-1]
    print("[best ckpt]", best)

    setup_cfg(args.base_path, args.case_name)
    logger.set_log_file(path=str(out), name="nsf_simft_standard_inference_log")
    infer_trainer = InvPhyTrainerWarp(
        data_path=f"{args.base_path}/{args.case_name}/final_data.pkl",
        base_dir=str(out),
        train_frame=args.train_frame,
    )
    load_ckpt(infer_trainer, str(best))
    save_standard_inference(infer_trainer, out / "inference.pkl")

    summary = {
        "case_name": args.case_name,
        "init_checkpoint": args.init_checkpoint,
        "best_checkpoint": str(best),
        "output_dir": str(out),
        "inference": str(out / "inference.pkl"),
        "finetune_steps": args.finetune_steps,
    }
    with open(out / "nsf_simft_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
