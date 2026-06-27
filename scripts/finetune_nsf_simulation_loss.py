#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_nsf_simulation_loss.py

Driver for enhanced NSF workflow:
1) Fit enhanced NSF to a raw checkpoint.
2) Export an NSF-initialized checkpoint.
3) Optionally call train_warp.py for simulation-loss fine-tuning.

The final simulation fine-tuning command depends on your train_warp.py arguments.
Run `python train_warp.py --help` to check whether it supports resume/init checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Allow running from /content/PhysTwin/scripts or /content/PhysTwin
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from enhanced_neural_spring_field import fit_enhanced_nsf_to_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case-name", required=True)
    p.add_argument("--base-path", required=True)
    p.add_argument("--train-frame", type=int, default=134)
    p.add_argument("--topology", required=True)
    p.add_argument("--raw-checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--field-fit-steps", type=int, default=2000)
    p.add_argument("--smooth-weight", type=float, default=1e-3)
    p.add_argument("--num-regions", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--region-embed-dim", type=int, default=8)
    p.add_argument("--run-simulation-finetune", action="store_true")
    p.add_argument("--train-extra-args", default="", help="Extra args for train_warp.py. Use {checkpoint} placeholder. Example: '--resume_checkpoint {checkpoint} --max_iter 100'")
    p.add_argument("--project-dir", default="/content/PhysTwin")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_field = out_dir / "enhanced_neural_spring_field.pt"
    output_checkpoint = out_dir / "enhanced_nsf_init_checkpoint.pth"
    summary_path = out_dir / "enhanced_nsf_summary.json"

    print("=" * 100)
    print("Step 1: fitting enhanced NSF")
    result = fit_enhanced_nsf_to_raw(topology_path=args.topology, raw_checkpoint=args.raw_checkpoint, output_field=output_field, output_checkpoint=output_checkpoint, num_regions=args.num_regions, steps=args.field_fit_steps, lr=args.lr, smooth_weight=args.smooth_weight, hidden_dim=args.hidden_dim, region_embed_dim=args.region_embed_dim, device="cuda")

    summary = {"case_name": args.case_name, "base_path": args.base_path, "topology": args.topology, "raw_checkpoint": args.raw_checkpoint, "out_dir": str(out_dir), "enhanced_nsf_fit": result, "simulation_finetune": None}

    if args.run_simulation_finetune:
        print("=" * 100)
        print("Step 2: simulation-loss fine-tuning")
        extra = args.train_extra_args.format(checkpoint=str(output_checkpoint))
        cmd = [sys.executable, "train_warp.py", "--case_name", args.case_name, "--base_path", args.base_path, "--train_frame", str(args.train_frame)] + shlex.split(extra)
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        env["WANDB_DISABLED"] = "true"
        env["SKIP_OPEN3D_VIDEO"] = "1"
        print("Command:", " ".join(cmd))
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=args.project_dir, env=env, text=True, capture_output=True)
        t1 = time.time()
        log_path = out_dir / "simulation_finetune_stdout_stderr.txt"
        log_path.write_text("STDOUT\n" + proc.stdout + "\n\nSTDERR\n" + proc.stderr, encoding="utf-8")
        summary["simulation_finetune"] = {"cmd": cmd, "return_code": proc.returncode, "seconds": t1 - t0, "log": str(log_path)}
        print("return code:", proc.returncode)
        print("seconds:", t1 - t0)
        print("STDOUT tail:\n", proc.stdout[-3000:])
        print("STDERR tail:\n", proc.stderr[-3000:])
    else:
        print("Skipping simulation-loss fine-tuning. Add --run-simulation-finetune after confirming train_warp.py resume arguments.")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
