#!/usr/bin/env python3
"""Loss-driven NeuSpring-style topology search for PhysTwin.

Pipeline for each candidate topology:
1. Build a piecewise topology candidate with per-region KNN/radius settings.
2. Run PhysTwin first-order adaptation on that fixed topology. This optimizes
   physical parameters for the candidate topology.
3. Fit a canonical-coordinate Neural Spring Field to the learned spring_Y and
   write a smoothed/field-regularized checkpoint.
4. Re-run inference with the field-regularized checkpoint.
5. Evaluate CD + tracking error and select the topology with the lowest score.

This creates an optimized topology .npz before any pruning.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_piecewise_candidate_topology import build_piecewise_candidate_topology
from eval_geometry import evaluate_geometry
from neural_spring_field import SpringFieldFitConfig, fit_neural_spring_field_to_checkpoint
from run_pipeline import clear_experiment_train_dir, copy_checkpoint_to_default, run_adaptation_and_inference, run_command


def _latest_train_checkpoint(scene: str, project_dir: Path) -> Path | None:
    train_dir = project_dir / "experiments" / scene / "train"
    ckpts = sorted(train_dir.glob("iter_*.pth"))
    if not ckpts:
        return None

    def iter_id(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except Exception:
            return -1

    return sorted(ckpts, key=iter_id)[-1]


def _write_method_table(rows: list[dict], csv_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def _make_env(topology_path: Path) -> dict[str, str]:
    torch_lib = Path(torch.__file__).parent / "lib"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(torch_lib) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["WANDB_SILENT"] = "true"
    env["SKIP_OPEN3D_VIDEO"] = "1"
    env["DISABLE_PYTORCH3D"] = "1"
    env["EXTERNAL_TOPOLOGY_NPZ"] = str(topology_path)
    return env


def run_inference_with_checkpoint(
    *,
    method_name: str,
    checkpoint_path: Path,
    topology_path: Path,
    project_dir: Path,
    base_path: Path,
    scene: str,
    method_dir: Path,
    python_bin: str,
) -> dict:
    method_dir.mkdir(parents=True, exist_ok=True)
    clear_experiment_train_dir(scene, project_dir)
    copy_checkpoint_to_default(scene, project_dir, checkpoint_path)

    default_inference = project_dir / "experiments" / scene / "inference.pkl"
    if default_inference.exists():
        default_inference.unlink()

    env = _make_env(topology_path)
    log_path = method_dir / "inference_stdout_stderr.txt"
    cmd = [python_bin, "inference_warp.py", "--base_path", str(base_path), "--case_name", scene]
    return_code, seconds = run_command(cmd, cwd=project_dir, env=env, log_path=log_path)

    saved_inference = method_dir / "inference.pkl"
    if return_code == 0 and default_inference.exists():
        shutil.copy2(default_inference, saved_inference)

    frames = np.nan
    fps = np.nan
    if saved_inference.exists() and seconds > 0:
        with open(saved_inference, "rb") as f:
            verts = pickle.load(f)
        frames = int(np.asarray(verts).shape[0])
        fps = frames / seconds

    return {
        "Method": method_name,
        "Topology Path": str(topology_path),
        "Method Dir": str(method_dir),
        "Checkpoint Path": str(checkpoint_path),
        "Inference Path": str(saved_inference) if saved_inference.exists() else None,
        "Train Return Code": 0,
        "Inference Return Code": return_code,
        "Used Reused Checkpoint": True,
        "Train Seconds": 0.0,
        "Inference Seconds": seconds,
        "Frames": frames,
        "Simulation FPS": fps,
        "Train Log": None,
        "Inference Log": str(log_path),
    }


def _candidate_parameters(candidate_idx: int, num_regions: int, rng: np.random.Generator) -> tuple[list[float], list[int]]:
    if candidate_idx == 0:
        return [1.0] * num_regions, [16] * num_regions

    # Random search around a reasonable K/radius region. We keep ranges conservative
    # so early experiments do not generate huge unstable graphs.
    radius_scale = rng.uniform(0.70, 1.55, size=num_regions)
    region_knn = rng.integers(8, 34, size=num_regions)
    return radius_scale.tolist(), region_knn.astype(int).tolist()


def _score_from_eval(eval_df: pd.DataFrame, method: str, track_weight: float, test_weight: float) -> float:
    row = eval_df[eval_df["Method"] == method]
    if row.empty:
        return float("inf")
    r = row.iloc[0]
    cd_train = float(r.get("CD Train", np.nan))
    cd_test = float(r.get("CD Test", np.nan))
    te_train = float(r.get("Track Error Train", np.nan))
    te_test = float(r.get("Track Error Test", np.nan))
    vals = []
    if np.isfinite(cd_train):
        vals.append(cd_train)
    if np.isfinite(te_train):
        vals.append(track_weight * te_train)
    if np.isfinite(cd_test):
        vals.append(test_weight * cd_test)
    if np.isfinite(te_test):
        vals.append(test_weight * track_weight * te_test)
    return float(np.sum(vals)) if vals else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default="/content/PhysTwin")
    parser.add_argument("--case-name", default="double_stretch_sloth")
    parser.add_argument("--base-path", default="/content/PhysTwin/data/different_types")
    parser.add_argument("--original-topology", default="/content/PhysTwin/results/double_stretch_sloth_phystwin_topology_open3d.npz")
    parser.add_argument("--search-root", default=None)
    parser.add_argument("--num-regions", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--bridges-per-region-pair", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--reuse-existing-candidates", action="store_true")
    parser.add_argument("--skip-field-fit", action="store_true", help="debug only; score raw PhysTwin checkpoints")
    parser.add_argument("--field-fit-steps", type=int, default=800)
    parser.add_argument("--field-fit-lr", type=float, default=3e-3)
    parser.add_argument("--field-smooth-weight", type=float, default=1e-4)
    parser.add_argument("--track-weight", type=float, default=1.0)
    parser.add_argument("--test-weight", type=float, default=0.5, help="lower to mainly optimize train frames; increase for future prediction")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    base_path = Path(args.base_path).resolve()
    scene = args.case_name
    data_root = base_path / scene
    original_topology = Path(args.original_topology).resolve()
    search_root = Path(args.search_root).resolve() if args.search_root else project_dir / "results" / "neuspring_topology_search" / scene
    topology_root = search_root / "topologies"
    candidates_root = search_root / "candidates"
    python_bin = args.python_bin or shutil.which("python") or "python"

    required = [project_dir, data_root, data_root / "final_data.pkl", data_root / "split.json", data_root / "gt_track_3d.pkl", original_topology]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    search_root.mkdir(parents=True, exist_ok=True)
    topology_root.mkdir(parents=True, exist_ok=True)
    candidates_root.mkdir(parents=True, exist_ok=True)

    with open(data_root / "split.json", "r") as f:
        split = json.load(f)
    train_frame = int(split["train"][1])
    print("scene:", scene)
    print("train_frame:", train_frame)
    print("search_root:", search_root)

    rng = np.random.default_rng(args.random_state)
    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    for candidate_idx in range(args.max_candidates):
        candidate_id = f"cand_{candidate_idx:03d}"
        radius_scale, region_knn = _candidate_parameters(candidate_idx, args.num_regions, rng)
        candidate_dir = candidates_root / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        topology_path = topology_root / f"{scene}_{candidate_id}_piecewise_topology.npz"

        print("\n" + "#" * 110)
        print("CANDIDATE:", candidate_id)
        print("radius_scale:", radius_scale)
        print("region_knn:", region_knn)

        if topology_path.exists() and args.reuse_existing_candidates:
            print("[reuse topology]", topology_path)
        else:
            build_piecewise_candidate_topology(
                data_root=data_root,
                original_topology=original_topology,
                output_path=topology_path,
                num_regions=args.num_regions,
                region_radius_scale=radius_scale,
                region_knn=region_knn,
                bridges_per_region_pair=args.bridges_per_region_pair,
                random_state=args.random_state,
                candidate_id=candidate_id,
            )

        raw_method = f"{candidate_id}_raw_param_opt"
        raw_result = run_adaptation_and_inference(
            method_name=raw_method,
            topology_path=topology_path,
            project_dir=project_dir,
            base_path=base_path,
            scene=scene,
            train_frame=train_frame,
            adapt_root=candidate_dir,
            python_bin=python_bin,
            reuse_checkpoints=False,
            skip_existing_inference=False,
            force_train=True,
        )
        all_rows.append(raw_result)

        score_method = raw_method
        nsf_result = None
        fit_stats = None
        nsf_ckpt = None
        nsf_field = None
        if (not args.skip_field_fit) and raw_result.get("Checkpoint Path") and Path(raw_result["Checkpoint Path"]).exists():
            nsf_ckpt = candidate_dir / "nsf_field_regularized_iter_199.pth"
            nsf_field = candidate_dir / "neural_spring_field.pt"
            fit_cfg = SpringFieldFitConfig(
                steps=args.field_fit_steps,
                lr=args.field_fit_lr,
                smooth_weight=args.field_smooth_weight,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            fit_stats = fit_neural_spring_field_to_checkpoint(
                checkpoint_path=Path(raw_result["Checkpoint Path"]),
                topology_path=topology_path,
                data_root=data_root,
                output_checkpoint_path=nsf_ckpt,
                output_field_path=nsf_field,
                config=fit_cfg,
            )
            nsf_method = f"{candidate_id}_nsf_param_opt"
            nsf_result = run_inference_with_checkpoint(
                method_name=nsf_method,
                checkpoint_path=nsf_ckpt,
                topology_path=topology_path,
                project_dir=project_dir,
                base_path=base_path,
                scene=scene,
                method_dir=candidate_dir / nsf_method,
                python_bin=python_bin,
            )
            all_rows.append(nsf_result)
            score_method = nsf_method if nsf_result.get("Inference Path") else raw_method

        method_csv = candidate_dir / "methods_for_eval.csv"
        eval_csv = candidate_dir / "geometry_eval.csv"
        method_df = _write_method_table([r for r in [raw_result, nsf_result] if r is not None], method_csv)
        eval_df = evaluate_geometry(data_root=data_root, method_table_csv=method_csv, output_csv=eval_csv)
        score = _score_from_eval(eval_df, score_method, args.track_weight, args.test_weight)

        topo = np.load(topology_path, allow_pickle=True)
        springs = topo["springs"]
        num_object_springs = int(topo["num_object_springs"])
        row = {
            "candidate_id": candidate_id,
            "score_method": score_method,
            "score": score,
            "topology_path": str(topology_path),
            "candidate_dir": str(candidate_dir),
            "num_object_springs": num_object_springs,
            "num_total_springs": int(len(springs)),
            "region_radius_scale": " ".join(f"{x:.4f}" for x in radius_scale),
            "region_knn": " ".join(str(int(x)) for x in region_knn),
            "raw_checkpoint": raw_result.get("Checkpoint Path"),
            "nsf_checkpoint": str(nsf_ckpt) if (nsf_ckpt is not None and nsf_ckpt.exists()) else None,
            "nsf_field": str(nsf_field) if (nsf_field is not None and nsf_field.exists()) else None,
            "nsf_fit_loss": fit_stats.get("fit_loss") if fit_stats else None,
        }
        # Attach eval metrics of selected method.
        selected_eval = eval_df[eval_df["Method"] == score_method]
        if not selected_eval.empty:
            for col in ["CD Train", "CD Test", "Track Error Train", "Track Error Test"]:
                row[col] = float(selected_eval.iloc[0][col])
        summary_rows.append(row)
        pd.DataFrame(summary_rows).sort_values("score").to_csv(search_root / "candidate_summary.csv", index=False)
        pd.DataFrame(all_rows).to_csv(search_root / "all_method_runs.csv", index=False)
        print("[candidate result]", json.dumps(row, indent=2))

    summary = pd.DataFrame(summary_rows).sort_values("score")
    summary_csv = search_root / "candidate_summary.csv"
    summary.to_csv(summary_csv, index=False)

    best = summary.iloc[0].to_dict()
    best_topology = Path(best["topology_path"])
    best_out = search_root / f"{scene}_best_optimized_topology_no_pruning.npz"
    shutil.copy2(best_topology, best_out)

    best_manifest = search_root / "best_topology_manifest.json"
    with open(best_manifest, "w") as f:
        json.dump(best, f, indent=2)

    print("\n" + "=" * 110)
    print("BEST CANDIDATE")
    print(json.dumps(best, indent=2))
    print("Saved best topology:", best_out)
    print("Saved summary:", summary_csv)
    print("Saved manifest:", best_manifest)


if __name__ == "__main__":
    main()
