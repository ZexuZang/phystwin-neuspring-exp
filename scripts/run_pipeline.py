#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_regionwise_topology import build_regionwise_topology
from eval_geometry import evaluate_geometry


def unzip_if_exists(zip_path: Path | None, dst: Path) -> None:
    if zip_path is None:
        return
    if not zip_path.exists():
        print(f"[skip] zip not found: {zip_path}")
        return
    print(f"[unzip] {zip_path} -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)


def copy_dir_contents(src: Path | None, dst: Path) -> None:
    if src is None:
        return
    if not src.exists():
        print(f"[skip] artifacts dir not found: {src}")
        return
    print(f"[copy artifacts] {src} -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def latest_checkpoint(scene: str, project_dir: Path) -> Path | None:
    ckpts = list((project_dir / "experiments" / scene / "train").glob("iter_*.pth"))
    if not ckpts:
        return None

    def iter_num(p: Path) -> int:
        m = re.search(r"iter_(\d+)\.pth", p.name)
        return int(m.group(1)) if m else -1

    return sorted(ckpts, key=iter_num)[-1]


def clear_experiment_train_dir(scene: str, project_dir: Path) -> None:
    train_dir = project_dir / "experiments" / scene / "train"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)


def copy_checkpoint_to_default(scene: str, project_dir: Path, ckpt_path: Path) -> None:
    train_dir = project_dir / "experiments" / scene / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_path, train_dir / ckpt_path.name)
    shutil.copy2(ckpt_path, train_dir / "best_199.pth")
    shutil.copy2(ckpt_path, train_dir / "iter_199.pth")
    print(f"[reuse ckpt] {ckpt_path} -> {train_dir}")


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> tuple[int, float]:
    print("cmd:", " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("STDOUT\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR\n")
        f.write(result.stderr)
    print("return code:", result.returncode)
    print("seconds:", elapsed)
    print("log:", log_path)
    print("\nSTDOUT tail:\n", result.stdout[-2000:])
    print("\nSTDERR tail:\n", result.stderr[-2000:])
    return result.returncode, elapsed


def run_adaptation_and_inference(
    *,
    method_name: str,
    topology_path: Path,
    project_dir: Path,
    base_path: Path,
    scene: str,
    train_frame: int,
    adapt_root: Path,
    python_bin: str,
    reuse_checkpoints: bool,
    skip_existing_inference: bool,
    force_train: bool,
) -> dict:
    assert topology_path.exists(), topology_path

    method_dir = adapt_root / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = method_dir / "train_stdout_stderr.txt"
    inference_log_path = method_dir / "inference_stdout_stderr.txt"
    saved_ckpt_path = method_dir / "iter_199.pth"
    saved_inference_path = method_dir / "inference.pkl"

    torch_lib = Path(torch.__file__).parent / "lib"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(torch_lib) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["WANDB_SILENT"] = "true"
    env["SKIP_OPEN3D_VIDEO"] = "1"
    env["DISABLE_PYTORCH3D"] = "1"
    env["EXTERNAL_TOPOLOGY_NPZ"] = str(topology_path)

    print("=" * 100)
    print("METHOD:", method_name)
    print("Topology:", topology_path)
    print("Method dir:", method_dir)

    train_return_code = None
    train_seconds = np.nan
    used_reused_checkpoint = False

    if skip_existing_inference and saved_inference_path.exists():
        print("[skip] existing inference:", saved_inference_path)
    else:
        can_reuse_ckpt = reuse_checkpoints and saved_ckpt_path.exists() and not force_train
        if can_reuse_ckpt:
            clear_experiment_train_dir(scene, project_dir)
            copy_checkpoint_to_default(scene, project_dir, saved_ckpt_path)
            used_reused_checkpoint = True
            train_return_code = 0
        else:
            clear_experiment_train_dir(scene, project_dir)
            train_cmd = [
                python_bin,
                "train_warp.py",
                "--case_name",
                scene,
                "--base_path",
                str(base_path),
                "--train_frame",
                str(train_frame),
            ]
            train_return_code, train_seconds = run_command(
                train_cmd,
                cwd=project_dir,
                env=env,
                log_path=train_log_path,
            )

            if train_return_code != 0:
                return {
                    "Method": method_name,
                    "Topology Path": str(topology_path),
                    "Method Dir": str(method_dir),
                    "Checkpoint Path": None,
                    "Inference Path": None,
                    "Train Return Code": train_return_code,
                    "Inference Return Code": None,
                    "Used Reused Checkpoint": used_reused_checkpoint,
                    "Train Seconds": train_seconds,
                    "Inference Seconds": np.nan,
                    "Frames": np.nan,
                    "Simulation FPS": np.nan,
                    "Train Log": str(train_log_path),
                    "Inference Log": str(inference_log_path),
                }

            ckpt_path = latest_checkpoint(scene, project_dir)
            if ckpt_path and ckpt_path.exists():
                shutil.copy2(ckpt_path, saved_ckpt_path)
                copy_checkpoint_to_default(scene, project_dir, saved_ckpt_path)
                print("Saved checkpoint:", saved_ckpt_path)
            else:
                print("[warning] no latest iter_*.pth found")

        default_inference = project_dir / "experiments" / scene / "inference.pkl"
        if default_inference.exists():
            default_inference.unlink()

        inference_cmd = [
            python_bin,
            "inference_warp.py",
            "--base_path",
            str(base_path),
            "--case_name",
            scene,
        ]
        inference_return_code, inference_seconds = run_command(
            inference_cmd,
            cwd=project_dir,
            env=env,
            log_path=inference_log_path,
        )

        if inference_return_code == 0 and default_inference.exists():
            shutil.copy2(default_inference, saved_inference_path)
            print("Saved inference:", saved_inference_path)
        else:
            print("[warning] inference.pkl not found:", default_inference)

    frame_num = np.nan
    sim_fps = np.nan
    inference_seconds_final = np.nan
    if inference_log_path.exists():
        # We do not parse elapsed from old logs. The current run stores it in this process only.
        pass

    if saved_inference_path.exists():
        with open(saved_inference_path, "rb") as f:
            verts = pickle.load(f)
        verts = np.asarray(verts)
        frame_num = int(verts.shape[0])
        # sim_fps is only meaningful when inference was run in this call.
        # If reusing an old inference, leave it as NaN.
        if "inference_seconds" in locals() and inference_seconds > 0:
            inference_seconds_final = inference_seconds
            sim_fps = frame_num / inference_seconds

    return {
        "Method": method_name,
        "Topology Path": str(topology_path),
        "Method Dir": str(method_dir),
        "Checkpoint Path": str(saved_ckpt_path) if saved_ckpt_path.exists() else None,
        "Inference Path": str(saved_inference_path) if saved_inference_path.exists() else None,
        "Train Return Code": train_return_code,
        "Inference Return Code": locals().get("inference_return_code", 0 if saved_inference_path.exists() else None),
        "Used Reused Checkpoint": used_reused_checkpoint,
        "Train Seconds": train_seconds,
        "Inference Seconds": inference_seconds_final,
        "Frames": frame_num,
        "Simulation FPS": sim_fps,
        "Train Log": str(train_log_path),
        "Inference Log": str(inference_log_path),
    }


def find_or_build_regionwise_topology(
    *,
    project_dir: Path,
    base_path: Path,
    scene: str,
    original_topology: Path,
    topology_root: Path,
    num_regions: int,
) -> Path:
    pattern = topology_root / f"{scene}_regionwise_optimized_K{num_regions}_keep_controller_full_topology.npz"
    if pattern.exists():
        print("[reuse topology]", pattern)
        return pattern

    # Some old artifacts may have a matching .npz under topologies.
    candidates = sorted(topology_root.glob(f"*regionwise*{num_regions}*.npz"))
    if candidates:
        print("[reuse topology candidate]", candidates[0])
        return candidates[0]

    output_path = pattern
    return build_regionwise_topology(
        data_root=base_path / scene,
        original_topology_path=original_topology,
        output_path=output_path,
        num_regions=num_regions,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default="/content/PhysTwin")
    parser.add_argument("--case-name", default="double_stretch_sloth")
    parser.add_argument("--base-path", default="/content/PhysTwin/data/different_types")
    parser.add_argument("--original-topology", default="/content/PhysTwin/results/double_stretch_sloth_phystwin_topology_open3d.npz")
    parser.add_argument("--adapt-root", default=None)
    parser.add_argument("--num-regions", type=int, default=5)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--reuse-artifact-zip", default=None)
    parser.add_argument("--reuse-checkpoint-zip", default=None)
    parser.add_argument("--reuse-artifacts-dir", default=None)
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--skip-existing-inference", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    base_path = Path(args.base_path).resolve()
    scene = args.case_name
    data_root = base_path / scene
    original_topology = Path(args.original_topology).resolve()
    adapt_root = (
        Path(args.adapt_root).resolve()
        if args.adapt_root
        else project_dir / "results" / "regionwise_optimization_final" / scene
    )
    topology_root = adapt_root / "topologies"
    python_bin = args.python_bin or shutil.which("python") or "python"

    for p in [project_dir, data_root, data_root / "final_data.pkl", data_root / "split.json", original_topology]:
        if not p.exists():
            raise FileNotFoundError(p)

    adapt_root.mkdir(parents=True, exist_ok=True)
    topology_root.mkdir(parents=True, exist_ok=True)

    # Restore old artifacts/checkpoints if provided.
    unzip_if_exists(Path(args.reuse_artifact_zip) if args.reuse_artifact_zip else None, adapt_root)
    unzip_if_exists(Path(args.reuse_checkpoint_zip) if args.reuse_checkpoint_zip else None, adapt_root)
    copy_dir_contents(Path(args.reuse_artifacts_dir) if args.reuse_artifacts_dir else None, adapt_root)

    with open(data_root / "split.json", "r") as f:
        split = json.load(f)
    train_frame = int(split["train"][1])
    print("scene:", scene)
    print("train_frame:", train_frame)
    print("adapt_root:", adapt_root)

    regionwise_topology = find_or_build_regionwise_topology(
        project_dir=project_dir,
        base_path=base_path,
        scene=scene,
        original_topology=original_topology,
        topology_root=topology_root,
        num_regions=args.num_regions,
    )

    rows = []
    rows.append(
        run_adaptation_and_inference(
            method_name="OriginalTopology_Adapt",
            topology_path=original_topology,
            project_dir=project_dir,
            base_path=base_path,
            scene=scene,
            train_frame=train_frame,
            adapt_root=adapt_root,
            python_bin=python_bin,
            reuse_checkpoints=args.reuse_checkpoints,
            skip_existing_inference=args.skip_existing_inference,
            force_train=args.force_train,
        )
    )
    rows.append(
        run_adaptation_and_inference(
            method_name="RegionWiseOptimizedTopology_Adapt",
            topology_path=regionwise_topology,
            project_dir=project_dir,
            base_path=base_path,
            scene=scene,
            train_frame=train_frame,
            adapt_root=adapt_root,
            python_bin=python_bin,
            reuse_checkpoints=args.reuse_checkpoints,
            skip_existing_inference=args.skip_existing_inference,
            force_train=args.force_train,
        )
    )

    base_df = pd.DataFrame(rows)
    base_csv = adapt_root / "base_adaptation_summary.csv"
    base_df.to_csv(base_csv, index=False)
    print(base_df)
    print("Saved:", base_csv)

    # Topology comparison.
    topo_rows = []
    for method_name, topology_path in [
        ("OriginalTopology_Adapt", original_topology),
        ("RegionWiseOptimizedTopology_Adapt", regionwise_topology),
    ]:
        topo = np.load(topology_path, allow_pickle=True)
        springs = topo["springs"]
        num_object_springs = int(topo["num_object_springs"])
        topo_rows.append(
            {
                "Method": method_name,
                "Topology Path": str(topology_path),
                "Object Springs": num_object_springs,
                "Controller Springs": int(len(springs) - num_object_springs),
                "Total Springs": int(len(springs)),
            }
        )
    topo_df = pd.DataFrame(topo_rows)
    topo_csv = adapt_root / "topology_compare.csv"
    topo_df.to_csv(topo_csv, index=False)
    print(topo_df)

    cd_csv = adapt_root / "cd_track.csv"
    cd_df = evaluate_geometry(
        data_root=data_root,
        method_table_csv=base_csv,
        output_csv=cd_csv,
    )

    summary_df = topo_df.merge(cd_df, on="Method", how="left")
    if "Simulation FPS" in base_df.columns:
        fps_df = base_df[["Method", "Simulation FPS", "Inference Seconds", "Frames"]].copy()
        summary_df = summary_df.merge(fps_df, on="Method", how="left")

    final_csv = adapt_root / "final_summary.csv"
    summary_df.to_csv(final_csv, index=False)
    print(summary_df)
    print("Saved:", final_csv)


if __name__ == "__main__":
    main()
