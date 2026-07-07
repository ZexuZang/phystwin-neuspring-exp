#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-rollout Gaussian residual fine-tuning for PhysTwin / NeuSpring.

This script implements a safer alternative to back-propagating render loss
through the official dynamic rollout.

Core idea
---------
1. Run the official PhysTwin rollout under torch.no_grad().
2. Freeze the rollout outputs: xyz_t, rgb_t/features_dc, quat_t, opacity_t, scale_raw.
3. Add small trainable residuals after rollout:
      delta_xyz, delta_rgb, delta_opacity, delta_scale, delta_rotation
4. Optimize those residuals through the differentiable Gaussian renderer.
5. Save residual checkpoint and re-render corrected frames.

This is Gaussian-state optimization, not PNG post-processing.

Recommended first experiment
----------------------------
Optimize only color / opacity / scale:

python scripts/post_rollout_gaussian_residual_finetune.py \
  --source_path /content/PhysTwin/data/gaussian_data/double_stretch_sloth \
  --model_path /content/PhysTwin/gaussian_output/double_stretch_sloth/init=30000 \
  --name double_stretch_sloth \
  --inference_path /content/PhysTwin/results/neuspring_topology_search/double_stretch_sloth/candidates/cand_000/cand_000_nsf_sim_ft_param_opt/inference.pkl \
  --gt_root /content/PhysTwin/data/different_types/double_stretch_sloth \
  --output_dir /content/PhysTwin/results/post_rollout_gaussian_residual/cand_000_nsf_sim_ft \
  --opt_rgb --opt_opacity --opt_scale \
  --steps 300 \
  --batch_size 1 \
  --num_train_frames 32 \
  --lambda_ssim 0.2 \
  --lambda_bg 0.1 \
  --lambda_delta 0.01
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import random
import re
import sys
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaussian_splatting.scene import Scene
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.gaussian_renderer import GaussianModel
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.utils.general_utils import safe_state

try:
    from gaussian_splatting.utils.loss_utils import ssim as gs_ssim
except Exception:
    gs_ssim = None

from gs_render import remove_gaussians_with_mask, remove_gaussians_with_low_opacity
from gs_render_dynamics import rollout


def numeric_stem(path: str | Path) -> Optional[int]:
    nums = re.findall(r"\d+", Path(path).stem)
    return int(nums[-1]) if nums else None


def collect_images(root: str | Path) -> List[str]:
    root = Path(root)
    if not root.exists():
        return []
    files: List[str] = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        files.extend(glob(str(root / "**" / ext), recursive=True))
    return sorted(files)


def build_frame_map(files: Sequence[str]) -> Dict[int, str]:
    mp: Dict[int, str] = {}
    for f in files:
        idx = numeric_stem(f)
        if idx is not None and idx not in mp:
            mp[idx] = f
    return mp


def load_rgb(path: str | Path) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img[:, :, :3].astype(np.uint8)


def load_mask(path: Optional[str | Path], fallback_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    if path is None or not Path(path).exists():
        if fallback_rgb is None:
            raise FileNotFoundError("Mask path is missing and fallback_rgb is None.")
        return (fallback_rgb.astype(np.float32).sum(axis=-1) > 5).astype(np.float32)
    mask = np.array(Image.open(path)).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.max() > 1.0:
        mask /= 255.0
    return mask.astype(np.float32)


def load_split(gt_root: Path, n_frames: int) -> Tuple[List[int], List[int]]:
    split_path = gt_root / "split.json"
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_indices = list(range(split["train"][0] + 1, split["train"][1]))
        test_indices = list(range(split["test"][0], split["test"][1]))
        train_indices = [i for i in train_indices if 0 <= i < n_frames]
        test_indices = [i for i in test_indices if 0 <= i < n_frames]
        return train_indices, test_indices
    train_end = int(round(n_frames * 0.7))
    return list(range(train_end)), list(range(train_end, n_frames))


def to_chw_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)


def resize_tensor_image(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    h, w = hw
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x[0] if squeeze else x


def resize_tensor_mask(m: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    h, w = hw
    if m.ndim == 2:
        m = m[None, None]
    elif m.ndim == 3:
        m = m[None]
    m = F.interpolate(m.float(), size=(h, w), mode="nearest")
    return m[0, 0]


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(0)
    return (x * mask).sum() / (mask.sum() * x.shape[-3] + eps)


def l1_masked(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(torch.abs(pred - gt), mask)


def smooth_rollout_outputs(xyz, rgb, quat, opa):
    with torch.no_grad():
        diffs = xyz - torch.cat([xyz[0:1], xyz[:-1]], dim=0)
        change_points = diffs.norm(dim=-1).sum(dim=-1).nonzero().flatten().cpu()
        if len(change_points) == 0 or change_points[0].item() != 0:
            change_points = torch.cat([torch.tensor([0]), change_points])
        for i in range(1, len(change_points)):
            start = int(change_points[i - 1].item())
            end = int(change_points[i].item())
            if end - start < 2:
                continue
            alpha = torch.linspace(0, 1, end - start + 1, device=xyz.device)[:, None, None]
            xyz[start:end] = torch.lerp(xyz[start][None], xyz[end][None], alpha)[:-1]
            rgb[start:end] = torch.lerp(rgb[start][None], rgb[end][None], alpha)[:-1]
            quat[start:end] = torch.lerp(quat[start][None], quat[end][None], alpha)[:-1]
            opa[start:end] = torch.lerp(opa[start][None], opa[end][None], alpha)[:-1]
        quat = F.normalize(quat, dim=-1)
    return xyz, rgb, quat, opa


def load_ctrl_pts(inference_path: Path, device: torch.device) -> torch.Tensor:
    with open(inference_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, np.ndarray):
        arr = obj
    elif torch.is_tensor(obj):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        arr = None
        for k in ["vertices", "trajectory", "pred", "pred_traj", "points", "ctrl_pts"]:
            if k in obj:
                arr = obj[k]
                break
        if arr is None:
            for v in obj.values():
                vv = v.detach().cpu().numpy() if torch.is_tensor(v) else v
                if isinstance(vv, np.ndarray) and vv.ndim == 3 and vv.shape[-1] == 3:
                    arr = vv
                    break
        if arr is None:
            raise RuntimeError(f"Cannot find trajectory array in {inference_path}")
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
    else:
        raise RuntimeError(f"Unsupported inference.pkl object type: {type(obj)}")
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise RuntimeError(f"Expected inference trajectory [T,N,3], got {arr.shape}")
    return torch.tensor(arr, dtype=torch.float32, device=device)


class GaussianResiduals(nn.Module):
    def __init__(self, n_gaussians, scale_shape, device, opt_xyz, opt_rgb, opt_opacity, opt_scale, opt_rotation):
        super().__init__()
        def make_param(shape, enabled):
            if enabled:
                return nn.Parameter(torch.zeros(*shape, device=device))
            return None
        self.delta_xyz = make_param((n_gaussians, 3), opt_xyz)
        self.delta_rgb = make_param((n_gaussians, 3), opt_rgb)
        self.delta_opacity = make_param((n_gaussians, 1), opt_opacity)
        self.delta_scale = make_param(scale_shape, opt_scale)
        self.delta_rotation = make_param((n_gaussians, 4), opt_rotation)
        self.register_buffer("zero_xyz", torch.zeros(n_gaussians, 3, device=device))
        self.register_buffer("zero_rgb", torch.zeros(n_gaussians, 3, device=device))
        self.register_buffer("zero_opacity", torch.zeros(n_gaussians, 1, device=device))
        self.register_buffer("zero_scale", torch.zeros(*scale_shape, device=device))
        self.register_buffer("zero_rotation", torch.zeros(n_gaussians, 4, device=device))

    def get_delta_xyz(self): return self.delta_xyz if self.delta_xyz is not None else self.zero_xyz
    def get_delta_rgb(self): return self.delta_rgb if self.delta_rgb is not None else self.zero_rgb
    def get_delta_opacity(self): return self.delta_opacity if self.delta_opacity is not None else self.zero_opacity
    def get_delta_scale(self): return self.delta_scale if self.delta_scale is not None else self.zero_scale
    def get_delta_rotation(self): return self.delta_rotation if self.delta_rotation is not None else self.zero_rotation

    def regularization(self):
        vals = [self.get_delta_xyz(), self.get_delta_rgb(), self.get_delta_opacity(), self.get_delta_scale(), self.get_delta_rotation()]
        return sum(v.square().mean() for v in vals)


def build_gaussians_for_frame(base_gaussians, residuals, xyz_t, rgb_t, quat_t, opa_t, scale_raw_base, args):
    g = copy.deepcopy(base_gaussians)
    xyz_t = xyz_t.cuda(); rgb_t = rgb_t.cuda(); quat_t = quat_t.cuda(); opa_t = opa_t.cuda()
    dxyz = torch.clamp(residuals.get_delta_xyz(), -args.max_delta_xyz, args.max_delta_xyz) if args.max_delta_xyz > 0 else residuals.get_delta_xyz()
    drgb = residuals.get_delta_rgb()
    dopa = torch.clamp(residuals.get_delta_opacity(), -args.max_delta_opacity, args.max_delta_opacity) if args.max_delta_opacity > 0 else residuals.get_delta_opacity()
    dscale = torch.clamp(residuals.get_delta_scale(), -args.max_delta_scale, args.max_delta_scale) if args.max_delta_scale > 0 else residuals.get_delta_scale()
    drot = torch.clamp(residuals.get_delta_rotation(), -args.max_delta_rotation, args.max_delta_rotation) if args.max_delta_rotation > 0 else residuals.get_delta_rotation()
    new_xyz = xyz_t + dxyz
    new_rgb = rgb_t + drgb
    if args.clamp_rgb:
        new_rgb = torch.clamp(new_rgb, 0.0, 1.0)
    new_quat = F.normalize(quat_t + drot, dim=-1)
    new_opa = torch.clamp(opa_t + dopa, 1e-4, 0.999)
    new_opacity_raw = g.inverse_opacity_activation(new_opa)
    if hasattr(g, "_features_rest") and g._features_rest is not None:
        g._features_rest = g._features_rest.cuda()
    g._xyz = new_xyz
    g._features_dc = new_rgb.unsqueeze(1)
    g._rotation = new_quat
    g._opacity = new_opacity_raw
    g._scaling = scale_raw_base + dscale
    return g


class GTProvider:
    def __init__(self, gt_root: Path, device: torch.device):
        self.gt_root = gt_root
        self.device = device
        self.color_map = build_frame_map(collect_images(gt_root / "color"))
        self.mask_map = build_frame_map(collect_images(gt_root / "mask"))

    def get(self, frame_idx: int, render_hw: Tuple[int, int]):
        color_path = self.color_map.get(frame_idx)
        if color_path is None:
            raise FileNotFoundError(f"Cannot find GT color for frame {frame_idx}")
        rgb_np = load_rgb(color_path)
        mask_np = load_mask(self.mask_map.get(frame_idx), fallback_rgb=rgb_np)
        gt = to_chw_tensor(rgb_np, self.device)
        mask = torch.from_numpy(mask_np.astype(np.float32)).to(self.device)
        gt = resize_tensor_image(gt, render_hw)
        mask = resize_tensor_mask(mask, render_hw)
        gt = gt * mask.unsqueeze(0)
        return gt, mask


def prepare_state(args, dataset, pipeline, device):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if args.remove_gaussians:
        gaussians = remove_gaussians_with_mask(gaussians, scene.getTrainCameras())
        gaussians = remove_gaussians_with_low_opacity(gaussians)
    inference_path = Path(args.inference_path)
    if not inference_path.exists():
        exp_name = Path(dataset.source_path).name
        inference_path = PROJECT_ROOT / "experiments" / exp_name / "inference.pkl"
    if not inference_path.exists():
        raise FileNotFoundError(f"inference.pkl not found: {args.inference_path}")
    ctrl_pts = load_ctrl_pts(inference_path, device)
    n_steps = ctrl_pts.shape[0]
    xyz_0 = gaussians.get_xyz.detach()
    rgb_0 = gaussians.get_features_dc.squeeze(1).detach()
    quat_0 = gaussians.get_rotation.detach()
    opa_0 = gaussians.get_opacity.detach()
    scale_raw_0 = gaussians._scaling.detach().clone().cuda()
    print("===== Number of steps:", n_steps)
    print("===== Number of control points:", ctrl_pts.shape[1])
    print("===== Number of gaussians:", xyz_0.shape[0])
    print("===== scaling raw shape:", tuple(scale_raw_0.shape))
    with torch.no_grad():
        xyz, rgb, quat, opa = rollout(xyz_0, rgb_0, quat_0, opa_0, ctrl_pts, n_steps)
        xyz = xyz.cuda(); rgb = rgb.cuda(); quat = quat.cuda(); opa = opa.cuda()
        xyz, rgb, quat, opa = smooth_rollout_outputs(xyz, rgb, quat, opa)
    views = scene.getTestCameras()
    if len(views) == 0:
        raise RuntimeError("scene.getTestCameras() returned empty list.")
    return {"background": background, "gaussians": gaussians, "views": views, "xyz": xyz, "rgb": rgb, "quat": quat, "opa": opa, "scale_raw": scale_raw_0, "n_steps": n_steps}


def render_one(args, frame_idx, view, state, residuals, pipeline):
    g = build_gaussians_for_frame(state["gaussians"], residuals, state["xyz"][frame_idx], state["rgb"][frame_idx], state["quat"][frame_idx], state["opa"][frame_idx], state["scale_raw"], args)
    if args.disable_sh_override or getattr(args, "disable_sh", False):
        override_color = g.get_features_dc.squeeze()
        result = render(view, g, pipeline, state["background"], override_color=override_color, use_trained_exp=args.train_test_exp, separate_sh=args.separate_sh)
    else:
        result = render(view, g, pipeline, state["background"], use_trained_exp=args.train_test_exp, separate_sh=args.separate_sh)
    return result["render"].clamp(0.0, 1.0)


def compute_loss_for_frame(args, frame_idx, view, state, residuals, gt_provider, pipeline):
    pred = render_one(args, frame_idx, view, state, residuals, pipeline)
    gt, mask = gt_provider.get(frame_idx, pred.shape[-2:])
    pred_masked = pred * mask.unsqueeze(0)
    fg_l1 = l1_masked(pred_masked, gt, mask)
    bg_mask = 1.0 - mask
    bg_l1 = l1_masked(pred, torch.zeros_like(pred), bg_mask)
    loss = fg_l1 + args.lambda_bg * bg_l1
    if args.lambda_ssim > 0 and gs_ssim is not None:
        loss = loss + args.lambda_ssim * (1.0 - gs_ssim(pred_masked[None], gt[None]))
    if args.lambda_delta > 0:
        loss = loss + args.lambda_delta * residuals.regularization()
    return loss, {"fg_l1": float(fg_l1.detach().cpu()), "bg_l1": float(bg_l1.detach().cpu())}


def save_frame(args, frame_idx, view, state, residuals, pipeline, output_dir, view_out_idx):
    with torch.no_grad():
        img = render_one(args, frame_idx, view, state, residuals, pipeline)
    out_dir = output_dir / args.name / f"{view_out_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(img, out_dir / f"{frame_idx:05d}.png")


def main():
    parser = ArgumentParser(description="Post-rollout Gaussian residual fine-tuning")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--remove_gaussians", action="store_true")
    parser.add_argument("--name", default="double_stretch_sloth", type=str)
    parser.add_argument("--output_dir", default="./results/post_rollout_gaussian_residual", type=str)
    parser.add_argument("--inference_path", required=True, type=str)
    parser.add_argument("--gt_root", required=True, type=str)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_train_frames", default=32, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--view_indices", default="0", type=str)
    parser.add_argument("--render_all_views", action="store_true")
    parser.add_argument("--opt_xyz", action="store_true")
    parser.add_argument("--opt_rgb", action="store_true")
    parser.add_argument("--opt_opacity", action="store_true")
    parser.add_argument("--opt_scale", action="store_true")
    parser.add_argument("--opt_rotation", action="store_true")
    parser.add_argument("--clamp_rgb", action="store_true")
    parser.add_argument("--disable_sh_override", action="store_true")
    parser.add_argument("--lambda_ssim", default=0.2, type=float)
    parser.add_argument("--lambda_bg", default=0.1, type=float)
    parser.add_argument("--lambda_delta", default=0.01, type=float)
    parser.add_argument("--max_delta_xyz", default=0.002, type=float)
    parser.add_argument("--max_delta_opacity", default=0.05, type=float)
    parser.add_argument("--max_delta_scale", default=0.02, type=float)
    parser.add_argument("--max_delta_rotation", default=0.02, type=float)
    parser.add_argument("--save_every", default=50, type=int)
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    args.train_test_exp = dataset.train_test_exp
    args.disable_sh = dataset.disable_sh
    args.separate_sh = False

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    safe_state(args.quiet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    state = prepare_state(args, dataset, pipe, device)
    gt_root = Path(args.gt_root)
    gt_provider = GTProvider(gt_root, device)
    train_indices, test_indices = load_split(gt_root, state["n_steps"])
    if args.num_train_frames > 0 and len(train_indices) > args.num_train_frames:
        train_frames = sorted(random.sample(train_indices, args.num_train_frames))
    else:
        train_frames = train_indices
    view_indices = [int(x) for x in args.view_indices.split(",") if x.strip()]
    views = state["views"]
    for vi in view_indices:
        if vi < 0 or vi >= len(views):
            raise IndexError(f"view index {vi} out of range for {len(views)} test cameras")
    train_view = views[view_indices[0]]
    print("===== Training frames:", len(train_frames), train_frames[:10], "...")
    print("===== View indices:", view_indices)
    print("===== Optimizing:", {"xyz": args.opt_xyz, "rgb": args.opt_rgb, "opacity": args.opt_opacity, "scale": args.opt_scale, "rotation": args.opt_rotation})
    n_gaussians = state["xyz"].shape[1]
    residuals = GaussianResiduals(n_gaussians, tuple(state["scale_raw"].shape), device, args.opt_xyz, args.opt_rgb, args.opt_opacity, args.opt_scale, args.opt_rotation).to(device)
    trainable = [p for p in residuals.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable residual. Add --opt_rgb / --opt_opacity / --opt_scale / --opt_xyz / --opt_rotation")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    curve = []
    for step in range(args.steps):
        batch = random.sample(train_frames, min(args.batch_size, len(train_frames)))
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0; parts = []
        for frame_idx in batch:
            loss, info = compute_loss_for_frame(args, frame_idx, train_view, state, residuals, gt_provider, pipe)
            total_loss = total_loss + loss; parts.append(info)
        total_loss = total_loss / len(batch)
        total_loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == args.steps - 1:
            row = {"step": step, "loss": float(total_loss.detach().cpu()), "fg_l1": float(np.mean([p["fg_l1"] for p in parts])), "bg_l1": float(np.mean([p["bg_l1"] for p in parts])), "delta_reg": float(residuals.regularization().detach().cpu())}
            curve.append(row); print(row)
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            torch.save({"step": step, "args": vars(args), "state_dict": residuals.state_dict()}, output_dir / f"residual_step_{step+1}.pt")
    torch.save({"step": args.steps, "args": vars(args), "state_dict": residuals.state_dict()}, output_dir / "residual_final.pt")
    import pandas as pd
    pd.DataFrame(curve).to_csv(output_dir / "train_curve.csv", index=False)
    with open(output_dir / "residual_config.json", "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "inference_path": args.inference_path, "gt_root": args.gt_root, "output_dir": args.output_dir, "view_indices": view_indices, "train_frames": train_frames, "opt_xyz": args.opt_xyz, "opt_rgb": args.opt_rgb, "opt_opacity": args.opt_opacity, "opt_scale": args.opt_scale, "opt_rotation": args.opt_rotation}, f, indent=2)
    print("===== Rendering corrected frames")
    render_views = view_indices if args.render_all_views else [view_indices[0]]
    all_frames = sorted(set(train_indices + test_indices))
    for view_out_idx, vi in enumerate(render_views):
        view = views[vi]
        for frame_idx in tqdm(all_frames, desc=f"Rendering view {vi}"):
            save_frame(args, frame_idx, view, state, residuals, pipe, output_dir, view_out_idx)
    print("Saved residual checkpoint:", output_dir / "residual_final.pt")
    print("Saved rendered frames:", output_dir / args.name)


if __name__ == "__main__":
    main()
