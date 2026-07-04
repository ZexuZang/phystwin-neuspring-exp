#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render loss fine-tuning for PhysTwin / NeuSpring experiments.

This script is a practical render-side fine-tuning module that can be added to:
    https://github.com/ZexuZang/phystwin-neuspring-exp

It fine-tunes a lightweight image-space appearance adapter on top of already-rendered
Gaussian frames. It is designed for the situation where NSF simulation fine-tuning
improves geometry metrics but RGB render metrics are still weaker.

What it optimizes:
    - channel-wise gamma
    - 3x3 color affine matrix
    - RGB bias

What it preserves:
    - dynamic trajectory
    - Gaussian positions
    - simulation result / inference.pkl

This is intentionally conservative. It improves RGB appearance without destroying
the physical trajectory learned by NSF simulation loss fine-tuning.

Typical usage in Colab:

    %cd /content/PhysTwin

    !python scripts/render_loss_finetune.py \
      --method-name cand_000_nsf_sim_ft_param_opt \
      --pred-root /content/PhysTwin/results/nsf_three_methods_gaussian_renders/cand_000_nsf_sim_ft_param_opt \
      --gt-root /content/PhysTwin/data/different_types/double_stretch_sloth \
      --out-dir /content/PhysTwin/results/render_loss_ft/cand_000_nsf_sim_ft_param_opt \
      --steps 400 \
      --lr 0.03 \
      --lambda-ssim 0.2 \
      --batch-size 4

Outputs:
    out_dir/
      corrected_frames/
      render_ft_metrics.csv
      render_ft_train_curve.csv
      render_ft_params.pt
      render_ft_params.json

Notes:
    This script does not directly update the 3D Gaussian checkpoint. It is a robust
    first-stage render-loss fine-tuning that can be run immediately on the current
    PhysTwin outputs. A deeper version can later move this loss inside the Gaussian
    optimizer to update SH/color/opacity/scale directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# File helpers
# -----------------------------

def numeric_stem(path: str | Path) -> Optional[int]:
    nums = re.findall(r"\d+", Path(path).stem)
    return int(nums[-1]) if nums else None


def collect_images(root: str | Path) -> List[str]:
    root = str(root)
    files: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(files)


def build_frame_map(files: List[str]) -> Dict[int, str]:
    mp: Dict[int, str] = {}
    for f in files:
        idx = numeric_stem(f)
        if idx is not None and idx not in mp:
            mp[idx] = f
    return mp


def choose_pred_files(pred_root: str | Path, scene_name: Optional[str] = None) -> Tuple[List[str], str]:
    pred_root = Path(pred_root)

    candidates = [pred_root]
    if scene_name:
        candidates = [
            pred_root / scene_name / "0",
            pred_root / scene_name,
            pred_root,
        ]

    best_files: List[str] = []
    used_root = str(pred_root)

    for r in candidates:
        files = collect_images(r)
        if len(files) > len(best_files):
            best_files = files
            used_root = str(r)

    return best_files, used_root


def load_split_indices(split_path: Path, total_frames: int) -> Tuple[List[int], List[int]]:
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)

        # Match the existing evaluation convention used in the user's Colab cells.
        train_indices = list(range(split_info["train"][0] + 1, split_info["train"][1]))
        test_indices = list(range(split_info["test"][0], split_info["test"][1]))
        return train_indices, test_indices

    train_end = int(total_frames * 0.7)
    return list(range(train_end)), list(range(train_end, total_frames))


def load_rgb_np(path: str | Path) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img[:, :, :3].astype(np.uint8)


def load_mask_np(path: Optional[str | Path], fallback_rgb: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if path is None or not Path(path).exists():
        if fallback_rgb is None:
            return None
        mask = (fallback_rgb.astype(np.float32).sum(axis=-1) > 5).astype(np.float32)
        return mask

    mask = np.array(Image.open(path)).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.max() > 1.0:
        mask /= 255.0
    return mask


def resize_rgb_to(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    if rgb.shape[:2] == (height, width):
        return rgb
    return np.array(Image.fromarray(rgb).resize((width, height), Image.BILINEAR))


def resize_mask_to(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape[:2] == (height, width):
        return mask
    m = Image.fromarray((mask * 255).astype(np.uint8))
    m = m.resize((width, height), Image.NEAREST)
    return np.array(m).astype(np.float32) / 255.0


# -----------------------------
# Dataset
# -----------------------------

@dataclass
class FramePair:
    frame_idx: int
    split: str
    pred_path: str
    gt_path: str
    gt_mask_path: Optional[str]


class RenderPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: List[FramePair], image_size: Optional[int] = None):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]

        gt = load_rgb_np(pair.gt_path)
        pred = load_rgb_np(pair.pred_path)

        h, w = gt.shape[:2]
        pred = resize_rgb_to(pred, h, w)

        mask = load_mask_np(pair.gt_mask_path, fallback_rgb=gt)
        if mask is None:
            mask = np.ones((h, w), dtype=np.float32)
        else:
            mask = resize_mask_to(mask, h, w)

        if self.image_size is not None:
            # Keep aspect ratio by resizing shorter side to image_size then center crop.
            gt, pred, mask = resize_triplet(gt, pred, mask, self.image_size)

        gt_t = torch.from_numpy(gt.astype(np.float32) / 255.0).permute(2, 0, 1)
        pred_t = torch.from_numpy(pred.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return {
            "frame_idx": pair.frame_idx,
            "split": pair.split,
            "pred": pred_t,
            "gt": gt_t,
            "mask": mask_t,
        }


def resize_triplet(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, size: int):
    h, w = gt.shape[:2]
    scale = size / min(h, w)
    new_h = max(size, int(round(h * scale)))
    new_w = max(size, int(round(w * scale)))

    gt_r = np.array(Image.fromarray(gt).resize((new_w, new_h), Image.BILINEAR))
    pred_r = np.array(Image.fromarray(pred).resize((new_w, new_h), Image.BILINEAR))
    mask_r = resize_mask_to(mask, new_h, new_w)

    top = (new_h - size) // 2
    left = (new_w - size) // 2

    return (
        gt_r[top : top + size, left : left + size],
        pred_r[top : top + size, left : left + size],
        mask_r[top : top + size, left : left + size],
    )


def make_pairs(
    pred_root: Path,
    gt_root: Path,
    scene_name: Optional[str],
    split_path: Path,
) -> Tuple[List[FramePair], List[FramePair], str]:
    gt_color_files = collect_images(gt_root / "color")
    gt_mask_files = collect_images(gt_root / "mask")

    pred_files, used_pred_root = choose_pred_files(pred_root, scene_name=scene_name)

    if not pred_files:
        raise FileNotFoundError(f"No rendered images found under {pred_root}")

    gt_color_map = build_frame_map(gt_color_files)
    gt_mask_map = build_frame_map(gt_mask_files)
    pred_map = build_frame_map(pred_files)

    if not gt_color_map:
        raise FileNotFoundError(f"No GT color images found under {gt_root / 'color'}")

    total_frames = max(max(gt_color_map.keys()), max(pred_map.keys())) + 1
    train_indices, test_indices = load_split_indices(split_path, total_frames)

    train_pairs: List[FramePair] = []
    test_pairs: List[FramePair] = []

    for split_name, indices, out_list in [
        ("train", train_indices, train_pairs),
        ("test", test_indices, test_pairs),
    ]:
        for frame_idx in indices:
            gt_path = gt_color_map.get(frame_idx)
            pred_path = pred_map.get(frame_idx)

            if gt_path is None or pred_path is None:
                continue

            out_list.append(
                FramePair(
                    frame_idx=frame_idx,
                    split=split_name,
                    pred_path=pred_path,
                    gt_path=gt_path,
                    gt_mask_path=gt_mask_map.get(frame_idx),
                )
            )

    return train_pairs, test_pairs, used_pred_root


# -----------------------------
# Model and losses
# -----------------------------

class AppearanceAdapter(nn.Module):
    """Small image-space appearance adapter.

    The adapter is deliberately low-capacity:
        - gamma per channel
        - 3x3 color matrix
        - RGB bias

    This reduces the risk of overfitting while still improving brightness,
    contrast, and color mismatch caused by trajectory changes.
    """

    def __init__(self):
        super().__init__()
        self.color_matrix = nn.Parameter(torch.eye(3))
        self.bias = nn.Parameter(torch.zeros(3))
        self.log_gamma = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,3,H,W in [0,1]
        gamma = torch.exp(self.log_gamma).view(1, 3, 1, 1)
        x = torch.clamp(x, 1e-4, 1.0) ** gamma

        # Apply 3x3 matrix per pixel.
        # b c h w, d c -> b d h w
        y = torch.einsum("bchw,dc->bdhw", x, self.color_matrix)
        y = y + self.bias.view(1, 3, 1, 1)
        return torch.clamp(y, 0.0, 1.0)


def masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum() * pred.shape[1] + 1e-8
    return (torch.abs(pred - gt) * mask).sum() / denom


def ssim_torch(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Differentiable SSIM approximation.

    Returns mean SSIM over the batch.
    """
    if mask is not None:
        x = x * mask
        y = y * mask

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)

    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )

    if mask is not None:
        return (ssim_map * mask).sum() / (mask.sum() * x.shape[1] + 1e-8)
    return ssim_map.mean()


def psnr_np(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    pred_f = pred.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0

    if mask is not None:
        m = mask.astype(np.float32)[..., None]
        mse = ((pred_f - gt_f) ** 2 * m).sum() / (m.sum() * 3.0 + 1e-8)
    else:
        mse = np.mean((pred_f - gt_f) ** 2)

    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def simple_ssim_np(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    pred_t = torch.from_numpy(pred.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    gt_t = torch.from_numpy(gt.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    mask_t = None
    if mask is not None:
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        return float(ssim_torch(pred_t, gt_t, mask_t).item())


def compute_iou_np(pred: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_mask = pred.astype(np.float32).sum(axis=-1) > 5.0
    gt_bool = gt_mask > 0.5
    inter = np.logical_and(pred_mask, gt_bool).sum()
    union = np.logical_or(pred_mask, gt_bool).sum()
    return float(inter / union) if union > 0 else 1.0


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr


# -----------------------------
# Training and evaluation
# -----------------------------

def train_adapter(
    train_pairs: List[FramePair],
    out_dir: Path,
    steps: int,
    lr: float,
    batch_size: int,
    lambda_ssim: float,
    image_size: Optional[int],
    device: torch.device,
    seed: int,
) -> AppearanceAdapter:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = RenderPairDataset(train_pairs, image_size=image_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = AppearanceAdapter().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    curve = []
    step = 0

    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            pred = batch["pred"].to(device)
            gt = batch["gt"].to(device)
            mask = batch["mask"].to(device)

            out = model(pred)
            l1 = masked_l1(out, gt, mask)
            ssim_val = ssim_torch(out, gt, mask)
            loss = l1 + lambda_ssim * (1.0 - ssim_val)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 10 == 0 or step == steps - 1:
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "l1": float(l1.detach().cpu()),
                    "ssim": float(ssim_val.detach().cpu()),
                    "gamma_r": float(torch.exp(model.log_gamma[0]).detach().cpu()),
                    "gamma_g": float(torch.exp(model.log_gamma[1]).detach().cpu()),
                    "gamma_b": float(torch.exp(model.log_gamma[2]).detach().cpu()),
                }
                curve.append(row)
                print(row)

            step += 1

    pd.DataFrame(curve).to_csv(out_dir / "render_ft_train_curve.csv", index=False)
    torch.save(model.state_dict(), out_dir / "render_ft_params.pt")

    params_json = {
        "color_matrix": model.color_matrix.detach().cpu().tolist(),
        "bias": model.bias.detach().cpu().tolist(),
        "gamma": torch.exp(model.log_gamma).detach().cpu().tolist(),
    }
    with open(out_dir / "render_ft_params.json", "w", encoding="utf-8") as f:
        json.dump(params_json, f, indent=2)

    return model


def evaluate_and_save(
    model: AppearanceAdapter,
    pairs: List[FramePair],
    split: str,
    out_dir: Path,
    device: torch.device,
    save_frames: bool,
) -> Dict[str, float]:
    model.eval()

    corrected_dir = out_dir / "corrected_frames" / split
    if save_frames:
        corrected_dir.mkdir(parents=True, exist_ok=True)

    before_psnr, before_ssim, before_iou = [], [], []
    after_psnr, after_ssim, after_iou = [], [], []

    for pair in pairs:
        gt = load_rgb_np(pair.gt_path)
        pred = load_rgb_np(pair.pred_path)
        h, w = gt.shape[:2]
        pred = resize_rgb_to(pred, h, w)

        gt_mask = load_mask_np(pair.gt_mask_path, fallback_rgb=gt)
        if gt_mask is None:
            gt_mask = np.ones((h, w), dtype=np.float32)
        else:
            gt_mask = resize_mask_to(gt_mask, h, w)

        pred_t = torch.from_numpy(pred.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            corrected_t = model(pred_t)[0]
        corrected = tensor_to_uint8(corrected_t)

        before_psnr.append(psnr_np(pred, gt, gt_mask))
        before_ssim.append(simple_ssim_np(pred, gt, gt_mask))
        before_iou.append(compute_iou_np(pred, gt_mask))

        after_psnr.append(psnr_np(corrected, gt, gt_mask))
        after_ssim.append(simple_ssim_np(corrected, gt, gt_mask))
        after_iou.append(compute_iou_np(corrected, gt_mask))

        if save_frames:
            Image.fromarray(corrected).save(corrected_dir / f"{pair.frame_idx:05d}.png")

    def mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "split": split,
        "num_frames": len(pairs),
        "PSNR_before": mean(before_psnr),
        "SSIM_before": mean(before_ssim),
        "IoU_before": mean(before_iou),
        "PSNR_after": mean(after_psnr),
        "SSIM_after": mean(after_ssim),
        "IoU_after": mean(after_iou),
        "PSNR_delta": mean(after_psnr) - mean(before_psnr),
        "SSIM_delta": mean(after_ssim) - mean(before_ssim),
        "IoU_delta": mean(after_iou) - mean(before_iou),
    }


def main():
    parser = argparse.ArgumentParser(description="Render loss fine-tuning from existing Gaussian render frames.")
    parser.add_argument("--method-name", type=str, default="cand_000_nsf_sim_ft_param_opt")
    parser.add_argument("--pred-root", type=str, required=True, help="Root directory of rendered prediction frames.")
    parser.add_argument("--gt-root", type=str, required=True, help="Case root, e.g. data/different_types/double_stretch_sloth.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--scene-name", type=str, default=None, help="Optional scene name used to search pred-root/scene/0.")
    parser.add_argument("--split-json", type=str, default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None, help="Optional crop size for training memory control.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate current renders without fine-tuning.")
    parser.add_argument("--save-corrected-frames", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)
    split_path = Path(args.split_json) if args.split_json else gt_root / "split.json"

    if args.scene_name is None:
        args.scene_name = gt_root.name

    train_pairs, test_pairs, used_pred_root = make_pairs(
        pred_root=pred_root,
        gt_root=gt_root,
        scene_name=args.scene_name,
        split_path=split_path,
    )

    print("method:", args.method_name)
    print("pred_root:", pred_root)
    print("used_pred_root:", used_pred_root)
    print("gt_root:", gt_root)
    print("split_path:", split_path, split_path.exists())
    print("train pairs:", len(train_pairs))
    print("test pairs:", len(test_pairs))

    metadata = {
        "method_name": args.method_name,
        "pred_root": str(pred_root),
        "used_pred_root": used_pred_root,
        "gt_root": str(gt_root),
        "split_path": str(split_path),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "steps": args.steps,
        "lr": args.lr,
        "lambda_ssim": args.lambda_ssim,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
    }
    with open(out_dir / "render_ft_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    if args.eval_only:
        model = AppearanceAdapter().to(device)
    else:
        if len(train_pairs) == 0:
            raise RuntimeError("No train pairs found; cannot fine-tune render appearance.")
        model = train_adapter(
            train_pairs=train_pairs,
            out_dir=out_dir,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            lambda_ssim=args.lambda_ssim,
            image_size=args.image_size,
            device=device,
            seed=args.seed,
        )

    rows = []
    rows.append(evaluate_and_save(model, train_pairs, "train", out_dir, device, args.save_corrected_frames))
    rows.append(evaluate_and_save(model, test_pairs, "test", out_dir, device, args.save_corrected_frames))

    metrics = pd.DataFrame(rows)
    metrics.insert(0, "Method", args.method_name)
    metrics.to_csv(out_dir / "render_ft_metrics.csv", index=False)

    print("\n=== Render loss fine-tuning metrics ===")
    print(metrics.to_string(index=False))
    print("\nSaved:", out_dir / "render_ft_metrics.csv")
    print("Corrected frames:", out_dir / "corrected_frames")


if __name__ == "__main__":
    main()
