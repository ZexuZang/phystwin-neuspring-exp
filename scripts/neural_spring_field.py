#!/usr/bin/env python3
"""Neural Spring Field utilities for PhysTwin checkpoints.

This implements the NeuSpring-style canonical-coordinate spring field:
    S(e) = S0 + F_theta(x_mid(e))
where x_mid(e) is the canonical midpoint of a spring.

Important practical note:
This file is a safe, non-invasive integration stage. It fits a neural field to
PhysTwin's learned per-spring stiffness after an inner train_warp.py run, then
writes a smoothed checkpoint. This gives you a field-regularized checkpoint and
is used by the loss-driven topology search. A deeper end-to-end integration can
later replace the per-spring trainable tensor inside trainer_warp.py with this
module, but this file already lets us build and rank optimized topologies.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpringFieldFitConfig:
    steps: int = 1200
    lr: float = 3e-3
    hidden_dim: int = 128
    fourier_frequencies: int = 6
    smooth_weight: float = 1e-4
    min_stiffness: float = 1e-6
    max_stiffness: float = 1e8
    device: str = "cuda"


class FourierSpringMLP(nn.Module):
    """Small canonical-coordinate MLP for log spring stiffness."""

    def __init__(self, hidden_dim: int = 128, fourier_frequencies: int = 6):
        super().__init__()
        self.fourier_frequencies = int(fourier_frequencies)
        in_dim = 3 * (1 + 2 * self.fourier_frequencies)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x is normalized to roughly [-1, 1].
        feats = [x]
        for k in range(self.fourier_frequencies):
            freq = float(2**k) * torch.pi
            feats.append(torch.sin(freq * x))
            feats.append(torch.cos(freq * x))
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(x)).squeeze(-1)


def _safe_torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_object_points(data_root: Path) -> np.ndarray:
    with open(data_root / "final_data.pkl", "rb") as f:
        data = pickle.load(f)
    return np.asarray(data["object_points"])[0].astype(np.float64)


def load_topology_points(topology_path: Path, data_root: Path | None = None) -> np.ndarray:
    topo = np.load(topology_path, allow_pickle=True)
    if "points_full" in topo.files and topo["points_full"] is not None:
        points_full = np.asarray(topo["points_full"])
        if points_full.ndim == 2 and points_full.shape[1] == 3:
            return points_full.astype(np.float64)
    if data_root is None:
        raise ValueError("topology has no points_full; please pass --data-root")
    return load_object_points(data_root)


def canonical_midpoints_for_object_springs(topology_path: Path, data_root: Path | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    topo = np.load(topology_path, allow_pickle=True)
    springs = topo["springs"].astype(np.int64)
    num_object_springs = int(topo["num_object_springs"])
    object_springs = springs[:num_object_springs]
    points = load_topology_points(topology_path, data_root)

    if object_springs.max() >= points.shape[0]:
        # Usually this means points_full is missing but object spring indexes are still valid
        # within object_points. Raise early so the user can inspect the topology.
        raise ValueError(
            f"spring index {object_springs.max()} >= points count {points.shape[0]}. "
            "The topology .npz does not match final_data.pkl/points_full."
        )
    mids = 0.5 * (points[object_springs[:, 0]] + points[object_springs[:, 1]])
    return mids.astype(np.float32), object_springs, num_object_springs


def normalize_points(x: np.ndarray) -> tuple[np.ndarray, dict]:
    center = x.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(x - center, axis=1)) + 1e-8
    out = (x - center) / scale
    stats = {"center": center.squeeze(0).tolist(), "scale": float(scale)}
    return out.astype(np.float32), stats


def fit_neural_spring_field_to_checkpoint(
    *,
    checkpoint_path: Path,
    topology_path: Path,
    data_root: Path,
    output_checkpoint_path: Path,
    output_field_path: Path,
    config: SpringFieldFitConfig,
) -> dict:
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt = _safe_torch_load(checkpoint_path, device)
    if "spring_Y" not in ckpt:
        raise KeyError(f"checkpoint does not contain spring_Y: {checkpoint_path}")

    mids_np, object_springs, num_object_springs = canonical_midpoints_for_object_springs(topology_path, data_root)
    mids_norm_np, norm_stats = normalize_points(mids_np)

    spring_Y = ckpt["spring_Y"].detach().clone().to(device).float()
    if spring_Y.numel() < num_object_springs:
        raise ValueError(f"checkpoint spring_Y length {spring_Y.numel()} < topology object springs {num_object_springs}")

    target = spring_Y[:num_object_springs].clamp(config.min_stiffness, config.max_stiffness)
    target_log = torch.log(target).detach()
    base_log = torch.median(target_log).detach()

    x = torch.tensor(mids_norm_np, dtype=torch.float32, device=device)
    model = FourierSpringMLP(config.hidden_dim, config.fourier_frequencies).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-6)

    best_loss = float("inf")
    best_state = None
    for step in range(config.steps):
        pred_delta = model(x)
        pred_log = base_log + pred_delta
        data_loss = F.mse_loss(pred_log, target_log)
        smooth_loss = (pred_delta ** 2).mean()
        loss = data_loss + config.smooth_weight * smooth_loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step % max(1, config.steps // 10) == 0 or step == config.steps - 1:
            print(
                f"[nsf fit] step={step:04d} loss={float(loss.detach()):.6e} "
                f"data={float(data_loss.detach()):.6e} smooth={float(smooth_loss.detach()):.6e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        smoothed_log = base_log + model(x)
        smoothed_Y = torch.exp(smoothed_log).clamp(config.min_stiffness, config.max_stiffness)

    new_ckpt = dict(ckpt)
    new_spring_Y = spring_Y.detach().clone()
    new_spring_Y[:num_object_springs] = smoothed_Y
    new_ckpt["spring_Y"] = new_spring_Y.detach().cpu()
    new_ckpt["neural_spring_field"] = {
        "type": "FourierSpringMLP_postfit",
        "config": asdict(config),
        "normalization": norm_stats,
        "base_log_stiffness": float(base_log.detach().cpu()),
        "topology_path": str(topology_path),
        "source_checkpoint_path": str(checkpoint_path),
        "num_object_springs": int(num_object_springs),
        "fit_loss": best_loss,
    }

    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_field_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, output_checkpoint_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "normalization": norm_stats,
            "base_log_stiffness": float(base_log.detach().cpu()),
            "topology_path": str(topology_path),
            "source_checkpoint_path": str(checkpoint_path),
            "num_object_springs": int(num_object_springs),
            "fit_loss": best_loss,
        },
        output_field_path,
    )

    stats = {
        "checkpoint_path": str(checkpoint_path),
        "topology_path": str(topology_path),
        "output_checkpoint_path": str(output_checkpoint_path),
        "output_field_path": str(output_field_path),
        "num_object_springs": int(num_object_springs),
        "target_mean": float(target.mean().detach().cpu()),
        "smoothed_mean": float(smoothed_Y.mean().detach().cpu()),
        "target_std": float(target.std().detach().cpu()),
        "smoothed_std": float(smoothed_Y.std().detach().cpu()),
        "fit_loss": best_loss,
    }
    with open(output_field_path.with_suffix(".json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[nsf fit] saved checkpoint:", output_checkpoint_path)
    print("[nsf fit] saved field:", output_field_path)
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-field", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--fourier-frequencies", type=int, default=6)
    parser.add_argument("--smooth-weight", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = SpringFieldFitConfig(
        steps=args.steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        fourier_frequencies=args.fourier_frequencies,
        smooth_weight=args.smooth_weight,
        device=args.device,
    )
    fit_neural_spring_field_to_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        topology_path=Path(args.topology),
        data_root=Path(args.data_root),
        output_checkpoint_path=Path(args.output_checkpoint),
        output_field_path=Path(args.output_field),
        config=cfg,
    )


if __name__ == "__main__":
    main()
