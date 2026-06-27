#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_neural_spring_field.py

Enhanced Neural Spring Field for PhysTwin / NeuSpring-style spring-mass systems.

Changes:
1) Input features = midpoint + length + direction + region id embedding.
2) Lower smoothing strength by default: smooth_weight=1e-3.
3) Utilities to fit NSF from a raw checkpoint and export an NSF checkpoint.

This is for PhysTwin / NeuSpring. It is not a VGGT model file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POINT_KEYS = ["object_points", "points", "vertices", "mass_points", "point_xyz", "xyz", "x"]
EDGE_KEYS = ["object_springs", "springs", "edges", "spring_indices", "spring_ij", "edge_index"]
REGION_KEYS = ["region_id", "region_ids", "spring_region_id", "spring_region_ids", "object_spring_region_id", "object_spring_region_ids"]
SPRING_PARAM_KEYS = ["spring_Y", "spring_y", "spring_k", "spring_K", "spring_stiffness", "stiffness", "spring_params"]


def _find_first_key(data: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in data:
            return k
    return None


def load_topology_npz(topology_path: str | Path) -> Dict[str, np.ndarray]:
    topology_path = Path(topology_path)
    with np.load(topology_path, allow_pickle=True) as raw:
        data = {k: raw[k] for k in raw.keys()}

    point_key = _find_first_key(data, POINT_KEYS)
    edge_key = _find_first_key(data, EDGE_KEYS)
    region_key = _find_first_key(data, REGION_KEYS)

    if point_key is None:
        raise KeyError(f"Cannot find point key in {topology_path}. Available keys: {list(data.keys())}")
    if edge_key is None:
        raise KeyError(f"Cannot find edge/spring key in {topology_path}. Available keys: {list(data.keys())}")

    points = np.asarray(data[point_key], dtype=np.float32)
    edges = np.asarray(data[edge_key])
    if edges.ndim == 2 and edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T
    edges = edges[:, :2].astype(np.int64)

    out = {"points": points, "edges": edges, "point_key": np.array(point_key), "edge_key": np.array(edge_key)}
    if region_key is not None:
        region_ids = np.asarray(data[region_key]).astype(np.int64).reshape(-1)
        out["region_ids"] = region_ids
        out["region_key"] = np.array(region_key)
    return out


def simple_kmeans_numpy(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int64)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = d2.argmin(axis=1)
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(axis=0)
    return labels


def build_spring_features(points: np.ndarray, edges: np.ndarray, region_ids: Optional[np.ndarray] = None, num_regions: int = 8) -> Dict[str, np.ndarray]:
    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]
    midpoint = 0.5 * (p0 + p1)
    vec = p1 - p0
    length = np.linalg.norm(vec, axis=-1, keepdims=True).astype(np.float32)
    direction = vec / np.maximum(length, 1e-8)

    center = midpoint.mean(axis=0, keepdims=True)
    scale = np.maximum(np.std(midpoint, axis=0, keepdims=True).mean(), 1e-6)
    midpoint_norm = (midpoint - center) / scale
    length_norm = length / np.maximum(np.mean(length), 1e-8)

    if region_ids is None or len(region_ids) != len(edges):
        region_ids = simple_kmeans_numpy(midpoint, k=num_regions, iters=30, seed=0)

    region_ids = np.asarray(region_ids, dtype=np.int64).reshape(-1)
    num_regions = int(max(num_regions, int(region_ids.max()) + 1))
    numeric = np.concatenate([midpoint_norm, length_norm, direction], axis=-1).astype(np.float32)

    return {
        "numeric": numeric,
        "midpoint": midpoint.astype(np.float32),
        "length": length.astype(np.float32),
        "direction": direction.astype(np.float32),
        "region_ids": region_ids,
        "num_regions": np.array(num_regions, dtype=np.int64),
        "norm_center": center.astype(np.float32),
        "norm_scale": np.asarray(scale, dtype=np.float32),
    }


class EnhancedNeuralSpringField(nn.Module):
    """NSF: [midpoint_xyz_norm, length_norm, direction_xyz, region_embedding] -> positive spring parameter."""
    def __init__(self, numeric_dim: int = 7, num_regions: int = 8, region_embed_dim: int = 8, hidden_dim: int = 128, num_layers: int = 4, out_dim: int = 1, min_value: float = 1e-6):
        super().__init__()
        self.region_embedding = nn.Embedding(num_regions, region_embed_dim)
        layers: List[nn.Module] = []
        dim = numeric_dim + region_embed_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.SiLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.min_value = min_value
        self.config = dict(numeric_dim=numeric_dim, num_regions=num_regions, region_embed_dim=region_embed_dim, hidden_dim=hidden_dim, num_layers=num_layers, out_dim=out_dim, min_value=min_value)

    def forward(self, numeric: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        emb = self.region_embedding(region_ids.long())
        x = torch.cat([numeric, emb], dim=-1)
        return F.softplus(self.net(x)) + self.min_value


def find_spring_param_key(ckpt: Dict[str, Any]) -> str:
    for k in SPRING_PARAM_KEYS:
        if k in ckpt:
            return k
    for outer in ["state_dict", "model", "params"]:
        if outer in ckpt and isinstance(ckpt[outer], dict):
            for k in SPRING_PARAM_KEYS:
                if k in ckpt[outer]:
                    return f"{outer}.{k}"
    raise KeyError(f"Cannot find spring parameter key. Top keys: {list(ckpt.keys())[:80]}")


def get_nested(d: Dict[str, Any], dotted: str):
    cur = d
    for p in dotted.split("."):
        cur = cur[p]
    return cur


def set_nested(d: Dict[str, Any], dotted: str, value):
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def load_spring_params_from_checkpoint(ckpt_path: str | Path) -> Tuple[Dict[str, Any], str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    key = find_spring_param_key(ckpt)
    val = get_nested(ckpt, key)
    if not isinstance(val, torch.Tensor):
        val = torch.as_tensor(val)
    return ckpt, key, val.float().reshape(-1, 1)


def edge_smoothness_loss(pred: torch.Tensor, region_ids: torch.Tensor, max_pairs_per_region: int = 4096) -> torch.Tensor:
    losses = []
    for r in torch.unique(region_ids):
        idx = torch.where(region_ids == r)[0]
        if idx.numel() < 2:
            continue
        n = min(max_pairs_per_region, idx.numel())
        a = idx[torch.randint(0, idx.numel(), (n,), device=pred.device)]
        b = idx[torch.randint(0, idx.numel(), (n,), device=pred.device)]
        losses.append((pred[a] - pred[b]).pow(2).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def fit_enhanced_nsf_to_raw(topology_path: str | Path, raw_checkpoint: str | Path, output_field: str | Path, output_checkpoint: str | Path, num_regions: int = 8, steps: int = 2000, lr: float = 1e-3, smooth_weight: float = 1e-3, hidden_dim: int = 128, region_embed_dim: int = 8, device: str = "cuda") -> Dict[str, Any]:
    topo = load_topology_npz(topology_path)
    features = build_spring_features(topo["points"], topo["edges"], region_ids=topo.get("region_ids"), num_regions=num_regions)
    numeric_np = features["numeric"]
    region_np = features["region_ids"]
    num_regions = int(features["num_regions"])

    ckpt, spring_key, spring_params = load_spring_params_from_checkpoint(raw_checkpoint)
    num_object_springs = len(topo["edges"])
    if spring_params.shape[0] < num_object_springs:
        raise ValueError(f"checkpoint spring length {spring_params.shape[0]} < topology object springs {num_object_springs}. Raw checkpoint and topology do not match.")

    target = spring_params[:num_object_springs].clone()
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    numeric = torch.from_numpy(numeric_np).to(device)
    region = torch.from_numpy(region_np).long().to(device)
    target = target.to(device)

    model = EnhancedNeuralSpringField(numeric_dim=numeric.shape[-1], num_regions=num_regions, region_embed_dim=region_embed_dim, hidden_dim=hidden_dim, num_layers=4, out_dim=target.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    log = []
    for step in range(steps):
        pred = model(numeric, region)
        fit_loss = F.mse_loss(torch.log(pred + 1e-8), torch.log(target + 1e-8))
        smooth_loss = edge_smoothness_loss(pred, region)
        loss = fit_loss + smooth_weight * smooth_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            row = {"step": step, "loss": float(loss.detach().cpu()), "fit_loss": float(fit_loss.detach().cpu()), "smooth_loss": float(smooth_loss.detach().cpu()), "smooth_weight": smooth_weight}
            print(row)
            log.append(row)

    with torch.no_grad():
        pred_full = model(numeric, region).detach().cpu()

    old_val = get_nested(ckpt, spring_key)
    old_tensor = old_val if isinstance(old_val, torch.Tensor) else torch.as_tensor(old_val)
    new_tensor = old_tensor.clone().float().reshape(-1, 1)
    new_tensor[:num_object_springs] = pred_full
    new_tensor = new_tensor.reshape(old_tensor.shape)
    set_nested(ckpt, spring_key, new_tensor)

    output_field = Path(output_field)
    output_checkpoint = Path(output_checkpoint)
    output_field.parent.mkdir(parents=True, exist_ok=True)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.cpu().state_dict(), "model_config": model.config, "feature_stats": {"norm_center": features["norm_center"], "norm_scale": features["norm_scale"]}, "topology_path": str(topology_path), "raw_checkpoint": str(raw_checkpoint), "spring_key": spring_key, "num_object_springs": int(num_object_springs), "feature_description": "midpoint_xyz_norm + length_norm + direction_xyz + region_embedding", "training_log": log}, output_field)
    torch.save(ckpt, output_checkpoint)

    return {"output_field": str(output_field), "output_checkpoint": str(output_checkpoint), "spring_key": spring_key, "num_object_springs": int(num_object_springs), "final_fit_loss": log[-1]["fit_loss"] if log else None, "final_smooth_loss": log[-1]["smooth_loss"] if log else None, "smooth_weight": smooth_weight}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", required=True)
    p.add_argument("--raw-checkpoint", required=True)
    p.add_argument("--output-field", required=True)
    p.add_argument("--output-checkpoint", required=True)
    p.add_argument("--num-regions", type=int, default=8)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--smooth-weight", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--region-embed-dim", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    result = fit_enhanced_nsf_to_raw(args.topology, args.raw_checkpoint, args.output_field, args.output_checkpoint, num_regions=args.num_regions, steps=args.steps, lr=args.lr, smooth_weight=args.smooth_weight, hidden_dim=args.hidden_dim, region_embed_dim=args.region_embed_dim, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
