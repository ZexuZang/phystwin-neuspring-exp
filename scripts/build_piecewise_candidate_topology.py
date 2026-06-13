#!/usr/bin/env python3
"""Build one NeuSpring-style piecewise topology candidate.

This is the topology side of the NeuSpring idea:
- cluster mass points into regions;
- give every region its own KNN/radius hyper-parameters;
- keep controller springs from the original PhysTwin topology;
- save a PhysTwin-compatible topology .npz.

The loss-driven search script calls this file many times with different
region_knn / region_radius_scale values, evaluates each candidate after
parameter optimization, and keeps the best topology.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans


def _as_float_list(value: str | None, length: int, default: float) -> list[float]:
    if value is None or value == "":
        return [float(default)] * length
    xs = [float(x) for x in value.replace(",", " ").split()]
    if len(xs) == 1:
        xs = xs * length
    if len(xs) != length:
        raise ValueError(f"Expected {length} values, got {len(xs)}: {value}")
    return xs


def _as_int_list(value: str | None, length: int, default: int) -> list[int]:
    if value is None or value == "":
        return [int(default)] * length
    xs = [int(round(float(x))) for x in value.replace(",", " ").split()]
    if len(xs) == 1:
        xs = xs * length
    if len(xs) != length:
        raise ValueError(f"Expected {length} values, got {len(xs)}: {value}")
    return xs


def _add_edge(edge_set: set[tuple[int, int]], i: int, j: int) -> bool:
    if int(i) == int(j):
        return False
    a, b = sorted((int(i), int(j)))
    if (a, b) in edge_set:
        return False
    edge_set.add((a, b))
    return True


def _components(num_nodes: int, edges: Iterable[tuple[int, int]]) -> list[np.ndarray]:
    parent = np.arange(num_nodes)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(num_nodes):
        groups.setdefault(find(i), []).append(i)
    return sorted([np.asarray(v, dtype=np.int64) for v in groups.values()], key=len, reverse=True)


def _load_points_and_topology(data_root: Path, original_topology: Path) -> dict:
    with open(data_root / "final_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_root / "split.json", "r") as f:
        split = json.load(f)

    object_points_all = np.asarray(data["object_points"])
    object_points = object_points_all[0].astype(np.float64)
    train_frame = int(split["train"][1])

    topo = np.load(original_topology, allow_pickle=True)
    orig_springs = topo["springs"].astype(np.int64)
    orig_rest_lengths = topo["rest_lengths"].astype(np.float64)
    orig_num_object_springs = int(topo["num_object_springs"])

    return {
        "data": data,
        "object_points_all": object_points_all,
        "object_points": object_points,
        "train_frame": train_frame,
        "topo": topo,
        "orig_springs": orig_springs,
        "orig_rest_lengths": orig_rest_lengths,
        "orig_num_object_springs": orig_num_object_springs,
        "original_object_springs": orig_springs[:orig_num_object_springs],
        "original_controller_springs": orig_springs[orig_num_object_springs:],
        "original_controller_rest": orig_rest_lengths[orig_num_object_springs:],
    }


def _cluster_regions(object_points_all: np.ndarray, train_frame: int, num_regions: int, random_state: int) -> np.ndarray:
    object_points = object_points_all[0].astype(np.float64)
    pos_feat = (object_points - object_points.mean(axis=0, keepdims=True)) / (
        object_points.std(axis=0, keepdims=True) + 1e-8
    )

    max_motion_frame = max(2, min(train_frame, object_points_all.shape[0] - 1))
    motion = object_points_all[1:max_motion_frame] - object_points_all[: max_motion_frame - 1]
    motion_mag = np.linalg.norm(motion, axis=-1)
    if motion_mag.size == 0:
        motion_feat = np.zeros((object_points.shape[0], 2), dtype=np.float64)
    else:
        motion_mean_raw = motion_mag.mean(axis=0)
        motion_std_raw = motion_mag.std(axis=0)
        motion_mean = (motion_mean_raw - motion_mean_raw.mean()) / (motion_mean_raw.std() + 1e-8)
        motion_std = (motion_std_raw - motion_std_raw.mean()) / (motion_std_raw.std() + 1e-8)
        motion_feat = np.stack([motion_mean, motion_std], axis=1)

    features = np.concatenate([pos_feat, 0.5 * motion_feat], axis=1)
    return KMeans(n_clusters=num_regions, random_state=random_state, n_init=10).fit_predict(features)


def _base_region_params(object_points: np.ndarray, object_points_all: np.ndarray, train_frame: int, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_regions = int(labels.max()) + 1
    tree = cKDTree(object_points)
    nn_dists, _ = tree.query(object_points, k=2)
    local_spacing = nn_dists[:, 1]

    max_motion_frame = max(2, min(train_frame, object_points_all.shape[0] - 1))
    motion = object_points_all[1:max_motion_frame] - object_points_all[: max_motion_frame - 1]
    motion_mag = np.linalg.norm(motion, axis=-1)
    if motion_mag.size == 0:
        complexity = np.zeros(object_points.shape[0], dtype=np.float64)
    else:
        complexity = motion_mag.mean(axis=0)
        complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-8)

    base_radius = np.zeros(num_regions, dtype=np.float64)
    base_k = np.zeros(num_regions, dtype=np.int64)
    for r in range(num_regions):
        idx = np.where(labels == r)[0]
        if len(idx) == 0:
            base_radius[r] = 0.02
            base_k[r] = 16
            continue
        spacing_r = float(np.median(local_spacing[idx]))
        motion_r = float(np.mean(complexity[idx]))
        base_radius[r] = np.clip(spacing_r * (2.6 + 1.2 * motion_r), 0.012, 0.035)
        base_k[r] = int(np.clip(12 + 16 * motion_r + np.sqrt(len(idx)) * 0.25, 8, 40))
    return base_radius, base_k


def build_piecewise_candidate_topology(
    *,
    data_root: Path,
    original_topology: Path,
    output_path: Path,
    num_regions: int = 5,
    region_radius_scale: list[float] | None = None,
    region_knn: list[int] | None = None,
    bridges_per_region_pair: int = 3,
    min_degree: int = 1,
    random_state: int = 0,
    candidate_id: str = "candidate",
) -> Path:
    loaded = _load_points_and_topology(data_root, original_topology)
    object_points_all = loaded["object_points_all"]
    object_points = loaded["object_points"]
    train_frame = loaded["train_frame"]
    topo = loaded["topo"]

    labels = _cluster_regions(object_points_all, train_frame, num_regions, random_state)
    base_radius, base_k = _base_region_params(object_points, object_points_all, train_frame, labels)

    if region_radius_scale is None:
        region_radius_scale = [1.0] * num_regions
    if region_knn is None:
        region_knn = [int(x) for x in base_k]
    if len(region_radius_scale) != num_regions or len(region_knn) != num_regions:
        raise ValueError("region_radius_scale and region_knn must have length num_regions")

    region_radius = np.clip(base_radius * np.asarray(region_radius_scale, dtype=np.float64), 0.006, 0.080)
    region_knn = np.clip(np.asarray(region_knn, dtype=np.int64), 3, 80)

    tree = cKDTree(object_points)
    edges: set[tuple[int, int]] = set()
    num_points = object_points.shape[0]

    for i in range(num_points):
        ri = int(labels[i])
        cand = tree.query_ball_point(object_points[i], r=float(region_radius[ri]))
        cand = [int(j) for j in cand if int(j) != i]
        cand = sorted(cand, key=lambda j: np.linalg.norm(object_points[i] - object_points[j]))[: int(region_knn[ri])]
        for j in cand:
            rj = int(labels[j])
            d = float(np.linalg.norm(object_points[i] - object_points[j]))
            if d <= max(float(region_radius[ri]), float(region_radius[rj])):
                _add_edge(edges, i, j)

    # local degree repair
    degree = np.zeros(num_points, dtype=np.int64)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    for i in range(num_points):
        if degree[i] >= min_degree:
            continue
        _, idxs = tree.query(object_points[i], k=min(num_points, max(min_degree + 8, 12)))
        idxs = np.atleast_1d(idxs)
        for j in idxs:
            if int(j) == i:
                continue
            if _add_edge(edges, i, int(j)):
                degree[i] += 1
                degree[int(j)] += 1
            if degree[i] >= min_degree:
                break

    # inter-region bridge springs
    added_region_bridges = 0
    for ra in range(num_regions):
        idx_a = np.where(labels == ra)[0]
        if len(idx_a) == 0:
            continue
        for rb in range(ra + 1, num_regions):
            idx_b = np.where(labels == rb)[0]
            if len(idx_b) == 0:
                continue
            tree_b = cKDTree(object_points[idx_b])
            dists, nn = tree_b.query(object_points[idx_a], k=1)
            order = np.argsort(dists)
            count = 0
            for local_i in order:
                ia = int(idx_a[int(local_i)])
                ib = int(idx_b[int(nn[int(local_i)])])
                threshold = max(float(region_radius[ra]), float(region_radius[rb])) * 1.5
                if float(dists[int(local_i)]) > threshold:
                    continue
                if _add_edge(edges, ia, ib):
                    added_region_bridges += 1
                    count += 1
                if count >= bridges_per_region_pair:
                    break

    # global component repair
    comps = _components(num_points, edges)
    added_component_edges = 0
    if len(comps) > 1:
        main = comps[0]
        for comp in comps[1:]:
            main_tree = cKDTree(object_points[main])
            dists, nn_idx = main_tree.query(object_points[comp], k=1)
            local_min = int(np.argmin(dists))
            a = int(comp[local_min])
            b = int(main[int(nn_idx[local_min])])
            if _add_edge(edges, a, b):
                added_component_edges += 1

    comps_after = _components(num_points, edges)
    if len(comps_after) != 1:
        raise RuntimeError(f"Topology graph still disconnected: {len(comps_after)} components")

    object_springs = np.asarray(sorted(edges), dtype=np.int64)
    object_rest = np.linalg.norm(object_points[object_springs[:, 0]] - object_points[object_springs[:, 1]], axis=1)
    object_rest = np.maximum(object_rest, 1e-8)

    controller_springs = loaded["original_controller_springs"]
    controller_rest = loaded["original_controller_rest"]
    springs = np.concatenate([object_springs, controller_springs], axis=0)
    rest_lengths = np.concatenate([object_rest, controller_rest], axis=0)

    points_full = topo["points_full"] if "points_full" in topo.files else None
    masses = topo["masses"].astype(np.float32) if "masses" in topo.files else np.ones(int(springs.max()) + 1, dtype=np.float32)
    spring_Y = np.ones(len(springs), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        points_full=points_full,
        springs=springs,
        rest_lengths=rest_lengths,
        masses=masses,
        spring_Y=spring_Y,
        num_object_springs=len(object_springs),
        topology_type="neuspring_piecewise_candidate_no_pruning",
        candidate_id=candidate_id,
        num_regions=num_regions,
        region_labels=labels,
        base_region_radius=base_radius,
        base_region_knn=base_k,
        region_radius_scale=np.asarray(region_radius_scale, dtype=np.float64),
        region_radius=region_radius,
        region_knn=region_knn,
        bridges_per_region_pair=bridges_per_region_pair,
        min_degree=min_degree,
        added_region_bridges=added_region_bridges,
        added_component_edges=added_component_edges,
        original_topology_path=str(original_topology),
        original_controller_springs=controller_springs,
        original_controller_rest=controller_rest,
    )

    summary = pd.DataFrame([
        {
            "candidate_id": candidate_id,
            "topology_path": str(output_path),
            "object_springs": len(object_springs),
            "controller_springs": len(controller_springs),
            "total_springs": len(springs),
            "num_regions": num_regions,
            "region_radius_scale": " ".join(f"{x:.3f}" for x in region_radius_scale),
            "region_knn": " ".join(str(int(x)) for x in region_knn),
        }
    ])
    summary.to_csv(output_path.with_suffix(".summary.csv"), index=False)

    print("[topology] saved:", output_path)
    print(summary.to_string(index=False))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--original-topology", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-regions", type=int, default=5)
    parser.add_argument("--region-radius-scale", default=None, help="single value or N values, comma/space separated")
    parser.add_argument("--region-knn", default=None, help="single value or N values, comma/space separated")
    parser.add_argument("--bridges-per-region-pair", type=int, default=3)
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--candidate-id", default="manual")
    args = parser.parse_args()

    radius_scale = _as_float_list(args.region_radius_scale, args.num_regions, 1.0)
    region_knn = _as_int_list(args.region_knn, args.num_regions, 16) if args.region_knn else None

    build_piecewise_candidate_topology(
        data_root=Path(args.data_root),
        original_topology=Path(args.original_topology),
        output_path=Path(args.output_path),
        num_regions=args.num_regions,
        region_radius_scale=radius_scale,
        region_knn=region_knn,
        bridges_per_region_pair=args.bridges_per_region_pair,
        min_degree=args.min_degree,
        random_state=args.random_state,
        candidate_id=args.candidate_id,
    )


if __name__ == "__main__":
    main()
