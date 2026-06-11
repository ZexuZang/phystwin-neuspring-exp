#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans


def get_components(num_nodes: int, edge_set: set[tuple[int, int]]) -> list[np.ndarray]:
    parent = np.arange(num_nodes)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edge_set:
        union(int(a), int(b))

    comp_dict: dict[int, list[int]] = {}
    for i in range(num_nodes):
        r = find(i)
        comp_dict.setdefault(r, []).append(i)

    comps = [np.asarray(v, dtype=np.int64) for v in comp_dict.values()]
    return sorted(comps, key=len, reverse=True)


def add_edge(edge_set: set[tuple[int, int]], i: int, j: int) -> bool:
    if i == j:
        return False
    a, b = sorted((int(i), int(j)))
    if (a, b) not in edge_set:
        edge_set.add((a, b))
        return True
    return False


def build_regionwise_topology(
    *,
    data_root: Path,
    original_topology_path: Path,
    output_path: Path,
    num_regions: int = 5,
    bridges_per_region_pair: int = 3,
    random_state: int = 0,
) -> Path:
    with open(data_root / "final_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_root / "split.json", "r") as f:
        split = json.load(f)

    train_frame = int(split["train"][1])
    object_points_all = np.asarray(data["object_points"])
    object_points = object_points_all[0].astype(np.float64)

    orig_topo = np.load(original_topology_path, allow_pickle=True)
    orig_springs = orig_topo["springs"].astype(np.int64)
    orig_rest_lengths = orig_topo["rest_lengths"].astype(np.float64)
    orig_num_object_springs = int(orig_topo["num_object_springs"])
    points_full = orig_topo["points_full"] if "points_full" in orig_topo.files else None
    masses = (
        orig_topo["masses"].astype(np.float32)
        if "masses" in orig_topo.files
        else np.ones(int(orig_springs.max()) + 1, dtype=np.float32)
    )

    original_object_springs = orig_springs[:orig_num_object_springs]
    original_controller_springs = orig_springs[orig_num_object_springs:]
    original_controller_rest = orig_rest_lengths[orig_num_object_springs:]

    num_object_points = object_points.shape[0]
    print("object_points:", object_points.shape)
    print("original object springs:", len(original_object_springs))
    print("original controller springs:", len(original_controller_springs))

    # KMeans feature = canonical position + simple motion features.
    pos_feat = (object_points - object_points.mean(axis=0, keepdims=True)) / (
        object_points.std(axis=0, keepdims=True) + 1e-8
    )
    max_motion_frame = min(train_frame, object_points_all.shape[0] - 1)
    motion = object_points_all[1:max_motion_frame] - object_points_all[: max_motion_frame - 1]
    motion_mag = np.linalg.norm(motion, axis=-1)

    motion_mean = (motion_mag.mean(axis=0) - motion_mag.mean()) / (
        motion_mag.mean(axis=0).std() + 1e-8
    )
    motion_std_raw = motion_mag.std(axis=0)
    motion_std = (motion_std_raw - motion_std_raw.mean()) / (
        motion_std_raw.std() + 1e-8
    )
    motion_feat = np.stack([motion_mean, motion_std], axis=1)
    features = np.concatenate([pos_feat, 0.5 * motion_feat], axis=1)
    region_labels = KMeans(n_clusters=num_regions, random_state=random_state, n_init=10).fit_predict(features)

    tree = cKDTree(object_points)
    nn_dists, _ = tree.query(object_points, k=2)
    local_spacing = nn_dists[:, 1]

    motion_complexity = motion_mag.mean(axis=0)
    motion_complexity = (motion_complexity - motion_complexity.min()) / (
        motion_complexity.max() - motion_complexity.min() + 1e-8
    )

    region_radius = np.zeros(num_regions)
    region_max_neighbors = np.zeros(num_regions, dtype=np.int64)

    for r in range(num_regions):
        idx = np.where(region_labels == r)[0]
        if len(idx) == 0:
            region_radius[r] = 0.02
            region_max_neighbors[r] = 16
            continue
        spacing_r = float(np.median(local_spacing[idx]))
        motion_r = float(np.mean(motion_complexity[idx]))
        region_radius[r] = np.clip(spacing_r * (2.6 + 1.2 * motion_r), 0.012, 0.035)
        region_max_neighbors[r] = int(np.clip(12 + 16 * motion_r + np.sqrt(len(idx)) * 0.25, 10, 32))

    print("region_radius:", region_radius)
    print("region_max_neighbors:", region_max_neighbors)

    edges: set[tuple[int, int]] = set()
    for i in range(num_object_points):
        ri = int(region_labels[i])
        cand = tree.query_ball_point(object_points[i], r=float(region_radius[ri]))
        cand = [j for j in cand if j != i]
        cand = sorted(cand, key=lambda j: np.linalg.norm(object_points[i] - object_points[j]))[
            : int(region_max_neighbors[ri])
        ]
        for j in cand:
            rj = int(region_labels[j])
            d = np.linalg.norm(object_points[i] - object_points[j])
            if d <= max(region_radius[ri], region_radius[rj]):
                edges.add(tuple(sorted((int(i), int(j)))))

    # Min-degree repair.
    min_degree = 1
    degree = np.zeros(num_object_points, dtype=np.int64)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    for i in range(num_object_points):
        if degree[i] >= min_degree:
            continue
        _, idxs = tree.query(object_points[i], k=min(num_object_points, min_degree + 10))
        if np.isscalar(idxs):
            idxs = [int(idxs)]
        for j in idxs:
            j = int(j)
            if j == i:
                continue
            if add_edge(edges, i, j):
                degree[i] += 1
                degree[j] += 1
            if degree[i] >= min_degree:
                break

    # Inter-region bridge springs.
    added_region_bridges = 0
    for ra in range(num_regions):
        idx_a = np.where(region_labels == ra)[0]
        if len(idx_a) == 0:
            continue
        pts_a = object_points[idx_a]

        for rb in range(ra + 1, num_regions):
            idx_b = np.where(region_labels == rb)[0]
            if len(idx_b) == 0:
                continue
            pts_b = object_points[idx_b]
            tree_b = cKDTree(pts_b)
            dists, nn = tree_b.query(pts_a, k=1)
            order = np.argsort(dists)
            added_for_pair = 0

            for local_i in order:
                ia = int(idx_a[int(local_i)])
                ib = int(idx_b[int(nn[int(local_i)])])
                threshold = max(region_radius[ra], region_radius[rb]) * 1.5
                if float(dists[int(local_i)]) > threshold:
                    continue
                if add_edge(edges, ia, ib):
                    added_region_bridges += 1
                    added_for_pair += 1
                if added_for_pair >= bridges_per_region_pair:
                    break

    print("Added inter-region bridge edges:", added_region_bridges)

    # Global component repair.
    components_before = get_components(num_object_points, edges)
    print("Connected components before global repair:", len(components_before))
    print("Largest component size before repair:", len(components_before[0]) if components_before else 0)

    added_component_edges = 0
    if len(components_before) > 1:
        main_comp = components_before[0]
        for comp in components_before[1:]:
            main_tree = cKDTree(object_points[main_comp])
            dists, nn_idx = main_tree.query(object_points[comp], k=1)
            local_min = int(np.argmin(dists))
            node_in_comp = int(comp[local_min])
            node_in_main = int(main_comp[int(nn_idx[local_min])])
            if add_edge(edges, node_in_comp, node_in_main):
                added_component_edges += 1

    components_after = get_components(num_object_points, edges)
    print("Added global component repair edges:", added_component_edges)
    print("Connected components after global repair:", len(components_after))
    assert len(components_after) == 1, "Object spring graph is still disconnected after repair."

    regionwise_object_springs = np.asarray(sorted(edges), dtype=np.int64)
    regionwise_object_rest = np.linalg.norm(
        object_points[regionwise_object_springs[:, 0]]
        - object_points[regionwise_object_springs[:, 1]],
        axis=1,
    )
    regionwise_object_rest = np.maximum(regionwise_object_rest, 1e-8)

    regionwise_springs = np.concatenate([regionwise_object_springs, original_controller_springs], axis=0)
    regionwise_rest_lengths = np.concatenate([regionwise_object_rest, original_controller_rest], axis=0)
    regionwise_spring_Y = np.ones(len(regionwise_springs), dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        points_full=points_full,
        springs=regionwise_springs,
        rest_lengths=regionwise_rest_lengths,
        masses=masses,
        spring_Y=regionwise_spring_Y,
        num_object_springs=len(regionwise_object_springs),
        topology_type="regionwise_optimized_no_pruning",
        num_regions=num_regions,
        region_labels=region_labels,
        region_radius=region_radius,
        region_max_neighbors=region_max_neighbors,
        original_topology_path=str(original_topology_path),
        original_controller_springs=original_controller_springs,
        original_controller_rest=original_controller_rest,
    )

    assert regionwise_springs.shape[0] == regionwise_rest_lengths.shape[0]
    assert regionwise_springs.min() >= 0
    if points_full is not None:
        assert regionwise_springs.max() < points_full.shape[0]

    print("Saved regionwise topology:", output_path)
    print("Regionwise object springs:", len(regionwise_object_springs))
    print("Controller springs kept:", len(original_controller_springs))
    print("Total springs:", len(regionwise_springs))

    summary = pd.DataFrame(
        [
            {
                "Method": "OriginalTopology",
                "Topology Path": str(original_topology_path),
                "Object Springs": len(original_object_springs),
                "Controller Springs": len(original_controller_springs),
                "Total Springs": len(orig_springs),
            },
            {
                "Method": "RegionWiseOptimizedTopology",
                "Topology Path": str(output_path),
                "Object Springs": len(regionwise_object_springs),
                "Controller Springs": len(original_controller_springs),
                "Total Springs": len(regionwise_springs),
            },
        ]
    )
    summary.to_csv(output_path.parent / "topology_compare.csv", index=False)
    print(summary)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--original-topology", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-regions", type=int, default=5)
    parser.add_argument("--bridges-per-region-pair", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    build_regionwise_topology(
        data_root=Path(args.data_root),
        original_topology_path=Path(args.original_topology),
        output_path=Path(args.output_path),
        num_regions=args.num_regions,
        bridges_per_region_pair=args.bridges_per_region_pair,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
