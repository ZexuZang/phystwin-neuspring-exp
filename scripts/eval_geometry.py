#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def chamfer_single_direction_l1(pred_points: np.ndarray, gt_visible_points: np.ndarray) -> float:
    tree = cKDTree(pred_points)
    dists, _ = tree.query(gt_visible_points, k=1, p=1)
    return float(np.mean(dists))


def evaluate_geometry(
    *,
    data_root: Path,
    method_table_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    final_data_path = data_root / "final_data.pkl"
    gt_track_path = data_root / "gt_track_3d.pkl"
    split_path = data_root / "split.json"

    for p in [final_data_path, gt_track_path, split_path, method_table_csv]:
        if not p.exists():
            raise FileNotFoundError(p)

    with open(final_data_path, "rb") as f:
        data = pickle.load(f)
    with open(gt_track_path, "rb") as f:
        gt_track_3d = pickle.load(f)
    with open(split_path, "r") as f:
        split = json.load(f)

    object_points_gt = np.asarray(data["object_points"])
    object_visibilities = np.asarray(data["object_visibilities"])
    surface_points = np.asarray(data["surface_points"])
    gt_track_3d = np.asarray(gt_track_3d)

    num_original_points = object_points_gt.shape[1]
    num_surface_points = num_original_points + surface_points.shape[0]
    train_frame = int(split["train"][1])
    test_frame = int(split["test"][1])

    print("num_surface_points:", num_surface_points)
    print("train_frame:", train_frame)
    print("test_frame:", test_frame)

    def evaluate_cd(pred_vertices: np.ndarray, start_frame: int, end_frame: int) -> dict[str, float]:
        cd_values = []
        max_frame = min(end_frame, pred_vertices.shape[0], object_points_gt.shape[0])
        for frame_idx in range(start_frame, max_frame):
            pred_x = pred_vertices[frame_idx]
            gt_points = object_points_gt[frame_idx]
            gt_vis = object_visibilities[frame_idx]
            gt_visible_points = gt_points[gt_vis]
            pred_surface_points = (
                pred_x[:num_surface_points]
                if pred_x.shape[0] >= num_surface_points
                else pred_x
            )
            if len(gt_visible_points) == 0 or len(pred_surface_points) == 0:
                continue
            cd_values.append(chamfer_single_direction_l1(pred_surface_points, gt_visible_points))
        return {
            "frame_num": len(cd_values),
            "cd": float(np.mean(cd_values)) if cd_values else np.nan,
        }

    def evaluate_track(pred_vertices: np.ndarray, start_frame: int, end_frame: int) -> float:
        track_errors = []
        mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
        tree = cKDTree(pred_vertices[0])
        _, idx = tree.query(gt_track_3d[0][mask], k=1)
        max_frame = min(end_frame, pred_vertices.shape[0], gt_track_3d.shape[0])
        for frame_idx in range(start_frame, max_frame):
            new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
            gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
            pred_track_points = pred_vertices[frame_idx][idx][new_mask]
            track_error = (
                0.0
                if len(pred_track_points) == 0
                else np.mean(np.linalg.norm(pred_track_points - gt_track_points, axis=1))
            )
            track_errors.append(track_error)
        return float(np.mean(track_errors)) if track_errors else np.nan

    method_df = pd.read_csv(method_table_csv)
    rows = []

    for _, row in method_df.iterrows():
        method = row["Method"]
        path = row["Inference Path"]
        print("=" * 100)
        print("Method:", method)
        print("Inference:", path)
        if not isinstance(path, str) or not Path(path).exists():
            print("[skip] missing inference")
            continue

        with open(path, "rb") as f:
            pred_vertices = pickle.load(f)
        pred_vertices = np.asarray(pred_vertices)

        cd_train = evaluate_cd(pred_vertices, 1, train_frame)
        cd_test = evaluate_cd(pred_vertices, train_frame, test_frame)
        track_train = evaluate_track(pred_vertices, 1, train_frame)
        track_test = evaluate_track(pred_vertices, train_frame, test_frame)

        rows.append(
            {
                "Method": method,
                "Train Frame Num": cd_train["frame_num"],
                "CD Train": cd_train["cd"],
                "Test Frame Num": cd_test["frame_num"],
                "CD Test": cd_test["cd"],
                "Track Error Train": track_train,
                "Track Error Test": track_test,
                "Inference Path": path,
            }
        )

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(out)
    print("Saved:", output_csv)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--method-table-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    evaluate_geometry(
        data_root=Path(args.data_root),
        method_table_csv=Path(args.method_table_csv),
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
