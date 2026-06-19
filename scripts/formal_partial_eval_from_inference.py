#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
formal_partial_eval_from_inference.py

Partial evaluation for NeuSpring / PhysTwin candidate results.

Use this when topology search stopped after some candidates, but these methods already have inference.pkl:

  cand_000_raw_param_opt
  cand_000_nsf_param_opt
  cand_001_raw_param_opt
  cand_001_nsf_param_opt
  cand_002_raw_param_opt
  cand_002_nsf_param_opt

Inputs:
  - Prediction trajectory: <search-root>/candidates/<cand>/<method>/inference.pkl
  - GT trajectory:         <base-path>/<case-name>/final_data.pkl

Outputs:
  - FULL_metrics_train_test.csv
  - FULL_metrics_per_frame.csv
  - COMPARE_raw_vs_nsf_train_test.csv
  - CANDIDATE_test_ranking_all_metrics.csv
  - GT | Prediction | Abs Error videos
  - projected render frames
  - zip package

Important:
  This computes projected point-cloud smoke render metrics.
  It is NOT official Gaussian Splatting RGB evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


LOWER_IS_BETTER = {"CD", "Track Error", "LPIPS", "AbsErrorMean3D", "AbsErrorP95_3D"}
HIGHER_IS_BETTER = {"PSNR", "SSIM", "IoU"}


def to_numpy(x: Any) -> Any:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return x


def flatten_arrays(obj: Any, prefix: str = "") -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    obj = to_numpy(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten_arrays(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0 and all(isinstance(to_numpy(x), np.ndarray) for x in obj):
            try:
                out.append((prefix or "<list_stack>", np.stack([to_numpy(x) for x in obj], axis=0)))
            except Exception:
                pass
        for i, v in enumerate(obj[:10]):
            out.extend(flatten_arrays(v, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    elif isinstance(obj, np.ndarray):
        out.append((prefix or "<root>", obj))
    return out


def standardize_traj(arr: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    # T,N,3
    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] >= 2 and arr.shape[1] >= 10:
        return arr.astype(np.float32)

    # C,T,N,3, use C=0
    if (
        arr.ndim == 4
        and arr.shape[-1] == 3
        and arr.shape[0] <= 8
        and arr.shape[1] >= 2
        and arr.shape[2] >= 10
    ):
        return arr[0].astype(np.float32)

    return None


def get_prediction_from_inference(pkl_path: Path) -> Tuple[str, np.ndarray]:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    candidates = []
    for key, arr in flatten_arrays(obj):
        traj = standardize_traj(arr)
        if traj is not None:
            candidates.append((key, traj))

    if not candidates:
        raise RuntimeError(f"No trajectory-like array found in {pkl_path}")

    def score(item):
        key, traj = item
        s = key.lower()
        val = 0
        for w in ["pred", "predict", "sim", "simulate", "infer", "trajectory", "output"]:
            if w in s:
                val += 3
        for w in ["gt", "target", "obs", "real"]:
            if w in s:
                val -= 5
        val += min(traj.shape[0], 500) / 1000
        return val

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def score_gt_key(key: str) -> float:
    s = key.lower()
    score = 0.0
    good = ["gt", "target", "obs", "observ", "surface", "pc", "point", "points", "xyz", "track", "trajectory", "state", "data", "final"]
    bad = ["color", "rgb", "image", "mask", "depth", "camera", "intr", "extr", "calib", "metadata"]
    for w in good:
        if w in s:
            score += 3
    for w in bad:
        if w in s:
            score -= 8
    return score


def get_gt_from_final_data(final_data_path: Path, pred_T: int, pred_N: int) -> Tuple[str, np.ndarray, pd.DataFrame]:
    with open(final_data_path, "rb") as f:
        obj = pickle.load(f)

    candidates = []
    for key, arr in flatten_arrays(obj):
        traj = standardize_traj(arr)
        if traj is None:
            continue
        T, N, _ = traj.shape
        if abs(T - pred_T) > max(10, int(pred_T * 0.25)):
            continue
        score = score_gt_key(key)
        score -= abs(T - pred_T) * 2.0
        score -= abs(N - pred_N) / 200.0
        candidates.append({"score": score, "key": key, "shape": str(traj.shape), "T": T, "N": N, "traj": traj})

    if not candidates:
        raise RuntimeError(f"No GT trajectory found in {final_data_path}")

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    table = pd.DataFrame([{k: v for k, v in c.items() if k != "traj"} for c in candidates])
    best = candidates[0]
    return best["key"], best["traj"], table


def load_split(case_dir: Path, n_frames: int) -> Tuple[set, set]:
    split_path = case_dir / "split.json"
    if not split_path.exists():
        train_end = 134 if n_frames >= 192 else int(round(n_frames * 0.7))
        return set(range(train_end)), set(range(train_end, n_frames))

    with open(split_path, "r") as f:
        split = json.load(f)

    def expand(v):
        if isinstance(v, list):
            if len(v) == 2 and all(isinstance(x, int) for x in v):
                return list(range(v[0], v[1]))
            return [int(x) for x in v]
        if isinstance(v, int):
            return list(range(v))
        return []

    train_ids = set(expand(split.get("train", [])))
    test_ids = set(expand(split.get("test", [])))

    if len(train_ids) == 0 and "train_frame" in split:
        train_ids = set(range(int(split["train_frame"])))
    if len(train_ids) == 0:
        train_end = 134 if n_frames >= 192 else int(round(n_frames * 0.7))
        train_ids = set(range(train_end))
    if len(test_ids) == 0:
        test_ids = set(range(n_frames)) - train_ids
    return train_ids, test_ids


def symmetric_chamfer(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    tree_gt = cKDTree(gt_pts)
    d_pg, _ = tree_gt.query(pred_pts, k=1)
    tree_pred = cKDTree(pred_pts)
    d_gp, _ = tree_pred.query(gt_pts, k=1)
    return float(0.5 * (np.mean(d_pg) + np.mean(d_gp)))


def track_error(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    n = min(len(pred_pts), len(gt_pts))
    return float(np.mean(np.linalg.norm(pred_pts[:n] - gt_pts[:n], axis=-1)))


def nearest_errors(pred_pts: np.ndarray, gt_pts: np.ndarray) -> np.ndarray:
    tree = cKDTree(gt_pts)
    dist, _ = tree.query(pred_pts, k=1)
    return dist


def make_projection_params(all_points: np.ndarray, img_size: int) -> Dict[str, Any]:
    center = all_points.mean(axis=0)
    azim = np.deg2rad(45)
    elev = np.deg2rad(20)
    Rz = np.array([[np.cos(azim), -np.sin(azim), 0],
                   [np.sin(azim),  np.cos(azim), 0],
                   [0, 0, 1]], dtype=np.float32)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(elev), -np.sin(elev)],
                   [0, np.sin(elev),  np.cos(elev)]], dtype=np.float32)
    R = Rx @ Rz
    pts = (all_points - center) @ R.T
    xy = pts[:, :2]
    extent = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    scale = img_size * 0.82 / max(extent, 1e-8)
    return {"center": center, "R": R, "scale": scale, "img_size": img_size}


def project_points(points: np.ndarray, params: Dict[str, Any]):
    pts = (points - params["center"]) @ params["R"].T
    xy = pts[:, :2] * params["scale"]
    img_size = params["img_size"]
    px = np.round(xy[:, 0] + img_size / 2).astype(np.int32)
    py = np.round(-xy[:, 1] + img_size / 2).astype(np.int32)
    valid = (px >= 0) & (px < img_size) & (py >= 0) & (py < img_size)
    return px[valid], py[valid], valid


def draw_points(points: np.ndarray, params: Dict[str, Any], radius: int = 1):
    img_size = params["img_size"]
    img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    px, py, _ = project_points(points, params)
    for x, y in zip(px, py):
        x0, x1 = max(0, x - radius), min(img_size, x + radius + 1)
        y0, y1 = max(0, y - radius), min(img_size, y + radius + 1)
        img[y0:y1, x0:x1] = (0, 0, 0)
        mask[y0:y1, x0:x1] = 1
    return img, mask


def draw_error(points: np.ndarray, errors: np.ndarray, params: Dict[str, Any], radius: int = 1):
    import matplotlib.cm as cm
    img_size = params["img_size"]
    img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    px, py, valid = project_points(points, params)
    err_valid = errors[valid]
    scale = max(float(np.percentile(err_valid, 95)), 1e-8) if len(err_valid) else 1.0
    colors = (cm.inferno(np.clip(err_valid / scale, 0, 1))[:, :3] * 255).astype(np.uint8)
    for x, y, c in zip(px, py, colors):
        x0, x1 = max(0, x - radius), min(img_size, x + radius + 1)
        y0, y1 = max(0, y - radius), min(img_size, y + radius + 1)
        img[y0:y1, x0:x1] = c
        mask[y0:y1, x0:x1] = 1
    return img, mask


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else float("nan")


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, pil.width, 34], fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def make_panel(gt_img: np.ndarray, pred_img: np.ndarray, err_img: np.ndarray, method: str, frame_id: int) -> np.ndarray:
    panel = np.concatenate([add_label(gt_img, "GT"), add_label(pred_img, "Prediction"), add_label(err_img, "Abs Error")], axis=1)
    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, pil.height - 30, pil.width, pil.height], fill=(0, 0, 0))
    draw.text((8, pil.height - 24), f"{method} | frame {frame_id}", fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def init_lpips(use_lpips: bool):
    if not use_lpips:
        return None, None
    try:
        import torch
        import lpips
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fn = lpips.LPIPS(net="alex").to(device)
        fn.eval()
        return fn, device
    except Exception as e:
        print(f"[WARN] LPIPS disabled: {repr(e)}")
        return None, None


def compute_lpips(fn, device, img_a: np.ndarray, img_b: np.ndarray, lpips_size: int) -> float:
    if fn is None:
        return float("nan")
    import torch
    a = Image.fromarray(img_a).resize((lpips_size, lpips_size), Image.BILINEAR)
    b = Image.fromarray(img_b).resize((lpips_size, lpips_size), Image.BILINEAR)
    ta = torch.from_numpy(np.asarray(a)).float() / 127.5 - 1.0
    tb = torch.from_numpy(np.asarray(b)).float() / 127.5 - 1.0
    ta = ta.permute(2, 0, 1).unsqueeze(0).to(device)
    tb = tb.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(fn(ta, tb).item())


def candidate_id(method: str) -> str:
    m = re.search(r"(cand_\d+)", method)
    return m.group(1) if m else "unknown"


def variant(method: str) -> str:
    if "raw_param_opt" in method:
        return "raw"
    if "nsf_param_opt" in method:
        return "nsf"
    return "unknown"


def build_raw_vs_nsf(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cand in sorted(summary["candidate"].dropna().unique()):
        raw_m = f"{cand}_raw_param_opt"
        nsf_m = f"{cand}_nsf_param_opt"
        for split in ["train", "test"]:
            raw = summary[(summary["Method"] == raw_m) & (summary["split"] == split)]
            nsf = summary[(summary["Method"] == nsf_m) & (summary["split"] == split)]
            if raw.empty or nsf.empty:
                continue
            raw, nsf = raw.iloc[0], nsf.iloc[0]
            row = {"candidate": cand, "split": split, "raw_method": raw_m, "nsf_method": nsf_m}
            for metric in ["CD", "Track Error", "PSNR", "SSIM", "LPIPS", "IoU"]:
                row[f"{metric}_raw"] = raw.get(metric, np.nan)
                row[f"{metric}_nsf"] = nsf.get(metric, np.nan)
                row[f"{metric}_delta_nsf_minus_raw"] = row[f"{metric}_nsf"] - row[f"{metric}_raw"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    rank = summary[summary["split"] == "test"].copy()
    if rank.empty:
        return rank
    rank_specs = [("CD", True), ("Track Error", True), ("LPIPS", True), ("PSNR", False), ("SSIM", False), ("IoU", False)]
    for col, asc in rank_specs:
        if col in rank.columns and rank[col].notna().any():
            rank[f"{col}_rank"] = rank[col].rank(ascending=asc, method="average")
    rank_cols = [c for c in rank.columns if c.endswith("_rank")]
    if rank_cols:
        rank["overall_rank_score"] = rank[rank_cols].mean(axis=1)
        rank = rank.sort_values("overall_rank_score").reset_index(drop=True)
    return rank


def zip_dir(out_dir: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(out_dir))
    print(f"[ZIP] {zip_path}")
    print(f"[ZIP] size MB: {zip_path.stat().st_size / 1024 / 1024:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", default="double_stretch_sloth")
    parser.add_argument("--base-path", default="/content/PhysTwin/data/different_types")
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--video-fps", type=int, default=24)
    parser.add_argument("--max-points-for-render", type=int, default=6000)
    parser.add_argument("--lpips-size", type=int, default=256)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--methods", nargs="+", default=[
        "cand_000_raw_param_opt",
        "cand_000_nsf_param_opt",
        "cand_001_raw_param_opt",
        "cand_001_nsf_param_opt",
        "cand_002_raw_param_opt",
        "cand_002_nsf_param_opt",
    ])
    args = parser.parse_args()

    search_root = Path(args.search_root)
    cand_root = search_root / "candidates"
    case_dir = Path(args.base_path) / args.case_name
    final_data = case_dir / "final_data.pkl"
    out_dir = Path(args.out_dir)
    render_root = out_dir / "render_eval_projected"
    video_dir = out_dir / "videos_GT_Prediction_AbsError"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    if not final_data.exists():
        raise FileNotFoundError(f"Missing final_data.pkl: {final_data}")

    lpips_fn, lpips_device = init_lpips(not args.no_lpips)

    loaded = {}
    selection_rows = []
    gt_candidate_tables = []

    for method in args.methods:
        cand = candidate_id(method)
        pkl_path = cand_root / cand / method / "inference.pkl"

        print("\n" + "=" * 100)
        print("Method:", method)
        print("inference:", pkl_path)

        if not pkl_path.exists():
            print(f"[WARN] Missing inference.pkl for {method}; skipping.")
            continue

        pred_key, pred_traj = get_prediction_from_inference(pkl_path)
        pred_T, pred_N, _ = pred_traj.shape
        gt_key, gt_traj, gt_candidates = get_gt_from_final_data(final_data, pred_T, pred_N)
        gt_candidates.insert(0, "Method", method)
        gt_candidate_tables.append(gt_candidates)

        T = min(pred_traj.shape[0], gt_traj.shape[0])
        if args.max_frames > 0:
            T = min(T, args.max_frames)

        pred_traj = pred_traj[:T]
        gt_traj = gt_traj[:T]

        n_eval = min(pred_traj.shape[1], gt_traj.shape[1])
        pred_eval = pred_traj[:, :n_eval]
        gt_eval = gt_traj[:, :n_eval]

        if n_eval > args.max_points_for_render:
            rng = np.random.default_rng(0)
            point_idx = rng.choice(n_eval, size=args.max_points_for_render, replace=False)
        else:
            point_idx = np.arange(n_eval)

        loaded[method] = {
            "pred": pred_eval,
            "gt": gt_eval,
            "point_idx": point_idx,
            "pred_key": pred_key,
            "gt_key": gt_key,
        }

        selection_rows.append({
            "Method": method,
            "candidate": cand,
            "variant": variant(method),
            "pred_key": pred_key,
            "pred_shape": str(pred_traj.shape),
            "gt_key": gt_key,
            "gt_shape": str(gt_traj.shape),
            "eval_shape": str(pred_eval.shape),
            "render_points": len(point_idx),
        })

    if not loaded:
        raise RuntimeError("No valid methods were found. Check inference.pkl paths.")

    selection = pd.DataFrame(selection_rows)
    selection.to_csv(out_dir / "selected_pred_gt_keys.csv", index=False)
    if gt_candidate_tables:
        pd.concat(gt_candidate_tables, ignore_index=True).to_csv(out_dir / "gt_candidate_key_scores.csv", index=False)

    # Global projection for comparable render.
    samples = []
    for pack in loaded.values():
        pred, gt, idx = pack["pred"], pack["gt"], pack["point_idx"]
        T = pred.shape[0]
        frames = np.linspace(0, T - 1, min(T, 16)).astype(int)
        samples.append(pred[frames][:, idx].reshape(-1, 3))
        samples.append(gt[frames][:, idx].reshape(-1, 3))
    proj = make_projection_params(np.concatenate(samples, axis=0), args.img_size)

    frame_rows = []
    summary_rows = []
    video_rows = []

    for method, pack in loaded.items():
        print("\nRendering and evaluating:", method)
        pred, gt, idx = pack["pred"], pack["gt"], pack["point_idx"]
        pred_vis = pred[:, idx]
        gt_vis = gt[:, idx]
        T = pred_vis.shape[0]
        train_ids, test_ids = load_split(case_dir, T)

        method_root = render_root / method
        gt_dir = method_root / "gt"
        pred_dir = method_root / "renders"
        err_dir = method_root / "abs_error"
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        err_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_dir / f"{method}_GT_Prediction_AbsError.mp4"
        writer = imageio.get_writer(str(video_path), fps=args.video_fps, codec="libx264", quality=8)

        method_rows = []
        for t in range(T):
            gt_pts = gt_vis[t]
            pred_pts = pred_vis[t]

            cd = symmetric_chamfer(pred_pts, gt_pts)
            te = track_error(pred_pts, gt_pts)

            gt_img, gt_mask = draw_points(gt_pts, proj, radius=args.point_radius)
            pred_img, pred_mask = draw_points(pred_pts, proj, radius=args.point_radius)
            err_values = nearest_errors(pred_pts, gt_pts)
            err_img, _ = draw_error(pred_pts, err_values, proj, radius=args.point_radius)

            psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=255)
            ssim = structural_similarity(gt_img, pred_img, channel_axis=2, data_range=255)
            lp = compute_lpips(lpips_fn, lpips_device, gt_img, pred_img, args.lpips_size)
            iou = compute_iou(gt_mask, pred_mask)

            split = "train" if t in train_ids else "test"
            row = {
                "Method": method,
                "candidate": candidate_id(method),
                "variant": variant(method),
                "frame": t,
                "split": split,
                "CD": float(cd),
                "Track Error": float(te),
                "PSNR": float(psnr),
                "SSIM": float(ssim),
                "LPIPS": float(lp),
                "IoU": float(iou),
                "AbsErrorMean3D": float(np.mean(err_values)),
                "AbsErrorP95_3D": float(np.percentile(err_values, 95)),
            }
            frame_rows.append(row)
            method_rows.append(row)

            Image.fromarray(gt_img).save(gt_dir / f"{t:04d}.png")
            Image.fromarray(pred_img).save(pred_dir / f"{t:04d}.png")
            Image.fromarray(err_img).save(err_dir / f"{t:04d}.png")
            writer.append_data(make_panel(gt_img, pred_img, err_img, method, t))

            if t % 25 == 0:
                print(f"[{method}] frame {t}/{T}")

        writer.close()
        video_rows.append({"Method": method, "video": str(video_path)})

        mdf = pd.DataFrame(method_rows)
        mdf.to_csv(method_root / f"{method}_per_frame_metrics.csv", index=False)

        for split in ["train", "test"]:
            sub = mdf[mdf["split"] == split]
            if sub.empty:
                continue
            summary_rows.append({
                "Method": method,
                "candidate": candidate_id(method),
                "variant": variant(method),
                "split": split,
                "CD": float(sub["CD"].mean()),
                "Track Error": float(sub["Track Error"].mean()),
                "PSNR": float(sub["PSNR"].mean()),
                "SSIM": float(sub["SSIM"].mean()),
                "LPIPS": float(sub["LPIPS"].mean()),
                "IoU": float(sub["IoU"].mean()),
                "AbsErrorMean3D": float(sub["AbsErrorMean3D"].mean()),
                "AbsErrorP95_3D": float(sub["AbsErrorP95_3D"].mean()),
                "num_frames": int(len(sub)),
            })

    frame_metrics = pd.DataFrame(frame_rows)
    summary = pd.DataFrame(summary_rows)
    video_manifest = pd.DataFrame(video_rows)
    raw_vs_nsf = build_raw_vs_nsf(summary)
    ranking = build_ranking(summary)

    frame_metrics.to_csv(out_dir / "FULL_metrics_per_frame.csv", index=False)
    summary.to_csv(out_dir / "FULL_metrics_train_test.csv", index=False)
    raw_vs_nsf.to_csv(out_dir / "COMPARE_raw_vs_nsf_train_test.csv", index=False)
    ranking.to_csv(out_dir / "CANDIDATE_test_ranking_all_metrics.csv", index=False)
    video_manifest.to_csv(out_dir / "video_manifest.csv", index=False)

    readme = """# Partial Candidate Evaluation from Inference

This package was generated from `inference.pkl` trajectories and `final_data.pkl`.

Important:
These are projected point-cloud smoke render metrics, not official Gaussian Splatting RGB metrics.

Use this to compare partially completed topology candidates and select candidates for later official Gaussian RGB rendering.

Lower is better:
- CD
- Track Error
- LPIPS
- AbsErrorMean3D
- AbsErrorP95_3D

Higher is better:
- PSNR
- SSIM
- IoU
"""
    with open(out_dir / "README_RESULTS.md", "w", encoding="utf-8") as f:
        f.write(readme)
        f.write("\n\n## Full metrics\n\n")
        f.write(summary.to_markdown(index=False) if not summary.empty else "No full metrics.")
        f.write("\n\n## Raw vs NSF\n\n")
        f.write(raw_vs_nsf.to_markdown(index=False) if not raw_vs_nsf.empty else "No raw-vs-NSF comparison.")
        f.write("\n\n## Test ranking\n\n")
        f.write(ranking.to_markdown(index=False) if not ranking.empty else "No ranking.")

    print("\n=== FULL METRICS TRAIN/TEST ===")
    print(summary)
    print("\n=== RAW VS NSF ===")
    print(raw_vs_nsf)
    print("\n=== TEST RANKING ===")
    print(ranking)

    zip_dir(out_dir, Path(args.zip_path))


if __name__ == "__main__":
    main()
