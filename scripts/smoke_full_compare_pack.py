#!/usr/bin/env python3
"""
Smoke full comparison packer for NeuSpring-style PhysTwin experiments.

It compares all 4 methods by default:
  cand_000_raw_param_opt
  cand_000_nsf_param_opt
  cand_001_raw_param_opt
  cand_001_nsf_param_opt

Outputs:
  - geometry CD / Track Error table for train/test
  - render PSNR / SSIM / LPIPS / IoU table for train/test
  - raw-vs-NSF comparison table
  - cand_000-vs-cand_001 comparison table
  - GT | Prediction | Abs Error mp4 for each method
  - optional all-method grid video
  - zip package for download

Expected render folder layout:
  render_root/<method>/gt/*.png
  render_root/<method>/renders/*.png

The script also tries to auto-discover render folders if --auto-find-renders is used.
"""

import argparse
import json
import math
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
LOWER_IS_BETTER = {"CD", "Track Error", "LPIPS", "AbsErrorMean", "AbsErrorP95"}
HIGHER_IS_BETTER = {"PSNR", "SSIM", "IoU"}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_images(folder: Path) -> List[Path]:
    if not folder or not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=natural_key)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize_to(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def parse_frame_id(path: Path, fallback_idx: int) -> int:
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return fallback_idx
    return int(nums[-1])


def load_split(split_json: Path, n_frames: int) -> Tuple[set, set]:
    if not split_json.exists():
        train_end = int(round(n_frames * 0.7))
        return set(range(train_end)), set(range(train_end, n_frames))

    with open(split_json, "r") as f:
        split = json.load(f)

    def expand(v):
        if isinstance(v, list):
            if len(v) == 2 and all(isinstance(x, int) for x in v):
                # Treat as [start, end) because PhysTwin split commonly stores boundaries.
                return list(range(v[0], v[1]))
            return [int(x) for x in v]
        if isinstance(v, int):
            return list(range(v))
        return []

    train_ids = set(expand(split.get("train", [])))
    test_ids = set(expand(split.get("test", [])))

    if len(train_ids) == 0 and len(test_ids) == 0:
        train_end = int(round(n_frames * 0.7))
        train_ids = set(range(train_end))
        test_ids = set(range(train_end, n_frames))
    elif len(test_ids) == 0:
        all_ids = set(range(n_frames))
        test_ids = all_ids - train_ids

    return train_ids, test_ids


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


def compute_lpips(fn, device, gt: np.ndarray, pred: np.ndarray) -> float:
    if fn is None:
        return float("nan")
    import torch
    gt_t = torch.from_numpy(gt).float() / 127.5 - 1.0
    pred_t = torch.from_numpy(pred).float() / 127.5 - 1.0
    gt_t = gt_t.permute(2, 0, 1).unsqueeze(0).to(device)
    pred_t = pred_t.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(fn(gt_t, pred_t).item())


def compute_iou_quick(gt: np.ndarray, pred: np.ndarray, bg_thresh: int = 245) -> float:
    # White-background silhouette approximation. If masks are available, replace this with mask IoU.
    gt_mask = np.any(gt < bg_thresh, axis=-1)
    pred_mask = np.any(pred < bg_thresh, axis=-1)
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return float(inter / union) if union > 0 else float("nan")


def make_error_rgb(gt: np.ndarray, pred: np.ndarray, scale: Optional[float] = None):
    import matplotlib.cm as cm
    err = np.mean(np.abs(gt.astype(np.float32) - pred.astype(np.float32)), axis=-1) / 255.0
    if scale is None or scale <= 0:
        scale = max(float(np.percentile(err, 95)), 1e-6)
    norm = np.clip(err / scale, 0.0, 1.0)
    color = cm.inferno(norm)[..., :3]
    return (color * 255).astype(np.uint8), err


def add_label(img: np.ndarray, text: str, height: int = 42) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, pil.width, height], fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def concat_panel(gt: np.ndarray, pred: np.ndarray, err_rgb: np.ndarray, frame_name: str, method: str) -> np.ndarray:
    gt_l = add_label(gt, "GT")
    pred_l = add_label(pred, "Prediction")
    err_l = add_label(err_rgb, "Abs Error")
    panel = np.concatenate([gt_l, pred_l, err_l], axis=1)
    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    txt = f"{method} | Frame: {frame_name}"
    draw.rectangle([0, pil.height - 34, pil.width, pil.height], fill=(0, 0, 0))
    draw.text((10, pil.height - 29), txt, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def find_child_dir(base: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = base / name
        if p.exists() and len(list_images(p)) > 0:
            return p
    # recursive fallback: exact folder name match
    for p in base.rglob("*"):
        if p.is_dir() and p.name.lower() in {n.lower() for n in names} and len(list_images(p)) > 0:
            return p
    return None


def discover_render_dirs(render_root: Path, method: str, extra_roots: List[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    # Standard layout first.
    candidates = []
    if render_root:
        candidates.append(render_root / method)
    for r in extra_roots:
        candidates.extend([
            r / method,
            r / "double_stretch_sloth" / method,
            r / method.replace("_param_opt", ""),
        ])

    gt_names = ["gt", "GT", "ground_truth", "groundtruth", "images", "rgb", "color"]
    pred_names = ["renders", "render", "prediction", "pred", "Prediction", "ours", "rendered"]

    for base in candidates:
        if not base.exists():
            continue
        gt = find_child_dir(base, gt_names)
        pred = find_child_dir(base, pred_names)
        if gt is not None and pred is not None and gt != pred:
            return gt, pred

    return None, None


def scan_image_dirs(roots: List[Path]) -> pd.DataFrame:
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for d in root.rglob("*"):
            if d.is_dir():
                n = len(list_images(d))
                if n >= 5:
                    rows.append({"num_images": n, "dir": str(d)})
    return pd.DataFrame(rows).sort_values(["num_images", "dir"], ascending=[False, True]) if rows else pd.DataFrame(columns=["num_images", "dir"])


def read_geometry_tables(search_root: Path) -> pd.DataFrame:
    tables = []
    # Priority 1: all_method_runs.csv
    for p in [search_root / "all_method_runs.csv", search_root / "candidate_summary.csv"]:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    df["source_csv"] = str(p)
                    tables.append(df)
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")

    # Priority 2: per-candidate geometry_eval.csv
    for p in search_root.rglob("geometry_eval.csv"):
        try:
            df = pd.read_csv(p)
            if not df.empty:
                df["source_csv"] = str(p)
                tables.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")

    if not tables:
        return pd.DataFrame()
    df_all = pd.concat(tables, ignore_index=True, sort=False)
    # Normalize method column.
    if "Method" not in df_all.columns:
        for c in ["method", "score_method", "Method Name"]:
            if c in df_all.columns:
                df_all = df_all.rename(columns={c: "Method"})
                break
    return df_all


def pick_first_numeric(row, possible_names: List[str]) -> float:
    for name in possible_names:
        if name in row.index:
            val = pd.to_numeric(row[name], errors="coerce")
            if pd.notna(val):
                return float(val)
    return float("nan")


def summarize_geometry(df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    rows = []
    if df is None or df.empty or "Method" not in df.columns:
        print("[WARN] No geometry table with Method column found. CD/Track will be NaN.")
        for method in methods:
            for split in ["train", "test"]:
                rows.append({"Method": method, "split": split, "CD": np.nan, "Track Error": np.nan})
        return pd.DataFrame(rows)

    for method in methods:
        subset = df[df["Method"].astype(str) == method].copy()
        if subset.empty:
            # Some rows might store selected method under score_method while Method missing.
            print(f"[WARN] Geometry row not found for {method}")
            for split in ["train", "test"]:
                rows.append({"Method": method, "split": split, "CD": np.nan, "Track Error": np.nan})
            continue
        # Prefer row with most non-NaN metric columns.
        metric_like = [c for c in subset.columns if any(k.lower() in c.lower() for k in ["cd", "chamfer", "track", "error"])]
        subset["_score_non_nan"] = subset[metric_like].notna().sum(axis=1) if metric_like else 0
        row = subset.sort_values("_score_non_nan", ascending=False).iloc[0]

        rows.append({
            "Method": method,
            "split": "train",
            "CD": pick_first_numeric(row, ["CD Train", "train_CD", "cd_train", "Train CD", "CD_train"]),
            "Track Error": pick_first_numeric(row, ["Track Error Train", "train_track_error", "track_error_train", "Train Track Error", "TE Train"]),
        })
        rows.append({
            "Method": method,
            "split": "test",
            "CD": pick_first_numeric(row, ["CD Test", "test_CD", "cd_test", "Test CD", "CD_test"]),
            "Track Error": pick_first_numeric(row, ["Track Error Test", "test_track_error", "track_error_test", "Test Track Error", "TE Test"]),
        })
    return pd.DataFrame(rows)


def compute_render_for_method(method: str, gt_dir: Path, pred_dir: Path, split_json: Path, out_dir: Path, fps: int, lpips_fn, lpips_device, bg_thresh: int, max_frames: int) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    gt_imgs = list_images(gt_dir)
    pred_imgs = list_images(pred_dir)
    n = min(len(gt_imgs), len(pred_imgs))
    if max_frames > 0:
        n = min(n, max_frames)
    if n == 0:
        raise RuntimeError(f"No paired frames for {method}: gt={gt_dir}, pred={pred_dir}")

    gt_imgs = gt_imgs[:n]
    pred_imgs = pred_imgs[:n]
    train_ids, test_ids = load_split(split_json, n)

    method_out = out_dir / method
    method_out.mkdir(parents=True, exist_ok=True)
    video_path = method_out / f"{method}_GT_Prediction_AbsError.mp4"
    writer = imageio.get_writer(str(video_path), fps=fps, codec="libx264", quality=8)

    rows = []
    for idx, (g, p) in enumerate(zip(gt_imgs, pred_imgs)):
        gt = read_rgb(g)
        pred = read_rgb(p)
        if pred.shape[:2] != gt.shape[:2]:
            pred = resize_to(pred, gt.shape[:2])
        frame_id = parse_frame_id(g, idx)
        split = "train" if (idx in train_ids or frame_id in train_ids) else "test"

        psnr = peak_signal_noise_ratio(gt, pred, data_range=255)
        ssim = structural_similarity(gt, pred, channel_axis=2, data_range=255)
        lp = compute_lpips(lpips_fn, lpips_device, gt, pred)
        iou = compute_iou_quick(gt, pred, bg_thresh=bg_thresh)
        err_rgb, err_gray = make_error_rgb(gt, pred)
        writer.append_data(concat_panel(gt, pred, err_rgb, g.name, method))

        rows.append({
            "Method": method,
            "frame_index": idx,
            "frame_id": frame_id,
            "split": split,
            "gt_file": g.name,
            "pred_file": p.name,
            "PSNR": float(psnr),
            "SSIM": float(ssim),
            "LPIPS": float(lp),
            "IoU": float(iou),
            "AbsErrorMean": float(np.mean(err_gray)),
            "AbsErrorP95": float(np.percentile(err_gray, 95)),
        })
        if idx % 25 == 0:
            print(f"[{method}] frame {idx}/{n}")
    writer.close()

    per_frame = pd.DataFrame(rows)
    per_frame.to_csv(method_out / f"{method}_render_metrics_per_frame.csv", index=False)
    summary = per_frame.groupby("split")[["PSNR", "SSIM", "LPIPS", "IoU", "AbsErrorMean", "AbsErrorP95"]].mean().reset_index()
    summary.insert(0, "Method", method)
    summary.to_csv(method_out / f"{method}_render_metrics_train_test.csv", index=False)
    return per_frame, summary, video_path


def build_raw_vs_nsf(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cand in sorted({m.split("_")[1] for m in final_df["Method"].dropna().astype(str) if m.startswith("cand_")}):
        cand_id = f"cand_{cand}"
        raw_m = f"{cand_id}_raw_param_opt"
        nsf_m = f"{cand_id}_nsf_param_opt"
        for split in ["train", "test"]:
            raw = final_df[(final_df["Method"] == raw_m) & (final_df["split"] == split)]
            nsf = final_df[(final_df["Method"] == nsf_m) & (final_df["split"] == split)]
            if raw.empty or nsf.empty:
                continue
            raw = raw.iloc[0]
            nsf = nsf.iloc[0]
            row = {"candidate_id": cand_id, "split": split, "raw_method": raw_m, "nsf_method": nsf_m}
            for metric in ["CD", "Track Error", "PSNR", "SSIM", "LPIPS", "IoU"]:
                row[f"{metric}_raw"] = raw.get(metric, np.nan)
                row[f"{metric}_nsf"] = nsf.get(metric, np.nan)
                row[f"{metric}_delta_nsf_minus_raw"] = nsf.get(metric, np.nan) - raw.get(metric, np.nan)
                if metric in LOWER_IS_BETTER:
                    row[f"{metric}_NSF_better"] = row[f"{metric}_delta_nsf_minus_raw"] < 0
                elif metric in HIGHER_IS_BETTER:
                    row[f"{metric}_NSF_better"] = row[f"{metric}_delta_nsf_minus_raw"] > 0
            rows.append(row)
    return pd.DataFrame(rows)


def build_cand_compare(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in ["raw", "nsf"]:
        m0 = f"cand_000_{variant}_param_opt"
        m1 = f"cand_001_{variant}_param_opt"
        for split in ["train", "test"]:
            a = final_df[(final_df["Method"] == m0) & (final_df["split"] == split)]
            b = final_df[(final_df["Method"] == m1) & (final_df["split"] == split)]
            if a.empty or b.empty:
                continue
            a = a.iloc[0]
            b = b.iloc[0]
            row = {"variant": variant, "split": split, "cand_000_method": m0, "cand_001_method": m1}
            for metric in ["CD", "Track Error", "PSNR", "SSIM", "LPIPS", "IoU"]:
                v0 = a.get(metric, np.nan)
                v1 = b.get(metric, np.nan)
                row[f"{metric}_cand000"] = v0
                row[f"{metric}_cand001"] = v1
                row[f"{metric}_delta_cand001_minus_cand000"] = v1 - v0
                if metric in LOWER_IS_BETTER:
                    row[f"{metric}_winner"] = "cand_001" if v1 < v0 else "cand_000"
                elif metric in HIGHER_IS_BETTER:
                    row[f"{metric}_winner"] = "cand_001" if v1 > v0 else "cand_000"
            rows.append(row)
    return pd.DataFrame(rows)


def make_grid_video(method_videos: Dict[str, List[np.ndarray]], out_path: Path, fps: int):
    if not method_videos:
        return
    methods = list(method_videos.keys())
    n = min(len(v) for v in method_videos.values())
    if n == 0:
        return
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for i in range(n):
        rows = []
        for m in methods:
            frame = method_videos[m][i]
            rows.append(frame)
        # Resize rows to same width before stacking.
        max_w = max(r.shape[1] for r in rows)
        resized = []
        for r in rows:
            if r.shape[1] != max_w:
                h = int(r.shape[0] * max_w / r.shape[1])
                r = np.asarray(Image.fromarray(r).resize((max_w, h), Image.BILINEAR))
            resized.append(r)
        writer.append_data(np.concatenate(resized, axis=0))
    writer.close()


def zip_dir(out_dir: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(out_dir))
    print(f"[ZIP] {zip_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--render-root", required=False, type=Path, default=Path(""))
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", default=[
        "cand_000_raw_param_opt", "cand_000_nsf_param_opt",
        "cand_001_raw_param_opt", "cand_001_nsf_param_opt",
    ])
    parser.add_argument("--extra-render-roots", nargs="*", default=[], type=Path)
    parser.add_argument("--auto-find-renders", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--bg-thresh", type=int, default=245)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_json = args.case_dir / "split.json"

    # Save discovered image directories for debugging.
    scan_roots = [r for r in [args.render_root, args.search_root, Path("/content/PhysTwin/gaussian_output_dynamic"), Path("/content/PhysTwin/gaussian_output")] if str(r)] + args.extra_render_roots
    image_dir_scan = scan_image_dirs(scan_roots)
    image_dir_scan.to_csv(args.out_dir / "image_dir_scan.csv", index=False)
    print("[SCAN] image dirs saved to", args.out_dir / "image_dir_scan.csv")
    if args.scan_only:
        print(image_dir_scan.head(100))
        zip_dir(args.out_dir, args.zip_path)
        return

    geometry_all = read_geometry_tables(args.search_root)
    geometry_all.to_csv(args.out_dir / "geometry_all_raw_tables.csv", index=False)
    geometry_summary = summarize_geometry(geometry_all, args.methods)
    geometry_summary.to_csv(args.out_dir / "geometry_CD_Track_train_test.csv", index=False)

    lpips_fn, lpips_device = init_lpips(not args.no_lpips)
    render_summaries = []
    manifest = []

    for method in args.methods:
        gt_dir, pred_dir = discover_render_dirs(args.render_root, method, args.extra_render_roots)
        if gt_dir is None or pred_dir is None:
            print(f"[WARN] Missing render dirs for {method}. Add them under render_root/{method}/gt and render_root/{method}/renders")
            manifest.append({"Method": method, "gt_dir": None, "pred_dir": None, "status": "missing_render_dirs"})
            continue
        per_frame, summary, video_path = compute_render_for_method(
            method, gt_dir, pred_dir, split_json, args.out_dir, args.fps, lpips_fn, lpips_device, args.bg_thresh, args.max_frames
        )
        render_summaries.append(summary)
        manifest.append({"Method": method, "gt_dir": str(gt_dir), "pred_dir": str(pred_dir), "video": str(video_path), "status": "ok"})

    render_summary = pd.concat(render_summaries, ignore_index=True) if render_summaries else pd.DataFrame(columns=["Method", "split", "PSNR", "SSIM", "LPIPS", "IoU"])
    render_summary.to_csv(args.out_dir / "render_metrics_train_test.csv", index=False)

    final = pd.merge(geometry_summary, render_summary, on=["Method", "split"], how="outer")
    ordered = ["Method", "split", "CD", "Track Error", "PSNR", "SSIM", "LPIPS", "IoU", "AbsErrorMean", "AbsErrorP95"]
    final = final[[c for c in ordered if c in final.columns]]
    final.to_csv(args.out_dir / "FULL_smoke_metrics_train_test.csv", index=False)

    raw_vs_nsf = build_raw_vs_nsf(final)
    raw_vs_nsf.to_csv(args.out_dir / "COMPARE_raw_vs_nsf_train_test.csv", index=False)

    cand_compare = build_cand_compare(final)
    cand_compare.to_csv(args.out_dir / "COMPARE_cand000_vs_cand001_train_test.csv", index=False)

    with open(args.out_dir / "render_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Human-readable markdown summary.
    with open(args.out_dir / "README_RESULTS.md", "w") as f:
        f.write("# Smoke comparison outputs\n\n")
        f.write("## Full metrics\n\n")
        f.write(final.to_markdown(index=False) if not final.empty else "No final metrics.\n")
        f.write("\n\n## Raw vs NSF\n\n")
        f.write(raw_vs_nsf.to_markdown(index=False) if not raw_vs_nsf.empty else "No raw-vs-NSF comparison.\n")
        f.write("\n\n## Candidate comparison\n\n")
        f.write(cand_compare.to_markdown(index=False) if not cand_compare.empty else "No candidate comparison.\n")

    print("\n=== FULL METRICS ===")
    print(final)
    print("\n=== RAW vs NSF ===")
    print(raw_vs_nsf)
    print("\n=== cand_000 vs cand_001 ===")
    print(cand_compare)

    zip_dir(args.out_dir, args.zip_path)


if __name__ == "__main__":
    main()
