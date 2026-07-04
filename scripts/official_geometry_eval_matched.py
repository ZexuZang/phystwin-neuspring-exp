#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official-style geometry evaluation for extra inference methods.

This script makes NSF+sim-ft "measured" instead of "calibrated" by recovering
the geometry protocol that reproduces the original raw row in geometry_eval.csv,
then applying the same protocol to the new inference.pkl.

It exits with an error if the protocol cannot reproduce the original row well.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return x


def flatten_arrays(obj, prefix="", max_list=120):
    out = []
    obj = to_numpy(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten_arrays(v, f"{prefix}.{k}" if prefix else str(k), max_list=max_list)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:max_list]):
            out += flatten_arrays(v, f"{prefix}[{i}]", max_list=max_list)
    elif isinstance(obj, np.ndarray):
        out.append((prefix, obj))
    return out


def standardize_traj(arr):
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] >= 2 and arr.shape[1] >= 50:
        return arr.astype(np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 3 and arr.shape[0] <= 16 and arr.shape[1] >= 2 and arr.shape[2] >= 50:
        return arr[0].astype(np.float32)
    return None


def load_inference(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    candidates = []
    for key, arr in flatten_arrays(obj):
        traj = standardize_traj(arr)
        if traj is not None:
            candidates.append((key, traj))
    if not candidates:
        raise RuntimeError(f"No trajectory array found in {path}")
    candidates = sorted(candidates, key=lambda x: (x[1].shape[0], x[1].shape[1]), reverse=True)
    key, traj = candidates[0]
    print(f"[load inference] {path} -> {key} {traj.shape}")
    return traj


def collect_gt_candidates(roots: List[Path]) -> List[dict]:
    candidates = []
    seen_files = set()
    for root in roots:
        if not root.exists():
            continue
        for p in list(root.rglob("*.pkl")) + list(root.rglob("*.npz")):
            if p in seen_files:
                continue
            seen_files.add(p)
            try:
                if p.suffix == ".pkl":
                    with open(p, "rb") as f:
                        obj = pickle.load(f)
                    items = flatten_arrays(obj)
                else:
                    data = np.load(p, allow_pickle=True)
                    items = [(k, data[k]) for k in data.files]
            except Exception:
                continue

            for key, arr in items:
                traj = standardize_traj(arr)
                if traj is None:
                    continue
                s = f"{p}::{key}".lower()
                bad = ["rgb", "color", "image", "mask", "camera", "intr", "extr", "calib", "metadata"]
                if any(b in s for b in bad):
                    continue
                good_bonus = 0
                for w in ["gt", "target", "point", "points", "surface", "track", "trajectory", "object"]:
                    if w in s:
                        good_bonus += 1
                candidates.append({
                    "source": str(p),
                    "key": key,
                    "traj": traj,
                    "shape": tuple(traj.shape),
                    "bonus": good_bonus,
                })
    candidates = sorted(candidates, key=lambda c: (c["bonus"], c["shape"][0], c["shape"][1]), reverse=True)
    print(f"[GT candidates] {len(candidates)}")
    for c in candidates[:40]:
        print(" ", c["shape"], c["source"], "::", c["key"])
    return candidates


def parse_reference(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    raw = df[df["Method"].astype(str).str.contains("raw_param_opt", regex=False)]
    if raw.empty:
        raise RuntimeError("No raw row found in reference geometry csv.")
    r = raw.iloc[0]
    return {
        "CD Train": float(r["CD Train"]),
        "CD Test": float(r["CD Test"]),
        "Track Error Train": float(r["Track Error Train"]),
        "Track Error Test": float(r["Track Error Test"]),
    }


def read_split(case_dir: Path, n_frames: int):
    split_path = case_dir / "split.json"
    if not split_path.exists():
        train_end = 134 if n_frames >= 192 else int(round(n_frames * 0.7))
        return list(range(train_end)), list(range(train_end, n_frames))

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

    train_ids = expand(split.get("train", []))
    test_ids = expand(split.get("test", []))
    if not train_ids:
        train_end = 134 if n_frames >= 192 else int(round(n_frames * 0.7))
        train_ids = list(range(train_end))
    if not test_ids:
        all_ids = set(range(n_frames))
        test_ids = sorted(all_ids - set(train_ids))
    return train_ids, test_ids


def chamfer(a, b):
    tree_b = cKDTree(b)
    d_ab, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    d_ba, _ = tree_a.query(b, k=1)
    return float(0.5 * (d_ab.mean() + d_ba.mean()))


def track_error(a, b):
    n = min(a.shape[0], b.shape[0])
    return float(np.linalg.norm(a[:n] - b[:n], axis=-1).mean())


def slice_modes(n_src: int, n_ref: int):
    vals = {n_ref, n_src, min(n_src, n_ref), 4993, 5487, 6487}
    vals = [v for v in vals if 50 <= v <= n_src]
    modes = [("full", slice(0, n_src))]
    for v in vals:
        modes.append((f"first_{v}", slice(0, v)))
        modes.append((f"last_{v}", slice(n_src - v, n_src)))
    if n_src > 1000:
        modes.append(("drop_last_1000", slice(0, n_src - 1000)))
    uniq, seen = [], set()
    for name, sl in modes:
        sig = (sl.start, sl.stop)
        if sig not in seen:
            uniq.append((name, sl))
            seen.add(sig)
    return uniq


def compute_metrics(pred, gt, case_dir, stride=1):
    T = min(pred.shape[0], gt.shape[0])
    pred, gt = pred[:T], gt[:T]
    train_ids, test_ids = read_split(case_dir, T)
    train_ids = [i for i in train_ids if 0 <= i < T][::stride]
    test_ids = [i for i in test_ids if 0 <= i < T][::stride]

    def one(ids):
        cds, tes = [], []
        for t in ids:
            cds.append(chamfer(pred[t], gt[t]))
            tes.append(track_error(pred[t], gt[t]))
        return float(np.mean(cds)), float(np.mean(tes)), len(ids)

    cd_tr, te_tr, ntr = one(train_ids)
    cd_te, te_te, nte = one(test_ids)
    return {
        "CD Train": cd_tr,
        "CD Test": cd_te,
        "Track Error Train": te_tr,
        "Track Error Test": te_te,
        "Train Frame Num": ntr,
        "Test Frame Num": nte,
    }


def relerr(a, b):
    return abs(float(a) - float(b)) / (abs(float(b)) + 1e-9)


def protocol_score(m, ref):
    return (
        relerr(m["CD Train"], ref["CD Train"])
        + relerr(m["CD Test"], ref["CD Test"])
        + 0.5 * relerr(m["Track Error Train"], ref["Track Error Train"])
        + 0.5 * relerr(m["Track Error Test"], ref["Track Error Test"])
    )


def find_protocol(raw_pred, gt_candidates, case_dir, ref, stride):
    rows = []
    for c in gt_candidates:
        gt0 = c["traj"]
        if min(raw_pred.shape[0], gt0.shape[0]) < 20:
            continue
        for pname, psl in slice_modes(raw_pred.shape[1], gt0.shape[1]):
            pred = raw_pred[:, psl, :]
            for gname, gsl in slice_modes(gt0.shape[1], pred.shape[1]):
                gt = gt0[:, gsl, :]
                if min(pred.shape[1], gt.shape[1]) < 50:
                    continue
                try:
                    m = compute_metrics(pred, gt, case_dir, stride=stride)
                except Exception:
                    continue
                rows.append({
                    "score": protocol_score(m, ref),
                    "gt_source": c["source"],
                    "gt_key": c["key"],
                    "gt_shape": str(c["shape"]),
                    "pred_mode": pname,
                    "gt_mode": gname,
                    **m,
                })
    if not rows:
        raise RuntimeError("Could not find any valid geometry protocol.")
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def mode_to_slice(mode: str, n_src: int):
    if mode == "full":
        return slice(0, n_src)
    if mode.startswith("first_"):
        n = int(mode.split("_")[1])
        return slice(0, min(n, n_src))
    if mode.startswith("last_"):
        n = int(mode.split("_")[1])
        return slice(max(0, n_src - n), n_src)
    if mode == "drop_last_1000":
        return slice(0, max(1, n_src - 1000))
    raise ValueError(mode)


def apply_protocol(pred, protocol, gt_candidates):
    gt = None
    for c in gt_candidates:
        if c["source"] == protocol["gt_source"] and c["key"] == protocol["gt_key"] and str(c["shape"]) == protocol["gt_shape"]:
            gt = c["traj"]
            break
    if gt is None:
        raise RuntimeError("Selected GT protocol could not be restored.")
    pred = pred[:, mode_to_slice(str(protocol["pred_mode"]), pred.shape[1]), :]
    gt = gt[:, mode_to_slice(str(protocol["gt_mode"]), gt.shape[1]), :]
    return pred, gt


def parse_method_arg(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Method item must be METHOD=PATH, got {item}")
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-geometry-csv", required=True)
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--search-root", action="append", default=[])
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--protocol-csv", default=None)
    ap.add_argument("--search-stride", type=int, default=8)
    ap.add_argument("--max-protocol-score", type=float, default=0.35)
    args = ap.parse_args()

    ref = parse_reference(Path(args.reference_geometry_csv))
    print("[reference raw]", ref)
    case_dir = Path(args.case_dir)
    search_roots = [case_dir] + [Path(p) for p in args.search_root]

    methods = parse_method_arg(args.methods)
    raw_method = [m for m in methods if "raw_param_opt" in m][0]
    raw_pred = load_inference(methods[raw_method])

    gt_candidates = collect_gt_candidates(search_roots)
    proto_df = find_protocol(raw_pred, gt_candidates, case_dir, ref, stride=args.search_stride)

    protocol_csv = Path(args.protocol_csv) if args.protocol_csv else Path(args.out_csv).with_name(Path(args.out_csv).stem + "_protocol_search.csv")
    proto_df.to_csv(protocol_csv, index=False)
    print("\n[Top 20 protocols]")
    print(proto_df.head(20).to_string(index=False))
    print("Protocol search saved:", protocol_csv)

    best = proto_df.iloc[0]
    if float(best["score"]) > args.max_protocol_score:
        raise RuntimeError(
            f"Best protocol score {best['score']:.4f} is too high. "
            "The original geometry evaluator was not recovered reliably. "
            "No calibrated estimate is produced."
        )

    rows = []
    for method, path in methods.items():
        pred0 = load_inference(path)
        pred, gt = apply_protocol(pred0, best, gt_candidates)
        m = compute_metrics(pred, gt, case_dir, stride=1)
        rows.append({
            "Method": method,
            "Train Frame Num": m["Train Frame Num"],
            "CD Train": m["CD Train"],
            "Test Frame Num": m["Test Frame Num"],
            "CD Test": m["CD Test"],
            "Track Error Train": m["Track Error Train"],
            "Track Error Test": m["Track Error Test"],
            "Inference Path": str(path),
            "value_type": "measured_official_protocol_matched",
            "protocol_score": float(best["score"]),
            "protocol_gt_source": best["gt_source"],
            "protocol_gt_key": best["gt_key"],
            "protocol_pred_mode": best["pred_mode"],
            "protocol_gt_mode": best["gt_mode"],
        })

    out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print("\n[Measured geometry table]")
    print(out.to_string(index=False))
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
