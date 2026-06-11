#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import py_compile
import re
import shutil
from pathlib import Path

def backup_once(path: Path, suffix: str) -> Path:
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        shutil.copy(path, backup)
        print(f"[backup] {path} -> {backup}")
    return backup

def ensure_import(text: str, module_line: str) -> str:
    if re.search(rf"^\s*{re.escape(module_line)}\s*$", text, flags=re.M):
        return text
    m = re.search(r"^(import |from )", text, flags=re.M)
    if m:
        return text[: m.start()] + module_line + "\n" + text[m.start():]
    return module_line + "\n" + text

def patch_trainer_external_topology(project_dir: Path) -> None:
    path = project_dir / "qqtt" / "engine" / "trainer_warp.py"
    if not path.exists():
        raise FileNotFoundError(path)

    backup_once(path, ".backup_before_external_topology_npz")
    text = path.read_text(encoding="utf-8")
    text = ensure_import(text, "import os")
    text = ensure_import(text, "import numpy as np")

    if "[PATCH_EXTERNAL_TOPOLOGY_NPZ_TRAINER_FULL_SPRINGS]" in text:
        print("[skip] external topology patch already exists")
    else:
        new_block = '''            # [PATCH_EXTERNAL_TOPOLOGY_NPZ_TRAINER_FULL_SPRINGS]
            external_topology_path = os.environ.get("EXTERNAL_TOPOLOGY_NPZ", None)
            if external_topology_path is not None and os.path.exists(external_topology_path):
                ext_topo = np.load(external_topology_path, allow_pickle=True)
                ext_springs = ext_topo["springs"].astype(np.int32)
                ext_rest_lengths = ext_topo["rest_lengths"].astype(np.float32)
                ext_num_object_springs = int(ext_topo["num_object_springs"])

                assert ext_springs.ndim == 2 and ext_springs.shape[1] == 2
                assert ext_rest_lengths.shape[0] == ext_springs.shape[0]
                assert ext_springs.min() >= 0
                assert ext_springs.max() < points.shape[0], (ext_springs.max(), points.shape[0])

                springs = ext_springs
                rest_lengths = ext_rest_lengths
                num_object_springs = ext_num_object_springs

                print("[PATCH_EXTERNAL_TOPOLOGY_NPZ_TRAINER_FULL_SPRINGS] Loaded:", external_topology_path)
                print("[PATCH_EXTERNAL_TOPOLOGY_NPZ_TRAINER_FULL_SPRINGS] trainer points kept:", points.shape)
                print("[PATCH_EXTERNAL_TOPOLOGY_NPZ_TRAINER_FULL_SPRINGS] external springs:", springs.shape)
                print("[PATCH_EXTERNAL_TOPOLOGY_NPZ_TRAINER_FULL_SPRINGS] num_object_springs:", num_object_springs)

'''
        old_return = '''            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
'''
        if old_return in text:
            text = text.replace(old_return, new_block + old_return)
        else:
            pat = re.compile(
                r"(?P<indent>[ \t]+)return\s*\(\s*\n"
                r"[ \t]+torch\.tensor\(points,.*?\n"
                r"[ \t]+torch\.tensor\(springs,.*?\n"
                r"[ \t]+torch\.tensor\(rest_lengths,.*?\n"
                r"[ \t]+torch\.tensor\(masses,.*?\n"
                r"[ \t]+num_object_springs,\s*\n"
                r"[ \t]+\)",
                flags=re.S,
            )
            m = pat.search(text)
            if not m:
                raise RuntimeError("Cannot find _init_start return block in trainer_warp.py.")
            text = text[: m.start()] + new_block + text[m.start():]
        print("[patch] inserted external topology patch")

    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print("[ok] trainer_warp.py syntax")

def patch_dummy_wandb(project_dir: Path) -> None:
    path = project_dir / "qqtt" / "engine" / "trainer_warp.py"
    backup_once(path, ".backup_before_dummy_wandb")
    text = path.read_text(encoding="utf-8")

    if "[PATCH_DUMMY_WANDB]" in text:
        print("[skip] dummy wandb patch already exists")
        return

    old = "import wandb\n"
    new = '''# [PATCH_DUMMY_WANDB]
class _DummyWandbVideo:
    def __init__(self, *args, **kwargs):
        pass

class _DummyWandb:
    Video = _DummyWandbVideo

    @staticmethod
    def log(*args, **kwargs):
        pass

    @staticmethod
    def init(*args, **kwargs):
        return None

    @staticmethod
    def finish(*args, **kwargs):
        pass

wandb = _DummyWandb()
'''
    if old not in text:
        print("[warn] plain 'import wandb' not found; leave trainer_warp.py unchanged for wandb")
        return

    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print("[patch] dummy wandb")

def patch_visualize_headless(project_dir: Path) -> None:
    vis_path = project_dir / "qqtt" / "utils" / "visualize.py"
    if not vis_path.exists():
        print(f"[skip] visualize.py not found: {vis_path}")
        return

    backup_once(vis_path, ".backup_before_headless_visualize_pc")
    text = vis_path.read_text(encoding="utf-8")
    text = ensure_import(text, "import os")

    if "[PATCH_SKIP_VISUALIZE_PC_HEADLESS]" in text:
        print("[skip] visualize_pc headless patch already exists")
    else:
        marker = "def visualize_pc("
        start = text.find(marker)
        if start < 0:
            print("[warn] cannot find visualize_pc; skip")
        else:
            colon_pos = text.find("):", start)
            if colon_pos < 0:
                print("[warn] cannot find visualize_pc signature end; skip")
            else:
                insert_pos = colon_pos + 2
                insert = '''
    # [PATCH_SKIP_VISUALIZE_PC_HEADLESS]
    if os.environ.get("SKIP_OPEN3D_VIDEO", "0") == "1":
        print("[PATCH_SKIP_VISUALIZE_PC_HEADLESS] Skip Open3D point-cloud video rendering.")
        return
'''
                text = text[:insert_pos] + insert + text[insert_pos:]
                print("[patch] visualize_pc headless skip")

    vis_path.write_text(text, encoding="utf-8")
    py_compile.compile(str(vis_path), doraise=True)

def patch_pytorch3d_fallback(project_dir: Path) -> None:
    for name in ["gs_render.py", "gs_render_dynamics.py"]:
        path = project_dir / name
        if not path.exists():
            continue

        backup_once(path, ".backup_before_pytorch3d_fallback")
        text = path.read_text(encoding="utf-8")
        if "[PATCH_PYTORCH3D_FALLBACK]" in text:
            print(f"[skip] {name} pytorch3d fallback already exists")
            continue

        fallback = '''
# [PATCH_PYTORCH3D_FALLBACK]
try:
    import pytorch3d
    import pytorch3d.ops as ops
except ModuleNotFoundError:
    print("[PATCH] pytorch3d not found; using torch fallback for ops.knn_points")
    import torch

    class _KNNResult:
        pass

    class _FallbackOps:
        @staticmethod
        def knn_points(p1, p2, K=1, return_nn=False, **kwargs):
            dists_all = torch.cdist(p1, p2) ** 2
            dists, idx = torch.topk(dists_all, k=K, dim=-1, largest=False)
            result = _KNNResult()
            result.dists = dists
            result.idx = idx
            if return_nn:
                B = p2.shape[0]
                batch_idx = torch.arange(B, device=p2.device)[:, None, None]
                result.knn = p2[batch_idx, idx]
            return result

    ops = _FallbackOps()
'''
        replaced = False
        for old in ["import pytorch3d\nimport pytorch3d.ops as ops\n", "import pytorch3d.ops as ops\n", "import pytorch3d\n"]:
            if old in text:
                text = text.replace(old, fallback + "\n", 1)
                replaced = True
                break
        if replaced:
            path.write_text(text, encoding="utf-8")
            py_compile.compile(str(path), doraise=True)
            print(f"[patch] {name} pytorch3d fallback")
        else:
            print(f"[warn] no pytorch3d import found in {name}; skip")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".", help="PhysTwin project root")
    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()

    patch_trainer_external_topology(project_dir)
    patch_dummy_wandb(project_dir)
    patch_visualize_headless(project_dir)
    patch_pytorch3d_fallback(project_dir)

    print("\\n[done] patches applied")

if __name__ == "__main__":
    main()
