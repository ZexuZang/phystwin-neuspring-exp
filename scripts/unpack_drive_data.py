#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

DEFAULT_ZIPS = [
    "data.zip",
    "experiments.zip",
    "experiments_optimization.zip",
    "gaussian_output.zip",
]

def unzip_file(zip_path: Path, dst_dir: Path) -> None:
    print(f"[unzip] {zip_path} -> {dst_dir}")
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(dst_dir)], check=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-dir", required=True, help="Google Drive folder, e.g. /content/drive/MyDrive/PhysTwin_Data")
    parser.add_argument("--project-dir", default="/content/PhysTwin", help="PhysTwin project root")
    parser.add_argument("--zip-names", nargs="*", default=DEFAULT_ZIPS)
    parser.add_argument("--copy-to-content", action="store_true", help="Copy zips to /content before unzipping")
    args = parser.parse_args()

    drive_dir = Path(args.drive_dir)
    project_dir = Path(args.project_dir)

    if not drive_dir.exists():
        raise FileNotFoundError(f"Drive dir not found: {drive_dir}")
    project_dir.mkdir(parents=True, exist_ok=True)

    for name in args.zip_names:
        src = drive_dir / name
        if not src.exists():
            print(f"[skip] missing {src}")
            continue

        zip_path = src
        if args.copy_to_content:
            dst = Path("/content") / name
            print(f"[copy] {src} -> {dst}")
            shutil.copy2(src, dst)
            zip_path = dst

        unzip_file(zip_path, project_dir)

    print("\n[check]")
    for rel in [
        "data/different_types",
        "experiments",
        "experiments_optimization",
        "gaussian_output",
        "results",
    ]:
        p = project_dir / rel
        print(f"{p}: {'OK' if p.exists() else 'missing'}")

if __name__ == "__main__":
    main()
