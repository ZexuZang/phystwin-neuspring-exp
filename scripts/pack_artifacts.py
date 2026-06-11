#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

INCLUDE_EXTS = {
    ".pth",
    ".pkl",
    ".npz",
    ".csv",
    ".txt",
    ".json",
    ".log",
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapt-root", required=True)
    parser.add_argument("--out-zip", required=True)
    args = parser.parse_args()

    adapt_root = Path(args.adapt_root)
    out_zip = Path(args.out_zip)
    if not adapt_root.exists():
        raise FileNotFoundError(adapt_root)

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, filenames in os.walk(adapt_root):
            for name in filenames:
                p = Path(root) / name
                if p.suffix in INCLUDE_EXTS:
                    arcname = p.relative_to(adapt_root)
                    zf.write(p, arcname.as_posix())
                    print("Added:", arcname)

    print("Saved:", out_zip)

if __name__ == "__main__":
    main()
