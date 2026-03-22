#!/usr/bin/env python3
"""Create a minimal ImageNet-style tree (1000 wnids) for local smoke tests.

This does **not** replace the real ImageNet training set. It writes one small
JPEG per class so ``ImageFolder`` and label ranges match the release configs.

Real training: set ``DRIFTING_IMAGENET_PATH`` to your ILSVRC ``train/`` and ``val/`` roots.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else REPO_ROOT / "data" / "imagenet")
    if (root / "train" / "n01440764").is_dir() and any((root / "train" / "n01440764").iterdir()):
        print(f"Already populated: {root / 'train' / 'n01440764'} — skipping.")
        return 0

    with urllib.request.urlopen(CLASS_INDEX_URL, timeout=120) as r:
        meta = json.loads(r.read().decode("utf-8"))
    wnids = [meta[k][0] for k in sorted(meta.keys(), key=int)]

    for split in ("train", "val"):
        for i, wnid in enumerate(wnids):
            d = root / split / wnid
            d.mkdir(parents=True, exist_ok=True)
            out = d / "000.jpg"
            if out.exists():
                continue
            # Deterministic 256×256 RGB pattern from class index
            r = (i * 37) % 256
            g = (i * 91) % 256
            b = (i * 53) % 256
            img = Image.new("RGB", (256, 256), (r, g, b))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            out.write_bytes(buf.getvalue())
        print(f"Wrote {len(wnids)} classes under {root / split}")

    print(f"Smoke ImageNet root: {root}")
    print("Export DRIFTING_IMAGENET_PATH=" + str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
