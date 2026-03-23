#!/usr/bin/env python3
"""Export Hugging Face `ILSVRC/imagenet-1k` into ImageFolder-style folders.

Writes:
  <out_root>/train/<wnid>/*.JPEG
  <out_root>/val/<wnid>/*.JPEG
  <out_root>/test/<wnid>/*.JPEG   (if the Hub split exists)

Requires: `pip install datasets tqdm` (in addition to project deps).

You must accept the dataset terms and authenticate, e.g.:
  https://huggingface.co/datasets/ILSVRC/imagenet-1k
  export HF_TOKEN=...   # or `huggingface-cli login`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm


CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)


def load_wnid_table(url: str) -> dict[int, str]:
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = json.load(r)
    # keys are "0".."999", values are [wnid, human_name]
    out = {int(k): v[0] for k, v in raw.items()}
    if len(out) != 1000:
        raise ValueError(f"expected 1000 classes, got {len(out)}")
    return out


def hub_split_to_dir(name: str) -> str:
    if name == "validation":
        return "val"
    return name


def export_split(split_ds, out_dir: Path, idx_to_wnid: dict[int, str], split_label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bad_label = 0
    for i, ex in enumerate(tqdm(split_ds, desc=f"export {split_label}", unit="img")):
        img = ex["image"]
        if hasattr(img, "convert"):
            if img.mode != "RGB":
                img = img.convert("RGB")
        else:
            raise TypeError(f"unexpected image type: {type(img)}")

        lab = ex["label"]
        if lab is None or lab < 0 or lab >= 1000:
            wnid = "_unknown"
            bad_label += 1
        else:
            wnid = idx_to_wnid[int(lab)]

        class_dir = out_dir / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / f"{split_label}_{i:08d}.JPEG"
        if dest.exists():
            continue
        img.save(dest, format="JPEG", quality=95, optimize=True)

    if bad_label:
        print(f"[{split_label}] wrote {bad_label} images without a valid label to `_unknown/`", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "imagenet",
        help="Output root (train/ val/ test/ created here).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--class-index-url",
        default=CLASS_INDEX_URL,
        help="JSON mapping class index -> [wnid, name] (PyTorch/ImageNet order).",
    )
    parser.add_argument(
        "--dataset",
        default="ILSVRC/imagenet-1k",
        help="Hub dataset id.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        tok_path = Path.home() / ".cache" / "huggingface" / "token"
        if not tok_path.is_file():
            print(
                "No Hugging Face token found. Set HF_TOKEN or run `huggingface-cli login` "
                "after accepting the dataset license.",
                file=sys.stderr,
            )

    idx_to_wnid = load_wnid_table(args.class_index_url)

    from datasets import load_dataset

    kwargs: dict = {"token": token if token else None}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)

    ds = load_dataset(args.dataset, **kwargs)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for split_name in ds.keys():
        sub = hub_split_to_dir(split_name)
        target = args.out_root / sub
        print(f"{split_name} -> {target}")
        export_split(ds[split_name], target, idx_to_wnid, split_label=sub)

    print("Done.")


if __name__ == "__main__":
    main()
