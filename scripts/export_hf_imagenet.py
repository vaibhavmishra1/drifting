#!/usr/bin/env python3
"""Export Hugging Face `ILSVRC/imagenet-1k` into ImageFolder layout.

Writes:
  <out_root>/train/<wnid>/*.JPEG
  <out_root>/val/<wnid>/*.JPEG
  <out_root>/test/<wnid>/*.JPEG   (labels are -1 for test; saved under _unknown/)

Speed notes:
  - Raw bytes are written directly (no JPEG decode/re-encode).
  - A thread pool writes files in parallel.
  - Set HF_HUB_ENABLE_HF_TRANSFER=1 (done automatically below) for multi-part
    downloads via hf_transfer if it is installed.

Usage:
  export HF_TOKEN=hf_...   # after accepting terms at huggingface.co/datasets/ILSVRC/imagenet-1k
  python scripts/export_hf_imagenet.py \\
      --out-root /workspace/drifting/data/imagenet \\
      --cache-dir /workspace/drifting/data/hf_cache/hub_datasets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)
DEFAULT_WORKERS = min(32, (os.cpu_count() or 4) * 2)


def load_wnid_table(url: str) -> dict[int, str]:
    """Fetch the public PyTorch class-index JSON and return {int_label: wnid}."""
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = json.load(r)
    out = {int(k): v[0] for k, v in raw.items()}
    if len(out) != 1000:
        raise ValueError(f"expected 1000 classes, got {len(out)}")
    return out


def hub_split_to_dir(name: str) -> str:
    if name == "validation":
        return "val"
    return name


def _write_one(args: tuple) -> None:
    dest, raw_bytes = args
    dest.write_bytes(raw_bytes)


def export_split(
    split_ds,
    out_dir: Path,
    idx_to_wnid: dict[int, str],
    split_label: str,
    num_workers: int = DEFAULT_WORKERS,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bad_label = 0
    skipped = 0

    # Pre-create all 1000 class dirs to avoid races in the thread pool.
    # Do NOT pre-create _unknown — it must only exist if bad labels are actually found,
    # otherwise torchvision.ImageFolder treats the empty dir as an invalid class.
    for wnid in idx_to_wnid.values():
        (out_dir / wnid).mkdir(exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {}
        pbar = tqdm(
            enumerate(split_ds),
            total=len(split_ds),
            desc=f"export {split_label}",
            unit="img",
        )
        for i, ex in pbar:
            lab = ex.get("label")
            if lab is None or int(lab) < 0 or int(lab) >= 1000:
                wnid = "_unknown"
                bad_label += 1
                (out_dir / "_unknown").mkdir(exist_ok=True)  # create only on first bad label
            else:
                wnid = idx_to_wnid[int(lab)]

            dest = out_dir / wnid / f"{split_label}_{i:08d}.JPEG"
            if dest.exists():
                skipped += 1
                continue

            # ex["image"] is a dict with key "bytes" when decode=False.
            raw = ex["image"]
            raw_bytes: bytes = raw["bytes"] if isinstance(raw, dict) else raw

            fut = pool.submit(_write_one, (dest, raw_bytes))
            futures[fut] = dest

        # Drain futures and surface errors.
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print(f"\nERROR writing {futures[fut]}: {exc}", file=sys.stderr)

    if skipped:
        print(f"[{split_label}] skipped {skipped} already-existing files.")
    if bad_label:
        print(
            f"[{split_label}] {bad_label} images had no valid label → saved to `_unknown/`",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "imagenet",
        help="Output root directory. train/, val/, test/ are created here.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face datasets cache directory (optional).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel file-write threads (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Subset of splits to download, e.g. --splits train validation. Default: all.",
    )
    parser.add_argument(
        "--class-index-url",
        default=CLASS_INDEX_URL,
    )
    parser.add_argument(
        "--dataset",
        default="ILSVRC/imagenet-1k",
    )
    args = parser.parse_args()

    # Enable fast multi-part downloads if hf_transfer is installed.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        hf_home = os.environ.get("HF_HOME", "")
        candidates = [
            Path.home() / ".cache" / "huggingface" / "token",
            Path(hf_home) / "token" if hf_home else None,
        ]
        if not any(p and p.is_file() for p in candidates if p):
            print(
                "WARNING: No Hugging Face token found.\n"
                "  Set HF_TOKEN or run `huggingface-cli login` after accepting\n"
                "  the dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k",
                file=sys.stderr,
            )

    print("Fetching ImageNet class index...")
    idx_to_wnid = load_wnid_table(args.class_index_url)

    from datasets import load_dataset, Image

    load_kwargs: dict = {"token": token or None}
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)

    print(f"Loading dataset '{args.dataset}'...")
    ds = load_dataset(args.dataset, **load_kwargs)

    # Cast image column to raw bytes — skips JPEG decode and re-encode entirely.
    ds = ds.cast_column("image", Image(decode=False))

    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {args.out_root}")
    print(f"Write workers: {args.num_workers}")

    splits_to_run = args.splits or list(ds.keys())
    for split_name in splits_to_run:
        if split_name not in ds:
            print(f"WARNING: split '{split_name}' not found, skipping.")
            continue
        sub = hub_split_to_dir(split_name)
        target = args.out_root / sub
        print(f"\n[{split_name}] → {target}  ({len(ds[split_name]):,} examples)")
        export_split(
            ds[split_name],
            target,
            idx_to_wnid,
            split_label=sub,
            num_workers=args.num_workers,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
