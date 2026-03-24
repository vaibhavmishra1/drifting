"""Build Inception reference statistics (.npz) for a class subset of ImageNet val.

Usage:
    python -m utils.build_ref_stats \
        --class-subset 0,1,2,3,4,5,6,7,8,9 \
        --out data/fid/imagenet_256_10cls_fid_stats.npz \
        --eval-batch-size 256
"""
from __future__ import annotations

import argparse

import jax

from dataset.dataset import create_imagenet_split
from utils.fid_util import build_ref_stats_from_loader
from utils.hsdp_util import set_global_mesh
from utils.misc import run_init


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FID reference stats for a class subset.")
    parser.add_argument("--class-subset", type=str, required=True,
                        help="Comma-separated ImageNet class indices, e.g. '0,1,2,3,4,5,6,7,8,9'.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output .npz path, e.g. data/fid/imagenet_256_10cls_fid_stats.npz")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Max real images to use (0 = all available for the subset).")
    parser.add_argument("--hsdp-dim", type=int, default=None)
    args = parser.parse_args()

    run_init()
    hsdp = args.hsdp_dim or min(8, jax.local_device_count() * jax.process_count())
    set_global_mesh(hsdp)

    class_subset = [int(c.strip()) for c in args.class_subset.split(",")]

    eval_loader, _, _ = create_imagenet_split(
        resolution=256,
        split="val",
        batch_size=args.eval_batch_size // jax.process_count(),
        num_workers=4,
        use_cache=True,
        class_subset=class_subset,
    )

    n_available = len(eval_loader.dataset)
    num_samples = args.num_samples if args.num_samples > 0 else n_available
    print(f"Building reference stats from {num_samples} real images "
          f"({len(class_subset)} classes) -> {args.out}")

    build_ref_stats_from_loader(eval_loader, num_samples=num_samples, out_npz=args.out)


if __name__ == "__main__":
    main()
