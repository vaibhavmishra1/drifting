#!/usr/bin/env python3
"""Download ImageNet-256 FID reference statistics (compatible with this codebase)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "fid" / "imagenet_256_fid_stats.npz"


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(
        repo_id="Yuanzhi/DiMO",
        filename="fid_stats_imagenet256_guided_diffusion.npz",
        repo_type="model",
    )
    data = np.load(src, allow_pickle=True)
    # Normalize keys expected by utils/fid_util._load_ref_stats
    if "ref_mu" in data:
        mu, sigma = data["ref_mu"], data["ref_sigma"]
    elif "mu" in data:
        mu, sigma = data["mu"], data["sigma"]
    else:
        print("Unexpected keys in FID npz:", list(data.keys()), file=sys.stderr)
        return 1
    np.savez_compressed(str(out), mu=mu, sigma=sigma)
    print(f"Wrote {out} (mu={mu.shape}, sigma={sigma.shape})")
    meta = {
        "source_repo": "Yuanzhi/DiMO",
        "source_file": "fid_stats_imagenet256_guided_diffusion.npz",
        "note": "Re-saved with mu/sigma keys for drifting fid_util.",
    }
    (out.parent / "imagenet_256_fid_stats.source.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
