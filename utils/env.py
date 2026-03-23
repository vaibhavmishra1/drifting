"""Global paths for the public Drift release."""

from __future__ import annotations

import os

IMAGENET_PATH = "/workspace/drifting/data/imagenet"
IMAGENET_CACHE_PATH = "/workspace/drifting/data/latent_cache"
IMAGENET_FID_NPZ = "/workspace/drifting/data/fid/imagenet_256_fid_stats.npz"
IMAGENET_PR_NPZ = "/workspace/drifting/data/fid/imagenet_val_prc_arr0.npz"

HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = os.environ.get("HF_ROOT", "/workspace/drifting/data/hf_cache")
