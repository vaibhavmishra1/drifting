"""Global paths for the public Drift release.

Override any path with environment variables (see names below) or edit defaults
after running ``scripts/setup_training.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = _REPO_ROOT / "data"

IMAGENET_PATH = os.environ.get(
    "DRIFTING_IMAGENET_PATH", str(_DEFAULT_DATA / "imagenet")
)
IMAGENET_CACHE_PATH = os.environ.get(
    "DRIFTING_IMAGENET_CACHE_PATH", str(_DEFAULT_DATA / "latent_cache")
)
IMAGENET_FID_NPZ = os.environ.get(
    "DRIFTING_IMAGENET_FID_NPZ",
    str(_DEFAULT_DATA / "fid" / "imagenet_256_fid_stats.npz"),
)
# Only read when precision/recall eval is enabled (50k-sample runs).
IMAGENET_PR_NPZ = os.environ.get(
    "DRIFTING_IMAGENET_PR_NPZ",
    str(_DEFAULT_DATA / "fid" / "imagenet_val_prc_arr0.npz"),
)

HF_REPO_ID = os.environ.get("DRIFTING_HF_REPO_ID", "Goodeat/drifting")
HF_ROOT = os.environ.get(
    "DRIFTING_HF_ROOT",
    os.environ.get("HF_ROOT", str(_DEFAULT_DATA / "hf_cache")),
)
