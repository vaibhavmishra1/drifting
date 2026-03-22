#!/usr/bin/env bash
# One-shot local GPU setup for drifting (conda env, JAX+CUDA, deps, data dirs, FID stats).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${DRIFTING_CONDA_ENV:-drifting-release}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; install Miniforge/Miniconda first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

echo "Installing JAX (CUDA 12)…"
pip install -U "jax[cuda12_pip]==0.4.37" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "numpy<2"
pip install -r "$ROOT/requirements-gpu.txt"

mkdir -p "$ROOT/data/imagenet" "$ROOT/data/fid" "$ROOT/data/hf_cache" "$ROOT/data/latent_cache"

echo "Downloading FID reference stats…"
python "$ROOT/scripts/download_fid_stats.py"

if [[ "${DRIFTING_SKIP_SMOKE_IMAGENET:-0}" != "1" ]]; then
  echo "Creating smoke ImageNet tree (set DRIFTING_SKIP_SMOKE_IMAGENET=1 to skip)…"
  python "$ROOT/scripts/create_smoke_imagenet.py" "$ROOT/data/imagenet"
fi

cat <<EOF

Done. Activate and run:

  conda activate $ENV_NAME
  cd $ROOT
  python main.py --gen --config configs/gen/local_1gpu.yaml --workdir runs/local_dev

Defaults use paths under $ROOT/data/ (override with DRIFTING_* env vars in utils/env.py).

For real ImageNet, replace $ROOT/data/imagenet with your ILSVRC tree and set
DRIFTING_IMAGENET_PATH accordingly. Latent training with use_cache=true requires
building the cache (see README) and DRIFTING_IMAGENET_CACHE_PATH.
EOF
