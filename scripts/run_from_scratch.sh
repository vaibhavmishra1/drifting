#!/usr/bin/env bash
# End-to-end: create conda env, install JAX+deps, prepare data/FID, then start generator training.
#
# Usage:
#   bash scripts/run_from_scratch.sh
#   bash scripts/run_from_scratch.sh --setup-only
#   bash scripts/run_from_scratch.sh --train-only
#   DRIFTING_CONFIG=configs/gen/latent_sota_L.yaml DRIFTING_WORKDIR=runs/my_run bash scripts/run_from_scratch.sh
#   bash scripts/run_from_scratch.sh -- --init-from hf://latent_L_sota
#
# Environment (optional):
#   DRIFTING_CONDA_ENV     conda env name (default: drifting-release)
#   DRIFTING_CONFIG        YAML config path (default: configs/gen/local_1gpu.yaml)
#   DRIFTING_WORKDIR       checkpoint/log root (default: runs/from_scratch)
#   DRIFTING_SKIP_SMOKE_IMAGENET=1  skip synthetic ImageNet tree (you must set DRIFTING_IMAGENET_PATH)
#   DRIFTING_SKIP_SETUP=1           same as --train-only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${DRIFTING_CONDA_ENV:-drifting-release}"
CONFIG="${DRIFTING_CONFIG:-configs/gen/local_1gpu.yaml}"
WORKDIR="${DRIFTING_WORKDIR:-runs/from_scratch}"

SETUP_ONLY=0
TRAIN_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-only) SETUP_ONLY=1; shift ;;
    --train-only) TRAIN_ONLY=1; shift ;;
    --) shift; break ;;
    -*)
      echo "Unknown option: $1" >&2
      echo "Use --setup-only, --train-only, or -- before args for main.py." >&2
      exit 1
      ;;
    *) break ;;
  esac
done

if [[ "${DRIFTING_SKIP_SETUP:-0}" == "1" ]]; then
  TRAIN_ONLY=1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; install Miniforge/Miniconda first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

run_setup() {
  echo "=== [1/5] Conda env: $ENV_NAME ==="
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.10 -y
  fi
  conda activate "$ENV_NAME"

  echo "=== [2/5] JAX (CUDA 12) + Python deps ==="
  pip install -U "jax[cuda12_pip]==0.4.37" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  pip install "numpy<2"
  pip install -r "$ROOT/requirements-gpu.txt"

  echo "=== [3/5] Data directories ==="
  mkdir -p "$ROOT/data/imagenet" "$ROOT/data/fid" "$ROOT/data/hf_cache" "$ROOT/data/latent_cache"

  echo "=== [4/5] FID reference stats ==="
  python "$ROOT/scripts/download_fid_stats.py"

  echo "=== [5/5] ImageNet tree ==="
  if [[ "${DRIFTING_SKIP_SMOKE_IMAGENET:-0}" != "1" ]]; then
    python "$ROOT/scripts/create_smoke_imagenet.py" "$ROOT/data/imagenet"
  else
    echo "Skipped smoke ImageNet (DRIFTING_SKIP_SMOKE_IMAGENET=1). Ensure DRIFTING_IMAGENET_PATH is valid."
  fi
}

run_train() {
  conda activate "$ENV_NAME"
  export HF_HOME="${HF_HOME:-$ROOT/data/hf_cache}"
  echo "=== Training ==="
  echo "  config:  $CONFIG"
  echo "  workdir: $WORKDIR"
  echo "  HF_HOME: $HF_HOME"
  python "$ROOT/main.py" --gen --config "$CONFIG" --workdir "$WORKDIR" "$@"
}

if [[ "$TRAIN_ONLY" -eq 0 ]]; then
  run_setup
fi

if [[ "$SETUP_ONLY" -eq 1 ]]; then
  cat <<EOF

Setup finished. Train later with:
  conda activate $ENV_NAME
  cd $ROOT
  export HF_HOME=\$PWD/data/hf_cache
  python main.py --gen --config $CONFIG --workdir $WORKDIR

Or: bash scripts/run_from_scratch.sh --train-only
EOF
  exit 0
fi

run_train "$@"
