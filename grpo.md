# Advantage-Weighted Drifting (AWD) — GRPO Post-Training

GRPO-based post-training for Drifting Models. Reduces FID via group-relative
distributional reweighting of the drifting loss.

**Baseline:** Drift-L latent, FID 1.54, IS 260.1 on ImageNet 256x256 (1 NFE, CFG 1.0)

**Target:** FID 1.45–1.50 via AWD post-training (confidence 72–78%)

---

## Algorithm

Standard drifting trains with uniform MSE: every generated sample contributes
equally regardless of whether its drift direction is reliable or noisy.

AWD replaces this with advantage-weighted loss:

```
L_AWD = sum_i  w_i * || f_theta(eps_i) - target_i ||^2
```

where:
- `r_i = log p_kde(x_i; real) - lambda * log q_kde(x_i; generated\{i})`
- `A_i = (r_i - mean(r)) / std(r)`   (group-relative advantage)
- `w_i = softmax(A_i / tau) * G`      (normalized weights)

The reward reuses the KDE kernel infrastructure from the drifting pipeline:
- **Quality term** (`log p_kde`): is x_i near real data?
- **Diversity penalty** (`log q_kde`, leave-one-out): is x_i redundant?

Both are computed on scale-normalized features (same space as the drift loss),
making the `awd_bandwidth` parameter directly comparable to `R_list` values.

Temperature `tau` is annealed linearly from `awd_tau_max` to `awd_tau_min`
over training, starting near-uniform and gradually sharpening.

---

## Changes Made

### `drift_loss.py`

**New function: `compute_kde_rewards(gen, fixed_pos, bandwidth, diversity_weight)`**

Computes per-sample distributional quality rewards using Gaussian KDE.
- Quality: `log p_kde(x_i; real)` via `logsumexp(-d^2 / 2h^2)` over positive samples
- Diversity: leave-one-out `log q_kde(x_i; gen\{i})` with diagonal masking
- Returns `rewards = log_p - lambda * log_q` of shape `[B, C_g]`

**Modified function: `drift_loss(...)`**

Added parameters:
| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_awd` | bool (static) | `False` | Enable AWD reweighting |
| `awd_tau` | float (dynamic) | `1.0` | Temperature for advantage softmax |
| `awd_lambda` | float | `0.1` | Diversity penalty weight |
| `awd_bandwidth` | float | `0.05` | KDE bandwidth (in normalized feature space) |

When `use_awd=True`, the loss computation changes from uniform MSE to:
1. Per-sample MSE: `mean(diff^2, axis=-1)` over feature dim only
2. KDE rewards on scale-normalized features (`old_gen / scale_inputs`)
3. Group-relative advantages with epsilon-stabilized std
4. Softmax weighting scaled by group size C_g
5. Weighted mean over generated samples

All reward/weight computations are `stop_gradient`-ed. When `use_awd=False`
(default), behavior is identical to the original code.

New logged metrics when AWD is active:
- `awd_reward_mean`, `awd_reward_std`: reward distribution statistics
- `awd_adv_abs_max`: advantage magnitude (monitors sharpness)
- `awd_weight_max`: maximum weight (monitors concentration)

### `train.py`

**`train_step()`** — Added `awd_tau=1.0` keyword parameter. Passed through
closure chain to `drift_loss()` in `feature_loss()`. Dynamic (not curried)
so it can change per step for temperature annealing.

**`train_gen()`** — Added parameters:
| Parameter | Default | Description |
|---|---|---|
| `awd_tau_max` | `10.0` | Initial (max) temperature |
| `awd_tau_min` | `0.5` | Final (min) temperature |

Training loop changes:
- Detects `use_awd` from `loss_kwargs` at startup with logging
- Computes linearly annealed `tau` per step: `tau_max - (tau_max - tau_min) * progress`
- Passes `awd_tau` as dynamic kwarg to `train_step_jit`
- Logs current `awd_tau` in metrics

### `configs/gen/latent_awd_L.yaml`

New config for AWD post-training of Drift-L. Key differences from `latent_sota_L.yaml`:
- `init_from: "hf://latent_L_sota"` — starts from pre-trained checkpoint
- `learning_rate: 0.00003` — 3e-5 (lower for fine-tuning)
- `warmup_steps: 500` — short warmup
- `total_steps: 20000` — shorter than base training (200k)
- `eval_per_step: 2000` — frequent evaluation to track FID
- `loss_kwargs.use_awd: true` — enables AWD
- `loss_kwargs.awd_lambda: 0.1` — diversity penalty weight
- `loss_kwargs.awd_bandwidth: 0.05` — KDE bandwidth
- `awd_tau_max: 10.0 -> awd_tau_min: 0.5` — tau annealing schedule

---

## Commands

### Environment Setup

```bash
conda create -n drifting-release python=3.10 -y
conda activate drifting-release
pip install -r requirements.txt
export JAX_PLATFORMS=tpu,cpu
```

### Build Latent Cache (if not already done)

```bash
python -m dataset.latent \
  --data-path /path/to/imagenet \
  --target-path /path/to/latent_cache \
  --local-batch-size 128 \
  --num-workers 8 \
  --pin-memory
```

### Reproduce Baseline FID (verify setup)

```bash
python inference.py --init-from "hf://latent_L_sota" --cfg-scale 1.0 \
  --num-samples 50000 --eval-batch-size 256 --json-out results_baseline.json
```

Expected: FID ~1.53–1.54, IS ~260

### Run AWD Post-Training

```bash
python main.py --gen --config configs/gen/latent_awd_L.yaml --workdir runs/awd_latent_L
```

Monitor AWD-specific metrics in logs:
- `awd_tau`: current temperature (should decay 10.0 -> 0.5)
- `awd_reward_mean/std`: reward distribution health
- `awd_weight_max`: weight concentration (should increase over training)
- `awd_adv_abs_max`: advantage magnitude

### Evaluate AWD Model

```bash
python inference.py --init-from runs/awd_latent_L --cfg-scale 1.0 \
  --num-samples 50000 --eval-batch-size 256 --json-out results_awd.json
```

### With W&B Logging

```bash
python main.py --gen --config configs/gen/latent_awd_L.yaml --workdir runs/awd_latent_L

python inference.py --init-from runs/awd_latent_L --cfg-scale 1.0 \
  --num-samples 50000 --eval-batch-size 256 --json-out results_awd.json \
  --use-wandb --wandb-entity YOUR_ENTITY --wandb-project YOUR_PROJECT
```

---

## Hyperparameter Guide

### Primary Hyperparameters

| Parameter | Config Path | Default | Range | Notes |
|---|---|---|---|---|
| `awd_tau_max` | `train.awd_tau_max` | `10.0` | 5.0–20.0 | Higher = more uniform start |
| `awd_tau_min` | `train.awd_tau_min` | `0.5` | 0.1–2.0 | Lower = sharper final weighting |
| `awd_lambda` | `train.loss_kwargs.awd_lambda` | `0.1` | 0.0–0.5 | 0.0 = quality only, no diversity penalty |
| `awd_bandwidth` | `train.loss_kwargs.awd_bandwidth` | `0.05` | 0.02–0.2 | Match to R_list middle value |
| Learning rate | `optimizer.lr_schedule.learning_rate` | `3e-5` | 1e-5–5e-5 | Lower than base training |
| Total steps | `train.total_steps` | `20000` | 5k–20k | Track FID at checkpoints |

### Ablation Sweeps

**Temperature schedule:**
```yaml
# Fixed tau (no annealing)
awd_tau_max: 2.0
awd_tau_min: 2.0

# Aggressive annealing
awd_tau_max: 20.0
awd_tau_min: 0.1
```

**Diversity weight:**
```yaml
# Quality-only (no diversity penalty)
awd_lambda: 0.0

# Moderate diversity
awd_lambda: 0.1

# Strong diversity
awd_lambda: 0.5
```

**Kernel choice for rewards:**

The `awd_bandwidth` operates on scale-normalized features (same as R_list).
Recommended values mirror R_list entries:
- `0.02`: fine-grained, sensitive to local structure
- `0.05`: balanced (default, matches middle R)
- `0.2`: coarse-grained, captures global density

---

## Hardware Notes

AWD post-training has similar per-step cost to base training (extra KDE reward
computation is minor relative to generator forward/backward pass).

| Configuration | Feasibility | Notes |
|---|---|---|
| 8x H100 (80GB) | Recommended | Same as base Drift-L training |
| 4x H100 (80GB) | Feasible | Reduce group size and eval batch |
| TPU v4-8 | Feasible | Reduce `--eval-batch-size` for inference |

When using fewer hosts than the original training (e.g., DDP on 1 node = 8 hosts
vs. 32 hosts for SOTA), increase `push_per_step` to maintain memory bank update
rate.

---

## Evaluation Targets

| Metric | Baseline (Drift-L) | AWD Target |
|---|---|---|
| FID-50K | 1.54 | 1.45–1.50 |
| Inception Score | 260.1 | > 260.1 |
| Precision | baseline | maintain or improve |
| Recall | baseline | maintain or improve |

Track FID at checkpoints (2k, 4k, 6k, ..., 20k steps) to identify optimal
stopping point. Best checkpoint is typically earlier than total_steps.

---

## Baselines for Comparison

1. **Standard Drift-L** — `hf://latent_L_sota`, FID 1.54
2. **Extended standard training** — Continue base drifting for equivalent compute
3. **Larger KDE batch only** — Increase `pos_per_sample` without AWD reweighting

---

## References

- Deng et al. (2026). *Generative Modeling via Drifting.* arXiv:2602.04770
- Cao, Wei, Liu (2026). *Gradient Flow Drifting.* arXiv:2603.10592
- Lai, Murata, Nguyen et al. (2026). *A Unified View of Drifting and Score-Based Models.*
- Shao et al. (2024). *DeepSeekMath.* arXiv:2402.03300 (GRPO)
