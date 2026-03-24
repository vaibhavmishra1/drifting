# Build 10-class reference stats once (uses all 500 val images per class = 5000 total)
# python -m utils.build_ref_stats \
#   --class-subset 0,1,2,3,4,5,6,7,8,9 \
#   --out data/fid/imagenet_256_10cls_fid_stats.npz \
#   --eval-batch-size 256

python inference.py --init-from /workspace/drifting/runs/latent_B_grpo_scratch/params_ema_step_2000 \
  --cfg-scale 1.0 --num-samples 5000 --eval-batch-size 256 \
  --class-subset 0,1,2,3,4,5,6,7,8,9 \
  --ref-npz data/fid/imagenet_256_10cls_fid_stats.npz \
  --json-out results_finetune_latent_B_grpo_scratch_step_2000.json

python inference.py --init-from /workspace/drifting/runs/latent_B_grpo_scratch/params_ema_step_4000 \
  --cfg-scale 1.0 --num-samples 5000 --eval-batch-size 256 \
  --class-subset 0,1,2,3,4,5,6,7,8,9 \
  --ref-npz data/fid/imagenet_256_10cls_fid_stats.npz \
  --json-out results_finetune_latent_B_grpo_scratch_step_4000.json

python inference.py --init-from /workspace/drifting/runs/latent_B_grpo_scratch/params_ema_step_6000 \
  --cfg-scale 1.0 --num-samples 5000 --eval-batch-size 256 \
  --class-subset 0,1,2,3,4,5,6,7,8,9 \
  --ref-npz data/fid/imagenet_256_10cls_fid_stats.npz \
  --json-out results_finetune_latent_B_grpo_scratch_step_6000.json

python inference.py --init-from /workspace/drifting/runs/latent_B_grpo_scratch/params_ema_step_8000 \
  --cfg-scale 1.0 --num-samples 5000 --eval-batch-size 256 \
  --class-subset 0,1,2,3,4,5,6,7,8,9 \
  --ref-npz data/fid/imagenet_256_10cls_fid_stats.npz \
  --json-out results_finetune_latent_B_grpo_scratch_step_8000.json

python inference.py --init-from /workspace/drifting/runs/latent_B_grpo_scratch/params_ema_step_10000 \
  --cfg-scale 1.0 --num-samples 5000 --eval-batch-size 256 \
  --class-subset 0,1,2,3,4,5,6,7,8,9 \
  --ref-npz data/fid/imagenet_256_10cls_fid_stats.npz \
  --json-out results_finetune_latent_B_grpo_scratch_step_10000.json