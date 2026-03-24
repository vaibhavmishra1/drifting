python inference.py --init-from runs/finetune_latent_L_baseline_full/params_ema_step_4000 \
  --cfg-scale 1.0 --num-samples 49900 --eval-batch-size 256 \
  --json-out results_finetune_latent_L_baseline_full_step_4000.json

python inference.py --init-from runs/finetune_latent_L_baseline_full/params_ema_step_16000 \
  --cfg-scale 1.0 --num-samples 49900 --eval-batch-size 256 \
  --json-out results_finetune_latent_L_baseline_full_step_16000.json

python inference.py --init-from runs/finetune_latent_L_grpo_full/params_ema_step_4000 \
  --cfg-scale 1.0 --num-samples 49900 --eval-batch-size 256 \
  --json-out results_finetune_latent_L_grpo_full_step_4000.json

python inference.py --init-from runs/finetune_latent_L_grpo_full/params_ema_step_12000 \
  --cfg-scale 1.0 --num-samples 49900 --eval-batch-size 256 \
  --json-out results_finetune_latent_L_grpo_full_step_12000.json

python inference.py --init-from runs/finetune_latent_L_grpo_full/params_ema_step_16000 \
  --cfg-scale 1.0 --num-samples 49900 --eval-batch-size 256 \
  --json-out results_finetune_latent_L_grpo_full_step_16000.json