#!/usr/bin/env bash
set -euo pipefail

SAMPLES_JSON="datasets/all_tasks_samples_5pct.json"

python scripts/make_samples.py \
  --output "$SAMPLES_JSON" \
  --pct 0.05 \
  --min_cap 10 \
  --max_cap 1000 \
  --seed 42 \
  --tasks "*" \
  --include_path lm-evaluation-harness/lm_eval/tasks

lm_eval \
  --model sglang \
  --model_args pretrained=/shared/huggingface/models/Qwen2.5-7B-Instruct,device=cuda,tp_size=1,dp_size=1,max_model_len=32768,trust_remote_code=True \
  --tasks "*" \
  --batch_size auto \
  --output output/qwen25_sglang \
  --log_samples \
  --samples "$SAMPLES_JSON"