#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-lerobot-train}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/cup_to_sink}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/cup_to_sink_diffusion}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"

"${LEROBOT_TRAIN}" \
  --dataset.repo_id=yxzhan/cup_to_sink \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --policy.resize_shape='[128,128]' \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq=20000 \
  --num_workers="${NUM_WORKERS}" \
  --wandb.enable=false \
  --output_dir="${OUTPUT_DIR}"
