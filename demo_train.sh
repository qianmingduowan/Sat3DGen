#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-Please_input_the_vigor_dataset_path}"
JOB_NAME="${JOB_NAME:-Please_set_the_job_name}"
OUT_BASE_DIR="${OUT_BASE_DIR:-./output}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
DATA_TXT="${DATA_TXT:-${DATA_DIR}/train__corrected_all_3city_remove_building.txt}"
CONF_DIR="${CONF_DIR:-configs/Sat3DGen_dino_v3.json}"
OUT_DIR="${OUT_BASE_DIR}/${JOB_NAME}"

accelerate launch --num_processes "${NUM_PROCESSES}" train.py \
  --train_data_dir "${DATA_DIR}" \
  --data_txt "${DATA_TXT}" \
  --output_dir "${OUT_DIR}" \
  --model_config_name_or_path "${CONF_DIR}" \
  --gradient_accumulation_steps 1 \
  --report_to tensorboard \
  --mixed_precision no \
  --checkpoints_total_limit 5 \
  --dataloader_num_workers 12 \
  --num_train_epochs 32 \
  --checkpointing_steps 100000 \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --use_ema \
  --adam_beta1 0.0 \
  --adam_beta2 0.9
