#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

for task in mre_mi mner_mi_plus; do
  config="configs/saver_${task}.yaml"
  for seed in 42 43 44 45 46; do
    python -u run_saver.py \
      --config "${config}" \
      --mode train \
      --seed "${seed}" \
      --output-dir "result/saver_${task}/seed_${seed}"
  done
done
