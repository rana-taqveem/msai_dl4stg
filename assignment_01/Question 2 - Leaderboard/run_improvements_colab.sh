#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUNS="${RUNS:-runs_improved}"
SEEDS="${SEEDS:-0 1 2}"
DESIGNS="${DESIGNS:-direct_exog level_residual direct_level}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
TRAIN_STRIDE="${TRAIN_STRIDE:-2}"
BATCH="${BATCH:-128}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))"
mkdir -p "$RUNS"

for design in $DESIGNS; do
  flags=""
  case "$design" in
    direct_exog) flags="--direct-exog" ;;
    level_residual) flags="--level-residual" ;;
    direct_level) flags="--direct-exog --level-residual" ;;
    *) echo "Unknown design: $design"; exit 2 ;;
  esac
  for seed in $SEEDS; do
    out="$RUNS/${design}_s${seed}"
    echo "Starting $out"
    python autoformer_task2.py --seed "$seed" --exog all --epochs "$EPOCHS" \
      --patience "$PATIENCE" --train-stride "$TRAIN_STRIDE" --batch "$BATCH" \
      --device cuda --amp --exog-hidden 16 --level-window 24 $flags --out "$out" \
      2>&1 | tee "$out.log"
  done
done

python summarize_runs.py --runs "$RUNS"
