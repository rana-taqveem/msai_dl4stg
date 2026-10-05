#!/usr/bin/env bash
set -euo pipefail

# Colab-ready validation/ablation runner. Override settings before invocation, for example:
#   SEEDS="0" EXOGS="all" EPOCHS=2 bash run_colab.sh       # quick smoke run
#   SEEDS="0 1 2" EXOGS="all none" bash run_colab.sh       # required ablation

cd "$(dirname "$0")"
mkdir -p runs_colab

SEEDS=${SEEDS:-"0 1 2"}
EXOGS=${EXOGS:-"all none"}
EPOCHS=${EPOCHS:-20}
PATIENCE=${PATIENCE:-5}
TRAIN_STRIDE=${TRAIN_STRIDE:-2}
BATCH=${BATCH:-128}

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. In Colab choose Runtime > Change runtime type > T4 GPU.")
print("GPU:", torch.cuda.get_device_name(0))
PY

for seed in $SEEDS; do
  for exog in $EXOGS; do
    run="runs_colab/${exog}_s${seed}"
    echo "Starting ${run}"
    python autoformer_task2.py \
      --seed "$seed" \
      --exog "$exog" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --train-stride "$TRAIN_STRIDE" \
      --batch "$BATCH" \
      --device cuda \
      --amp \
      --out "$run" 2>&1 | tee "${run}.log"
  done
done

python summarize_runs.py --runs runs_colab
