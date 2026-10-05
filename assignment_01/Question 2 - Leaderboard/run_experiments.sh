#!/usr/bin/env bash
# Ablation: with vs without the optional external file, 3 seeds each, chronological validation.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
for seed in 0 1 2; do
  for exog in all none; do
    $PY autoformer_task2.py --seed $seed --exog $exog --epochs 6 --patience 2 --train-stride 4 \
        --out runs/${exog}_s${seed} > runs/${exog}_s${seed}.log 2>&1
  done
done
echo done > runs/ALL_DONE
