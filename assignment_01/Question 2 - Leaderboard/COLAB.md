# Task 2 on Google Colab

## 1. Start a GPU runtime

In Colab select **Runtime > Change runtime type > T4 GPU**. Then verify:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
```

## 2. Clone and enter the repository

```python
!git clone https://github.com/rana-taqveem/msai_dl4stg.git
%cd "/content/msai_dl4stg/assignment_01/Question 2 - Leaderboard"
```

If the repository is private, use Colab's GitHub integration or upload a ZIP to Drive. Do not put a
GitHub token directly in a notebook cell.

## 3. Confirm the data

```python
import pandas as pd

train = pd.read_csv("Data/student_train.csv")
test = pd.read_csv("Data/student_test.csv")
external = pd.read_csv("Data/optional_external_data.csv", skipinitialspace=True)

print(len(train), train.time_idx.min(), train.time_idx.max())
print(len(test), test.time_idx.min(), test.time_idx.max(), test.value.isna().sum())
print(len(external), external.time_idx.min(), external.time_idx.max())
```

Expected ranges are train `1..43656`, test `43657..43824`, and external `1..43824`.

## 4. Run a short smoke test

This confirms CUDA, mixed precision, the data path, and checkpoint writing before launching long runs.

```python
!SEEDS="0" EXOGS="all" EPOCHS=1 PATIENCE=1 TRAIN_STRIDE=16 BATCH=128 bash run_colab.sh
```

The first log line must contain `"device": "cuda"`, `"amp": true`, and
`"future_covariates_in_decoder": true`.

## 5. Run the required three-seed external-data ablation

```python
!SEEDS="0 1 2" EXOGS="all none" EPOCHS=20 PATIENCE=5 TRAIN_STRIDE=2 BATCH=128 bash run_colab.sh
```

If a T4 runs out of memory, change `BATCH=128` to `BATCH=64`. Batch size does not alter the model's
parameter count.

## 6. Select the best epoch and refit on all history

First inspect the summary:

```python
!python summarize_runs.py --runs runs_colab
```

Suppose the selected all-external run has `best_epoch = 8` and its `epochs_run = 12`. Refit seed 0
for eight epochs and record the 12 selection epochs:

```python
!python autoformer_task2.py --seed 0 --exog all --final --epochs 8 \
  --selection-epochs 12 --train-stride 2 --batch 128 --device cuda --amp \
  --out runs_colab/final_all_s0
```

Replace the seed and epoch count with the values supported by validation. The final outputs are:

- `runs_colab/final_all_s0/forecast.csv`
- `runs_colab/final_all_s0/predictions.txt`
- `runs_colab/final_all_s0/model.pt`
- `runs_colab/final_all_s0/summary.json`

`summary.json` records `declared_epochs = selection_epochs + final_refit_epochs`. For an ensemble,
count all members' parameters and training epochs according to the assignment rules.

## 7. Preserve results in Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!cp -r runs_colab "/content/drive/MyDrive/AI651_Task2_runs"
```

Copy the run directory periodically because the Colab runtime is temporary.

## 8. Run the controlled improvement experiment

The baseline already sends future A-J values through the decoder. The experiment below tests two
small additions independently before combining them:

- `direct_exog`: a `10 -> 16 -> 1` residual head gives each known future A-J row a direct route to
  its corresponding forecast step.
- `level_residual`: one bounded learned gate uses the difference between the last 24 target values
  and the full 168-step context to correct a recent level shift.
- `direct_level`: enables both additions.

First run a one-seed, one-epoch smoke test:

```python
!SEEDS="0" DESIGNS="direct_level" EPOCHS=1 PATIENCE=1 TRAIN_STRIDE=16 BATCH=128 \
  bash run_improvements_colab.sh
```

Then run the three designs with three seeds:

```python
!SEEDS="0 1 2" DESIGNS="direct_exog level_residual direct_level" \
  EPOCHS=20 PATIENCE=5 TRAIN_STRIDE=2 BATCH=128 bash run_improvements_colab.sh
```

Inspect the grouped result table:

```python
!python summarize_runs.py --runs runs_colab runs_improved
```

Compare each design's three-seed mean and spread against the existing baseline mean RMSE `72.31`.
Choose a design only if the improvement is repeatable across seeds; then choose that design's best
individual seed and epoch for the final refit. For example, if `direct_level_s1` wins at epoch 14:

```python
!python autoformer_task2.py --seed 1 --exog all --direct-exog --level-residual \
  --exog-hidden 16 --level-window 24 --final --epochs 14 --selection-epochs 20 \
  --train-stride 2 --batch 128 --device cuda --amp \
  --out runs_improved/final_direct_level_s1
```

Every final run now also writes `forecast_unclipped.csv`. It is diagnostic only: submit
`forecast.csv`, whose negative values have been clipped to zero. The validation history and final
summary record how often clipping occurred, helping detect a model that only appears good because
many invalid negative forecasts were hidden.

Back up the new results to a newly timestamped directory:

```python
!cp -r runs_improved "/content/drive/MyDrive/AI651_Task2_improved_$(date +%Y%m%d_%H%M%S)"
```
