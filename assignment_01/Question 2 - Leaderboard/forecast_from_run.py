"""Forecast the hidden 168 steps with a model saved by a validation run (no further training).

The run's scaling is re-created exactly (statistics from positions before the validation start),
the context is the last 168 observed steps, and the external variables over the hidden horizon
enter through the decoder covariates.

    python forecast_from_run.py runs/all_s1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import autoformer_task2 as af

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
parser.add_argument("--amp", action="store_true")
args = parser.parse_args()

run = args.run
summary = json.loads((run / "summary.json").read_text())
config = summary["config"]
device = af.resolve_device(args.device)
use_amp = bool(args.amp and device.type == "cuda")
n_history = len(pd.read_csv(af.DATA / "student_train.csv"))
train_end = n_history - af.VAL_LEN                       # same scaling statistics as during that run
target, covariates, scaler, _ = af.load_for_config(config, train_end)
model = af.model_from_config(covariates.shape[1], config).to(device)
model.load_state_dict(torch.load(run / "model.pt", map_location=device, weights_only=True))
raw_forecast = af.predict(model, target, covariates, np.array([n_history]), config["seq_len"],
                          config["label_len"], scaler, device=device, amp=use_amp, clip=False)[0]
forecast = np.clip(raw_forecast, 0, None)
pd.DataFrame({"time_idx": np.arange(n_history + 1, n_history + af.PRED_LEN + 1),
              "value": forecast}).to_csv(run / "forecast.csv", index=False)
pd.DataFrame({"time_idx": np.arange(n_history + 1, n_history + af.PRED_LEN + 1),
              "value": raw_forecast}).to_csv(run / "forecast_unclipped.csv", index=False)
(run / "predictions.txt").write_text(", ".join(f"{v:.4f}" for v in forecast))
print(f"{len(forecast)} values, time_idx {n_history + 1}..{n_history + af.PRED_LEN}; "
      f"P={summary['parameters']} E={summary['epochs_run']}; "
      f"device={device} amp={use_amp}; "
      f"raw_negatives={(raw_forecast < 0).sum()} clipped_zeros={(forecast == 0).sum()}; "
      f"min {forecast.min():.1f} mean {forecast.mean():.1f} max {forecast.max():.1f}")
