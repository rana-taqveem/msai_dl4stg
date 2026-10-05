"""Experiment runner, results log and submission builder for Task 2.

Every experiment is a named set of command-line flags for autoformer_task2.py, run once per seed.
Each finished run is appended to <root>/results.csv, and a re-run skips seeds that already finished
(pass force=True to retrain). All paths stored in the logs are relative to <root>, so the whole
folder can be trained on Kaggle, downloaded, and analysed locally.

Layout under the chosen root:
    runs/<experiment>_s<seed>/        validation runs (summary.json, model.pt, val_predictions.npy)
    final/<experiment>_s<seed>/       all-history refits
    submissions/<label>/              averaged forecast, predictions.txt, submission.json (P and E)
    experiments.json                  flags and notes for every experiment name
    results.csv                       one row per validation run
    comparison.csv                    one row per experiment (written by compare())
    submissions.csv, leaderboard.csv  submission bookkeeping
"""
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import autoformer_task2 as af

HERE = Path(__file__).resolve().parent
ROOT = HERE / "experiments"
DEFAULTS = {"epochs": 20, "patience": 5, "train_stride": 2, "batch": 128, "device": "cuda",
            "amp": True}
HIDDEN_IDX = np.arange(43657, 43825)


def configure(root=None, **defaults):
    """Choose where results are written and change default training settings."""
    global ROOT
    if root is not None:
        ROOT = Path(root)
    DEFAULTS.update(defaults)
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"results root: {ROOT}\ndefaults: {DEFAULTS}")


def _registry():
    path = ROOT / "experiments.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _register(name, flags, notes, settings):
    
    registry = _registry()
    previous = registry.get(name)
    
    if previous and previous["flags"] != flags:
        raise ValueError(f"experiment '{name}' already exists with flags '{previous['flags']}'. "
                         "Use a new name for a different configuration.")
   
    registry[name] = {"flags": flags, "notes": notes or (previous or {}).get("notes", ""),
                      "settings": settings}
    
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments.json").write_text(json.dumps(registry, indent=2))


def _command(flags, settings, seed, out, extra=()):
    command = [sys.executable, str(HERE / "autoformer_task2.py"), "--seed", str(seed)]
    if "--exog" not in flags:
        command += ["--exog", "all"]
    for key in ("epochs", "patience", "train_stride", "batch", "device"):
        command += ["--" + key.replace("_", "-"), str(settings[key])]
    if settings.get("amp"):
        command.append("--amp")
    return command + shlex.split(flags) + list(extra) + ["--out", str(out)]


def _stream(command, log_path, quiet):
    """Run a command, echo its output (only per-epoch summaries if quiet), and keep a log file."""
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in process.stdout:
            log.write(line)
            
            if not quiet:
                print(line, end="")
            elif line.startswith('{"epoch"'):
                row = json.loads(line)
                print(f"  epoch {row['epoch']:>3}  train {row['train_mse_scaled']:.4f}"
                      + (f"  val RMSE {row['RMSE']:.2f}  recent {row['recent_RMSE']:.2f}"
                         if "RMSE" in row else ""), flush=True)
        if process.wait():
            raise RuntimeError(f"run failed (exit {process.returncode}); see {log_path}")


def _record(name, seed, summary, relative_dir):
    best = summary["best_val"]
    row = {"experiment": name, "seed": seed, "run_dir": relative_dir,
           "design": summary["design"], "features": summary["config"].get("features", "basic"),
           "target_transform": summary["config"].get("target_transform", "none"),
           "flags": _registry()[name]["flags"], "params": summary["parameters"],
           "best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"],
           "RMSE": best["RMSE"], "MAE": best["MAE"], "sMAPE": best["sMAPE"],
           "recent_RMSE": best.get("recent_RMSE"), "recent_MAE": best.get("recent_MAE"),
           "raw_negative_count": best.get("raw_negative_count"),
           "train_seconds": best.get("seconds"),
           "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = ROOT / "results.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table):
        table = table[~((table.experiment == name) & (table.seed == seed))]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)
    return row


def run_experiment(name, flags="", seeds=(0, 1, 2), notes="", force=False, quiet=True, **settings):
    
    """Train one validation run per seed and record each result in results.csv."""
    
    settings = {**DEFAULTS, **settings}
    
    _register(name, flags, notes, settings)
    
    """Run an experiment with the given settings and record the results."""
    
    rows = []
    for seed in seeds:
        relative = f"runs/{name}_s{seed}"
        out = ROOT / relative
        
        if (out / "summary.json").exists() and not force:
            print(f"[{name} seed {seed}] already finished; skipping (force=True to retrain)")
        else:
            print(f"[{name} seed {seed}] training: {flags or '(baseline flags)'}", flush=True)
            
            _stream(_command(flags, settings, seed, out), out.with_suffix(".log"), quiet)
        
        summary = json.loads((out / "summary.json").read_text())
        rows.append(_record(name, seed, summary, relative))
        
        print(f"[{name} seed {seed}] P={rows[-1]['params']} best epoch {rows[-1]['best_epoch']}"
              f"/{rows[-1]['epochs_run']}  RMSE {rows[-1]['RMSE']:.2f}  "
              f"recent {rows[-1]['recent_RMSE']:.2f}", flush=True)
    
    return pd.DataFrame(rows)


def results():
    """All recorded validation runs, one row per seed."""
    path = ROOT / "results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _runs_of(name, seeds=None):
    table = results()
    runs = table[table.experiment == name] if len(table) else table
    if seeds is not None:
        runs = runs[runs.seed.isin(seeds)]
    if runs.empty:
        raise ValueError(f"no recorded runs for '{name}'")
    return runs


def _validation_truth():
    raw = pd.read_csv(af.DATA / "student_train.csv")["value"].to_numpy(np.float64)
    n = len(raw)
    origins = np.arange(n - af.VAL_LEN, n - af.PRED_LEN + 1, af.VAL_STRIDE)
    return np.stack([raw[o:o + af.PRED_LEN] for o in origins])


def ensemble_validation(name, seeds=None, recent_blocks=24):
    """Score the average of an experiment's seed forecasts on the validation blocks."""
    runs = _runs_of(name, seeds)
    predictions = [np.load(ROOT / r / "val_predictions.npy") for r in runs.run_dir]
    truth = _validation_truth()
    average = np.mean(predictions, axis=0)
    recent = slice(-recent_blocks, None)
    return {"members": len(predictions),
            **{f"ens_{k}": v for k, v in af.metrics(average, truth).items()},
            **{f"ens_recent_{k}": v
               for k, v in af.metrics(average[recent], truth[recent]).items()}}


def compare(names=None, sort_by="RMSE_mean"):
    """One row per experiment: mean +/- std across seeds, plus the seed-ensemble score."""
    table = results()
    if table.empty:
        print("no results yet")
        return table
    if names is not None:
        table = table[table.experiment.isin(names)]
    registry = _registry()
    rows = []
    for name, group in table.groupby("experiment", sort=False):
        row = {"experiment": name, "seeds": len(group), "params": int(group.params.iloc[0])}
        for metric in ("RMSE", "MAE", "sMAPE", "recent_RMSE"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        row["best_epoch_mean"] = group.best_epoch.mean()
        row["epochs_run_total"] = int(group.epochs_run.sum())
        if len(group) > 1:
            ensemble = ensemble_validation(name)
            row["ens_RMSE"] = ensemble["ens_RMSE"]
            row["ens_recent_RMSE"] = ensemble["ens_recent_RMSE"]
        row["flags"] = registry.get(name, {}).get("flags", "")
        row["notes"] = registry.get(name, {}).get("notes", "")
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(sort_by).reset_index(drop=True)
    summary.to_csv(ROOT / "comparison.csv", index=False)
    return summary


def refit_experiment(name, seeds=None, force=False, quiet=True):
    """Refit each seed of an experiment on all history for its own best epoch count.

    Declared epochs per member = validation epochs actually run + refit epochs, as the rules require.
    """
    registry = _registry()[name]
    settings = {**DEFAULTS, **registry["settings"]}
    for _, run in _runs_of(name, seeds).iterrows():
        out = ROOT / "final" / f"{name}_s{run.seed}"
        if (out / "summary.json").exists() and not force:
            print(f"[final {name} seed {run.seed}] already refit; skipping")
            continue
        print(f"[final {name} seed {run.seed}] refit for {run.best_epoch} epochs "
              f"(+{run.epochs_run} selection epochs)", flush=True)
        extra = ["--final", "--selection-epochs", str(run.epochs_run)]
        command = _command(registry["flags"], {**settings, "epochs": int(run.best_epoch)},
                           int(run.seed), out, extra)
        _stream(command, out.with_suffix(".log"), quiet)


def _write_submission(label, experiment, kind, forecasts, members, validation=None):
    """Average member forecasts, write paste-ready predictions, and log P and E."""
    average = np.mean(forecasts, axis=0)
    P = int(sum(m["parameters"] for m in members))
    E = int(sum(m["declared_epochs"] for m in members))
    out = ROOT / "submissions" / label
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_idx": HIDDEN_IDX, "value": average}).to_csv(out / "forecast.csv",
                                                                     index=False)
    text = ", ".join(f"{v:.4f}" for v in average)
    (out / "predictions.txt").write_text(text)
    info = {"label": label, "experiment": experiment, "kind": kind, "members": len(members),
            "declared_parameters_P": P, "declared_epochs_E": E,
            **(validation or {}),
            "forecast_mean": float(average.mean()), "forecast_min": float(average.min()),
            "forecast_max": float(average.max()), "zeros": int((average == 0).sum()),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "submission.json").write_text(json.dumps({**info, "per_member": members}, indent=2))
    path = ROOT / "submissions.csv"
    log = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(log):
        log = log[log.label != label]
    pd.concat([log, pd.DataFrame([info])], ignore_index=True).to_csv(path, index=False)
    print(f"[{label}] {len(members)} member(s): declare P = {P}, E = {E}; "
          f"mean {average.mean():.1f} min {average.min():.1f} max {average.max():.1f}")
    return text, info


def checkpoint_submission(name, seeds=None):
    """Average forecasts from the saved validation checkpoints (no refit, no extra training).

    Each member forecasts from the last observed 168 steps with the scaling it was trained with.
    Declared epochs are the validation epochs actually run. Its validation estimate is exactly the
    experiment's seed-ensemble score, because these are the same models.
    """
    n = len(pd.read_csv(af.DATA / "student_train.csv"))
    forecasts, members = [], []
    for _, run in _runs_of(name, seeds).iterrows():
        summary = json.loads((ROOT / run.run_dir / "summary.json").read_text())
        config = summary["config"]
        target, covariates, scaler, _ = af.load_for_config(config, n - af.VAL_LEN)
        model = af.model_from_config(covariates.shape[1], config)
        model.load_state_dict(torch.load(ROOT / run.run_dir / "model.pt", map_location="cpu",
                                         weights_only=True))
        forecasts.append(af.predict(model, target, covariates, np.array([n]), config["seq_len"],
                                    config["label_len"], scaler, device=torch.device("cpu"))[0])
        members.append({"seed": int(run.seed), "parameters": summary["parameters"],
                        "declared_epochs": summary["epochs_run"]})
    validation = ensemble_validation(name, seeds) if len(members) > 1 else None
    return _write_submission(f"{name}__checkpoint", name, "checkpoint", forecasts, members,
                             validation)


def refit_submission(name, seeds=None):
    """Average the all-history refit forecasts of an experiment (run refit_experiment first)."""
    forecasts, members = [], []
    for _, run in _runs_of(name, seeds).iterrows():
        run_dir = ROOT / "final" / f"{name}_s{run.seed}"
        if not (run_dir / "summary.json").exists():
            raise ValueError(f"missing refit {run_dir}; run refit_experiment('{name}') first")
        summary = json.loads((run_dir / "summary.json").read_text())
        forecast = pd.read_csv(run_dir / "forecast.csv")
        assert forecast.time_idx.tolist() == HIDDEN_IDX.tolist(), run_dir
        forecasts.append(forecast.value.to_numpy())
        members.append({"seed": int(run.seed), "parameters": summary["parameters"],
                        "declared_epochs": summary["declared_epochs"]})
    validation = ensemble_validation(name, seeds) if len(members) > 1 else None
    return _write_submission(f"{name}__refit", name, "refit", forecasts, members, validation)


def submissions():
    """Every submission candidate built so far, with its validation estimate and P / E."""
    path = ROOT / "submissions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def record_leaderboard(attempt, label, rmse, mae, smape, params, epochs, note=""):
    """Keep your own log of what each leaderboard attempt actually contained."""
    path = ROOT / "leaderboard.csv"
    log = pd.read_csv(path) if path.exists() else pd.DataFrame()
    row = {"attempt": attempt, "submission": label, "RMSE": rmse, "MAE": mae,
           "sMAPE": smape, "P": params, "E": epochs, "note": note,
           "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if len(log):
        log = log[log.attempt != attempt]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True).sort_values("attempt")
    log.to_csv(path, index=False)
    return log
