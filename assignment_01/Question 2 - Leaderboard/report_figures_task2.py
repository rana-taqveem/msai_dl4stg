"""Build the Task 2 report figures and print every number the report's tables use.

Reads the Kaggle experiment folder (new_results/) and the data only; trains nothing.

    python report_figures_task2.py --results new_results --out ../output/pdf/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import autoformer_task2 as af
import task2_experiments as tx

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def save(fig, out, name):
    fig.savefig(out / name)
    plt.close(fig)
    print(f"saved {out / name}")


def members(name):
    runs = tx.results().query("experiment == @name")
    return [np.load(tx.ROOT / d / "val_predictions.npy") for d in runs.run_dir]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("new_results"))
    parser.add_argument("--out", type=Path, default=Path("../output/pdf/figures"))
    args = parser.parse_args()
    tx.ROOT = args.results
    args.out.mkdir(parents=True, exist_ok=True)

    y = pd.read_csv(af.DATA / "student_train.csv")["value"].to_numpy(float)
    n = len(y)
    val_start = n - af.VAL_LEN
    truth = tx._validation_truth()
    origins = np.arange(val_start, n - af.PRED_LEN + 1, af.VAL_STRIDE)
    numbers = {}

    # ---------------------------------------------------------------- data properties
    centred = y - y.mean()
    spectrum = np.fft.rfft(centred, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:337]
    acf = acf / acf[0]
    numbers["data"] = {"n": n, "min": y.min(), "median": float(np.median(y)), "mean": y.mean(),
                       "p99": float(np.percentile(y, 99)), "max": y.max(),
                       **{f"acf_{k}": float(acf[k]) for k in (1, 6, 12, 24, 48, 168, 336)}}
    train_mean = y[:val_start].mean()
    naive = {
        "Training mean": np.full_like(truth, train_mean),
        "Persistence": np.stack([np.full(af.PRED_LEN, y[o - 1]) for o in origins]),
        "Seasonal naive (24)": np.stack([np.tile(y[o - 24:o], 7) for o in origins]),
        "Seasonal naive (168)": np.stack([y[o - 168:o] for o in origins]),
    }
    numbers["naive"] = {k: af.metrics(v, truth) for k, v in naive.items()}

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    left.hist(y, bins=np.arange(0, 1001, 10), color=BLUE, edgecolor="white", linewidth=0.3)
    left.set_yscale("log")
    left.set_xlabel("observed value")
    left.set_ylabel("count (log scale)")
    left.set_title("(a) Heavy right tail")
    for q, label in ((np.median(y), "median 73"), (np.percentile(y, 99), "99th pct 418")):
        left.axvline(q, color=MUTED, ls=":", lw=1)
        left.text(q + 12, left.get_ylim()[1] * 0.3, label, color=MUTED, fontsize=7)
    right.plot(np.arange(337), acf, color=BLUE)
    for lag in (24, 168):
        right.axvline(lag, color=MUTED, ls=":", lw=1)
        right.text(lag + 4, 0.85, f"lag {lag}: {acf[lag]:.2f}", color=MUTED, fontsize=7)
    right.axhline(0, color=MUTED, lw=0.8)
    right.set_xlabel("lag (steps)")
    right.set_ylabel("autocorrelation")
    right.set_title("(b) Short memory, weak 24-step rhythm")
    save(fig, args.out, "T2.1-1.pdf")

    # ---------------------------------------------------------------- results tables
    results = tx.results()
    comparison = tx.compare()
    numbers["comparison"] = comparison.drop(columns=["notes"]).to_dict("records")
    numbers["per_seed"] = results[["experiment", "seed", "params", "best_epoch", "epochs_run",
                                   "RMSE", "MAE", "sMAPE", "recent_RMSE"]].to_dict("records")
    for name in comparison.experiment:
        numbers.setdefault("ensembles", {})[name] = tx.ensemble_validation(name)

    # ---------------------------------------------------------------- external-file ablation
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    pairs = {}
    for seed in (0, 1, 2):
        without = results.query("experiment == 'E0_none' and seed == @seed").RMSE.item()
        with_ = results.query("experiment == 'E0_baseline' and seed == @seed").RMSE.item()
        pairs[seed] = (without, with_)
        ax.plot([0, 1], [without, with_], color=BLUE, marker="o", ms=5, lw=1.2, alpha=0.8)
        ax.text(1.05, with_ + (1.2 if seed == 1 else -1.2 if seed == 2 else 0), f"seed {seed}",
                va="center", fontsize=7, color=MUTED)
    numbers["ablation_pairs"] = pairs
    ax.axhline(numbers["naive"]["Training mean"]["RMSE"], color=ORANGE, ls="--", lw=1.2)
    ax.text(-0.27, numbers["naive"]["Training mean"]["RMSE"] - 2.2, "training-mean\nforecast",
            color=ORANGE, ha="left", va="top", fontsize=7)
    ax.set_xticks([0, 1], ["without file\n(E0_none)", "with file\n(E0_baseline)"])
    ax.set_xlim(-0.3, 1.45)
    ax.set_ylabel("validation RMSE (176 blocks)")
    ax.set_title("Same architecture, same seeds")
    save(fig, args.out, "T2.2-1.pdf")

    # ---------------------------------------------------------------- design ladder
    order = comparison.sort_values("RMSE_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ypos = np.arange(len(order))
    chosen = order.experiment == "E7_ctx336"
    ax.errorbar(order.RMSE_mean, ypos, xerr=order.RMSE_std, fmt="none", ecolor=BLUE,
                elinewidth=1.4, capsize=3)
    ax.scatter(order.RMSE_mean[~chosen], ypos[~chosen], s=36, color=BLUE, zorder=3,
               edgecolor="white", linewidth=1, label="mean ± SD over seeds")
    ax.scatter(order.RMSE_mean[chosen], ypos[chosen], s=60, color=BLUE, zorder=3,
               edgecolor=INK, linewidth=1.2)
    ax.scatter(order.ens_RMSE, ypos, s=40, marker="D", color=ORANGE, zorder=3,
               edgecolor="white", linewidth=1, label="average of the seeds' forecasts")
    ax.axvline(numbers["naive"]["Training mean"]["RMSE"], color=MUTED, ls=":", lw=1)
    ax.text(numbers["naive"]["Training mean"]["RMSE"] - 0.5, len(order) - 0.6,
            "training mean", color=MUTED, fontsize=7, ha="right")
    labels = [f"{e}  ({s} seeds)" for e, s in zip(order.experiment, order.seeds)]
    ax.set_yticks(ypos, labels)
    ax.get_yticklabels()[int(np.flatnonzero(chosen)[0])].set_fontweight("bold")
    ax.set_xlabel("validation RMSE (176 blocks, lower is better)")
    ax.legend(loc=(0.47, 0.42), frameon=False)
    ax.grid(axis="y", visible=False)
    save(fig, args.out, "T2.3-1.pdf")

    # ---------------------------------------------------------------- error by horizon
    series = [("E0_none (no file)", np.mean(members("E0_none"), 0), ORANGE, "s"),
              ("E0_baseline", np.mean(members("E0_baseline"), 0), AQUA, "^"),
              ("E7_ctx336 (submitted)", np.mean(members("E7_ctx336"), 0), BLUE, "o")]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    horizon = np.arange(1, af.PRED_LEN + 1)
    by_horizon = {}
    for label, prediction, colour, marker in series:
        curve = np.sqrt(((prediction - truth) ** 2).mean(0))
        by_horizon[label] = curve
        ax.plot(horizon, curve, color=colour, marker=marker, markevery=24, ms=5)
        ax.text(170, curve[-12:].mean(), label, color=INK, fontsize=7, va="center")
    curve = np.sqrt(((naive["Training mean"] - truth) ** 2).mean(0))
    by_horizon["Training mean"] = curve
    ax.plot(horizon, curve, color=MUTED, ls="--", lw=1.2)
    ax.text(170, curve[-12:].mean(), "training mean", color=MUTED, fontsize=7, va="center")
    numbers["by_horizon"] = {k: {h: float(v[h - 1]) for h in (1, 6, 24, 72, 168)}
                             for k, v in by_horizon.items()}
    ax.set_xlim(0, 215)
    ax.set_xticks([1, 24, 48, 72, 96, 120, 144, 168])
    ax.set_xlabel("forecast step ahead")
    ax.set_ylabel("validation RMSE")
    ax.set_title("Seed-averaged forecasts, RMSE at each step of the 168-step horizon")
    save(fig, args.out, "T2.4-1.pdf")

    # ---------------------------------------------------------------- submitted forecast
    runs = results.query("experiment == 'E7_ctx336'")
    member_forecasts = []
    for run_dir in runs.run_dir:
        summary = json.loads((tx.ROOT / run_dir / "summary.json").read_text())
        config = summary["config"]
        target, covariates, scaler, _ = af.load_for_config(config, val_start)
        model = af.model_from_config(covariates.shape[1], config)
        model.load_state_dict(torch.load(tx.ROOT / run_dir / "model.pt", map_location="cpu",
                                         weights_only=True))
        member_forecasts.append(af.predict(model, target, covariates, np.array([n]),
                                           config["seq_len"], config["label_len"], scaler,
                                           device=torch.device("cpu"))[0])
    member_forecasts = np.array(member_forecasts)
    submitted = pd.read_csv(tx.ROOT / "submissions/E7_ctx336__checkpoint/forecast.csv").value
    assert np.allclose(member_forecasts.mean(0), submitted, atol=1e-3)
    numbers["submitted"] = {"mean": float(submitted.mean()), "min": float(submitted.min()),
                            "max": float(submitted.max()), "last168_mean": float(y[-168:].mean()),
                            "member_spread_mean": float((member_forecasts.max(0)
                                                         - member_forecasts.min(0)).mean())}
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    history = np.arange(n - 335, n + 1)
    future = np.arange(n + 1, n + af.PRED_LEN + 1)
    ax.plot(history, y[-336:], color=MUTED, lw=1.1)
    ax.text(history[5], 200, "observed\n(last 336 steps)", color=MUTED, fontsize=7)
    ax.fill_between(future, member_forecasts.min(0), member_forecasts.max(0), color=BLUE,
                    alpha=0.18, linewidth=0)
    ax.plot(future, submitted, color=BLUE)
    ax.text(future[70], 255, "submitted forecast (mean of 5 seeds)\n"
            "band = range of the 5 members", color=INK, fontsize=7)
    ax.axvline(n + 0.5, color=INK, ls=":", lw=1)
    ax.text(n - 3, ax.get_ylim()[1] * 0.97, "forecast origin", fontsize=7, va="top", ha="right")
    ax.set_xlabel("time_idx")
    ax.set_ylabel("value")
    save(fig, args.out, "T2.5-1.pdf")

    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
