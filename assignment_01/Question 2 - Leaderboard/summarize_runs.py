"""Summarise validation runs by design and external-data setting."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=Path, nargs="+",
                    default=[Path(__file__).resolve().parent / "runs"],
                    help="one or more run directories")
args = parser.parse_args()
rows = []
for run_dir in args.runs:
    for path in sorted(run_dir.glob("*/summary.json")):
        s = json.loads(path.read_text())
        if s.get("final"):
            continue
        best = s["best_val"]
        rows.append({"run": path.parent.name, "design": s.get("design", "baseline"),
                     "exog": s["exog"], "seed": s["seed"], "params": s["parameters"],
                     "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
                     "RMSE": best["RMSE"], "MAE": best["MAE"], "sMAPE": best["sMAPE"],
                     "recent_RMSE": best.get("recent_RMSE", float("nan"))})

print(f"{'run':<24}{'design':<16}{'exog':<6}{'seed':>5}{'params':>8}{'best_ep':>9}{'ran':>5}{'RMSE':>9}{'MAE':>9}{'sMAPE':>9}{'recent':>9}")
for r in rows:
    print(f"{r['run']:<24}{r['design']:<16}{r['exog']:<6}{r['seed']:>5}{r['params']:>8}{r['best_epoch']:>9}{r['epochs_run']:>5}"
          f"{r['RMSE']:>9.2f}{r['MAE']:>9.2f}{r['sMAPE']:>9.2f}{r['recent_RMSE']:>9.2f}")
print()
for design, exog in sorted({(r["design"], r["exog"]) for r in rows}):
    group = [r for r in rows if r["design"] == design and r["exog"] == exog]
    if not group:
        continue
    line = f"design={design:<16} exog={exog:<5} n={len(group)}"
    for metric in ("RMSE", "MAE", "sMAPE"):
        values = np.array([r[metric] for r in group])
        line += f"  {metric} {values.mean():.2f} +/- {values.std(ddof=1) if len(values) > 1 else 0:.2f}"
    print(line)
print("\nnaive baselines on the same 176 validation blocks (RMSE / MAE / sMAPE):")
print("  training mean 88.7 / 68.5 / 79.5   persistence 124.1 / 86.5 / 88.2   seasonal-naive-24 119.6 / 85.4 / 93.6")
