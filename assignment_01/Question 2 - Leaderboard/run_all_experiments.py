"""Run the whole Task 2 experiment plan in one unattended session (e.g. Kaggle "Save & Run All").

Stages (every result is recorded as soon as it exists, so a crash or timeout loses only the run in
progress; re-running the same command skips everything already finished):
  1. fixed experiments    E0 references, E1 exog trend, E2a/E2 engineered features, E3 log target
  2. adaptive             E4 anchor, E5 anchor + direct head, on top of the better of E2 / E3
  3. adaptive             E6 wider model, E7 336-step context, on top of the best of E1-E5
  4. extra seeds          for the top-k experiments, so the final choice rests on more runs
  5. checkpoint forecasts every experiment's seed-averaged forecast from its validation models
  6. refits              the top-k experiments refit on all history, seed-averaged
Nothing is submitted anywhere; candidates are written to <root>/submissions/ for local selection.

    python run_all_experiments.py --root /kaggle/working/task2_results
    python run_all_experiments.py --root experiments_smoke --smoke --device cpu --no-amp
"""
import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import task2_experiments as tx

FIXED = [
    ("E0_baseline", "", "original Autoformer, all external variables"),
    ("E0_none", "--exog none", "ablation: no external variables"),
    ("E0_direct_level", "--direct-exog --level-residual", "previous best design"),
    ("E1_exogtrend", "--trend-init exog", "covariate-conditioned decoder trend start"),
    ("E2a_features_only", "--features engineered", "engineered features, original architecture"),
    ("E2_exogtrend_feat", "--trend-init exog --features engineered",
     "exog trend start + engineered features"),
    ("E3_log", "--trend-init exog --features engineered --target-transform log1p",
     "E2 + log1p target with smearing"),
]


class Plan:
    def __init__(self, args):
        self.args = args
        self.started = time.time()
        self.log_path = Path(args.root) / "plan_log.json"
        self.log = {"args": vars(args), "events": [], "failures": []}

    def hours(self):
        return (time.time() - self.started) / 3600

    def note(self, message, **extra):
        event = {"t_hours": round(self.hours(), 3), "message": message, **extra}
        self.log["events"].append(event)
        print(f"\n=== [{event['t_hours']:.2f} h] {message}", flush=True)
        self.log_path.write_text(json.dumps(self.log, indent=2))

    def out_of_time(self):
        return self.hours() > self.args.max_hours

    def attempt(self, label, function, *args, **kwargs):
        """Run one step; record a failure and carry on instead of stopping the whole session."""
        try:
            return function(*args, **kwargs)
        
        except Exception as error:      # keep the session (and its GPU hours) productive
            self.log["failures"].append({"step": label, "error": repr(error),
                                         "traceback": traceback.format_exc()})
            self.note(f"FAILED {label}: {error!r}")
            return None

    def experiment(self, name, flags, notes, seeds=None):
        
        if self.out_of_time():
            self.note(f"skipping {name}: time budget of {self.args.max_hours} h reached")
            return
        
        self.note(f"experiment {name}", flags=flags)
        self.attempt(name, tx.run_experiment, name, flags, seeds=seeds or self.args.seeds, notes=notes)

    def best(self, names):
        table = tx.compare()
        table = table[table.experiment.isin(names)] if len(table) else table
        return None if table.empty else table.iloc[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--extra-seeds", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--top-k", type=int, default=3, help="experiments that get extra seeds and an all-history refit")
    parser.add_argument("--max-hours", type=float, default=6.0, help="stop starting new training after this long; still build outputs")
    parser.add_argument("--smoke", action="store_true", help="tiny settings to check the whole plan end to end")
    args = parser.parse_args(argv)
    
    if args.smoke:
        args.epochs, args.patience, args.train_stride = 1, 1, 64
        args.seeds, args.extra_seeds, args.top_k = [0, 1], [2], 2

    tx.configure(args.root, epochs=args.epochs, patience=args.patience,
                 train_stride=args.train_stride, batch=args.batch, device=args.device,
                 amp=not args.no_amp)
    
    plan = Plan(args)
    plan.note("plan started")

    for name, flags, notes in FIXED:
        plan.experiment(name, flags, notes)

    base = plan.best(["E2_exogtrend_feat", "E3_log"])
    
    if base is not None:
        plan.note(f"stage 2 builds on {base.experiment} (mean RMSE {base.RMSE_mean:.2f})")
        plan.experiment("E4_anchor", base["flags"] + " --anchor",
                        f"{base.experiment} + decaying anchor to last value")
        plan.experiment("E5_anchor_direct", base["flags"] + " --anchor --direct-exog",
                        f"{base.experiment} + anchor + direct per-step covariate head")

    best = plan.best(["E1_exogtrend", "E2_exogtrend_feat", "E3_log", "E4_anchor",
                      "E5_anchor_direct"])
    if best is not None:
        plan.note(f"stage 3 builds on {best.experiment} (mean RMSE {best.RMSE_mean:.2f})")
        plan.experiment("E6_wider", best["flags"] + " --d-model 64 --d-ff 128",
                        f"{best.experiment} with d_model 64, d_ff 128")
        plan.experiment("E7_ctx336", best["flags"] + " --seq-len 336",
                        f"{best.experiment} with a 336-step context")

    table = tx.compare()
    top = list(table.experiment[:args.top_k]) if len(table) else []
    plan.note(f"top {args.top_k} by mean validation RMSE: {top}")
    if args.extra_seeds:
        registry = tx._registry()
        for name in top:
            plan.experiment(name, registry[name]["flags"], registry[name]["notes"],
                            seeds=args.extra_seeds)

    plan.note("building checkpoint (no-refit) submissions for every experiment")
    for name in (tx.compare().experiment if len(tx.results()) else []):
        plan.attempt(f"checkpoint {name}", tx.checkpoint_submission, name)

    for name in top:
        if plan.out_of_time():
            plan.note(f"skipping refit of {name}: time budget reached")
            continue
        plan.note(f"refitting {name} on all history")
        failures_before = len(plan.log["failures"])
        plan.attempt(f"refit {name}", tx.refit_experiment, name)
        if len(plan.log["failures"]) == failures_before:
            plan.attempt(f"refit submission {name}", tx.refit_submission, name)

    comparison = tx.compare()
    plan.note("plan finished", total_hours=round(plan.hours(), 2),
              failures=len(plan.log["failures"]))
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(tx.submissions().to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    archive = shutil.make_archive(str(Path(args.root).resolve()), "zip", args.root)
    print(f"\nEverything is in {args.root}; zipped copy: {archive}")


if __name__ == "__main__":
    main()
