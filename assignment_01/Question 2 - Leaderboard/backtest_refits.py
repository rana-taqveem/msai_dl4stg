"""Refit selected validation configurations at chronological cutoffs; score known futures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import autoformer_task2 as af


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True,
                        help='individual validation run folders, not their parent')
    parser.add_argument('--out', type=Path, default=Path('backtests'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    truth = pd.read_csv(af.DATA / 'student_train.csv').value.to_numpy()
    # Separate target weeks across the validation tail. These are diagnostic:
    # the configurations were already selected using this validation region.
    cutoffs = [len(truth) - weeks * af.PRED_LEN for weeks in (13, 5, 1)]
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        summary = json.loads((run / 'summary.json').read_text())
        if summary.get('final'):
            parser.error(f'Use the original validation run, not a final refit: {run}')
        config = summary['config']
        for cutoff in cutoffs:
            dest = args.out / f'{run.name}_cut{cutoff}'
            command = [sys.executable, str(af.HERE / 'autoformer_task2.py'),
                       '--final', '--history-end', str(cutoff),
                       '--epochs', str(summary['best_epoch']), '--device', args.device,
                       '--out', str(dest)]
            for key in ('seed', 'exog', 'seq_len', 'label_len', 'd_model', 'exog_hidden',
                        'level_window', 'train_stride', 'batch', 'lr', 'lr_schedule',
                        'features', 'target_transform', 'trend_init', 'trend_hidden',
                        'trend_kernel', 'anchor_tau', 'heads', 'd_ff', 'e_layers', 'd_layers',
                        'kernel', 'factor', 'dropout'):
                if key in config:
                    command.extend(['--' + key.replace('_', '-'), str(config[key])])
            for key in ('direct_exog', 'level_residual', 'anchor'):
                if config.get(key):
                    command.append('--' + key.replace('_', '-'))
            if args.amp:
                command.append('--amp')
            print(f'Backtesting {run.name}: targets {cutoff + 1}..{cutoff + af.PRED_LEN}', flush=True)
            subprocess.run(command, check=True)
            forecast = pd.read_csv(dest / 'forecast.csv')
            assert forecast.time_idx.tolist() == list(range(cutoff + 1, cutoff + af.PRED_LEN + 1))
            pred = forecast.value.to_numpy()
            actual = truth[cutoff:cutoff + af.PRED_LEN]
            row = {'run': str(run), 'cutoff': cutoff, **af.metrics(pred, actual),
                   'pred_mean': float(pred.mean()), 'true_mean': float(actual.mean()),
                   'zero_count': int((pred == 0).sum())}
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out / 'scores.csv', index=False)
            print(json.dumps(row), flush=True)
    table = pd.DataFrame(rows)
    print('\nPer-cutoff results:\n' + table.to_string(index=False))
    print('\nMean metrics (inspect individual cutoffs too):\n' +
          table.groupby('run')[['RMSE', 'MAE', 'sMAPE']].mean().to_string())
    print('These are reused validation periods, not an independent test. No leaderboard submission was made.')


if __name__ == '__main__':
    main()
