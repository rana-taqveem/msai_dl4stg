"""Compare saved validation checkpoints; never evaluate all-history refits here."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import autoformer_task2 as af


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--recent-blocks', type=int, default=24)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--out', type=Path, default=Path('recent_validation.csv'))
    args = parser.parse_args()
    if args.recent_blocks < 1:
        parser.error('--recent-blocks must be positive')
    device = af.resolve_device(args.device)
    n = len(pd.read_csv(af.DATA / 'student_train.csv'))
    start = n - af.VAL_LEN
    origins = np.arange(start, n - af.PRED_LEN + 1, af.VAL_STRIDE)
    count = min(args.recent_blocks, len(origins))
    print(f'{len(origins)} validation blocks; recent subset: {count} blocks, '
          f'target time_idx {origins[-count] + 1}..{origins[-1] + af.PRED_LEN}')
    rows = []
    paths = sorted({p for root in args.runs for p in root.glob('*/summary.json')})
    for path in paths:
        summary = json.loads(path.read_text())
        if summary.get('final') or summary.get('config', {}).get('final'):
            print(f'Skipping final refit (trained on validation targets): {path.parent}')
            continue
        config = summary['config']
        target, cov, scaler, truth = af.load_for_config(config, start)
        model = af.model_from_config(cov.shape[1], config).to(device)
        model.load_state_dict(torch.load(path.parent / 'model.pt', map_location=device,
                                        weights_only=True))
        raw = af.predict(model, target, cov, origins, config['seq_len'],
                         config['label_len'], scaler, device=device, clip=False)
        pred = np.maximum(raw, 0)
        actual = np.stack([truth[o:o + af.PRED_LEN] for o in origins])
        row = {'run': str(path.parent), 'parameters': summary['parameters'],
               'best_epoch': summary['best_epoch'], 'epochs_run': summary['epochs_run']}
        for prefix, sl in [('all', slice(None)), ('recent', slice(-count, None))]:
            row.update({f'{prefix}_{k}': v for k, v in af.metrics(pred[sl], actual[sl]).items()})
            row[f'{prefix}_pred_mean'] = float(pred[sl].mean())
            row[f'{prefix}_true_mean'] = float(actual[sl].mean())
            row[f'{prefix}_negative_pct'] = float(100 * (raw[sl] < 0).mean())
        rows.append(row)
        print(f"{path.parent.name}: all RMSE={row['all_RMSE']:.2f}, "
              f"recent RMSE={row['recent_RMSE']:.2f}", flush=True)
        del model
    if not rows:
        parser.error('No validation runs found. Check --runs directory paths.')
    table = pd.DataFrame(rows).sort_values('recent_RMSE')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    columns = ['run', 'all_RMSE', 'recent_RMSE', 'recent_MAE',
               'recent_pred_mean', 'recent_true_mean', 'recent_negative_pct']
    print('\n' + table[columns].to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print(f'\nSaved {args.out}. Blocks overlap; this is reused validation, not a new test.')


if __name__ == '__main__':
    main()
