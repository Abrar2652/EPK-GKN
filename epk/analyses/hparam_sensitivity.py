"""Hyperparameter sensitivity analysis for EPK-GKN.

Sweeps K, cutoff, hidden_paths, dropout to show how the model responds to
hyperparameter choices.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from epk.data import load_dataset
from epk.preprocess import preprocess_dataset, build_feature_tensor
from epk.train import ten_fold_cv
from epk.run import DEFAULT_CFG, DATASET_CFG


def run_grid(name, device, cache_dir, grid_name, values, n_seeds=3):
    ds_cfg = DATASET_CFG.get(name, {})
    base_cfg = dict(DEFAULT_CFG)
    base_cfg.update({k: v for k, v in ds_cfg.items()
                     if k not in ('K', 'use_structural', 'cutoff',
                                  'normalize_rows', 'total_hidden',
                                  'include_sq')})

    dataset = load_dataset(name)
    raw, meta = preprocess_dataset(dataset, cache_dir, name)
    lens_full = sorted(meta['lens_histogram'].keys())
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    results = []
    for v in values:
        cfg = dict(base_cfg)
        cfg['n_seeds'] = n_seeds
        cfg['encoder'] = 'linear'
        cfg['use_attn'] = False
        cfg['use_len_embed'] = False
        cfg['use_len_gate'] = False

        K = 1
        use_struct = True
        cutoff = ds_cfg.get('cutoff', None)
        total_hidden = ds_cfg.get('total_hidden', 0)
        if grid_name == 'K':
            K = v
        elif grid_name == 'cutoff':
            cutoff = v
        elif grid_name == 'total_hidden':
            total_hidden = v
        elif grid_name == 'dropout':
            cfg['dropout'] = v
        else:
            raise ValueError(f'Unknown grid {grid_name}')

        lens = [l for l in lens_full if cutoff is None or l <= cutoff]
        if total_hidden > 0:
            lh = meta['lens_histogram']
            p = int(max(1, sum(lh[l] for l in lens) / total_hidden))
            cfg['per_len_channels'] = [max((lh[l] + p - 1) // p, 5)
                                         for l in lens]
        else:
            cfg['per_len_channels'] = None

        try:
            x, d_enr, segments, y = build_feature_tensor(
                raw, lens, K=K, use_structural=use_struct,
                normalize_rows=ds_cfg.get('normalize_rows', False),
            )
        except Exception as e:
            print(f'  build FAIL {grid_name}={v}: {e}', flush=True)
            continue

        t0 = time.time()
        try:
            mean, std, accs = ten_fold_cv(x, y, segments, d_enr,
                                            meta['num_classes'], cfg, dev,
                                            verbose=False)
        except Exception as e:
            print(f'  train FAIL {grid_name}={v}: {e}', flush=True)
            continue
        dt = time.time() - t0
        print(f'  [{name}] {grid_name}={v:<8} -> {mean*100:.2f} ± '
              f'{std*100:.2f}  ({dt:.1f}s)', flush=True)
        results.append({grid_name: v, 'mean': mean, 'std': std, 'time': dt})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache_dir', default='cache_epk')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_seeds', type=int, default=3)
    args = ap.parse_args()

    print(f'\n=== SENSITIVITY {args.dataset} ===', flush=True)
    all_results = {}

    grids = {
        'K': [0, 1, 2, 3],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'total_hidden': [100, 200, 400, 600, 800],
    }
    ds_cfg = DATASET_CFG.get(args.dataset, {})
    if ds_cfg.get('cutoff') is not None:
        base_cutoff = ds_cfg['cutoff']
        grids['cutoff'] = [max(2, base_cutoff // 2), base_cutoff,
                           base_cutoff + 5, base_cutoff * 2]

    for gname, vals in grids.items():
        print(f'--- grid {gname} ---', flush=True)
        res = run_grid(args.dataset, args.device, args.cache_dir, gname, vals,
                        n_seeds=args.n_seeds)
        all_results[gname] = res
        with open(args.out, 'w') as f:
            json.dump({'dataset': args.dataset, 'grids': all_results}, f, indent=2)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
