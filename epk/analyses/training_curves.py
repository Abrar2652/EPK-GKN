"""Training convergence curves: per-epoch loss + train/test accuracy.

Runs one representative fold per dataset (fold 0, seed 0) and records the
full training history. Used to show EPK-GKN converges quickly and smoothly.
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from epk.data import load_dataset
from epk.preprocess import preprocess_dataset, build_feature_tensor
from epk.train import train_one_fold
from epk.run import DEFAULT_CFG, DATASET_CFG


def record(name, device, cache_dir):
    ds_cfg = DATASET_CFG.get(name, {})
    base_cfg = dict(DEFAULT_CFG)
    base_cfg.update({k: v for k, v in ds_cfg.items()
                      if k not in ('K', 'use_structural', 'cutoff',
                                    'normalize_rows', 'total_hidden',
                                    'include_sq')})
    dataset = load_dataset(name)
    raw, meta = preprocess_dataset(dataset, cache_dir, name)
    cutoff = ds_cfg.get('cutoff', None)
    lens_full = sorted(meta['lens_histogram'].keys())
    lens = [l for l in lens_full if cutoff is None or l <= cutoff]
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    x, d_enr, segments, y = build_feature_tensor(
        raw, lens, K=ds_cfg.get('K', 2),
        use_structural=ds_cfg.get('use_structural', True),
        include_sq=ds_cfg.get('include_sq', False),
        normalize_rows=ds_cfg.get('normalize_rows', False),
    )

    if ds_cfg.get('total_hidden', 0) > 0:
        lh = meta['lens_histogram']
        p = int(max(1, sum(lh[l] for l in lens) / ds_cfg['total_hidden']))
        per_len = [max((lh[l] + p - 1) // p, 5) for l in lens]
    else:
        per_len = None

    cfg = dict(base_cfg)
    cfg['encoder'] = 'linear'
    cfg['use_attn'] = False
    cfg['use_len_embed'] = False
    cfg['use_len_gate'] = False
    cfg['per_len_channels'] = per_len

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    splits = list(skf.split(x.cpu().numpy(), y.cpu().numpy()))
    tr_idx, te_idx = splits[0]
    x_tr = x[tr_idx].to(dev); x_te = x[te_idx].to(dev)
    y_tr = y[tr_idx].to(dev); y_te = y[te_idx].to(dev)

    torch.manual_seed(0)
    best, history = train_one_fold(x_tr, y_tr, x_te, y_te, segments, d_enr,
                                     meta['num_classes'], cfg, dev,
                                     return_history=True)
    print(f'  [{name}] best={best:.4f}  epochs={len(history)}', flush=True)
    return {'dataset': name, 'best_acc': best, 'epochs': len(history),
            'history': history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache_dir', default='cache_epk')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    print('=== TRAINING CURVES ===', flush=True)
    results = [record(n, args.device, args.cache_dir) for n in args.datasets]
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
