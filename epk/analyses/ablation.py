"""Component ablation for EPK-GKN.

Sweeps the four independent design choices and reports accuracy for each
cell of the 2^4 grid, revealing the contribution of each component:

  - K ∈ {0, 1, 2}           polynomial propagation depth
  - structural ∈ {off, on}  degree / core / clustering features
  - include_sq ∈ {off, on}  second-moment features
  - attn ∈ {off, on}        cross-length Transformer

Reports 10-fold × 3-seed mean accuracy per cell.
"""
import argparse
import json
import os
import sys
import time
import itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from epk.data import load_dataset
from epk.preprocess import preprocess_dataset, build_feature_tensor
from epk.train import ten_fold_cv
from epk.run import DEFAULT_CFG, DATASET_CFG


def ablate(name, device, cache_dir, Ks=(0, 1, 2), structs=(False, True),
           sqs=(False, True), attns=(False,), n_seeds=3, epochs_override=None,
           _out_path=None):
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

    if ds_cfg.get('total_hidden', 0) > 0:
        lh = meta['lens_histogram']
        p = int(max(1, sum(lh[l] for l in lens) / ds_cfg['total_hidden']))
        per_len_channels = [max((lh[l] + p - 1) // p, 5) for l in lens]
    else:
        per_len_channels = None

    results = []
    for K, use_struct, sq, attn in itertools.product(Ks, structs, sqs, attns):
        cfg = dict(base_cfg)
        cfg['encoder'] = 'linear'
        cfg['use_attn'] = attn
        cfg['use_len_embed'] = attn
        cfg['use_len_gate'] = attn
        cfg['n_seeds'] = n_seeds
        if epochs_override is not None:
            cfg['epochs'] = epochs_override
        cfg['per_len_channels'] = per_len_channels

        x, d_enr, segments, y = build_feature_tensor(
            raw, lens, K=K, use_structural=use_struct,
            normalize_rows=ds_cfg.get('normalize_rows', False),
            include_sq=sq,
        )

        t0 = time.time()
        try:
            mean, std, accs = ten_fold_cv(x, y, segments, d_enr,
                                           meta['num_classes'], cfg, dev,
                                           verbose=False)
        except Exception as e:
            print(f'  FAILED K={K} struct={use_struct} sq={sq} attn={attn}: {e}',
                  flush=True)
            continue
        dt = time.time() - t0
        label = f'K={K}, struct={int(use_struct)}, sq={int(sq)}, attn={int(attn)}'
        print(f'  [{name}] {label:40s} -> {mean*100:.2f} ± '
              f'{std*100:.2f}  ({dt:.1f}s)', flush=True)
        results.append({'K': K, 'struct': use_struct, 'sq': sq,
                        'attn': attn, 'mean': mean, 'std': std,
                        'time': dt})
        # persist incrementally so we don't lose progress on crash
        if _out_path:
            with open(_out_path, 'w') as f:
                json.dump({'dataset': name, 'results': results}, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache_dir', default='cache_epk')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_seeds', type=int, default=3)
    ap.add_argument('--Ks', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()
    print(f'\n=== ABLATION {args.dataset} ===', flush=True)
    results = ablate(args.dataset, args.device, args.cache_dir,
                      Ks=tuple(args.Ks), n_seeds=args.n_seeds,
                      epochs_override=getattr(args, 'epochs', None),
                      _out_path=args.out)
    with open(args.out, 'w') as f:
        json.dump({'dataset': args.dataset, 'results': results}, f, indent=2)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
