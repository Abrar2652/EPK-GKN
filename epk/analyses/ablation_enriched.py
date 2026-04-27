"""Per-component ablation of enriched features for EPK-GKN.

Unlike the bundled `ablation.py` (which toggles {K, struct-bundle, sq, attn}
as on/off), this script ablates each ingredient of the enriched feature
vector individually. Starting from the pure-HSP baseline (just X), we add
one component at a time, then report the accuracy contribution of each.

Components:
  base          X only                    (K=0, no struct, no sq)
  +ÂX           + 1-hop propagation       (K=1)
  +Â²X          + 2-hop propagation       (K=2)
  +Â³X          + 3-hop propagation       (K=3)
  +deg          K=0 + degree only         (K=0, struct_mask=(1,0,0))
  +core         K=0 + k-core only         (K=0, struct_mask=(0,1,0))
  +clust        K=0 + clustering only     (K=0, struct_mask=(0,0,1))
  +all-struct   K=0 + all three           (K=0, struct_mask=(1,1,1))
  +sq           K=0 + second-moment       (K=0, include_sq=True)
  full          all on                    (K=2, struct, sq)
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


CONFIGS = [
    ('base',         dict(K=0, use_structural=False, include_sq=False)),
    ('+Ah X',        dict(K=1, use_structural=False, include_sq=False)),
    ('+Ah2 X',       dict(K=2, use_structural=False, include_sq=False)),
    ('+Ah3 X',       dict(K=3, use_structural=False, include_sq=False)),
    ('+deg',         dict(K=0, use_structural=True,  include_sq=False,
                          struct_mask=(1, 0, 0))),
    ('+core',        dict(K=0, use_structural=True,  include_sq=False,
                          struct_mask=(0, 1, 0))),
    ('+clust',       dict(K=0, use_structural=True,  include_sq=False,
                          struct_mask=(0, 0, 1))),
    ('+all-struct',  dict(K=0, use_structural=True,  include_sq=False,
                          struct_mask=(1, 1, 1))),
    ('+sq',          dict(K=0, use_structural=False, include_sq=True)),
    ('full',         dict(K=2, use_structural=True,  include_sq=True,
                          struct_mask=(1, 1, 1))),
]


def ablate_enriched(name, device, cache_dir, n_seeds=3, out_path=None):
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
        per_len = [max((lh[l] + p - 1) // p, 5) for l in lens]
    else:
        per_len = None

    results = []
    for label, feat_cfg in CONFIGS:
        cfg = dict(base_cfg)
        cfg['encoder'] = 'linear'
        cfg['use_attn'] = False
        cfg['use_len_embed'] = False
        cfg['use_len_gate'] = False
        cfg['n_seeds'] = n_seeds
        cfg['per_len_channels'] = per_len

        kwargs = dict(feat_cfg)
        kwargs.setdefault('struct_mask', (1, 1, 1))
        x, d_enr, segments, y = build_feature_tensor(
            raw, lens,
            normalize_rows=ds_cfg.get('normalize_rows', False),
            **kwargs,
        )

        t0 = time.time()
        try:
            mean, std, accs = ten_fold_cv(x, y, segments, d_enr,
                                           meta['num_classes'], cfg, dev,
                                           verbose=False)
        except Exception as e:
            print(f'  FAILED {label}: {e}', flush=True)
            continue
        dt = time.time() - t0
        print(f'  [{name}] {label:<14s} d_enr={d_enr:<4d} '
              f'-> {mean*100:.2f} ± {std*100:.2f}  ({dt:.1f}s)', flush=True)
        results.append({
            'label': label, 'd_enr': d_enr,
            'mean': mean, 'std': std, 'time': dt,
            **{k: (list(v) if isinstance(v, tuple) else v)
               for k, v in feat_cfg.items()},
        })
        if out_path:
            with open(out_path, 'w') as f:
                json.dump({'dataset': name, 'results': results}, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache_dir', default='cache_epk')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_seeds', type=int, default=3)
    args = ap.parse_args()
    print(f'\n=== ENRICHED ABLATION {args.dataset} ===', flush=True)
    results = ablate_enriched(args.dataset, args.device, args.cache_dir,
                                n_seeds=args.n_seeds, out_path=args.out)
    with open(args.out, 'w') as f:
        json.dump({'dataset': args.dataset, 'results': results}, f, indent=2)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
