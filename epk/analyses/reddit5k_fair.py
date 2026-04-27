"""REDDIT-MULTI-5K fair HSP-GKN baseline.

HSP-equivalent configuration: K=0, no structural features, no
second-moment, linear encoder, no attention — this matches HSP-GKN's
original feature map (φ_l(G) = X^T M_l) exactly. Hyperparameters from
HSP-GKN's published `config.yml`:
  lr=0.2, weight_decay=0.2, dropout=0.2, total_hidden=200, cutoff=20,
  norm=batch_norm, norm_attr=False.

Run with 3-seed-per-fold protocol to match main table.
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
from epk.run import DEFAULT_CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache_dir', default='cache_epk')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_seeds', type=int, default=3)
    args = ap.parse_args()

    name = 'REDDIT-MULTI-5K'
    cfg = dict(DEFAULT_CFG)
    cfg.update(dict(
        lr=0.2, weight_decay=0.2, dropout=0.2, epochs=500,
        encoder='linear', use_attn=False, use_len_embed=False,
        use_len_gate=False, norm='batch', label_smooth=0.0,
        cosine=False, grad_clip=0.0, optimizer='adam',
        n_seeds=args.n_seeds,
    ))

    dataset = load_dataset(name)
    raw, meta = preprocess_dataset(dataset, args.cache_dir, name)
    cutoff = 20
    lens_full = sorted(meta['lens_histogram'].keys())
    lens = [l for l in lens_full if l <= cutoff]

    # frequency-proportional channels with total_hidden = 200
    total_hidden = 200
    lh = meta['lens_histogram']
    p = int(max(1, sum(lh[l] for l in lens) / total_hidden))
    cfg['per_len_channels'] = [max((lh[l] + p - 1) // p, 5) for l in lens]
    print(f'per_len_channels: {cfg["per_len_channels"]}  sum={sum(cfg["per_len_channels"])}',
          flush=True)

    x, d_enr, segments, y = build_feature_tensor(
        raw, lens, K=0, use_structural=False, normalize_rows=False,
        include_sq=False,
    )
    print(f'feature tensor: {tuple(x.shape)}  d_enr={d_enr}', flush=True)

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    mean, std, accs = ten_fold_cv(x, y, segments, d_enr,
                                   meta['num_classes'], cfg, dev, verbose=True)
    dt = time.time() - t0
    print(f'\n[{name}] HSP-equiv fair: {mean*100:.2f} ± {std*100:.2f}  '
          f'({dt:.1f}s)', flush=True)
    with open(args.out, 'w') as f:
        json.dump({name: {'mean': mean, 'std': std, 'per_fold': accs}},
                   f, indent=2)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
