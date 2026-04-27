"""Robustness analysis for EPK-GKN.

Reports accuracy degradation under:
  - Gaussian node-attribute noise (σ ∈ {0, 0.05, 0.1, 0.25, 0.5})
  - Random edge dropout (p ∈ {0, 0.05, 0.1, 0.2, 0.3})
  - Random node feature masking (set feature to zero with p)

Each perturbation is applied per-graph at feature-build time, then the
usual 10-fold CV is run. Reveals how sensitive the model is to graph
corruption.
"""
import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from epk.data import load_dataset
from epk.preprocess import preprocess_dataset, enrich
from epk.train import ten_fold_cv
from epk.run import DEFAULT_CFG, DATASET_CFG
import torch.nn.functional as F


def build_feature_tensor_with_noise(raw, lens_to_keep, K=1,
                                     use_structural=True,
                                     normalize_rows=False,
                                     noise_type='none', noise_level=0.0,
                                     seed=0):
    """Like preprocess.build_feature_tensor but with optional perturbation."""
    rng = np.random.RandomState(seed)
    # determine d_enr once
    d_enr = None
    for g in raw:
        if g is None or g['n'] == 0:
            continue
        d_enr = enrich(g, K=K, use_structural=use_structural).shape[1]
        break
    assert d_enr is not None
    lens_to_keep = sorted(lens_to_keep)
    segments = []
    offset = 0
    for l in lens_to_keep:
        segments.append((l, offset, l * d_enr))
        offset += l * d_enr
    total_dim = offset
    out = torch.zeros((len(raw), total_dim), dtype=torch.float32)
    labels = torch.zeros(len(raw), dtype=torch.long)

    for i, g in enumerate(raw):
        if g is None or g['n'] == 0:
            labels[i] = g['label'].squeeze().long() if g is not None else 0
            continue
        # Perturb per-graph: modify X before enrichment
        X_orig = g['X']
        if noise_type == 'gaussian':
            noise = torch.tensor(
                rng.randn(*X_orig.shape).astype(np.float32)) * noise_level
            X_perturbed = X_orig + noise
        elif noise_type == 'node_mask':
            mask = torch.tensor(
                (rng.rand(X_orig.shape[0]) > noise_level).astype(np.float32)
            ).unsqueeze(1)
            X_perturbed = X_orig * mask
        else:
            X_perturbed = X_orig

        # edge dropout
        M = g['M']
        if noise_type == 'edge_drop' and 2 in M:
            # zero-out a fraction of entries in M_2 (proxy for edge removal)
            M = dict(M)
            M2 = M[2].clone()
            mask = torch.tensor(
                (rng.rand(*M2.shape) > noise_level).astype(np.float32))
            M[2] = M2 * mask

        # recompute enrichment with perturbed X
        g_local = dict(g)
        g_local['X'] = X_perturbed
        X_enr = enrich(g_local, K=K, use_structural=use_structural)

        for (l, start, length) in segments:
            Ml = M.get(l)
            if Ml is None:
                continue
            phi = X_enr.t() @ Ml
            out[i, start:start + length] = phi.t().reshape(-1)
        labels[i] = g['label'].squeeze().long()

    if normalize_rows:
        out = F.normalize(out, p=2, dim=1)

    return out, d_enr, segments, labels


def robustness_sweep(name, device, cache_dir, noise_types=('gaussian',
                                                              'edge_drop'),
                      levels=(0.0, 0.05, 0.1, 0.25, 0.5), n_seeds=3):
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

    results = {}
    for noise_type in noise_types:
        series = []
        for lv in levels:
            x, d_enr, segments, y = build_feature_tensor_with_noise(
                raw, lens, K=1, use_structural=True,
                normalize_rows=ds_cfg.get('normalize_rows', False),
                noise_type=noise_type, noise_level=lv, seed=42)
            cfg = dict(base_cfg)
            cfg['encoder'] = 'linear'
            cfg['use_attn'] = False
            cfg['use_len_embed'] = False
            cfg['use_len_gate'] = False
            cfg['n_seeds'] = n_seeds
            cfg['per_len_channels'] = per_len_channels
            t0 = time.time()
            mean, std, accs = ten_fold_cv(x, y, segments, d_enr,
                                           meta['num_classes'], cfg, dev,
                                           verbose=False)
            dt = time.time() - t0
            print(f'  [{name}] {noise_type}={lv} -> {mean*100:.2f} ± '
                  f'{std*100:.2f}  ({dt:.1f}s)', flush=True)
            series.append({'level': lv, 'mean': mean, 'std': std})
        results[noise_type] = series
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache_dir', default='cache_epk')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_seeds', type=int, default=3)
    args = ap.parse_args()

    print(f'\n=== ROBUSTNESS {args.dataset} ===', flush=True)
    results = robustness_sweep(args.dataset, args.device, args.cache_dir,
                                 n_seeds=args.n_seeds)
    with open(args.out, 'w') as f:
        json.dump({'dataset': args.dataset, 'results': results}, f, indent=2)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
