"""Parameter count, FLOP estimate, and memory usage for EPK-GKN.

Reports on every dataset:
  - total parameter count
  - per-component parameter breakdown
  - approximate FLOPs per forward pass
  - feature-tensor memory footprint
"""
import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from epk.data import load_dataset
from epk.preprocess import preprocess_dataset, build_feature_tensor
from epk.model import EPKGKN
from epk.run import DEFAULT_CFG, DATASET_CFG


def count_params(model):
    breakdown = {'encoders': 0, 'head': 0, 'other': 0}
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if 'encoders' in name:
            breakdown['encoders'] += n
        elif 'head' in name or 'classifier' in name:
            breakdown['head'] += n
        else:
            breakdown['other'] += n
    return total, breakdown


def estimate_flops_forward(model, x):
    """Rough FLOP estimate for one forward pass on batch x."""
    flops = 0
    with torch.no_grad():
        # per-length encoders: Linear(in, out) costs in*out per sample
        B = x.size(0)
        for i, (l, start, length) in enumerate(model.segments):
            in_dim = length
            out_dim = model.channels[i]
            flops += 2 * B * in_dim * out_dim
        # head sequential Linear layers (dominant)
        for layer in getattr(model.head, '__iter__', lambda: [])() or model.head:
            if isinstance(layer, torch.nn.Linear):
                flops += 2 * B * layer.in_features * layer.out_features
    return flops


def analyse(name, device, cache_dir):
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

    x, d_enr, segments, y = build_feature_tensor(
        raw, lens, K=1, use_structural=True,
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

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = EPKGKN(
        segments=segments,
        d_enr=d_enr,
        n_classes=meta['num_classes'],
        hid_per_len=cfg['hid_per_len'],
        fc_hidden=cfg['fc_hidden'],
        dropout=cfg['dropout'],
        encoder=cfg['encoder'],
        use_attn=cfg['use_attn'],
        use_len_embed=cfg['use_len_embed'],
        use_len_gate=cfg['use_len_gate'],
        per_len_channels=cfg['per_len_channels'],
        norm=cfg.get('norm', 'batch'),
    ).to(dev)

    total_p, breakdown = count_params(model)
    memory_bytes = x.numel() * x.element_size()
    flops = estimate_flops_forward(model, x[:1].to(dev))

    # forward-pass wall-clock throughput (samples / sec)
    model.eval()
    x_dev = x.to(dev)
    with torch.no_grad():
        for _ in range(3):
            _ = model(x_dev)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        import time as _t
        t0 = _t.time()
        n_iters = 20
        for _ in range(n_iters):
            _ = model(x_dev)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        dt = _t.time() - t0
    samples_per_sec = n_iters * x.size(0) / dt
    fwd_ms_per_batch = 1000.0 * dt / n_iters

    out = {
        'dataset': name,
        'num_graphs': len(raw),
        'num_classes': meta['num_classes'],
        'feat_dim': x.shape[1],
        'd_enr': d_enr,
        'n_path_lengths': len(segments),
        'total_params': total_p,
        'params_breakdown': breakdown,
        'feature_memory_bytes': int(memory_bytes),
        'forward_flops_per_sample': int(flops),
        'samples_per_sec': float(samples_per_sec),
        'fwd_ms_per_batch': float(fwd_ms_per_batch),
    }
    print(f'  {name:<18} params={total_p:7d}  feat_dim={x.shape[1]:>6}  '
          f'throughput={samples_per_sec:>9.0f} samples/s  '
          f'fwd={fwd_ms_per_batch:.2f}ms/batch',
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache_dir', default='cache_epk')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    print('=== EFFICIENCY ANALYSIS ===', flush=True)
    results = [analyse(n, args.device, args.cache_dir) for n in args.datasets]
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
