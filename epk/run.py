"""Default configs per dataset, rebuilt from memory of Optuna-tuned values."""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epk.data import load_dataset
from epk.preprocess import preprocess_dataset, build_feature_tensor
from epk.train import ten_fold_cv


DEFAULT_CFG = dict(
    lr=1e-3, weight_decay=1e-4, epochs=500, dropout=0.2,
    hid_per_len=32, fc_hidden=256, attn_heads=4, attn_depth=1,
    label_smooth=0.0, encoder='mlp', use_attn=True, use_len_embed=True,
    use_len_gate=True, norm='layer', batch_size=0, cosine=True,
    grad_clip=2.0,
)

DATASET_CFG = {
    'MUTAG': dict(K=1, use_structural=True, cutoff=5, normalize_rows=True,
                   total_hidden=600, hid_per_len=64, fc_hidden=300,
                   dropout=0.4, lr=7e-4, weight_decay=2.4e-3, epochs=500,
                   encoder='linear', use_attn=False, norm='batch',
                   label_smooth=0.05),
    'PTC_MR': dict(K=1, use_structural=True, cutoff=None,
                    normalize_rows=False, hid_per_len=32, fc_hidden=128,
                    dropout=0.15, lr=5e-3, weight_decay=2e-4, epochs=500,
                    encoder='linear', use_attn=False, norm='batch',
                    label_smooth=0.1),
    'BZR': dict(K=1, use_structural=True, cutoff=13, normalize_rows=True,
                 hid_per_len=32, fc_hidden=128, dropout=0.15, lr=2e-2,
                 weight_decay=1e-3, epochs=500, encoder='linear',
                 use_attn=False, norm='batch', label_smooth=0.05),
    'PROTEINS': dict(K=1, use_structural=True, cutoff=30,
                      normalize_rows=False, hid_per_len=32, fc_hidden=256,
                      dropout=0.3, lr=2e-3, weight_decay=1e-4, epochs=500,
                      encoder='linear', use_attn=False, norm='batch',
                      label_smooth=0.05),
    'DD': dict(K=1, use_structural=True, cutoff=40, normalize_rows=False,
                hid_per_len=24, fc_hidden=256, dropout=0.3, lr=1.5e-3,
                weight_decay=5e-2, epochs=300, encoder='linear',
                use_attn=False, norm='layer', label_smooth=0.05),
    'NCI1': dict(K=1, use_structural=True, cutoff=20, normalize_rows=False,
                  hid_per_len=32, fc_hidden=256, dropout=0.2, lr=1e-3,
                  weight_decay=1e-4, epochs=500, encoder='linear',
                  use_attn=False, norm='batch', label_smooth=0.05),
    'ENZYMES': dict(K=1, use_structural=True, cutoff=None,
                     normalize_rows=False, hid_per_len=32, fc_hidden=256,
                     dropout=0.3, lr=1e-3, weight_decay=1e-3, epochs=500,
                     encoder='linear', use_attn=False, norm='batch',
                     label_smooth=0.1),
    'IMDB-BINARY': dict(K=1, use_structural=True, cutoff=None,
                         normalize_rows=False, hid_per_len=32, fc_hidden=128,
                         dropout=0.3, lr=2e-3, weight_decay=1e-4, epochs=400,
                         encoder='linear', use_attn=False, norm='batch',
                         label_smooth=0.05),
    'IMDB-MULTI': dict(K=1, use_structural=True, cutoff=None,
                        normalize_rows=False, hid_per_len=32, fc_hidden=128,
                        dropout=0.3, lr=2e-3, weight_decay=1e-3, epochs=400,
                        encoder='linear', use_attn=False, norm='batch',
                        label_smooth=0.1),
    'COLLAB': dict(K=1, use_structural=True, cutoff=None,
                    normalize_rows=True, hid_per_len=32, fc_hidden=256,
                    dropout=0.4, lr=2e-3, weight_decay=5e-4, epochs=300,
                    encoder='linear', use_attn=False, norm='layer',
                    label_smooth=0.05),
    'REDDIT-BINARY': dict(K=1, use_structural=True, cutoff=20,
                           normalize_rows=True, hid_per_len=32, fc_hidden=128,
                           dropout=0.3, lr=1e-2, weight_decay=1e-4,
                           epochs=300, encoder='linear', use_attn=False,
                           norm='batch', label_smooth=0.05),
    'REDDIT-MULTI-5K': dict(K=1, use_structural=True, cutoff=20,
                             normalize_rows=False, total_hidden=200,
                             hid_per_len=24, fc_hidden=100, dropout=0.2,
                             lr=0.2, weight_decay=0.2, epochs=500,
                             encoder='linear', use_attn=False, norm='batch',
                             label_smooth=0.0),
}


def run(name, device='cuda:0', cache_dir='cache_epk', verbose=True,
         override=None):
    cfg = dict(DEFAULT_CFG)
    ds_cfg = DATASET_CFG.get(name, {})
    cfg.update({k: v for k, v in ds_cfg.items()
                 if k not in ('K', 'use_structural', 'cutoff',
                              'normalize_rows', 'total_hidden', 'include_sq')})
    if override:
        cfg.update(override)
    K = ds_cfg.get('K', 1)
    use_struct = ds_cfg.get('use_structural', True)
    cutoff = ds_cfg.get('cutoff', None)
    normalize_rows = ds_cfg.get('normalize_rows', False)
    if override:
        K = override.get('K', K)
        use_struct = override.get('use_structural', use_struct)
        cutoff = override.get('cutoff', cutoff)
        normalize_rows = override.get('normalize_rows', normalize_rows)
    dataset = load_dataset(name)
    raw, meta = preprocess_dataset(dataset, cache_dir, name)
    lens = sorted(meta['lens_histogram'].keys())
    if cutoff is not None:
        lens = [l for l in lens if l <= cutoff]
    if verbose:
        print(f'[{name}] lens={lens}  K={K}  struct={use_struct}  '
               f'num_graphs={len(raw)}  classes={meta["num_classes"]}')
    total_hidden = cfg.get('total_hidden', 0)
    if total_hidden > 0:
        lh = meta['lens_histogram']
        p = int(max(1, sum(lh[l] for l in lens) / total_hidden))
        cfg['per_len_channels'] = [max((lh[l] + p - 1) // p, 5) for l in lens]
    x, d_enr, segments, y = build_feature_tensor(
        raw, lens, K=K, use_structural=use_struct,
        normalize_rows=normalize_rows)
    if verbose:
        print(f'[{name}] feature tensor: {tuple(x.shape)}  d_enr={d_enr}')
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    mean, std, accs = ten_fold_cv(x, y, segments, d_enr, meta['num_classes'],
                                    cfg, dev, verbose=verbose)
    return mean, std, accs, cfg


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache_dir', default='cache_epk')
    args = ap.parse_args()
    mean, std, accs, cfg = run(args.dataset, device=args.device,
                                 cache_dir=args.cache_dir)
    print(f'\n=== {args.dataset} EPK-GKN: {mean*100:.2f} ± {std*100:.2f} ===')
