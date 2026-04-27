"""Replay HSP-GKN's original training loop under a multi-seed protocol.

Requires the authors' `utils.py`, `model.py`, and `config.yml` to be present in the
repo root (they were once checked in from the HSP-GKN GitHub release).

If those files are missing (e.g. after a fresh clone of just the epk/
package), this script cannot run; see `README_rebuild.md` for how to
re-seed them from the HSP-GKN repo.
"""
import argparse
import json
import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import dgl
import yaml
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _import_utils():
    """Late-import utils.py and config.yml to avoid crashing if absent."""
    try:
        from utils import (cat_attr_label, set_attr_as_label,
                            set_label_as_degree, set_node_degree_as_feature,
                            normalize_attr, process_dataset_matrix_ig)
    except ImportError:
        print('utils.py from HSP-GKN not present. See README_rebuild.md.',
               flush=True)
        sys.exit(2)
    return (cat_attr_label, set_attr_as_label, set_label_as_degree,
             set_node_degree_as_feature, normalize_attr,
             process_dataset_matrix_ig)


class Model(nn.Module):
    """Copy of HSP-GKN's Model with local init (so we don't import opt.py)."""
    def __init__(self, hid_ch, node_feat_dim, n_classes, hid_dims, Norm,
                  dropout):
        super().__init__()
        self.hid_ch = hid_ch
        self.node_feat_dim = node_feat_dim
        self.paths_layers = nn.ModuleList(
            [nn.Linear(k * node_feat_dim, v, bias=False)
              for k, v in hid_ch.items()])
        a = sum(hid_ch.values())
        self.fc = nn.Sequential(
            nn.Linear(a, hid_dims[0], bias=True),
            Norm(hid_dims[0]), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(hid_dims[0], hid_dims[1], bias=True),
            Norm(hid_dims[1]), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(hid_dims[1], n_classes, bias=True),
        )
        for p in self.paths_layers:
            init.xavier_uniform_(p.weight)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        split = list(self.hid_ch.keys())
        parts = torch.split(x, [v * self.node_feat_dim for v in split], dim=1)
        out = torch.cat([self.paths_layers[i](parts[i])
                          for i in range(len(parts))], dim=1)
        return self.fc(out)


def prep_dataset(name, utils_fns):
    (cat_attr_label, set_attr_as_label, set_label_as_degree,
      set_node_degree_as_feature, _, _) = utils_fns
    if name in ('PROTEINS', 'ENZYMES', 'BZR', 'COX2', 'DHFR'):
        ds = dgl.data.TUDataset(name); cat_attr_label(ds)
    elif name in ('NCI1', 'DD', 'MUTAG', 'PTC_MR'):
        ds = dgl.data.TUDataset(name); set_attr_as_label(ds)
    elif name in ('COLLAB', 'IMDB-BINARY', 'IMDB-MULTI'):
        ds = dgl.data.TUDataset(name)
        set_label_as_degree(ds); set_attr_as_label(ds)
    elif name in ('REDDIT-BINARY', 'REDDIT-MULTI-5K'):
        ds = dgl.data.TUDataset(name); set_node_degree_as_feature(ds)
    else:
        raise ValueError(name)
    return ds


def run(name, device, cfg, utils_fns, n_seeds=3, epochs=500, verbose=False):
    (_, _, _, _, normalize_attr, process_dataset_matrix_ig) = utils_fns
    ds = prep_dataset(name, utils_fns)
    node_feat_dim = ds[0][0].ndata['node_attr'].shape[1]
    Norm = nn.BatchNorm1d if cfg['norm'] == 'batch_norm' else nn.LayerNorm
    os.makedirs('cache', exist_ok=True)
    cache_attrs = f'cache/{name}.pt'
    cache_lens = f'cache/{name}_len_count.pt'
    if os.path.exists(cache_attrs):
        graphs_paths_attrs = torch.load(cache_attrs, weights_only=False)
        len_count = torch.load(cache_lens, weights_only=False)
    else:
        graphs_paths_attrs, len_count = process_dataset_matrix_ig(
            ds, cfg['cutoff'])
        torch.save(graphs_paths_attrs, cache_attrs)
        torch.save(len_count, cache_lens)
    cutoff = cfg['cutoff']
    for l in list(len_count.keys()):
        if cutoff is not None and l > cutoff:
            del len_count[l]
    hid_paths = cfg['hidden_paths']
    p = int(max(1, sum(len_count.values()) / hid_paths))
    hid_ch = {k: max((v + p - 1) // p, 5) for k, v in len_count.items()}
    a = sum(hid_ch.values())
    hid_dims = [int(a / 2), int(a / 4)]
    x = graphs_paths_attrs.to(device)
    if cutoff is not None:
        x = x[:, :int(cutoff * (cutoff + 1) / 2) * node_feat_dim]
    if cfg['norm_attr']:
        x = normalize_attr(x)
    y = ds.graph_labels.to(device)
    num_classes = ds.num_labels
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    fold_accs = []
    for fold, (tr, te) in enumerate(skf.split(x.cpu().clone(), y.cpu().clone())):
        seed_accs = []
        x_tr = x[tr]; x_te = x[te]
        y_tr = torch.squeeze(y[tr], 1).long()
        y_te = torch.squeeze(y[te], 1).long()
        for s in range(n_seeds):
            torch.manual_seed(s * 17 + fold * 3)
            model = Model(hid_ch, node_feat_dim, num_classes, hid_dims, Norm,
                           cfg['dropout']).to(device)
            loss_fn = nn.CrossEntropyLoss()
            opt = optim.Adam(model.parameters(), lr=cfg['lr'],
                              weight_decay=cfg['l2'])
            best = 0.0
            for ep in range(epochs):
                model.train()
                opt.zero_grad()
                loss = loss_fn(model(x_tr), y_tr); loss.backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    acc = (model(x_te).argmax(1) == y_te).float().mean().item()
                if acc > best:
                    best = acc
            seed_accs.append(best)
        fold_accs.append(float(np.mean(seed_accs)))
        if verbose:
            print(f'  fold {fold}: {np.mean(seed_accs):.4f}', flush=True)
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n_seeds', type=int, default=3)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    utils_fns = _import_utils()
    FALLBACK = {'ENZYMES': dict(norm='batch_norm', norm_attr=False,
                                  cutoff=None, hidden_paths=400, lr=1e-3,
                                  l2=1e-3, dropout=0.3)}
    try:
        yml = yaml.safe_load(open('config.yml'))
    except FileNotFoundError:
        yml = {'dataset': {}}
    results = {}
    for name in args.datasets:
        cfg = yml.get('dataset', {}).get(name) or FALLBACK.get(name)
        if cfg is None:
            print(f'No config for {name}', flush=True); continue
        print(f'\n=== HSP-GKN-original {name} ===', flush=True)
        t0 = time.time()
        mean, std, accs = run(name, args.device, cfg, utils_fns,
                                 n_seeds=args.n_seeds, verbose=True)
        print(f'[{name}] HSP-GKN-orig multi-seed: {mean*100:.2f} ± '
               f'{std*100:.2f}  ({time.time()-t0:.1f}s)', flush=True)
        results[name] = {'mean': mean, 'std': std, 'per_fold': accs}
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
