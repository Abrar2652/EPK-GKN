"""Scalability analysis for EPK-GKN: preprocessing and training time vs
graph size, density, and feature dimension.

Generates synthetic Erdős–Rényi graphs with controlled size and density
and measures preprocessing + forward-pass wall-clock.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import igraph as ig

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from epk.preprocess import _graph_to_raw, enrich
from epk.model import EPKGKN


class SyntheticGraph:
    """Mimics DGL graph interface for _graph_to_raw."""
    def __init__(self, n, edges, X):
        self.n = n
        self.src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        self.dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
        self.ndata = {'node_attr': X}

    def number_of_nodes(self):
        return self.n

    def edges(self):
        return self.src, self.dst


def gen_er_graph(n, p_edge, d):
    """Random Erdős–Rényi graph with n nodes, edge prob p_edge, d node attrs."""
    g = ig.Graph.Erdos_Renyi(n=n, p=p_edge, directed=False, loops=False)
    # ensure connected (else SP lengths sparse)
    if not g.is_connected():
        comps = g.connected_components()
        # keep largest component
        largest = max(comps, key=len)
        g = g.subgraph(largest)
        n = g.vcount()
    edges = g.get_edgelist()
    X = torch.randn(n, d)
    return SyntheticGraph(n, edges, X), n


def time_preproc_and_forward(g_syn, label_obj):
    """Time (preproc, forward-pass) on a single synthetic graph."""
    t0 = time.time()
    raw = _graph_to_raw(g_syn, label_obj)
    t_preproc = time.time() - t0

    # build feature tensor for single graph
    if raw['n'] == 0:
        return t_preproc, 0.0, 0
    X_enr = enrich(raw, K=1, use_structural=True)
    lens = sorted(raw['M'].keys())
    parts = []
    t0 = time.time()
    for l in lens:
        Ml = raw['M'][l]
        phi = X_enr.t() @ Ml
        parts.append(phi.t().reshape(-1))
    feat = torch.cat(parts, dim=0).unsqueeze(0)
    t_feat = time.time() - t0
    return t_preproc, t_feat, feat.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    class Lbl:
        def __init__(self): pass
        def squeeze(self): return torch.tensor(0)
        @property
        def long(self): return self

    lbl = torch.tensor([0])
    results = {'vs_n': [], 'vs_density': [], 'vs_d': []}

    # vary n
    print('--- vs graph size n ---', flush=True)
    for n in [50, 100, 200, 500, 1000, 2000]:
        trials = []
        for trial in range(3):
            g, actual_n = gen_er_graph(n, p_edge=0.1, d=10)
            tp, tf, dim = time_preproc_and_forward(g, lbl)
            trials.append({'n': actual_n, 'preproc_s': tp, 'feat_s': tf,
                            'dim': dim})
        mean_preproc = np.mean([t['preproc_s'] for t in trials])
        mean_feat = np.mean([t['feat_s'] for t in trials])
        print(f'  n={n:4d}: preproc={mean_preproc:.3f}s feat={mean_feat:.3f}s',
              flush=True)
        results['vs_n'].append({'n': n, 'preproc_s': mean_preproc,
                                 'feat_s': mean_feat,
                                 'trials': trials})

    # vary density (fixed n=200)
    print('--- vs graph density ---', flush=True)
    for p in [0.05, 0.1, 0.2, 0.4, 0.6, 0.9]:
        g, n = gen_er_graph(200, p_edge=p, d=10)
        tp, tf, dim = time_preproc_and_forward(g, lbl)
        m = g.n
        actual_density = len(list(g.edges()[0])) / (m * (m - 1) / 2) if m > 1 else 0
        print(f'  p={p}: n={m} preproc={tp:.3f}s feat={tf:.3f}s',
              flush=True)
        results['vs_density'].append({'p': p, 'n': m, 'preproc_s': tp,
                                       'feat_s': tf})

    # vary d (node attr dim)
    print('--- vs feature dim d ---', flush=True)
    for d in [1, 5, 10, 50, 100, 500]:
        g, n = gen_er_graph(100, p_edge=0.1, d=d)
        tp, tf, dim = time_preproc_and_forward(g, lbl)
        print(f'  d={d:4d}: preproc={tp:.3f}s feat={tf:.3f}s dim={dim}',
              flush=True)
        results['vs_d'].append({'d': d, 'preproc_s': tp, 'feat_s': tf,
                                 'dim': dim})

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
