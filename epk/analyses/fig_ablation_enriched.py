"""Figure: per-component enriched-feature ablation.

Waterfall-style bar chart showing accuracy delta (pts) vs base (X only) for
each additional component. One panel per dataset.
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from epk.analyses.style import setup_rc, PALETTE, corner_legend

SHORT = {'IMDB-BINARY': 'IMDB-B', 'IMDB-MULTI': 'IMDB-M',
          'REDDIT-BINARY': 'REDDIT-B', 'REDDIT-MULTI-5K': 'REDDIT-5K',
          'PROTEINS': 'PROT', 'ENZYMES': 'ENZ', 'PTC_MR': 'PTC',
          'IMDBB': 'IMDB-B'}


def load(glob_pat):
    out = {}
    for p in sorted(glob.glob(glob_pat)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        name = d.get('dataset') or os.path.basename(p).replace(
            'ablation_enriched_', '').replace('.json', '')
        name = SHORT.get(name, name)
        out[name] = d['results']
    return out


def fig_enriched(all_res, out_path):
    setup_rc()
    datasets = list(all_res.keys())
    labels = [r['label'] for r in all_res[datasets[0]]]
    mat = np.full((len(datasets), len(labels)), np.nan)
    for i, ds in enumerate(datasets):
        base = None
        for r in all_res[ds]:
            if r['label'] == 'base':
                base = r['mean']
                break
        if base is None:
            continue
        for j, r in enumerate(all_res[ds]):
            mat[i, j] = (r['mean'] - base) * 100

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(labels))
    width = 0.85 / max(1, len(datasets))
    for i, ds in enumerate(datasets):
        off = (i - (len(datasets) - 1) / 2) * width
        ax.bar(x + off, mat[i], width, color=PALETTE[i % 8],
                edgecolor='black', linewidth=0.4, label=ds)
    ax.axhline(0, color=PALETTE[6], linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('$\\Delta$ accuracy vs base (pts)')
    corner_legend(ax, loc='upper left', ncol=1)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Saved {out_path} (+ png)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='analysis_results/ablation_enriched_*.json')
    ap.add_argument('--out', default='figures/fig_ablation_enriched.pdf')
    args = ap.parse_args()
    all_res = load(args.glob)
    if not all_res:
        print('No enriched-ablation results yet.', flush=True)
        return
    fig_enriched(all_res, args.out)


if __name__ == '__main__':
    main()
