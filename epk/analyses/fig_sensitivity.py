"""Hyperparameter sensitivity figure: 2×2 grid over K, dropout,
total_hidden, (optional) cutoff. Lines per dataset.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='analysis_results/sens_*.json')
    ap.add_argument('--out', default='figures/fig_sensitivity.pdf')
    args = ap.parse_args()

    setup_rc()
    data = {}
    for path in sorted(glob.glob(args.glob)):
        name = os.path.basename(path).replace('sens_', '').replace('.json', '')
        try:
            d = json.load(open(path))
            data[name] = d.get('grids', {})
        except Exception:
            continue
    if not data:
        print('No sensitivity JSONs found.', flush=True)
        return

    grid_names = ['K', 'dropout', 'total_hidden', 'cutoff']
    available_grids = set()
    for ds in data.values():
        available_grids.update(ds.keys())
    grid_names = [g for g in grid_names if g in available_grids]
    if not grid_names:
        return
    ncols = len(grid_names)
    fig, axes = plt.subplots(1, ncols, figsize=(2.2 * ncols, 2.2),
                              squeeze=False)
    axes = axes[0]
    for i, gname in enumerate(grid_names):
        ax = axes[i]
        # determine the union of x-values across datasets, sorted, as
        # categorical positions (equal x-spacing avoids log-scale crowding)
        all_xv = sorted({r[gname] for d in data.values()
                          for r in d.get(gname, [])})
        x_to_pos = {v: idx for idx, v in enumerate(all_xv)}
        for j, (dsname, grids) in enumerate(data.items()):
            if gname not in grids:
                continue
            series = grids[gname]
            xs = [x_to_pos[r[gname]] for r in series]
            ys = [r['mean'] * 100 for r in series]
            stds = [r['std'] * 100 for r in series]
            ax.errorbar(xs, ys, yerr=stds, marker='o', color=PALETTE[j],
                         label=dsname, linewidth=1.2, markersize=4.5,
                         markeredgecolor='black', markeredgewidth=0.5,
                         capsize=2, elinewidth=0.6)
        ax.set_xticks(list(x_to_pos.values()))
        ax.set_xticklabels([str(v) for v in all_xv])
        ax.set_xlabel(gname)
        ax.set_ylabel('accuracy (%)')
        if i == 0:
            corner_legend(ax, loc='lower right', ncol=1)
    fig.tight_layout()
    fig.savefig(args.out)
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
