"""Scalability figure: 3-panel runtime vs N, n, d."""
import argparse
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
    ap.add_argument('--in_', default='analysis_results/scalability.json')
    ap.add_argument('--out', default='figures/fig_scalability.pdf')
    args = ap.parse_args()

    data = json.load(open(args.in_))
    setup_rc()
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1))

    # n sweep
    ax = axes[0]
    ns = [d['n'] for d in data['vs_n']]
    preps = [d['preproc_s'] for d in data['vs_n']]
    feats = [d['feat_s'] for d in data['vs_n']]
    ax.plot(ns, preps, marker='o', color=PALETTE[0], label='Dijkstra + M$_l$',
             markersize=4.5, markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(ns, feats, marker='s', color=PALETTE[1], label='X$^\\top$M$_l$',
             markersize=4.5, markeredgecolor='black', markeredgewidth=0.5)
    ax.set_xlabel('graph size $n$')
    ax.set_ylabel('runtime (s)')
    ax.set_xscale('log'); ax.set_yscale('log')
    corner_legend(ax, loc='upper left', ncol=1)

    # density sweep
    ax = axes[1]
    ps = [d['p'] for d in data['vs_density']]
    preps = [d['preproc_s'] for d in data['vs_density']]
    ax.plot(ps, preps, marker='o', color=PALETTE[0],
             markersize=4.5, markeredgecolor='black', markeredgewidth=0.5)
    ax.set_xlabel('edge density $c$')
    ax.set_ylabel('preproc runtime (s)')
    ax.set_xscale('linear')

    # d sweep
    ax = axes[2]
    ds = [d['d'] for d in data['vs_d']]
    preps = [d['preproc_s'] for d in data['vs_d']]
    feats = [d['feat_s'] for d in data['vs_d']]
    ax.plot(ds, preps, marker='o', color=PALETTE[0], label='Dijkstra + M$_l$')
    ax.plot(ds, feats, marker='s', color=PALETTE[1], label='X$^\\top$M$_l$')
    ax.set_xlabel('feature dim $d$')
    ax.set_ylabel('runtime (s)')
    ax.set_xscale('log')
    corner_legend(ax, loc='upper left', ncol=1)

    fig.tight_layout()
    fig.savefig(args.out)
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
