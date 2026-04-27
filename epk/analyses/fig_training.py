"""Figure: training convergence curves (loss + train/test accuracy)."""
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
    ap.add_argument('--in_', default='analysis_results/training_curves.json')
    ap.add_argument('--out', default='figures/fig_training.pdf')
    args = ap.parse_args()
    setup_rc()
    data = json.load(open(args.in_))
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.1), squeeze=False)
    axes = axes[0]
    for i, rec in enumerate(data):
        ax = axes[i]
        hist = rec['history']
        eps = [h['epoch'] for h in hist]
        loss = [h['loss'] for h in hist]
        tr = [h['train_acc'] * 100 for h in hist]
        te = [h['test_acc'] * 100 for h in hist]
        ax2 = ax.twinx()
        ax.plot(eps, loss, color=PALETTE[1], linewidth=1.2, label='loss')
        ax2.plot(eps, tr, color=PALETTE[0], linewidth=1.0,
                  linestyle='--', label='train acc')
        ax2.plot(eps, te, color=PALETTE[2], linewidth=1.2, label='test acc')
        ax.set_xlabel('epoch')
        if i == 0:
            ax.set_ylabel('loss', color=PALETTE[1])
        if i == n - 1:
            ax2.set_ylabel('accuracy (%)', color=PALETTE[2])
        ax.text(0.98, 0.05, rec['dataset'], transform=ax.transAxes,
                 ha='right', va='bottom', fontsize=7.5, fontweight='bold',
                 color=PALETTE[6],
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='none', alpha=0.85))
        if i == 0:
            lines1, labs1 = ax.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labs1 + labs2, loc='center right',
                       fontsize=6.5, frameon=True, framealpha=0.85,
                       edgecolor='none')
    fig.tight_layout()
    fig.savefig(args.out)
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
