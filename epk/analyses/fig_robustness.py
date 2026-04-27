"""Figure: robustness to Gaussian feature noise and edge dropout."""
import argparse
import glob
import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from epk.analyses.style import setup_rc, PALETTE, corner_legend

SHORT = {'IMDB-BINARY': 'IMDB-B', 'IMDBB': 'IMDB-B'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='analysis_results/robustness_*.json')
    ap.add_argument('--out', default='figures/fig_robustness.pdf')
    args = ap.parse_args()
    setup_rc()
    data = {}
    for p in sorted(glob.glob(args.glob)):
        d = json.load(open(p))
        name = d.get('dataset') or os.path.basename(p).replace(
            'robustness_', '').replace('.json', '')
        name = SHORT.get(name, name)
        data[name] = d['results']
    if not data:
        print('No robustness results yet.', flush=True)
        return

    noise_types = sorted({t for v in data.values() for t in v.keys()})
    fig, axes = plt.subplots(1, len(noise_types),
                                figsize=(2.6 * len(noise_types), 2.2),
                                squeeze=False)
    axes = axes[0]
    for i, nt in enumerate(noise_types):
        ax = axes[i]
        for j, (ds, res) in enumerate(data.items()):
            if nt not in res:
                continue
            pts = res[nt]
            xs = [p['level'] for p in pts]
            ys = [p['mean'] * 100 for p in pts]
            es = [p['std'] * 100 for p in pts]
            ax.errorbar(xs, ys, yerr=es, marker='o', color=PALETTE[j],
                         label=ds, linewidth=1.2, markersize=4.5,
                         markeredgecolor='black', markeredgewidth=0.5,
                         capsize=2, elinewidth=0.6)
        ax.set_xlabel(nt.replace('_', ' '))
        ax.set_ylabel('accuracy (%)')
        if i == 0:
            corner_legend(ax, loc='lower left', ncol=1)
    fig.tight_layout()
    fig.savefig(args.out)
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
