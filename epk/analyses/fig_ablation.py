"""Figure: component ablation across datasets.

For each dataset, shows a bar chart with 4 groups (K=0,1,2 × struct × sq ×
attn off/on). Bars are coloured by configuration. We render one panel per
dataset in a grid layout.

Alternative: a "contribution heatmap" showing improvement of each component
over the base (K=0, no struct, no sq, no attn).
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

from epk.analyses.style import setup_rc, PALETTE, BAR_PALETTE, corner_legend

SHORT = {'IMDBB': 'IMDB-B', 'IMDBM': 'IMDB-M'}


def _short(f):
    n = os.path.basename(f).replace('ablation_', '').replace('.json', '')
    return SHORT.get(n, n)


def load_abl(glob_pat='analysis_results/ablation_*.json'):
    out = {}
    for path in sorted(glob.glob(glob_pat)):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        name = d.get('dataset', _short(path))
        out[name] = d['results']
    return out


def fig_component_contrib(all_res, out_path):
    """Heatmap of Δ accuracy relative to base config (K=0, struct=0,
    sq=0, attn=0) for each (dataset, single-component-on) pair."""
    setup_rc()
    components = [
        ('K=1', dict(K=1, struct=False, sq=False, attn=False)),
        ('K=2', dict(K=2, struct=False, sq=False, attn=False)),
        ('+struct', dict(K=0, struct=True, sq=False, attn=False)),
        ('+sq', dict(K=0, struct=False, sq=True, attn=False)),
        ('K=1+struct', dict(K=1, struct=True, sq=False, attn=False)),
        ('K=2+struct+sq', dict(K=2, struct=True, sq=True, attn=False)),
    ]
    datasets = list(all_res.keys())
    mat = np.full((len(datasets), len(components)), np.nan)
    base = {}
    for i, name in enumerate(datasets):
        recs = all_res[name]
        for r in recs:
            if r['K'] == 0 and not r['struct'] and not r['sq'] and not r['attn']:
                base[name] = r['mean']
                break
    for i, name in enumerate(datasets):
        if name not in base:
            continue
        for j, (_, conf) in enumerate(components):
            for r in all_res[name]:
                if (r['K'] == conf['K'] and r['struct'] == conf['struct']
                        and r['sq'] == conf['sq'] and r['attn'] == conf['attn']):
                    mat[i, j] = (r['mean'] - base[name]) * 100
                    break
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([c[0] for c in components], fontsize=7.5,
                        rotation=25, ha='right')
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=7.5)
    for i in range(len(datasets)):
        for j in range(len(components)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                         fontsize=6, color=('white' if abs(v) > vmax * 0.55
                                                else PALETTE[6]))
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label('$\\Delta$ accuracy vs base (pts)', fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Saved {out_path}', flush=True)


def fig_ablation_bars(all_res, out_path):
    """Per-dataset bar chart for 4 chosen configs."""
    setup_rc()
    configs = [
        ('base', dict(K=0, struct=False, sq=False, attn=False)),
        ('K=1', dict(K=1, struct=False, sq=False, attn=False)),
        ('K=1+struct', dict(K=1, struct=True, sq=False, attn=False)),
        ('K=2+struct+sq', dict(K=2, struct=True, sq=True, attn=False)),
    ]
    datasets = list(all_res.keys())
    vals = np.full((len(datasets), len(configs)), np.nan)
    for i, name in enumerate(datasets):
        for j, (_, c) in enumerate(configs):
            for r in all_res[name]:
                if (r['K'] == c['K'] and r['struct'] == c['struct']
                        and r['sq'] == c['sq'] and r['attn'] == c['attn']):
                    vals[i, j] = r['mean'] * 100
                    break

    x = np.arange(len(datasets))
    width = 0.20
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    for j, (label, _) in enumerate(configs):
        off = (j - 1.5) * width
        ax.bar(x + off, vals[:, j], width, label=label,
                color=PALETTE[j], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=7)
    ax.set_ylabel('accuracy (%)')
    # auto-set y-limits
    lo = np.nanmin(vals) - 3
    hi = np.nanmax(vals) + 5
    ax.set_ylim(max(lo, 40), min(hi, 104))
    corner_legend(ax, loc='upper center', ncol=4,
                   bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Saved {out_path}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='analysis_results/ablation_*.json')
    ap.add_argument('--out_dir', default='figures')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    all_res = load_abl(args.glob)
    if not all_res:
        print('No ablation results found.', flush=True)
        return
    fig_component_contrib(all_res, f'{args.out_dir}/fig_ablation_heatmap.pdf')


if __name__ == '__main__':
    main()
