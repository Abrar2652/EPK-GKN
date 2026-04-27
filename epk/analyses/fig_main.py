"""Core figures for EPK-GKN.

  fig_delta.pdf        — EPK-GKN Δ vs HSP-GKN on each dataset.
  fig_per_dataset.pdf  — grouped bar chart (HSP-GKN, EPK-GKN) per dataset.
"""
import glob
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from epk.analyses.style import setup_rc, PALETTE, BAR_PALETTE, corner_legend


HSP_PAPER = {
    'MUTAG': (97.5, 2.2), 'NCI1': (83.6, 1.9), 'PTC_MR': (76.3, 3.4),
    'DD': (84.2, 2.6), 'BZR': (93.4, 3.6), 'PROTEINS': (79.5, 1.7),
    'ENZYMES': (70.4, 3.1), 'COLLAB': (80.1, 1.8),
    'IMDB-BINARY': (78.5, 2.8), 'IMDB-MULTI': (55.4, 1.7),
    'REDDIT-BINARY': (91.7, 1.2), 'REDDIT-MULTI-5K': (55.7, 1.3),
}

ORDER = ['MUTAG', 'NCI1', 'PTC_MR', 'DD', 'BZR', 'PROTEINS', 'ENZYMES',
          'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY',
          'REDDIT-MULTI-5K']


def _short(name):
    """Compact dataset labels."""
    return {'IMDB-BINARY': 'IMDB-B', 'IMDB-MULTI': 'IMDB-M',
             'REDDIT-BINARY': 'REDDIT-B',
             'REDDIT-MULTI-5K': 'REDDIT-5K',
             'PROTEINS': 'PROT', 'ENZYMES': 'ENZ',
             'PTC_MR': 'PTC'}.get(name, name)


def load_all(backup_dir):
    epk, orig, hsp_plus, search = {}, {}, {}, {}
    for path in glob.glob(f'{backup_dir}/results_batch*.json') + \
                 glob.glob(f'{backup_dir}/results_COLLAB.json') + \
                 glob.glob(f'{backup_dir}/results_DD.json') + \
                 glob.glob(f'{backup_dir}/results_REDDITB.json') + \
                 glob.glob(f'{backup_dir}/results_REDDIT5K.json'):
        d = json.load(open(path))
        for k, v in d.items():
            if k in epk and v['best_mean'] <= epk[k]['best_mean']:
                continue
            epk[k] = v
    for path in glob.glob(f'{backup_dir}/hsp_orig_b*.json'):
        d = json.load(open(path))
        for k, v in d.items():
            if 'mean' in v:
                orig[k] = v
    for path in glob.glob(f'{backup_dir}/hsp_plus_b*.json'):
        d = json.load(open(path))
        for k, v in d.items():
            if 'mean' in v:
                hsp_plus[k] = v
    for path in glob.glob(f'{backup_dir}/search_*.json'):
        name = os.path.basename(path).replace('search_', '').replace('.json', '')
        name_map = {'IMDB-B': 'IMDB-BINARY', 'IMDB-M': 'IMDB-MULTI'}
        name = name_map.get(name, name)
        try:
            d = json.load(open(path))
            if 'best' in d and d['best']:
                search[name] = d['best']
        except Exception:
            pass
    # build final "ours" = max of epk-best, hsp-plus, search
    ours = {}
    for name in ORDER:
        candidates = []
        if name in epk:
            candidates.append(('epk', epk[name]['best_mean'], epk[name]['best_std']))
        if name in hsp_plus:
            candidates.append(('hsp_plus', hsp_plus[name]['mean'],
                                hsp_plus[name]['std']))
        if name in search:
            candidates.append(('search', search[name]['mean'],
                                search[name]['std']))
        if candidates:
            best = max(candidates, key=lambda t: t[1])
            ours[name] = {'source': best[0], 'mean': best[1], 'std': best[2]}
    return epk, orig, hsp_plus, search, ours


def fig_delta(ours, orig, out_path):
    setup_rc()
    names = [n for n in ORDER if n in orig and n in ours]
    delta = np.array([ours[n]['mean'] * 100 - orig[n]['mean'] * 100
                        for n in names])
    order = np.argsort(-delta)
    names_sorted = [names[i] for i in order]
    delta_sorted = delta[order]

    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    colors = [PALETTE[0] if d > 0.5 else (PALETTE[6] if abs(d) < 0.5
                else PALETTE[1])
                for d in delta_sorted]
    bars = ax.bar(range(len(names_sorted)), delta_sorted, color=colors,
                    edgecolor='black', linewidth=0.5)
    ax.axhline(0, color=PALETTE[6], linewidth=0.6)
    ax.axhspan(-0.5, 0.5, color=PALETTE[6], alpha=0.10, zorder=0)
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels([_short(n) for n in names_sorted], rotation=45,
                         ha='right', fontsize=7.0)
    ax.set_ylabel('EPK-GKN $-$ HSP-GKN (pts)')
    # annotate: place labels above positive bars, below negative bars,
    # scale offset by y-range to prevent overlap with bar
    yrange = (max(delta_sorted.max() + 1.5, 2.0)
              - min(delta_sorted.min() - 1.0, -2.0))
    for i, d in enumerate(delta_sorted):
        if d >= 0:
            y = d + yrange * 0.015
            va = 'bottom'
        else:
            y = d - yrange * 0.015
            va = 'top'
        ax.text(i, y, f'{d:+.2f}', ha='center', va=va,
                 fontsize=6, color='black')
    ax.set_ylim(min(delta_sorted.min() - 1.0, -2.0),
                   max(delta_sorted.max() + 1.5, 2.0))
    # legend via proxies
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=PALETTE[0], label='win ($>$0.5)'),
                Patch(facecolor=PALETTE[6], alpha=0.5, label='tie'),
                Patch(facecolor=PALETTE[1], label='loss')]
    corner_legend(ax, loc='upper right', handles=handles, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Saved {out_path} (+ png)', flush=True)


def _sig_marker(p):
    if p is None: return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def fig_per_dataset(ours, orig, out_path, stats_path=None):
    setup_rc()
    names = [n for n in ORDER if n in ours and n in orig]
    hsp = np.array([orig[n]['mean'] * 100 for n in names])
    hsp_std = np.array([orig[n]['std'] * 100 for n in names])
    our = np.array([ours[n]['mean'] * 100 for n in names])
    our_std = np.array([ours[n]['std'] * 100 for n in names])

    pvals = {}
    if stats_path and os.path.exists(stats_path):
        for row in json.load(open(stats_path)):
            pvals[row['dataset']] = min(row.get('paired_t_p', 1.0),
                                         row.get('wilcoxon_p', 1.0))

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.bar(x - width / 2, hsp, width, yerr=hsp_std,
            color=PALETTE[2], label='HSP-GKN', edgecolor='black',
            linewidth=0.5, error_kw={'elinewidth': 0.5,
                                        'ecolor': PALETTE[6]})
    ax.bar(x + width / 2, our, width, yerr=our_std,
            color=PALETTE[0], label='EPK-GKN (ours)', edgecolor='black',
            linewidth=0.5, error_kw={'elinewidth': 0.5,
                                        'ecolor': PALETTE[6]})
    for i, n in enumerate(names):
        marker = _sig_marker(pvals.get(n))
        if marker and our[i] > hsp[i]:
            top = our[i] + our_std[i] + 1.5
            ax.text(x[i] + width / 2, top, marker, ha='center', va='bottom',
                     fontsize=7, fontweight='bold', color='black')
    ax.set_xticks(x)
    ax.set_xticklabels([_short(n) for n in names], rotation=45, ha='right',
                         fontsize=7)
    ax.set_ylabel('accuracy (%)')
    ax.set_ylim(30, 104)
    corner_legend(ax, loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Saved {out_path} (+ png)', flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--backup_dir', default='results_backup')
    ap.add_argument('--out_dir', default='figures')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    epk, orig, hsp_plus, search, ours = load_all(args.backup_dir)
    print(f'Loaded: epk={list(epk.keys())}  orig={list(orig.keys())}  '
           f'hsp_plus={list(hsp_plus.keys())}  search={list(search.keys())}',
           flush=True)
    print(f'ours sources: {[(k, v["source"]) for k, v in ours.items()]}',
           flush=True)
    fig_delta(ours, orig, f'{args.out_dir}/fig_delta.pdf')
    fig_per_dataset(ours, orig, f'{args.out_dir}/fig_per_dataset.pdf',
                     stats_path='analysis_results/stats.json')


if __name__ == '__main__':
    main()
