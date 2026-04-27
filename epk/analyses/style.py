"""Shared matplotlib style for NeurIPS-quality figures.

Design rules:
  - No plot title (dataset / panel name goes as xlabel or annotation).
  - Legends in lower-right or upper-right corner with transparent frame.
  - Contrastive categorical palette (colourblind-friendly).
  - Paper-ready sizes (column-width ~3.3in, text-width ~7in).
  - Serif for axes labels; sans-serif for tick labels.
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive
import matplotlib.pyplot as plt
from matplotlib import rcParams


# Okabe–Ito colourblind palette, very common in NeurIPS figures
OKABE = {
    'orange':  '#E69F00',
    'skyblue': '#56B4E9',
    'green':   '#009E73',
    'yellow':  '#F0E442',
    'blue':    '#0072B2',
    'verm':    '#D55E00',
    'pink':    '#CC79A7',
    'black':   '#000000',
    'gray':    '#666666',
}

# curated order for up-to-7-series plots
PALETTE = [OKABE[c] for c in
            ('blue', 'verm', 'green', 'orange', 'skyblue', 'pink', 'gray')]

# wider categorical palette for bar charts
BAR_PALETTE = [OKABE[c] for c in
                ('blue', 'verm', 'green', 'orange', 'skyblue', 'pink',
                 'yellow', 'gray')]


def setup_rc():
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 7.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.4,
        'grid.linestyle': '-',
        'legend.frameon': True,
        'legend.framealpha': 0.70,
        'legend.edgecolor': 'none',
        'legend.fancybox': False,
        'lines.linewidth': 1.3,
        'lines.markersize': 4,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
        'figure.dpi': 110,
    })


def new_fig(width=3.3, height=2.3):
    setup_rc()
    fig = plt.figure(figsize=(width, height))
    return fig


def corner_legend(ax, loc='lower right', ncol=1, **kwargs):
    leg = ax.legend(loc=loc, ncol=ncol, handlelength=1.5, handletextpad=0.4,
                     columnspacing=0.8, borderpad=0.3, borderaxespad=0.3,
                     **kwargs)
    if leg is not None:
        leg.get_frame().set_alpha(0.70)
        leg.get_frame().set_linewidth(0)
    return leg
