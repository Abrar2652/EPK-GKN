# EPK-GKN: Enriched Polynomial-Kernel Graph Kernel Network

EPK-GKN is a graph-classification model that combines three ideas
into a single end-to-end-trainable kernel network:

1. **Polynomial node-feature enrichment.** Each node's representation
   is a `K`-degree polynomial in the normalized adjacency `Â`,
   concatenated with three topology descriptors (degree, k-core,
   clustering coefficient) and an optional second-moment term:

   ```
   X_enr = [ X, Â X, Â² X, …, Â^K X, deg, core, clust, (X ⊙ X) ]
   ```

   where `Â = D^{-1/2}(A+I) D^{-1/2}`.

2. **Hierarchical shortest-path aggregation.** For each path length
   `l = 1, …, δ(G)` we maintain a count matrix `M_l` and form the
   length-`l` graph descriptor `φ_l(G) = X_enr^⊤ M_l`. This is a
   positive-definite kernel.

3. **Differentiable matching head.** Length-indexed learnable graph
   anchors are matched against `{φ_l(G)}` via inner product;
   frequency-proportional channel allocation distributes capacity
   across path lengths; an MLP closes the classification loop.

The combination is **strictly more expressive** than non-enriched
shortest-path kernels (we exhibit a graph pair the latter collapses
but EPK-GKN distinguishes).


## What's where

| | |
|---|---|
| `SUMMARY_FOR_USER.md` | One-page user-facing summary |
| `epk/` | EPK-GKN package: `data.py`, `preprocess.py`, `model.py`, `train.py`, `run.py` |
| `epk/analyses/` | All analysis scripts (ablation, sensitivity, scalability, efficiency, robustness, stats, training curves) and figure generators |
| `hsp_original_multiseed.py` | HSP-GKN baseline runner under our 10-fold × 3-seed protocol (uses `hsp_gkn_reference/`) |
| `hsp_gkn_reference/` | HSP-GKN authors' original code + paper PDF, kept verbatim for the baseline |
| `figures/` | All paper figures (PDF + PNG) |
| `analysis_results/` | All analysis output JSONs |
| `results_backup/` | Raw per-dataset benchmark JSONs |
| `cache_epk/` | Pre-computed shortest-path features (delete to regenerate; ≈85 min for REDDIT-5K) |

## Quick start

```bash
# Install deps (Python 3.10 recommended)
pip install torch dgl python-igraph scikit-learn matplotlib scipy tqdm pyyaml pandas

# Train EPK-GKN on one dataset
python3 -m epk.run --dataset COLLAB --device cuda:0

# Run HSP-GKN baseline under our protocol
python3 hsp_original_multiseed.py --dataset COLLAB --device cuda:0

# Regenerate all figures from existing JSONs
python3 -m epk.analyses.fig_main
python3 -m epk.analyses.fig_ablation
python3 -m epk.analyses.fig_ablation_enriched
python3 -m epk.analyses.fig_sensitivity --out figures/fig_sensitivity.pdf
python3 -m epk.analyses.fig_scalability --out figures/fig_scalability.pdf
python3 -m epk.analyses.fig_training
python3 -m epk.analyses.fig_robustness
```

