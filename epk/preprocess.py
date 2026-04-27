"""EPK-GKN preprocessing: shortest-path count matrices + enriched features."""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import igraph as ig
from collections import defaultdict
from tqdm import tqdm


def _graph_to_raw(g, label):
    n = g.number_of_nodes()
    X = g.ndata['node_attr'].to(torch.float32)
    src, dst = g.edges()
    src, dst = src.tolist(), dst.tolist()

    ig_g = ig.Graph(edges=list(zip(src, dst)), directed=False, n=n)
    edge_set = set()
    for u, v in zip(src, dst):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edge_set.add((a, b))
    if len(edge_set) == 0:
        A_idx = torch.zeros((2, 0), dtype=torch.long)
    else:
        ea = torch.tensor([e[0] for e in edge_set], dtype=torch.long)
        eb = torch.tensor([e[1] for e in edge_set], dtype=torch.long)
        A_idx = torch.stack([torch.cat([ea, eb]),
                             torch.cat([eb, ea])], dim=0)

    paths = []
    for i in range(n):
        p = ig_g.get_shortest_paths(i, output='vpath', algorithm='dijkstra')
        paths.extend(p)

    idx_by_len = defaultdict(list)
    for path in paths:
        if len(path) > 0:
            idx_by_len[len(path)].append(path)
    idx_by_len = {k: np.array(v) for k, v in idx_by_len.items()}
    idx_by_len.pop(0, None)
    idx_by_len.pop(1, None)
    self_paths = np.array([[i] for i in range(n)])
    idx_by_len[1] = self_paths

    dict_count = {}
    for key, value in idx_by_len.items():
        cols = [np.bincount(value[:, i], minlength=n) for i in range(value.shape[1])]
        dict_count[key] = torch.stack(
            [torch.as_tensor(c, dtype=torch.float32) for c in cols], dim=1
        )

    if len(dict_count) > 0:
        for i in range(min(dict_count.keys()), max(dict_count.keys()) + 1):
            if i not in dict_count:
                dict_count[i] = torch.zeros((n, i), dtype=torch.float32)
    dict_count = dict(sorted(dict_count.items()))

    deg = torch.tensor(ig_g.degree(), dtype=torch.float32)
    core = torch.tensor(ig_g.coreness(), dtype=torch.float32)
    try:
        clust = torch.tensor(ig_g.transitivity_local_undirected(mode="zero"),
                             dtype=torch.float32)
    except Exception:
        clust = torch.zeros(n, dtype=torch.float32)

    return {
        'X': X, 'A_idx': A_idx, 'M': dict_count,
        'deg': deg, 'core': core, 'clust': clust,
        'label': label, 'n': n,
    }


def preprocess_dataset(dataset, cache_dir, name):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{name}_epk_raw.pt')
    meta_path = os.path.join(cache_dir, f'{name}_epk_meta.pt')
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        print(f'[cache] loading {cache_path}')
        raw = torch.load(cache_path, weights_only=False)
        meta = torch.load(meta_path, weights_only=False)
        return raw, meta
    raw = []
    lens_histogram = defaultdict(int)
    start = time.time()
    for g, label in tqdm(dataset, desc=f'preproc {name}'):
        rec = _graph_to_raw(g, label)
        raw.append(rec)
        for l, Ml in rec['M'].items():
            lens_histogram[l] += int(Ml[:, 0].sum().item())
    meta = {
        'lens_histogram': dict(lens_histogram),
        'base_feat_dim': dataset[0][0].ndata['node_attr'].shape[1],
        'num_classes': int(dataset.num_labels),
    }
    torch.save(raw, cache_path)
    torch.save(meta, meta_path)
    print(f'[preproc] {name}: done in {time.time() - start:.1f}s')
    return raw, meta


def _A_hat(g_rec, add_self_loops=True):
    n = g_rec['n']
    A_idx = g_rec['A_idx']
    if add_self_loops:
        sl = torch.arange(n).unsqueeze(0).repeat(2, 1)
        A_idx = torch.cat([A_idx, sl], dim=1)
    vals = torch.ones(A_idx.size(1), dtype=torch.float32)
    A = torch.sparse_coo_tensor(A_idx, vals, (n, n)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    d_inv = torch.pow(deg.clamp(min=1e-6), -0.5)
    idx = A.indices()
    vals = A.values() * d_inv[idx[0]] * d_inv[idx[1]]
    return torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()


def enrich(g_rec, K=2, use_structural=True, include_sq=False,
            struct_mask=(1, 1, 1)):
    """struct_mask selects (deg, core, clust); ignored if use_structural=False."""
    X = g_rec['X']
    n = g_rec['n']
    if n == 0:
        return X
    Ah = _A_hat(g_rec)
    feats = [X]
    cur = X
    for _ in range(K):
        cur = torch.sparse.mm(Ah, cur)
        feats.append(cur)
    if use_structural:
        deg = g_rec['deg']; core = g_rec['core']; clust = g_rec['clust']
        d_max = deg.max().clamp(min=1.0)
        c_max = core.max().clamp(min=1.0)
        cols = []
        if struct_mask[0]: cols.append(deg / d_max)
        if struct_mask[1]: cols.append(core / c_max)
        if struct_mask[2]: cols.append(clust)
        if cols:
            feats.append(torch.stack(cols, dim=1))
    out = torch.cat(feats, dim=1)
    if include_sq:
        out = torch.cat([out, X * X], dim=1)
    return out


def build_feature_tensor(raw, lens_to_keep, K=2, use_structural=True,
                          normalize_rows=False, include_sq=False,
                          struct_mask=(1, 1, 1)):
    d_enr = None
    for g in raw:
        if g is None or g['n'] == 0:
            continue
        d_enr = enrich(g, K=K, use_structural=use_structural,
                       include_sq=include_sq,
                       struct_mask=struct_mask).shape[1]
        break
    assert d_enr is not None
    lens_to_keep = sorted(lens_to_keep)
    segments, offset = [], 0
    for l in lens_to_keep:
        segments.append((l, offset, l * d_enr))
        offset += l * d_enr
    total_dim = offset
    out = torch.zeros((len(raw), total_dim), dtype=torch.float32)
    labels = torch.zeros(len(raw), dtype=torch.long)
    for i, g in enumerate(raw):
        if g is None or g['n'] == 0:
            labels[i] = g['label'].squeeze().long() if g is not None else 0
            continue
        X = enrich(g, K=K, use_structural=use_structural,
                    include_sq=include_sq, struct_mask=struct_mask)
        for (l, start, length) in segments:
            Ml = g['M'].get(l)
            if Ml is None:
                continue
            phi = X.t() @ Ml
            out[i, start:start + length] = phi.t().reshape(-1)
        labels[i] = g['label'].squeeze().long()
    if normalize_rows:
        out = F.normalize(out, p=2, dim=1)
    return out, d_enr, segments, labels
