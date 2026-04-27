"""TUDataset loaders following HSP-GKN's feature-prep conventions."""
import torch
import torch.nn.functional as F
import dgl


def set_node_degree_as_feature(dataset, norm_degree=True):
    for g, _ in dataset:
        degrees = g.in_degrees().float()
        if norm_degree:
            degrees = F.normalize(degrees.unsqueeze(1), p=2, dim=0)
        else:
            degrees = degrees.unsqueeze(1)
        g.ndata['node_attr'] = degrees


def set_attr_as_label(dataset):
    all_labels = torch.cat([g.ndata['node_labels'] for g, _ in dataset])
    uniq = torch.unique(all_labels)
    nc = uniq.shape[0]
    label_map = {v.item(): i for i, v in enumerate(uniq)}
    for g, _ in dataset:
        ml = torch.tensor([label_map[x.item()] for x in g.ndata['node_labels']])
        g.ndata['node_attr'] = F.one_hot(ml, num_classes=nc).float()


def set_label_as_degree(dataset):
    for g, _ in dataset:
        g.ndata['node_labels'] = g.in_degrees().float().unsqueeze(1)


def cat_attr_label(dataset):
    all_labels = torch.cat([g.ndata['node_labels'] for g, _ in dataset])
    uniq = torch.unique(all_labels)
    nc = uniq.shape[0]
    label_map = {v.item(): i for i, v in enumerate(uniq)}
    for g, _ in dataset:
        ml = torch.tensor([label_map[x.item()] for x in g.ndata['node_labels']])
        feats = g.ndata.get('node_attr', torch.zeros(g.num_nodes(), 0))
        oh = F.one_hot(ml, num_classes=nc).float()
        g.ndata['node_attr'] = torch.cat([feats, oh], dim=1)


def load_dataset(name):
    ds = dgl.data.TUDataset(name)
    if name in ('PROTEINS', 'PROTEINS_full', 'ENZYMES', 'SYNTHETIC', 'BZR',
                'COX2', 'Synthie', 'DHFR'):
        cat_attr_label(ds)
    elif name in ('NCI1', 'DD', 'MUTAG', 'PTC_MR', 'NCI109'):
        set_attr_as_label(ds)
    elif name in ('COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'reddit_threads'):
        set_label_as_degree(ds)
        set_attr_as_label(ds)
    elif name in ('REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K'):
        set_node_degree_as_feature(ds, norm_degree=True)
    else:
        raise ValueError(f'unsupported dataset {name}')
    return ds
