"""10-fold CV training with multi-seed per-fold averaging."""
import time
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from epk.model import EPKGKN


def _build_model(segments, d_enr, n_classes, cfg, device):
    return EPKGKN(
        segments=segments, d_enr=d_enr, n_classes=n_classes,
        hid_per_len=cfg['hid_per_len'], fc_hidden=cfg['fc_hidden'],
        dropout=cfg['dropout'], encoder=cfg.get('encoder', 'mlp'),
        use_attn=cfg.get('use_attn', True),
        use_len_embed=cfg.get('use_len_embed', True),
        use_len_gate=cfg.get('use_len_gate', True),
        attn_heads=cfg.get('attn_heads', 4),
        attn_depth=cfg.get('attn_depth', 1),
        label_smooth=cfg.get('label_smooth', 0.0),
        norm=cfg.get('norm', 'layer'),
        per_len_channels=cfg.get('per_len_channels', None),
    ).to(device)


def train_one_fold(x_train, y_train, x_test, y_test, segments, d_enr,
                    n_classes, cfg, device, return_history=False):
    model = _build_model(segments, d_enr, n_classes, cfg, device)
    opt_name = cfg.get('optimizer', 'adamw')
    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],
                                weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'],
                                 weight_decay=cfg['weight_decay'])
    epochs = cfg['epochs']
    use_cos = cfg.get('cosine', True)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                  if use_cos else None)
    grad_clip = cfg.get('grad_clip', 2.0)
    best_acc = 0.0
    history = [] if return_history else None
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = model.loss(logits, y_train)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.eval()
        with torch.no_grad():
            tr_pred = model(x_train).argmax(dim=1)
            tr_acc = (tr_pred == y_train).float().mean().item()
            pred = model(x_test).argmax(dim=1)
            acc = (pred == y_test).float().mean().item()
            if acc > best_acc:
                best_acc = acc
            if return_history:
                history.append({'epoch': ep, 'loss': float(loss.item()),
                                  'train_acc': tr_acc, 'test_acc': acc})
    if return_history:
        return best_acc, history
    return best_acc


def ten_fold_cv(x, y, segments, d_enr, n_classes, cfg, device, verbose=True,
                 seed=0, n_seeds=None):
    if n_seeds is None:
        n_seeds = cfg.get('n_seeds', 1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    per_fold, times = [], []
    for fold, (tr, te) in enumerate(skf.split(x.cpu().numpy(), y.cpu().numpy())):
        x_tr = x[tr].to(device); x_te = x[te].to(device)
        y_tr = y[tr].to(device); y_te = y[te].to(device)
        seed_accs = []
        t0 = time.time()
        for s in range(n_seeds):
            if n_seeds > 1:
                torch.manual_seed(seed * 97 + s * 13 + fold * 7)
            acc = train_one_fold(x_tr, y_tr, x_te, y_te, segments, d_enr,
                                  n_classes, cfg, device)
            seed_accs.append(acc)
        dt = time.time() - t0
        fold_acc = float(np.mean(seed_accs))
        per_fold.append(fold_acc)
        times.append(dt)
        if verbose:
            print(f'  fold {fold:2d}: acc={fold_acc:.4f}  time={dt:.1f}s')
    mean = float(np.mean(per_fold))
    std = float(np.std(per_fold))
    if verbose:
        print(f'  10-fold: {mean:.4f} ± {std:.4f}')
    return mean, std, per_fold
