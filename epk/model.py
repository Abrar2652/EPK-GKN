"""EPK-GKN model: per-length encoder + optional cross-length attention + head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class LinearEncoder(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, h_dim, bias=False)
        init.xavier_uniform_(self.fc.weight)
    def forward(self, x):
        return self.fc(x)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim, bias=False)
        self.norm = nn.LayerNorm(h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim, bias=True)
        self.drop = nn.Dropout(dropout)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
    def forward(self, x):
        h = self.fc1(x)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)
        return h


class EPKGKN(nn.Module):
    def __init__(self, segments, d_enr, n_classes,
                 hid_per_len=32, fc_hidden=256, dropout=0.2,
                 encoder='mlp', use_attn=True, use_len_embed=True,
                 use_len_gate=True, attn_heads=4, attn_depth=1,
                 label_smooth=0.0, norm='layer',
                 per_len_channels=None):
        super().__init__()
        self.segments = segments
        self.d_enr = d_enr
        self.use_attn = use_attn
        self.use_len_embed = use_len_embed
        self.use_len_gate = use_len_gate
        self.label_smooth = label_smooth
        self.num_lens = len(segments)
        if per_len_channels is None:
            self.channels = [hid_per_len] * self.num_lens
        else:
            assert len(per_len_channels) == self.num_lens
            self.channels = list(per_len_channels)
        self.attn_dim = max(self.channels) if use_attn else 0

        self.encoders = nn.ModuleList()
        for (l, _, _), c in zip(segments, self.channels):
            if encoder == 'mlp':
                self.encoders.append(MLPEncoder(l * d_enr, c, dropout=dropout))
            else:
                self.encoders.append(LinearEncoder(l * d_enr, c))

        if use_attn:
            self.token_proj = nn.ModuleList(
                [nn.Linear(c, self.attn_dim) for c in self.channels])
            for mod in self.token_proj:
                init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    init.zeros_(mod.bias)
            if use_len_embed:
                self.len_embed = nn.Embedding(self.num_lens, self.attn_dim)
                init.normal_(self.len_embed.weight, std=0.02)
            if use_len_gate:
                self.len_gate = nn.Parameter(torch.ones(self.num_lens))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.attn_dim, nhead=attn_heads,
                dim_feedforward=self.attn_dim * 2, dropout=dropout,
                batch_first=True, activation='gelu', norm_first=True,
            )
            self.attn = nn.TransformerEncoder(enc_layer, num_layers=attn_depth)
            readout_dim = self.attn_dim * self.num_lens
        else:
            self.attn = None
            if use_len_gate:
                self.len_gate = nn.Parameter(torch.ones(self.num_lens))
            readout_dim = sum(self.channels)

        Norm = nn.LayerNorm if norm == 'layer' else nn.BatchNorm1d
        self.head = nn.Sequential(
            nn.Linear(readout_dim, fc_hidden),
            Norm(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            Norm(fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, n_classes),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        enc_outs = []
        for i, (l, start, length) in enumerate(self.segments):
            chunk = x[:, start:start + length]
            h = self.encoders[i](chunk)
            enc_outs.append(h)
        if self.attn is not None:
            projected = []
            for i, h in enumerate(enc_outs):
                p = self.token_proj[i](h)
                if self.use_len_embed:
                    p = p + self.len_embed.weight[i].unsqueeze(0)
                if self.use_len_gate:
                    p = p * self.len_gate[i]
                projected.append(p)
            tok = torch.stack(projected, dim=1)
            tok = self.attn(tok)
            flat = tok.reshape(tok.size(0), -1)
        else:
            if self.use_len_gate:
                enc_outs = [h * self.len_gate[i] for i, h in enumerate(enc_outs)]
            flat = torch.cat(enc_outs, dim=1)
        return self.head(flat)

    def loss(self, logits, y):
        return F.cross_entropy(logits, y, label_smoothing=self.label_smooth)
