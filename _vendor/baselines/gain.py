"""GAIN (Yoon et al., 2018) — generative adversarial imputation, deps-free.

Generator fills missing entries; discriminator predicts which entries are real vs
imputed, aided by a hint mask. Adapted to standardised data (linear generator
output, MSE reconstruction). Point imputer (m=1). A new deep baseline for the
journal comparison (not in the workshop table).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from baselines.base import Imputer


class _MLP(nn.Module):
    def __init__(self, din, dout, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


class GAINImputer(Imputer):
    name = "GAIN"

    def __init__(self, hidden=128, alpha=10.0, hint_rate=0.9, epochs=100,
                 lr=1e-3, batch_size=256, device="cpu"):
        self.hidden, self.alpha, self.hint_rate = hidden, alpha, hint_rate
        self.epochs, self.lr, self.batch_size, self.device = epochs, lr, batch_size, device

    def fit(self, Xobs_tr, M_tr):
        d = Xobs_tr.shape[1]; dev = self.device
        self.G = _MLP(2 * d, d, self.hidden).to(dev)
        self.D = _MLP(2 * d, d, self.hidden).to(dev)
        X = torch.tensor(np.nan_to_num(Xobs_tr, nan=0.0), dtype=torch.float32)
        M = torch.tensor(np.asarray(M_tr), dtype=torch.float32)
        oG = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        oD = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(X, M), batch_size=self.batch_size, shuffle=True)
        eps = 1e-7
        for _ in range(self.epochs):
            for xb, mb in loader:
                xb, mb = xb.to(dev), mb.to(dev)
                z = torch.randn_like(xb)
                xfill = mb * xb + (1 - mb) * z
                b = (torch.rand_like(mb) < self.hint_rate).float()
                hint = b * mb + 0.5 * (1 - b)
                # D step
                gh = self.G(torch.cat([xfill, mb], 1)).detach()
                xt = mb * xb + (1 - mb) * gh
                dp = torch.sigmoid(self.D(torch.cat([xt, hint], 1)))
                d_loss = -(mb * torch.log(dp + eps) + (1 - mb) * torch.log(1 - dp + eps)).mean()
                oD.zero_grad(); d_loss.backward(); oD.step()
                # G step
                gh = self.G(torch.cat([xfill, mb], 1))
                xt = mb * xb + (1 - mb) * gh
                dp = torch.sigmoid(self.D(torch.cat([xt, hint], 1)))
                g_adv = -((1 - mb) * torch.log(dp + eps)).mean()
                g_rec = (mb * (gh - xb) ** 2).sum() / (mb.sum() + eps)
                g_loss = g_adv + self.alpha * g_rec
                oG.zero_grad(); g_loss.backward(); oG.step()
        self.G.eval()
        return self

    @torch.no_grad()
    def impute(self, Xobs_te, M_te, m):
        X = torch.tensor(np.nan_to_num(Xobs_te, nan=0.0), dtype=torch.float32, device=self.device)
        M = torch.tensor(np.asarray(M_te), dtype=torch.float32, device=self.device)
        gh = self.G(torch.cat([M * X + (1 - M) * torch.randn_like(X), M], 1))
        out = (M * X + (1 - M) * gh).cpu().numpy()
        return out[None].astype(np.float32)
