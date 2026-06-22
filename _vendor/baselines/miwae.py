"""MIWAE (Mattei & Frellsen, 2019) — masked importance-weighted VAE imputer.

Gaussian encoder q(z|x_obs) and decoder p(x|z); trained on the masked IWAE bound
(observed dims only); single imputation via self-normalised importance weights over
K posterior samples. Config follows paper B.3: latent 5, hidden 64, K=5, 200 epochs.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from baselines.base import Imputer

_LOG2PI = float(np.log(2 * np.pi))


class _Net(nn.Module):
    def __init__(self, d, h=64, L=5):
        super().__init__()
        self.L = L
        self.enc = nn.Sequential(nn.Linear(2 * d, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2 * L))
        self.dec = nn.Sequential(nn.Linear(L, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2 * d))

    def encode(self, x, m):
        mu, logv = self.enc(torch.cat([x * m, m], 1)).chunk(2, 1)
        return mu, logv.clamp(-6, 6)

    def decode(self, z):
        mu, logv = self.dec(z).chunk(2, 1)
        return mu, logv.clamp(-6, 6)


def _logw(net, x, m, K):
    """Return log importance weights (K,B) and decoder means mu_x (K,B,d)."""
    B, d = x.shape
    mu_z, logv_z = net.encode(x, m)                       # (B,L)
    std_z = (0.5 * logv_z).exp()
    z = mu_z.unsqueeze(0) + std_z.unsqueeze(0) * torch.randn(K, B, net.L, device=x.device)
    mu_x, logv_x = net.decode(z.reshape(K * B, net.L))
    mu_x = mu_x.reshape(K, B, d); logv_x = logv_x.reshape(K, B, d)
    var_x = logv_x.exp()
    logpx = ((-0.5 * (((x - mu_x) ** 2) / var_x + logv_x + _LOG2PI)) * m).sum(-1)
    logpz = (-0.5 * (z ** 2 + _LOG2PI)).sum(-1)
    logqz = (-0.5 * (((z - mu_z) ** 2) / std_z.pow(2) + logv_z + _LOG2PI)).sum(-1)
    return logpx + logpz - logqz, mu_x


class MIWAEImputer(Imputer):
    name = "MIWAE"

    def __init__(self, latent=5, hidden=64, K=5, epochs=200, lr=1e-3,
                 batch_size=256, device="cpu"):
        self.latent, self.hidden, self.K = latent, hidden, K
        self.epochs, self.lr, self.batch_size, self.device = epochs, lr, batch_size, device

    def fit(self, Xobs_tr, M_tr):
        d = Xobs_tr.shape[1]
        self.net = _Net(d, self.hidden, self.latent).to(self.device)
        X = torch.tensor(np.nan_to_num(Xobs_tr, nan=0.0), dtype=torch.float32)
        M = torch.tensor(np.asarray(M_tr), dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(X, M), batch_size=self.batch_size, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for xb, mb in loader:
                xb, mb = xb.to(self.device), mb.to(self.device)
                logw, _ = _logw(self.net, xb, mb, self.K)
                loss = -(torch.logsumexp(logw, 0) - np.log(self.K)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def impute(self, Xobs_te, M_te, m):
        X = torch.tensor(np.nan_to_num(Xobs_te, nan=0.0), dtype=torch.float32, device=self.device)
        M = torch.tensor(np.asarray(M_te), dtype=torch.float32, device=self.device)
        obs = np.asarray(M_te) == 1; base = np.nan_to_num(Xobs_te, nan=0.0)
        if m == 1:                                          # point: IW posterior mean
            logw, mu_x = _logw(self.net, X, M, self.K * 4)
            ximp = (torch.softmax(logw, 0).unsqueeze(-1) * mu_x).sum(0).cpu().numpy()
            return np.where(obs, base, ximp)[None].astype(np.float32)
        outs = []                                           # MI: m sampled completions
        for _ in range(m):
            mu_z, logv_z = self.net.encode(X, M)
            z = mu_z + (0.5 * logv_z).exp() * torch.randn_like(mu_z)
            mu_x, logv_x = self.net.decode(z)
            xs = (mu_x + (0.5 * logv_x).exp() * torch.randn_like(mu_x)).cpu().numpy()
            outs.append(np.where(obs, base, xs))
        return np.stack(outs).astype(np.float32)            # (m, n, d)
