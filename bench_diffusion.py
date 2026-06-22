"""Self-contained GPU diffusion imputers behind the shared Imputer interface.

Both expose `fit(Xobs_tr, M_tr)` / `impute(Xobs_te, M_te, m) -> (m, n, d)` and an
`nfe` attribute, so the SAME harness scores them exactly like MissFlow.

- TabDiffImputer  : conditional DDPM for tabular imputation (CSDI / MissDiff family).
                    Trained MissDiff-style on incomplete data (loss masked to observed
                    entries); imputed by RePaint-style conditional reverse diffusion.
                    nfe = number of reverse steps (default 50; set 100-150 for the
                    "diffusion is accurate but slow" regime).
- DiffPuterImputer: the same denoiser wrapped in an EM loop (impute, retrain on the
                    completed data), following DiffPuter.

These are faithful *reference* implementations, not the official repos; swap in the
official code by giving a class with the same fit/impute/nfe contract.
"""
from __future__ import annotations
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:                                   # torch installed on the GPU box
    torch = None


# ----------------------------------------------------------------------- network
def _timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half, 1))
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _make_net(d, hidden, tdim, nlayers):
    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.tdim = tdim
            layers = [nn.Linear(3 * d + tdim, hidden), nn.SiLU()]
            for _ in range(nlayers - 1):
                layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            self.body = nn.Sequential(*layers)
            self.out = nn.Linear(hidden, d)

        def forward(self, xt, t, cond, mask):
            h = torch.cat([xt, cond, mask, _timestep_embedding(t, self.tdim)], dim=-1)
            return self.out(self.body(h))
    return Denoiser()


def _cosine_alpha_bar(T, device, s=0.008):
    steps = torch.arange(T + 1, device=device, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    return (f / f[0]).clamp(1e-5, 1.0).float()           # (T+1,), abar[0]=1, abar[T]≈0


# ----------------------------------------------------------------- TabDiff (CSDI/MissDiff)
class TabDiffImputer:
    name = "TabDiff"

    def __init__(self, T=50, epochs=200, hidden=256, tdim=128, nlayers=4,
                 lr=1e-3, batch=256, device="cuda", verbose=False):
        self.T, self.epochs, self.hidden, self.tdim, self.nlayers = T, epochs, hidden, tdim, nlayers
        self.lr, self.batch, self.device, self.verbose = lr, batch, device, verbose
        self.nfe = T
        self.net = None

    @staticmethod
    def _prep(Xobs, M):
        X = np.nan_to_num(np.asarray(Xobs, dtype=np.float32), nan=0.0)
        return X, np.asarray(M, dtype=np.float32)        # M: 1=observed

    def fit(self, Xobs_tr, M_tr):
        assert torch is not None, "PyTorch is required (install on the GPU server)."
        X, M = self._prep(Xobs_tr, M_tr)
        n, d = X.shape
        dev = self.device
        self.d = d
        self.ab = _cosine_alpha_bar(self.T, dev)
        self.net = _make_net(d, self.hidden, self.tdim, self.nlayers).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        Xt = torch.tensor(X, device=dev)
        Mt = torch.tensor(M, device=dev)
        cond = Xt * Mt                                   # observed values, 0 at missing
        self.net.train()
        for ep in range(self.epochs):
            idx = torch.randperm(n, device=dev)
            last = 0.0
            for s in range(0, n, self.batch):
                b = idx[s:s + self.batch]
                x0, mask, c = Xt[b], Mt[b], cond[b]
                t = torch.randint(1, self.T + 1, (len(b),), device=dev)
                abt = self.ab[t][:, None]
                eps = torch.randn_like(x0)
                xt = torch.sqrt(abt) * x0 + torch.sqrt(1 - abt) * eps
                pred = self.net(xt, t, c, mask)
                loss = (((pred - eps) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
                opt.zero_grad(); loss.backward(); opt.step()
                last = float(loss.item())
            if self.verbose and ep % max(self.epochs // 4, 1) == 0:
                print(f"[TabDiff] epoch {ep}  loss {last:.4f}", flush=True)
        return self

    @torch.no_grad() if torch is not None else (lambda f: f)
    def _sample_once(self, Xt, Mt):
        dev, ab, T = self.device, self.ab, self.T
        n, d = Xt.shape
        cond = Xt * Mt
        x = torch.randn(n, d, device=dev)
        for t in range(T, 0, -1):
            abt, abtm = ab[t], ab[t - 1]
            # RePaint: observed coords carry the forward-noised observed signal
            x = Mt * (torch.sqrt(abt) * cond + torch.sqrt(1 - abt) * torch.randn_like(x)) + (1 - Mt) * x
            tt = torch.full((n,), t, device=dev, dtype=torch.long)
            eps = self.net(x, tt, cond, Mt)
            x0 = ((x - torch.sqrt(1 - abt) * eps) / torch.sqrt(abt)).clamp(-8, 8)
            beta_t = 1 - abt / abtm
            mean = (torch.sqrt(abtm) * beta_t / (1 - abt)) * x0 \
                 + (torch.sqrt(abt / abtm) * (1 - abtm) / (1 - abt)) * x
            if t > 1:
                var = (beta_t * (1 - abtm) / (1 - abt)).clamp_min(1e-20)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        return Mt * cond + (1 - Mt) * x                  # observed coords = true observed

    def impute(self, Xobs_te, M_te, m):
        assert self.net is not None, "call fit() first"
        X, M = self._prep(Xobs_te, M_te)
        dev = self.device
        Xt = torch.tensor(X, device=dev)
        Mt = torch.tensor(M, device=dev)
        self.net.eval()
        outs = [self._sample_once(Xt, Mt).cpu().numpy() for _ in range(m)]
        return np.stack(outs, 0).astype(np.float32)      # (m, n, d)


# --------------------------------------------------------------------- DiffPuter (diffusion + EM)
class DiffPuterImputer:
    name = "DiffPuter"

    def __init__(self, em_iters=2, **tabdiff_kwargs):
        self.em_iters = em_iters
        self.kw = dict(tabdiff_kwargs)
        self.nfe = self.kw.get("T", 50)
        self.model = None

    def fit(self, Xobs_tr, M_tr):
        M = np.asarray(M_tr, dtype=np.float32)
        X0 = np.nan_to_num(np.asarray(Xobs_tr, dtype=np.float32), nan=0.0)
        cur = X0.copy()                                  # current completion (init: 0 = col mean)
        full = np.ones_like(M)
        for it in range(self.em_iters):
            td = TabDiffImputer(**self.kw)
            td.fit(cur, full)                            # M-step: learn the complete-data law
            comp = td.impute(np.where(M == 1, cur, np.nan), M, 1)[0]   # E-step: re-impute
            cur = np.where(M == 1, X0, comp).astype(np.float32)
            self.model = td
            if self.kw.get("verbose"):
                print(f"[DiffPuter] EM iter {it + 1}/{self.em_iters} done", flush=True)
        return self

    def impute(self, Xobs_te, M_te, m):
        return self.model.impute(Xobs_te, M_te, m)


# Registry used by the driver.  Each entry is a zero-arg factory (closure over config).
def build(name, *, device, smoke=False):
    cfg = dict(device=device, verbose=True)
    if smoke:
        cfg.update(epochs=3, hidden=32, nlayers=2, batch=128)
    nfe = 50
    if name == "tabdiff":
        return lambda: TabDiffImputer(T=nfe, epochs=cfg.get("epochs", 200), **{k: v for k, v in cfg.items() if k != "epochs"})
    if name == "missdiff":          # same conditional-DDPM engine, labelled MissDiff
        def _missdiff():
            imp = TabDiffImputer(T=nfe, epochs=cfg.get("epochs", 200),
                                 **{k: v for k, v in cfg.items() if k != "epochs"})
            imp.name = "MissDiff"
            return imp
        return _missdiff
    if name == "diffputer":
        em = 1 if smoke else 2
        return lambda: DiffPuterImputer(em_iters=em, T=nfe, epochs=cfg.get("epochs", 150),
                                        **{k: v for k, v in cfg.items() if k != "epochs"})
    raise ValueError(f"unknown diffusion baseline {name}")
