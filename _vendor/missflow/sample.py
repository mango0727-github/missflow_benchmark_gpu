"""
sample.py  —  MissFlow v3 Sampling (CFMI-style)
================================================
At inference:
  target_mask = 1 - M  (missing features)
  cond_mask   = M      (observed features)
  x0          = randn * target_mask  (noise only at missing)

Euler integration of the ODE from t=0 to t=1:
  x_{t+dt} = x_t + v_theta(x_t, t, x1_cond, cond_mask) * target_mask * dt

The velocity is masked to zero at observed dims, so they don't move.
Final imputation: X_obs + x_1 * target_mask (replace missing with ODE output).
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch

from .model import VelocityNetwork


@torch.no_grad()
def impute_one(
    model:       VelocityNetwork,
    X_obs:       torch.Tensor,   # (n, d) — zero at missing
    M:           torch.Tensor,   # (n, d) — 1=observed, 0=missing
    eps:         torch.Tensor,   # (n, d) — initial noise
    time_grid:   torch.Tensor,   # (steps+1,) from 0 to 1
    sigma:       float = 0.0,    # SDE noise scale; 0 = deterministic ODE (default)
) -> torch.Tensor:
    """One conditional completion — returns (n, d).

    sigma>0 switches to a marginal-preserving SDE (lever 1d). For the CFM path
    x_t=(1-t)eps+t*x1 the score is s=(t*v-x)/(1-t); the SDE that keeps the ODE
    marginals is dx=[v+0.5 g^2 s]dt + g dW. With g(t)=sigma*(1-t) the noise
    vanishes at t=1 and the 1/(1-t) in s cancels, so it stays well-behaved.
    """
    target_mask = 1.0 - M       # (n, d) — 1 at missing
    cond_mask   = M             # (n, d) — 1 at observed
    x1_cond     = X_obs * M    # conditioning: observed values

    # Start from noise at missing dims, zero at observed dims
    x = eps * target_mask

    for k in range(1, len(time_grid)):
        t_prev = time_grid[k - 1]
        t_curr = time_grid[k]
        dt     = (t_curr - t_prev).item()

        t_batch = t_prev.expand(x.size(0))
        v       = model(x, t_batch, x1_cond, cond_mask)  # (n, d)
        v       = v * target_mask                         # zero at observed

        if sigma > 0.0:
            t     = t_prev.item()
            # 0.5 * g^2 * score with g=sigma*(1-t), score=(t*v-x)/(1-t)
            drift = v + 0.5 * sigma * sigma * (1.0 - t) * (t * v - x)
            noise = sigma * (1.0 - t) * (dt ** 0.5) * torch.randn_like(x)
            x = x + (drift * dt + noise) * target_mask
        else:
            x = x + v * dt

    # Return: observed values at cond dims, ODE output at target dims
    return X_obs * M + x * target_mask


@torch.no_grad()
def impute_missflow(
    model:         VelocityNetwork,
    X_obs:         np.ndarray,
    M:             np.ndarray,
    *,
    n_imputations: int  = 20,
    n_steps:       int  = 10,
    use_rk4:       bool = False,
    sigma:         float = 0.0,
    device:        Optional[str] = None,
    batch_size:    int  = 2048,
) -> np.ndarray:
    """Draw n_imputations independent completions. Returns (n_imputations, n, d).

    sigma>0 uses the SDE sampler (lever 1d) for wider, better-calibrated draws.
    """
    if device is None:
        device = next(model.parameters()).device

    n, d      = X_obs.shape
    time_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

    X_clean = np.nan_to_num(X_obs, nan=0.0).astype(np.float32)
    X_t = torch.tensor(X_clean, dtype=torch.float32, device=device)
    M_t = torch.tensor(M,       dtype=torch.float32, device=device)

    model.eval()
    completions = []
    for _ in range(n_imputations):
        eps    = torch.randn(n, d, device=device)
        chunks = []
        for s in range(0, n, batch_size):
            sl = slice(s, s + batch_size)
            c  = impute_one(model, X_t[sl], M_t[sl], eps[sl], time_grid, sigma=sigma)
            chunks.append(c.cpu())
        completions.append(torch.cat(chunks, dim=0).numpy())

    return np.stack(completions, axis=0)   # (n_imputations, n, d)


def rubins_rules(
    estimates:        np.ndarray,
    within_variances: np.ndarray,
) -> Tuple[float, float, float]:
    m      = len(estimates)
    psi_mi = estimates.mean()
    W_m    = within_variances.mean()
    B_m    = np.var(estimates, ddof=1)
    V_mi   = W_m + (1.0 + 1.0 / m) * B_m
    ratio  = W_m / ((1.0 + 1.0 / m) * B_m + 1e-20)
    nu     = (m - 1) * (1.0 + ratio) ** 2
    return float(psi_mi), float(V_mi), float(nu)


def point_impute(
    model:         VelocityNetwork,
    X_obs:         np.ndarray,
    M:             np.ndarray,
    *,
    n_imputations: int  = 10,
    n_steps:       int  = 10,
    use_rk4:       bool = False,
    device:        Optional[str] = None,
) -> np.ndarray:
    comps     = impute_missflow(model, X_obs, M,
                                n_imputations=n_imputations,
                                n_steps=n_steps, device=device)
    mean_comp = comps.mean(axis=0)
    return np.where(M == 0, mean_comp, np.nan_to_num(X_obs, nan=0.0))
