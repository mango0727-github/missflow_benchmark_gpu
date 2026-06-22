"""
train.py  —  MissFlow v3.2 (corrected)
========================================
Critical fix: only train on OBSERVED features.

During training, we randomly split observed features (M=1) into:
  target  : random subset of observed — the model learns to impute these
  cond    : remaining observed features — used as conditioning

Actually missing features (M=0) are excluded from training entirely,
because we have no ground-truth for them. The model generalises at
inference time from "impute observed subset from complement" to
"impute missing features from all observed features".

This is the correct implementation of CFMI's shared conditional model.
Mode A (training on actually missing positions with NaN-filled zeros)
was the root cause of collapse in all previous versions.
"""
from __future__ import annotations
import time
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .model import VelocityNetwork


def train_missflow(
    X_obs,  M, *,
    hidden_dim=256, n_layers=4, time_dim=128,
    use_attention=False, n_heads=4,
    lr=1e-3, weight_decay=1e-5,
    n_epochs=300, batch_size=256,
    n_em_rounds=0, em_epochs=0, em_steps=0, em_draws=0,
    target_prob=0.5,
    actual_mask_prob=0.0,   # kept for API compat — must stay 0
    ipw_weight=None,        # (n,d) = 1/pi for the IPW-masked loss (Prop 1'); None = uniform
    device=None, verbose=True, log_every=50,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n, d = X_obs.shape
    X_clean = np.nan_to_num(X_obs, nan=0.0).astype(np.float32)
    X_t = torch.tensor(X_clean, dtype=torch.float32)
    M_t = torch.tensor(M,       dtype=torch.float32)
    W_t = (torch.ones_like(M_t) if ipw_weight is None
           else torch.tensor(np.asarray(ipw_weight, dtype=np.float32)))

    model     = VelocityNetwork(d, hidden_dim=hidden_dim,
                                n_layers=n_layers, time_dim=time_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    loader = DataLoader(TensorDataset(X_t, M_t, W_t),
                        batch_size=batch_size, shuffle=True)

    t0 = time.time()
    model.train()

    for epoch in range(1, n_epochs + 1):
        total_loss, n_total = 0.0, 0

        for X_batch, M_batch, W_batch in loader:
            X_batch = X_batch.to(device)   # (B, d); 0 at missing positions
            M_batch = M_batch.to(device)   # (B, d); 1=observed, 0=missing
            W_batch = W_batch.to(device)   # (B, d); 1/pi at IPW coords, else 1
            B = X_batch.size(0)

            # ----------------------------------------------------------
            # Random partition of OBSERVED features only.
            # Missing features (M=0) are never in target or cond.
            # ----------------------------------------------------------
            rand_b = torch.rand_like(M_batch) < target_prob
            # target: random subset of observed features
            target_mask = (rand_b & M_batch.bool()).float()    # (B, d)
            # cond: the complementary observed features
            cond_mask   = (~rand_b & M_batch.bool()).float()   # (B, d)

            # Guarantee at least 1 feature in cond per row
            # (rows with only 1 observed feature need special handling)
            n_target = target_mask.sum(dim=1)
            n_cond   = cond_mask.sum(dim=1)
            # If cond is empty but there are observed features, move one
            # from target → cond
            empty_cond = (n_cond == 0) & (M_batch.sum(dim=1) > 1)
            if empty_cond.any():
                rows = empty_cond.nonzero(as_tuple=True)[0]
                # Move first target feature of each such row to cond
                first_tgt = target_mask[rows].float().argmax(dim=1)
                cond_mask[rows, first_tgt]   = 1.0
                target_mask[rows, first_tgt] = 0.0

            # Mode A (collapse reproduction; OFF at the default 0.0). With prob
            # actual_mask_prob, also place genuinely-missing positions into the
            # target with their zero-fill value as the regression target. This
            # drives missing dims toward 0, narrowing the MI draws — the suspected
            # cause of the workshop's under-coverage. At 0.0 this block is inert,
            # so the reproduction path (Gate A) is unchanged.
            if actual_mask_prob > 0:
                add_miss = ((torch.rand_like(M_batch) < actual_mask_prob)
                            & (~M_batch.bool())).float()
                target_mask = torch.clamp(target_mask + add_miss, max=1.0)

            # Skip rows with no target features (e.g. all-missing rows)
            n_target = target_mask.sum(dim=1)
            valid_rows = n_target > 0
            if not valid_rows.any():
                continue

            # ----------------------------------------------------------
            # Flow matching quantities (only over target OBSERVED dims)
            # ----------------------------------------------------------
            x0      = torch.randn_like(X_batch) * target_mask
            x1      = X_batch * target_mask    # real observed values at target
            x1_cond = X_batch * cond_mask      # real observed values at cond

            t_s = torch.rand(B, device=device)
            x_t = ((1 - t_s[:,None]) * x0 + t_s[:,None] * x1) * target_mask
            u_t = (x1 - x0) * target_mask     # true velocity

            v_t = model(x_t, t_s, x1_cond, cond_mask) * target_mask

            # Per-sample normalised loss (IPW: target coords weighted by 1/pi)
            sq   = (v_t - u_t).pow(2) * W_batch
            loss = (sq.sum(dim=1)[valid_rows] /
                    n_target[valid_rows]).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * valid_rows.sum().item()
            n_total    += valid_rows.sum().item()

        scheduler.step()
        if verbose and epoch % log_every == 0:
            print(f"  [MissFlow v3.2] epoch {epoch}/{n_epochs} | "
                  f"loss={total_loss/max(n_total,1):.5f} | "
                  f"t={time.time()-t0:.1f}s")

    if verbose:
        print(f"  [MissFlow v3.2] done in {time.time()-t0:.1f}s")
    model.eval()
    return model
