"""MissFlow behind the common Imputer interface."""
from __future__ import annotations
import numpy as np
from missflow import train_missflow, impute_missflow
from baselines.base import Imputer


class MissFlowImputer(Imputer):
    name = "MissFlow"

    def __init__(self, n_epochs: int = 400, n_steps: int = 20,
                 hidden_dim: int = 256, n_layers: int = 4, time_dim: int = 128,
                 lr: float = 1e-3, batch_size: int = 256,
                 actual_mask_prob: float = 0.0, ipw: bool = False,
                 device: str = "cpu", verbose: bool = False):
        self.train_cfg = dict(n_epochs=n_epochs, hidden_dim=hidden_dim,
                              n_layers=n_layers, time_dim=time_dim, lr=lr,
                              batch_size=batch_size, actual_mask_prob=actual_mask_prob,
                              device=device, verbose=verbose)
        self.n_steps = n_steps
        self.device = device
        self.nfe = n_steps
        self.ipw = ipw
        self.model = None

    def fit(self, Xobs_tr, M_tr):
        ipw_weight = None
        if self.ipw:
            from missflow import estimate_propensity
            ipw_weight = 1.0 / estimate_propensity(Xobs_tr, M_tr)
        self.model = train_missflow(
            Xobs_tr, M_tr, ipw_weight=ipw_weight,
            log_every=max(self.train_cfg["n_epochs"] // 4, 1),
            **self.train_cfg)
        return self

    def impute(self, Xobs_te, M_te, m):
        return impute_missflow(self.model, Xobs_te, M_te,
                               n_imputations=m, n_steps=self.n_steps,
                               device=self.device)
