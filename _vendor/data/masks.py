"""Missingness mask generators.  Convention: M = 1 observed, 0 missing.

- mcar : entrywise Bernoulli(p) missing — bit-identical to reproduce.py so the
         MCAR harness reproduces the gate numbers.
- mar  : logistic MAR (Muzellec et al., 2020). A random subset of features is
         always observed; the rest go missing with prob logistic in the observed.
- mnar : logistic self-masking MNAR — each feature missing with prob logistic in
         its OWN (z-scored) value.

MAR/MNAR intercepts are calibrated per target feature by bisection to hit the
requested marginal missing rate p. All generators take the harness's rng so a
run is reproducible from a single seed.
"""
from __future__ import annotations
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return (X - mu) / sd


def _calibrate_intercept(logits: np.ndarray, target: float) -> float:
    """Find b s.t. mean(sigmoid(logits + b)) == target (sigmoid is monotone in b)."""
    lo, hi = -60.0, 60.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _sigmoid(logits + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mcar(shape, p: float, rng: np.random.Generator) -> np.ndarray:
    """1 = observed. Same draw order as reproduce.py: rng.random(shape) > p."""
    n, d = shape
    return (rng.random((n, d)) > p).astype(np.float32)


def mar(X: np.ndarray, p: float, rng: np.random.Generator,
        p_obs: float = 0.3) -> np.ndarray:
    """Logistic MAR. p_obs = fraction of features kept always-observed (inputs)."""
    n, d = X.shape
    Z = _zscore(X)
    d_obs = max(1, int(round(p_obs * d)))
    idx = rng.permutation(d)
    obs_idx, na_idx = idx[:d_obs], idx[d_obs:]
    # Only the n_na NA-eligible features can be missing, so to hit an OVERALL rate
    # p we inflate the per-NA-feature rate (inputs stay fully observed).
    n_na = len(na_idx)
    target_na = min(0.98, p * d / max(n_na, 1))
    miss = np.zeros((n, d), dtype=bool)
    W = rng.standard_normal((d_obs, n_na)) / np.sqrt(d_obs)
    logits = Z[:, obs_idx] @ W                               # (n, n_na)
    for k, j in enumerate(na_idx):
        b = _calibrate_intercept(logits[:, k], target_na)
        miss[:, j] = rng.random(n) < _sigmoid(logits[:, k] + b)
    return (~miss).astype(np.float32)


def mnar(X: np.ndarray, p: float, rng: np.random.Generator,
         strength: float = 1.0) -> np.ndarray:
    """Logistic self-masking MNAR: each feature missing w.p. logistic in its own value."""
    n, d = X.shape
    Z = _zscore(X)
    miss = np.zeros((n, d), dtype=bool)
    for j in range(d):
        a = strength * (1.0 if rng.random() < 0.5 else -1.0)  # random sign per feature
        logit = a * Z[:, j]
        b = _calibrate_intercept(logit, p)
        miss[:, j] = rng.random(n) < _sigmoid(logit + b)
    return (~miss).astype(np.float32)


def make_mask(mechanism: str, X: np.ndarray, p: float,
              rng: np.random.Generator) -> np.ndarray:
    if mechanism == "mcar":
        return mcar(X.shape, p, rng)
    if mechanism == "mar":
        return mar(X, p, rng)
    if mechanism == "mnar":
        return mnar(X, p, rng)
    raise ValueError(f"unknown mechanism {mechanism}")
