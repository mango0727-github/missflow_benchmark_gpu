"""Observation-propensity estimation for the IPW-masked CFM loss (Prop 1').

pi_i(x_obs) = Pr(M_i=1 | x_obs), estimated per feature by logistic regression of
the observed-indicator M_i on the other (NaN-filled) features. The IPW training
weight is 1/pi, clipped from below for positivity (pi >= clip_min). Under MAR this
de-biases the masked loss so the population minimiser is the true velocity.
"""
from __future__ import annotations
import numpy as np


def estimate_propensity(X, M, clip_min: float = 0.05) -> np.ndarray:
    """Return pi (n, d) with pi[b, i] = est. Pr(M_i=1 | x_obs of row b)."""
    from sklearn.linear_model import LogisticRegression
    Xf = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
    M = np.asarray(M)
    n, d = Xf.shape
    pi = np.ones((n, d), dtype=np.float64)
    for i in range(d):
        y = M[:, i].astype(int)
        if y.min() == y.max():                       # feature never/always missing
            pi[:, i] = np.clip(float(y.mean()), clip_min, 1.0)
            continue
        Xpred = np.delete(Xf, i, axis=1)             # other features as predictors
        clf = LogisticRegression(max_iter=500).fit(Xpred, y)
        pi[:, i] = clf.predict_proba(Xpred)[:, 1]
    return np.clip(pi, clip_min, 1.0)
