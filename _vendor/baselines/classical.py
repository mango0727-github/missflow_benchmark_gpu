"""Classical imputation baselines (the workshop set) through the Imputer API.

All are point imputers (m=1). Fit on the training rows (NaN at missing), transform
the test rows — out-of-sample, train statistics only. Configs follow the paper B.3.
"""
from __future__ import annotations
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from baselines.base import Imputer


class _Sklearn(Imputer):
    def __init__(self, name, make):
        self.name = name
        self._make = make

    def fit(self, Xobs_tr, M_tr):
        self._imp = self._make().fit(np.asarray(Xobs_tr, dtype=np.float64))
        return self

    def impute(self, Xobs_te, M_te, m):
        out = self._imp.transform(np.asarray(Xobs_te, dtype=np.float64))
        return out[None].astype(np.float32)          # (1, n, d), point method


class MICEMIImputer(Imputer):
    """MICE with multiple imputation (sample_posterior); m stochastic completions."""
    name = "MICE-MI"

    def __init__(self, max_iter=5):
        self.max_iter = max_iter

    def fit(self, Xobs_tr, M_tr):
        self._Xtr = np.asarray(Xobs_tr, dtype=np.float64)
        return self

    def impute(self, Xobs_te, M_te, m):
        Xte = np.asarray(Xobs_te, dtype=np.float64)
        outs = []
        for k in range(m):
            imp = IterativeImputer(estimator=BayesianRidge(), max_iter=self.max_iter,
                                   sample_posterior=True, random_state=k).fit(self._Xtr)
            outs.append(imp.transform(Xte))
        return np.stack(outs).astype(np.float32)


def make_classical():
    return {
        "Mean":   _Sklearn("Mean",   lambda: SimpleImputer(strategy="mean")),
        "Median": _Sklearn("Median", lambda: SimpleImputer(strategy="median")),
        "KNN":    _Sklearn("KNN",    lambda: KNNImputer(n_neighbors=5)),
        "MICE":   _Sklearn("MICE",   lambda: IterativeImputer(
                      estimator=BayesianRidge(), max_iter=5, random_state=0)),
        "MissForest": _Sklearn("MissForest", lambda: IterativeImputer(
                      estimator=ExtraTreesRegressor(n_estimators=10, random_state=0),
                      max_iter=3, random_state=0)),
    }
