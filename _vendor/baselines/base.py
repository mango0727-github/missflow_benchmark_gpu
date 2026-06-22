"""Common imputer interface — the fair-comparison contract.

Every method (MissFlow and every baseline) is `fit` then `impute`, returning m
independent completions of shape (m, n, d) so the SAME harness can score
accuracy AND coverage on all of them. Point methods return m=1.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Imputer(ABC):
    name: str = "imputer"
    nfe: int | None = None      # inference NFE per draw (speed table); None if N/A

    @abstractmethod
    def fit(self, Xobs_tr: np.ndarray, M_tr: np.ndarray) -> "Imputer":
        """Fit on standardized training data (NaN at missing); return self."""

    @abstractmethod
    def impute(self, Xobs_te: np.ndarray, M_te: np.ndarray, m: int) -> np.ndarray:
        """Return (m, n, d) completions in standardized space."""
