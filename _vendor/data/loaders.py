"""Dataset loaders + train-observed standardization.

Faithful to experiments/reproduce.py (Appendix B protocol): the same five
datasets and the same train-observed-only standardization, extracted so the
harness and every baseline share one loader.
"""
from __future__ import annotations
import io
import zipfile
import urllib.request
import numpy as np

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"
DATASETS = ["bean", "magic", "shoppers", "letter", "california"]


def _label_encode(df):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):   # robust across pandas versions
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    return df.values.astype(np.float64)


def load_dataset(name: str) -> np.ndarray:
    """Return data matrix X (n, d) as float64, matching the paper's d."""
    import pandas as pd
    if name == "california":
        from sklearn.datasets import fetch_california_housing
        return fetch_california_housing().data.astype(np.float64)        # d=8
    if name == "magic":
        df = pd.read_csv(f"{UCI}/magic/magic04.data", header=None)        # 10 feats + class
        return _label_encode(df)                                         # d=11
    if name == "letter":
        df = pd.read_csv(f"{UCI}/letter-recognition/letter-recognition.data",
                         header=None)                                    # class + 16 feats
        return _label_encode(df)                                         # d=17
    if name == "shoppers":
        df = pd.read_csv(f"{UCI}/00468/online_shoppers_intention.csv")   # 17 feats + Revenue
        return _label_encode(df)                                         # d=18
    if name == "bean":
        raw = urllib.request.urlopen(f"{UCI}/00602/DryBeanDataset.zip").read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        xls = [n for n in z.namelist() if n.endswith(".xlsx")][0]
        df = pd.read_excel(io.BytesIO(z.read(xls)))                      # 16 feats + Class
        return _label_encode(df)                                         # d=17
    raise ValueError(f"unknown dataset {name}")


def standardize_train_obs(X: np.ndarray, M: np.ndarray, tr: np.ndarray):
    """Standardize with TRAIN-OBSERVED stats only (reproduce.py protocol).

    Returns (Xs, mu, sd) where Xs = (X - mu) / sd over the whole matrix.
    """
    Xtr_obs = np.where(M[tr] == 1, X[tr], np.nan)
    mu = np.nanmean(Xtr_obs, axis=0)
    sd = np.nanstd(Xtr_obs, axis=0) + 1e-8
    return (X - mu) / sd, mu, sd
