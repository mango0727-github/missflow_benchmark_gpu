"""Evaluation metrics: accuracy (RMSE/MAE) and the coverage metric (Ext 1).

All operate in standardized space on the test missing entries.
  point : (n, d) point estimate (e.g. mean over draws)
  comps : (m, n, d) independent completions
  Xtrue : (n, d) standardized ground truth
  miss  : (n, d) bool, True where M==0 (genuinely missing in test)
"""
from __future__ import annotations
import numpy as np


def rmse(point, Xtrue, miss):
    return float(np.sqrt(((point[miss] - Xtrue[miss]) ** 2).mean()))


def mae(point, Xtrue, miss):
    return float(np.abs(point[miss] - Xtrue[miss]).mean())


def coverage_band(comps, Xtrue, miss, level: float = 0.95):
    """Per-coordinate empirical [α/2, 1-α/2] band over m draws.

    Returns (coverage, mean_width) averaged over missing entries.
    """
    a = (1.0 - level) / 2.0
    lo = np.percentile(comps, 100.0 * a, axis=0)            # (n, d)
    hi = np.percentile(comps, 100.0 * (1.0 - a), axis=0)
    inside = (Xtrue >= lo) & (Xtrue <= hi)
    return float(inside[miss].mean()), float((hi - lo)[miss].mean())


def reliability(comps, Xtrue, miss, levels=(0.90, 0.95, 0.99)):
    """{level: (coverage, width)} for a reliability diagram."""
    return {lvl: coverage_band(comps, Xtrue, miss, lvl) for lvl in levels}


def zscore_std(comps, Xtrue, miss):
    """Calibration sharpness: std of z=(truth-mean)/draw_std over missing cells.

    ~1.0 = well calibrated; >1 = draws too narrow (under-dispersed); <1 = too wide.
    """
    point = comps.mean(axis=0)
    sd = comps.std(axis=0) + 1e-8
    z = (Xtrue - point) / sd
    return float(z[miss].std())
