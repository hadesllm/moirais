# morie.fn -- function file (hadesllm/morie)
"""Cronbach's coefficient alpha with Feldt CI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from morie.fn._containers import RlbRes


def crba(
    data: pd.DataFrame | np.ndarray,
    *,
    ci: float = 0.95,
) -> RlbRes:
    """Cronbach's coefficient alpha with Feldt CI.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    ci : float
        Confidence level (default 0.95).

    Returns
    -------
    RlbRes
        Reliability result with raw alpha, standardised alpha, average
        inter-item correlation, and confidence interval bounds.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape

    if k < 2:
        return RlbRes(raw=np.nan, std=np.nan, avgr=np.nan, k=k, n=n)

    item_var = np.var(X, axis=0, ddof=1)
    total_var = np.var(X.sum(axis=1), ddof=1)
    if total_var < 1e-15:
        return RlbRes(raw=np.nan, std=np.nan, avgr=0.0, k=k, n=n)

    raw = (k / (k - 1)) * (1 - item_var.sum() / total_var)

    R = np.corrcoef(X, rowvar=False)
    mask = ~np.eye(k, dtype=bool)
    avgr = R[mask].mean()
    if np.isnan(avgr):
        avgr = 0.0
    std = (k * avgr) / (1 + (k - 1) * avgr)

    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    f_lo = sp.f.ppf(1 - (1 - ci) / 2, df1, df2)
    f_hi = sp.f.ppf((1 - ci) / 2, df1, df2)

    return RlbRes(
        raw=float(raw),
        std=float(std),
        avgr=float(avgr),
        k=k,
        n=n,
        ci_lo=float(1 - (1 - raw) * f_lo),
        ci_hi=float(1 - (1 - raw) * f_hi),
    )


def cheatsheet() -> str:
    return "crba({}) -> Cronbach's coefficient alpha with Feldt CI."
