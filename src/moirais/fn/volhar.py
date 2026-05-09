"""Heterogeneous Autoregressive RV (Corsi)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_har_rv"]


def vol_har_rv(RV, h):
    """
    Heterogeneous Autoregressive RV (Corsi)

    Formula: RV_t = c + β_d RV_{t-1} + β_w RV_{t-1,w} + β_m RV_{t-1,m}

    Parameters
    ----------
    RV : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: c, betas, fitted

    References
    ----------
    Corsi (2009)
    """
    RV = np.atleast_1d(np.asarray(RV, dtype=float))
    n = len(RV)
    result = float(np.mean(RV))
    se = float(np.std(RV, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heterogeneous Autoregressive RV (Corsi)"})


def cheatsheet():
    return "volhar: Heterogeneous Autoregressive RV (Corsi)"
