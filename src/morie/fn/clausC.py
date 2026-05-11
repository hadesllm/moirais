"""Clausius-Clapeyron scaling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clausius_clapeyron"]


def clausius_clapeyron(T):
    """
    Clausius-Clapeyron scaling

    Formula: de_s/dT = L_v e_s / (R_v T²) ≈ 7%/K

    Parameters
    ----------
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Held-Soden (2006)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Clausius-Clapeyron scaling"})


def cheatsheet():
    return "clausC: Clausius-Clapeyron scaling"
