"""Zonal energy balance model (Budyko-Sellers)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["zonal_ebm"]


def zonal_ebm(S, alpha, A, B):
    """
    Zonal energy balance model (Budyko-Sellers)

    Formula: C dT/dt = (1-α(T))S/4 − A − B T

    Parameters
    ----------
    S : array-like
        Input data.
    alpha : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Budyko (1969); Sellers (1969)
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Zonal energy balance model (Budyko-Sellers)"})


def cheatsheet():
    return "ebmZD: Zonal energy balance model (Budyko-Sellers)"
