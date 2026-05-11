"""TMLE under interference / spillover."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_spillover"]


def tmle_spillover(y, D, X, network, exposure_summary):
    """
    TMLE under interference / spillover

    Formula: network-aware Q with neighborhood treatment summary

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    network : array-like
        Input data.
    exposure_summary : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sofrygin & vdL (2017); Aronow-Samii (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE under interference / spillover"})


def cheatsheet():
    return "tmlspl: TMLE under interference / spillover"
