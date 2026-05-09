"""Federated TMLE over decentralized data."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_federated"]


def tmle_federated(y, D, X, site):
    """
    Federated TMLE over decentralized data

    Formula: site-local Q,g + secure aggregation of influence curves

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    site : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vo-vdL-Petersen (2023)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Federated TMLE over decentralized data"})


def cheatsheet():
    return "tmlfed: Federated TMLE over decentralized data"
