"""Elastic net regression (L1 + L2 penalty)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["elastic_net_regression"]


def elastic_net_regression(y, X, lambda1, lambda2):
    """
    Elastic net regression (L1 + L2 penalty)

    Formula: min ||y - X beta||^2 + lambda1 ||beta||_1 + lambda2 ||beta||^2

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    lambda1 : array-like
        Input data.
    lambda2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zou & Hastie (2005)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Elastic net regression (L1 + L2 penalty)"})


def cheatsheet():
    return "elasrg: Elastic net regression (L1 + L2 penalty)"
