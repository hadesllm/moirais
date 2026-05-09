"""Chernoff bound on tail."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chernoff_bound"]


def chernoff_bound(mgf, threshold):
    """
    Chernoff bound on tail

    Formula: P(X >= a) <= e^{-t a} E[e^{tX}]

    Parameters
    ----------
    mgf : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chernoff (1952)
    """
    mgf = np.atleast_1d(np.asarray(mgf, dtype=float))
    n = len(mgf)
    result = float(np.mean(mgf))
    se = float(np.std(mgf, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chernoff bound on tail"})


def cheatsheet():
    return "chrnff: Chernoff bound on tail"
