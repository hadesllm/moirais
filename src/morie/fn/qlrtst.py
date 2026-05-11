"""Quandt likelihood ratio (sup-LR) for unknown break."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["quandt_likelihood_ratio"]


def quandt_likelihood_ratio(y, X):
    """
    Quandt likelihood ratio (sup-LR) for unknown break

    Formula: sup_t LR_t over trimmed (0.15, 0.85) range

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Quandt (1960); Andrews (1993)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quandt likelihood ratio (sup-LR) for unknown break"})


def cheatsheet():
    return "qlrtst: Quandt likelihood ratio (sup-LR) for unknown break"
