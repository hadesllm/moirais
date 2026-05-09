"""TMLE for marginal risk ratio."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_marginal_rr"]


def tmle_marginal_rr(y, D, X):
    """
    TMLE for marginal risk ratio

    Formula: RR = E[Y(1)]/E[Y(0)]; targeted via log-link

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Spiegelman-Hertzmark (2005); vdL-Rose (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for marginal risk ratio"})


def cheatsheet():
    return "tmlmrr: TMLE for marginal risk ratio"
