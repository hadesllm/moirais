"""KL with smoothing for sparse."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kl_molecular_smooth"]


def kl_molecular_smooth(p, q, eps):
    """
    KL with smoothing for sparse

    Formula: add eps to zero probabilities

    Parameters
    ----------
    p : array-like
        Input data.
    q : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chen-Goodman (1996)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KL with smoothing for sparse"})


def cheatsheet():
    return "klmsm1: KL with smoothing for sparse"
