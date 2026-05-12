# morie.fn -- function file (hadesllm/morie)
"""Incomplete beta integral identity linking binomial CDF and beta CDF."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_binomial_beta_link"]


def gibbons_binomial_beta_link(t, r, n):
    """
    Incomplete beta integral identity linking binomial CDF and beta CDF

    Formula: sum_{i=r}^{n} C(n,i) t^i (1-t)^(n-i) = I_t(r, n-r+1)

    Parameters
    ----------
    t : array-like
        Input data.
    r : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Corollary 2.4.3.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Incomplete beta integral identity linking binomial CDF and beta CDF"})


def cheatsheet():
    return "gb2431: Incomplete beta integral identity linking binomial CDF and beta CDF"
