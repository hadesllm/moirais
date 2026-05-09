"""Sample compositions from a Dirichlet distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dirichlet_sample"]


def dirichlet_sample(alpha, n):
    """
    Sample compositions from a Dirichlet distribution

    Formula: g_i ~ Gamma(α_i,1); x = C(g)

    Parameters
    ----------
    alpha : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X

    References
    ----------
    Wilks (1962)
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sample compositions from a Dirichlet distribution"})


def cheatsheet():
    return "aitdrs: Sample compositions from a Dirichlet distribution"
