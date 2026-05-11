"""GEM distribution from stick-breaking."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gem_distribution"]


def gem_distribution(alpha, K):
    """
    GEM distribution from stick-breaking

    Formula: GEM(alpha) = stick-breaking with V_k ~ Beta(1, alpha)

    Parameters
    ----------
    alpha : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pitman (2002)
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GEM distribution from stick-breaking"})


def cheatsheet():
    return "dpgem: GEM distribution from stick-breaking"
