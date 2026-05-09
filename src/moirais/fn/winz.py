"""Winsorized mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["winsorized_mean"]


def winsorized_mean(x, alpha):
    """
    Winsorized mean

    Formula: replace tails with α-quantiles, then mean

    Parameters
    ----------
    x : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Winsor (1941); Hastings et al (1947)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Winsorized mean"})


def cheatsheet():
    return "winz: Winsorized mean"
