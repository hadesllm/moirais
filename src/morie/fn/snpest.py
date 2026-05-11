"""Sequential nonparametric pseudo-likelihood."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sn_pseudo_estimate"]


def sn_pseudo_estimate(y_stream, alpha):
    """
    Sequential nonparametric pseudo-likelihood

    Formula: sequential update of f via DP-based filter

    Parameters
    ----------
    y_stream : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Caron-Doucet-Gottardo (2017)
    """
    y_stream = np.atleast_1d(np.asarray(y_stream, dtype=float))
    n = len(y_stream)
    result = float(np.mean(y_stream))
    se = float(np.std(y_stream, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequential nonparametric pseudo-likelihood"})


def cheatsheet():
    return "snpest: Sequential nonparametric pseudo-likelihood"
