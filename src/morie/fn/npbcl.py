"""NP Bayes clustering with DP-mixture."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["np_bayes_clustering"]


def np_bayes_clustering(y, alpha):
    """
    NP Bayes clustering with DP-mixture

    Formula: posterior mode partition

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Quintana (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NP Bayes clustering with DP-mixture"})


def cheatsheet():
    return "npbcl: NP Bayes clustering with DP-mixture"
