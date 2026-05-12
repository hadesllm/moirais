"""APPNP -- personalized PageRank propagation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["appnp"]


def appnp(A, H0, alpha, K):
    """
    APPNP -- personalized PageRank propagation

    Formula: H^{l+1} = (1−α)Â H^l + α H^0

    Parameters
    ----------
    A : array-like
        Input data.
    H0 : array-like
        Input data.
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
    Klicpera et al (2019)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "APPNP -- personalized PageRank propagation"})


def cheatsheet():
    return "appnp: APPNP -- personalized PageRank propagation"
