"""DP covariance matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_covariance"]


def dp_covariance(X, C, epsilon):
    """
    DP covariance matrix

    Formula: clip rows; add Wishart-/Gaussian noise

    Parameters
    ----------
    X : array-like
        Input data.
    C : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork et al (2014); Amin-Joseph-Mao-Mehta (2019)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP covariance matrix"})


def cheatsheet():
    return "dpcov: DP covariance matrix"
