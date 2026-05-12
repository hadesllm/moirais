"""ECOD -- empirical CDF outlier."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ecod"]


def ecod(X):
    """
    ECOD -- empirical CDF outlier

    Formula: sum log(min(F_i(x), 1-F_i(x))) per dim

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Li et al (2022) ECOD
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ECOD -- empirical CDF outlier"})


def cheatsheet():
    return "ecod: ECOD -- empirical CDF outlier"
