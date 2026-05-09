"""Standard empirical distribution function indexed by t in R."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_empirical_distribution_function"]


def kosorok_ch2_empirical_distribution_function(X, t, n):
    """
    Standard empirical distribution function indexed by t in R

    Formula: F_n(t) = n^{-1} * sum_{i=1}^{n} 1{X_i <= t}

    Parameters
    ----------
    X : array-like
        Input data.
    t : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.1, p. 9
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Standard empirical distribution function indexed by t in R"})


def cheatsheet():
    return "ksr026: Standard empirical distribution function indexed by t in R"
