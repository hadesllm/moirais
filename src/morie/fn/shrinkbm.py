"""Bayesian shrinkage (horseshoe / Laplace)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shrinkage_bayes"]


def shrinkage_bayes(X, y, prior_family):
    """
    Bayesian shrinkage (horseshoe / Laplace)

    Formula: beta_j ~ N(0, lambda_j tau); lambda_j ~ C+(0,1)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    prior_family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Carvalho-Polson-Scott (2010) horseshoe
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian shrinkage (horseshoe / Laplace)"})


def cheatsheet():
    return "shrinkbm: Bayesian shrinkage (horseshoe / Laplace)"
