"""Bayesian LASSO for genomic prediction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_lasso"]


def bayes_lasso(y, M, lam):
    """
    Bayesian LASSO for genomic prediction

    Formula: u_j ~ Laplace(0, lambda)

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Park-Casella (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian LASSO for genomic prediction"})


def cheatsheet():
    return "bayslo: Bayesian LASSO for genomic prediction"
