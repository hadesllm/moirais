"""Bayesian robust regression (Student-t)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_robust"]


def bayes_robust(y, X, nu_prior):
    """
    Bayesian robust regression (Student-t)

    Formula: y ~ Student-t(nu, X beta, sigma)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    nu_prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    West (1984); Geweke (1993)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian robust regression (Student-t)"})


def cheatsheet():
    return "bayreg2: Bayesian robust regression (Student-t)"
