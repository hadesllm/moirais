"""Posterior predictive Poisson."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["poisson_predictive"]


def poisson_predictive(counts, alpha, beta):
    """
    Posterior predictive Poisson

    Formula: y_new ~ Poisson(lambda) | lambda ~ Gamma(alpha+sum y, beta+n)

    Parameters
    ----------
    counts : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman et al (2013) BDA3
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior predictive Poisson"})


def cheatsheet():
    return "poispr: Posterior predictive Poisson"
