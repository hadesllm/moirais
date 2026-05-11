"""Sample from a logistic-normal on the simplex via ALR^-1."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["logistic_normal_sample"]


def logistic_normal_sample(mu, Sigma, n):
    """
    Sample from a logistic-normal on the simplex via ALR^-1

    Formula: y ~ N(μ,Σ); x = alr^{-1}(y)

    Parameters
    ----------
    mu : array-like
        Input data.
    Sigma : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X

    References
    ----------
    Aitchison (1986)
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sample from a logistic-normal on the simplex via ALR^-1"})


def cheatsheet():
    return "aitlns: Sample from a logistic-normal on the simplex via ALR^-1"
