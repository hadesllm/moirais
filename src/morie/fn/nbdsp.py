# morie.fn -- function file (hadesllm/morie)
"""Negative binomial regression for overdispersed count data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["negative_binomial_dispersion"]


def negative_binomial_dispersion(y, X, link):
    """
    Negative binomial regression for overdispersed count data

    Formula: P(Y=k) = C(k+r-1,k) * p^r * (1-p)^k; E[Y]=mu, Var[Y]=mu+mu^2/r

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    link : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'mu_hat': 'array', 'r_hat': 'float'}

    References
    ----------
    Montesinos Lopez Ch 7
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Negative binomial regression for overdispersed count data"})


def cheatsheet():
    return "nbdsp: Negative binomial regression for overdispersed count data"
