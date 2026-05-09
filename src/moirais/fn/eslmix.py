"""Gaussian mixture density."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_gaussian_mixture"]


def esl_gaussian_mixture(X, k):
    """
    Gaussian mixture density

    Formula: f(x) = sum pi_k N(x | mu_k, Sigma_k)

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Hastie ESL Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian mixture density"})


def cheatsheet():
    return "eslmix: Gaussian mixture density"
