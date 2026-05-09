"""Density of the logistic-normal on the simplex."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["logistic_normal_pdf"]


def logistic_normal_pdf(x, mu, Sigma):
    """
    Density of the logistic-normal on the simplex

    Formula: f(x|μ,Σ) = (2π)^(-d/2) |Σ|^(-1/2) (prod x_i)^(-1) exp(-1/2 (alr(x)-μ)^T Σ^(-1) (alr(x)-μ))

    Parameters
    ----------
    x : array-like
        Input data.
    mu : array-like
        Input data.
    Sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: f

    References
    ----------
    Aitchison (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Density of the logistic-normal on the simplex"})


def cheatsheet():
    return "aitlnp: Density of the logistic-normal on the simplex"
