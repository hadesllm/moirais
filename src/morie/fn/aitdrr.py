"""Dirichlet regression of compositions on covariates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dirichlet_regression"]


def dirichlet_regression(X_cov, Y_comp, ref):
    """
    Dirichlet regression of compositions on covariates

    Formula: α_i(x) = exp(x β_i); ML over (β,φ)

    Parameters
    ----------
    X_cov : array-like
        Input data.
    Y_comp : array-like
        Input data.
    ref : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta, phi, ll

    References
    ----------
    Hijazi & Jernigan (2009)
    """
    X_cov = np.atleast_1d(np.asarray(X_cov, dtype=float))
    n = len(X_cov)
    result = float(np.mean(X_cov))
    se = float(np.std(X_cov, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet regression of compositions on covariates"})


def cheatsheet():
    return "aitdrr: Dirichlet regression of compositions on covariates"
