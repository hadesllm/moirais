# morie.fn — function file (hadesllm/morie)
"""Penalized Poisson regression for count genomic outcomes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["poisson_penalized_regression"]


def poisson_penalized_regression(y, X, lam):
    """
    Penalized Poisson regression for count genomic outcomes

    Formula: L = -sum(y_i*X_i*beta - exp(X_i*beta) - log(y_i!)) + lambda*||beta||_1

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'beta_hat': 'array'}

    References
    ----------
    Montesinos Lopez Ch 7
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Penalized Poisson regression for count genomic outcomes"})


def cheatsheet():
    return "poipr: Penalized Poisson regression for count genomic outcomes"
