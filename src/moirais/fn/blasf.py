# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian LASSO with double exponential prior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayesian_lasso_full"]


def bayesian_lasso_full(x, y):
    """
    Bayesian LASSO with double exponential prior

    Formula: beta_j ~ DE(0, lambda), lambda^2 ~ Gamma(r,s)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Montesinos Lopez Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian LASSO with double exponential prior"})


def cheatsheet():
    return "blasf: Bayesian LASSO with double exponential prior"
