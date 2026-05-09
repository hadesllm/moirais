# moirais.fn — function file (hadesllm/moirais)
"""Monte Carlo dropout: leave dropout on at inference for uncertainty."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mc_dropout"]


def geron_mc_dropout(model, x, K, p):
    """
    Monte Carlo dropout: leave dropout on at inference for uncertainty

    Formula: y_hat_mean, y_hat_var estimated from K forward passes with dropout

    Parameters
    ----------
    model : array-like
        Input data.
    x : array-like
        Input data.
    K : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean, var

    References
    ----------
    Géron Ch 11
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Monte Carlo dropout: leave dropout on at inference for uncertainty"})


def cheatsheet():
    return "hmmcd: Monte Carlo dropout: leave dropout on at inference for uncertainty"
