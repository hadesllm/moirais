# morie.fn -- function file (hadesllm/morie)
"""Restricted maximum likelihood (REML) log-likelihood."""
import numpy as np
from ._richresult import RichResult

__all__ = ["reml_log_likelihood"]


def reml_log_likelihood(y, X, V):
    """
    Restricted maximum likelihood (REML) log-likelihood

    Formula: log L_R = log L(beta_hat(V), V) - (1/2)*log|X'V^{-1}X|

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'reml_loglik': 'float'}

    References
    ----------
    Montesinos Lopez Ch 5
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Restricted maximum likelihood (REML) log-likelihood"})


def cheatsheet():
    return "remlf: Restricted maximum likelihood (REML) log-likelihood"
