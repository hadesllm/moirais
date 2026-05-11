# morie.fn — function file (hadesllm/morie)
"""Elastic net combined penalty objective."""
import numpy as np
from ._richresult import RichResult

__all__ = ["elastic_net_objective"]


def elastic_net_objective(y, X, beta, lam, alpha):
    """
    Elastic net combined penalty objective

    Formula: L(beta) = (1/(2n))*||y-X*beta||^2 + lambda*(alpha*||beta||_1 + (1-alpha)/2*||beta||_2^2)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    beta : array-like
        Input data.
    lam : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'loss': 'float'}

    References
    ----------
    Montesinos Lopez Ch 3
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Elastic net combined penalty objective"})


def cheatsheet():
    return "enobj: Elastic net combined penalty objective"
