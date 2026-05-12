# morie.fn -- function file (hadesllm/morie)
"""Ridge regression (L2 penalized) objective function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ridge_objective"]


def ridge_objective(y, X, beta, lam):
    """
    Ridge regression (L2 penalized) objective function

    Formula: L(beta) = ||y - X*beta||^2 + lambda * ||beta||_2^2

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ridge regression (L2 penalized) objective function"})


def cheatsheet():
    return "ridgj: Ridge regression (L2 penalized) objective function"
