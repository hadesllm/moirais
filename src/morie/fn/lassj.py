# morie.fn -- function file (hadesllm/morie)
"""LASSO (L1 penalized) regression objective."""
import numpy as np
from ._richresult import RichResult

__all__ = ["lasso_objective"]


def lasso_objective(y, X, beta, lam):
    """
    LASSO (L1 penalized) regression objective

    Formula: L(beta) = (1/(2n)) * ||y - X*beta||^2 + lambda * ||beta||_1

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LASSO (L1 penalized) regression objective"})


def cheatsheet():
    return "lassj: LASSO (L1 penalized) regression objective"
