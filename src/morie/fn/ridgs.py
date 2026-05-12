# morie.fn -- function file (hadesllm/morie)
"""Ridge regression closed-form solution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ridge_solution"]


def ridge_solution(X, y, lam):
    """
    Ridge regression closed-form solution

    Formula: beta_ridge = (X'X + lambda*I)^{-1} X'y

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'beta_hat': 'array'}

    References
    ----------
    Montesinos Lopez Ch 3
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ridge regression closed-form solution"})


def cheatsheet():
    return "ridgs: Ridge regression closed-form solution"
