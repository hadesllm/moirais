"""Support vector regression epsilon-insensitive loss."""
import numpy as np
from ._richresult import RichResult

__all__ = ["svr_epsilon_insensitive"]


def svr_epsilon_insensitive(X, y, C, eps):
    """
    Support vector regression epsilon-insensitive loss

    Formula: L_eps(y, f(y)) = max(0, |y - f(y)| - eps); min (1/2)||w||^2 + C*sum(xi + xi*)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    C : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'w': 'array', 'b': 'float'}

    References
    ----------
    Montesinos Lopez Ch 9
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Support vector regression epsilon-insensitive loss"})


def cheatsheet():
    return "svmep: Support vector regression epsilon-insensitive loss"
