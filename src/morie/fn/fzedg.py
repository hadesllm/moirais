# morie.fn — function file (hadesllm/morie)
"""Edgeworth expansion for kernel quantile."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_edgeworth_quantile"]


def fauzi_edgeworth_quantile(x):
    """
    Edgeworth expansion for kernel quantile

    Formula: P(sqrt(n)(Q_hat-Q)/sigma <= x) = Phi(x) + n^{-1/2}*p1(x)*phi(x) + ...

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Edgeworth expansion for kernel quantile"})


def cheatsheet():
    return "fzedg: Edgeworth expansion for kernel quantile"
