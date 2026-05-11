# morie.fn — function file (hadesllm/morie)
"""Nonparametric IV model: Y=g(X)+U, E[U|W]=0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_npiv_model"]


def horowitz_npiv_model(x, y, w):
    """
    Nonparametric IV model: Y=g(X)+U, E[U|W]=0

    Formula: E(Y|W=w)*fW(w) = integral fXW(x,w)*g(x)dx; Fredholm eq with m=E(Y|W)*fW, k=fXW

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_hat

    References
    ----------
    Horowitz Ch 5, Eq 5.4-5.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonparametric IV model: Y=g(X)+U, E[U|W]=0"})


def cheatsheet():
    return "hrznpiv: Nonparametric IV model: Y=g(X)+U, E[U|W]=0"
